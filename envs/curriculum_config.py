# envs/curriculum_config.py
"""
Curriculum Configuration for Trading RL Agent
==============================================

Defines curriculum progression from early learning to live-ready discipline.

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

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Callable, Dict, List, Optional, Tuple, Set


# =============================================================================
# CANONICAL METRIC REGISTRY (Split Namespaces)
# =============================================================================
# Separates OBSERVED METRICS (what evaluators produce) from THRESHOLD FIELDS
# (what CompetenceThresholds contains). This prevents silent failures from
# metric name mismatches.

# ---- Namespace 1: Observed Metrics ----
# These are the keys produced by evaluation (RollingStats, composite scoring)
OBSERVED_METRICS: Set[str] = {
    "win_rate",
    "profit_factor",
    "avg_pnl",
    "r_multiple",
    "max_drawdown",
    "dd_breach_rate",
    "consecutive_loss_rate",
    "win_rate_std",
    "pnl_std",
    "trade_count_avg",
    "entropy",
    "consistency",
    "trade_activity",
}

# Aliases within observed metrics namespace (observed -> observed canonical)
OBSERVED_ALIASES: Dict[str, str] = {
    "drawdown": "max_drawdown",
    "max_avg_drawdown": "max_drawdown",
    "dd": "max_drawdown",
    "pf": "profit_factor",
    "winrate": "win_rate",
    "r_mult": "r_multiple",
}

# ---- Namespace 2: Threshold Fields ----
# These are the field names in CompetenceThresholds
THRESHOLD_FIELDS: Set[str] = {
    "min_win_rate",
    "min_profit_factor",
    "min_avg_pnl",
    "min_avg_r_multiple",
    "max_avg_drawdown",
    "max_dd_breach_rate",
    "max_consecutive_loss_rate",
    "max_win_rate_std",
    "max_pnl_std",
    "min_trade_count_avg",
    "min_entropy",
}

# Aliases within threshold fields namespace (threshold -> threshold canonical)
THRESHOLD_ALIASES: Dict[str, str] = {
    "max_drawdown": "max_avg_drawdown",
    "min_r_multiple": "min_avg_r_multiple",
}

# ---- Bridge: Observed Metric -> Threshold Field ----
# Maps observed metric keys to corresponding threshold field names
METRIC_TO_THRESHOLD: Dict[str, str] = {
    "win_rate": "min_win_rate",
    "profit_factor": "min_profit_factor",
    "avg_pnl": "min_avg_pnl",
    "r_multiple": "min_avg_r_multiple",
    "max_drawdown": "max_avg_drawdown",
    "dd_breach_rate": "max_dd_breach_rate",
    "consecutive_loss_rate": "max_consecutive_loss_rate",
    "win_rate_std": "max_win_rate_std",
    "pnl_std": "max_pnl_std",
    "trade_count_avg": "min_trade_count_avg",
    "entropy": "min_entropy",
}

# ---- Composite Scoring Component Keys ----
# These are the exact keys produced by compute_composite_score() in curriculum_manager.py.
# CompositeScoringConfig.weights and .hard_floors MUST use only these keys.
COMPOSITE_WEIGHT_KEYS: Set[str] = {
    "win_rate",
    "profit_factor",
    "drawdown",
    "consistency",
    "r_multiple",
    "dd_breach_rate",
    "trade_activity",
    "consecutive_loss_rate",
}

COMPOSITE_HARD_FLOOR_KEYS: Set[str] = {
    "win_rate",
    "max_drawdown",
    "dd_breach_rate",
    "profit_factor",
    "r_multiple",
}

# Legacy: Combined set for backward compatibility
METRIC_CANONICAL_NAMES: Set[str] = OBSERVED_METRICS | THRESHOLD_FIELDS


def canonicalize_observed_metric(name: str) -> str:
    """
    Canonicalize an observed metric name.
    Always lowercases and applies observed aliases.
    """
    lower = name.lower()
    return OBSERVED_ALIASES.get(lower, lower)


def canonicalize_threshold_field(name: str) -> str:
    """
    Canonicalize a threshold field name.
    Always lowercases and applies threshold aliases.
    """
    lower = name.lower()
    return THRESHOLD_ALIASES.get(lower, lower)


def canonicalize_metric(name: str) -> str:
    """
    DEPRECATED: Use canonicalize_observed_metric or canonicalize_threshold_field.
    
    Legacy function that tries to canonicalize in observed namespace first,
    then threshold namespace. Always lowercases for case-insensitive matching.
    """
    lower = name.lower()
    # Try observed first
    if lower in OBSERVED_METRICS or lower in OBSERVED_ALIASES:
        return canonicalize_observed_metric(lower)
    # Then threshold
    if lower in THRESHOLD_FIELDS or lower in THRESHOLD_ALIASES:
        return canonicalize_threshold_field(lower)
    # Unknown metric - return lowercased
    return lower


def is_valid_observed_metric(name: str) -> bool:
    """Check if name is a valid observed metric (after canonicalization)."""
    canonical = canonicalize_observed_metric(name)
    return canonical in OBSERVED_METRICS


def is_valid_threshold_field(name: str) -> bool:
    """Check if name is a valid threshold field (after canonicalization)."""
    canonical = canonicalize_threshold_field(name)
    return canonical in THRESHOLD_FIELDS


def get_threshold_for_metric(metric: str) -> Optional[str]:
    """
    Get the threshold field name for an observed metric.
    Returns None if no mapping exists.
    """
    canonical = canonicalize_observed_metric(metric)
    return METRIC_TO_THRESHOLD.get(canonical)


class CurriculumStage(IntEnum):
    """
    10-Stage Curriculum: "First Grade to University"
    
    PHASE 0: DISCOVERY (Stages 0-1) - Pure exploration, learn market patterns
    PHASE 1: FOUNDATION (Stages 2-4) - One concept per stage
    PHASE 2: DEVELOPMENT (Stages 5-7) - Combine skills into strategies
    PHASE 3: MASTERY (Stages 8-9) - Prop firm constraints, live-ready
    """
    # Phase 0: DISCOVERY - "Kindergarten"
    EXPLORER = 0           # Pure observation, no penalties
    EXPERIMENTER = 1       # Light outcome signals
    
    # Phase 1: FOUNDATION - "Elementary School"
    TREND_STUDENT = 2      # Learn trend alignment
    SESSION_STUDENT = 3    # Learn session awareness
    TIMING_STUDENT = 4     # Learn entry quality
    
    # Phase 2: DEVELOPMENT - "High School"
    INTEGRATOR = 5         # Combine trend + session + entry
    RISK_MANAGER = 6       # Add risk control
    STRATEGIST = 7         # Full strategy formation
    
    # Phase 3: MASTERY - "University"
    PROFESSIONAL = 8       # Prop firm constraints
    LIVE_READY = 9         # Live execution robustness


class TradingSkill(Enum):
    """Decomposed trading competencies for granular assessment."""
    ENTRY_TIMING = "entry_timing"           # Enters at good prices
    EXIT_QUALITY = "exit_quality"           # Trailing stops > hard stops
    DRAWDOWN_CONTROL = "drawdown_control"   # Stays within limits
    POSITION_SIZING = "position_sizing"     # Uses appropriate size
    PATIENCE = "patience"                   # Doesn't overtrade
    TREND_ALIGNMENT = "trend_alignment"     # Trades with trend
    RISK_REWARD = "risk_reward"             # Good R-multiples
    CONSISTENCY = "consistency"             # Low variance
    LOSS_MANAGEMENT = "loss_management"     # Handles losing streaks
    ADAPTATION = "adaptation"               # Adjusts to market regimes


class MarketRegime(Enum):
    """Market regime classification for validation."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class ExecutionDifficulty:
    base_spread_points: float = 0.05
    spread_mult_range: Tuple[float, float] = (0.9, 1.1)
    max_spread_points: float = 0.5

    slippage_points_sigma: float = 0.0
    slippage_mult_range: Tuple[float, float] = (0.5, 1.5)
    max_slippage_points: float = 0.1

    commission_per_lot: float = 0.0
    latency_bars: int = 0

    enable_randomization: bool = False
    spread_randomization_range: Tuple[float, float] = (0.95, 1.05)
    slippage_randomization_range: Tuple[float, float] = (0.95, 1.05)
    latency_randomization_range: Tuple[int, int] = (0, 0)
    volatility_scale_range: Tuple[float, float] = (1.0, 1.0)
    
    # Spread shock events (for live-robustness testing)
    spread_shock_enabled: bool = False
    spread_shock_probability: float = 0.02  # 2% of steps
    spread_shock_multiplier: float = 3.0    # 3x normal spread during shock


@dataclass
class DataDifficulty:
    """
    Data difficulty settings for curriculum-based data filtering.
    
    Allows early stages to train on "easier" market conditions
    (clear trends, lower volatility) before introducing complex regimes.
    """
    volatility_percentile_range: Tuple[float, float] = (0.0, 1.0)
    min_trend_clarity: float = 0.0
    
    include_asian_session: bool = True
    include_london_session: bool = True
    include_ny_session: bool = True
    include_overlap_sessions: bool = True
    
    exclude_high_impact_news: bool = False
    exclude_market_open_close: bool = False
    
    prefer_recent_data: bool = False
    recent_data_weight: float = 1.0


@dataclass
class TransitionSettings:
    """
    Settings for smooth stage transitions.
    
    Prevents sudden destabilization when moving to harder stages.
    """
    lr_warmup_enabled: bool = True
    lr_warmup_factor: float = 0.3
    lr_warmup_steps: int = 10_000
    
    reward_blend_enabled: bool = True
    reward_blend_episodes: int = 20
    
    checkpoint_on_transition: bool = True
    transition_cooldown_episodes: int = 50


@dataclass
class RewardShaping:
    reward_scale: float = 10.0
    loss_multiplier: float = 1.0

    r_multiple_bonus_threshold: float = 1.5
    r_multiple_bonus_scale: float = 0.3
    r_multiple_bonus_cap: float = 0.6

    mae_efficiency_enabled: bool = False
    mae_efficiency_scale: float = 0.25
    mae_efficiency_threshold: float = 2.0

    time_efficiency_enabled: bool = False
    time_efficiency_scale: float = 0.15
    optimal_trade_bars: int = 8
    max_trade_bars_for_bonus: int = 24

    exit_quality_enabled: bool = False
    trailing_stop_bonus: float = 0.15
    agent_close_bonus: float = 0.05
    hard_stop_penalty: float = 0.15
    risk_liquidation_penalty: float = 0.30

    truncation_winner_discount: float = 0.30
    truncation_loser_extra_penalty: float = 0.15

    entry_quality_integration: bool = False
    entry_quality_weight: float = 0.2

    dd_shaping_enabled: bool = False
    dd_threshold: float = 0.02
    dd_penalty_scale: float = 1.0
    dd_severity_exponent: float = 1.5
    dd_severity_cap: float = 1.5

    streak_modifier_enabled: bool = False
    win_streak_bonus_per_win: float = 0.02
    loss_streak_penalty_per_loss: float = 0.03

    anti_churn_enabled: bool = False
    daily_trade_soft_limit: int = 20
    churn_penalty_per_trade: float = 0.02

    hard_block_penalty: float = 0.02
    soft_block_penalty: float = 0.01

    per_step_shaping_enabled: bool = False
    holding_cost_per_bar: float = 0.0
    opportunity_bonus_scale: float = 0.0

    min_reward: float = -5.0
    max_reward: float = 5.0

    exploration_bonus: float = 0.0
    directional_accuracy_weight: float = 1.0


@dataclass
class TradingConstraints:
    max_positions: int = 1

    max_trades_per_day: int = 100
    max_trades_per_session: int = 50
    max_consecutive_losses: int = 10

    enforce_session_windows: bool = False
    enforce_no_new_trades_window: bool = False
    enforce_weekend_block: bool = False
    enforce_hard_close: bool = False
    min_minutes_between_entries: int = 0
    min_minutes_after_loss: int = 0

    daily_drawdown_limit: float = 1.0
    max_drawdown_limit: float = 1.0
    daily_dd_safety_buffer: float = 0.0
    max_dd_safety_buffer: float = 0.0
    emergency_close_threshold: float = 1.0

    entry_quality_gate_enabled: bool = False
    entry_quality_threshold: float = 0.0

    hard_stop_loss_eur: float = 10000.0
    soft_stop_loss_eur: float = 10000.0
    trailing_activation_eur: float = 10000.0
    trailing_retrace_pct: float = 0.50
    time_decay_hours: float = 24.0

    risk_per_trade_pct: float = 0.01
    max_risk_per_trade_pct: float = 0.02


MIN_EVALUATION_EPISODES = 25


@dataclass
class CompetenceThresholds:
    min_episodes: int = 100
    min_timesteps: int = 50_000

    min_win_rate: float = 0.40
    min_profit_factor: float = 0.8
    max_avg_drawdown: float = 0.20
    min_avg_pnl: float = -1000.0
    
    min_avg_r_multiple: float = 0.0
    min_entropy: float = 0.0

    max_win_rate_std: float = 0.30
    max_pnl_std: float = 10000.0
    min_trade_count_avg: float = 1.0

    max_dd_breach_rate: float = 0.50
    max_consecutive_loss_rate: float = 0.30

    evaluation_window: int = 50


@dataclass
class SkillRequirements:
    """
    Per-stage skill requirements for promotion.
    
    Maps skills to minimum scores [0, 1] required to pass.
    """
    required_skills: Dict[TradingSkill, float] = field(default_factory=dict)
    
    # Minimum confidence required for skill assessment to count
    min_confidence: float = 0.5
    
    # Whether all skills must pass or just weighted average
    require_all_skills: bool = False
    weighted_threshold: float = 0.6  # If not require_all_skills, weighted avg must exceed this
    
    # Skill weights for weighted average (default equal weights)
    skill_weights: Dict[TradingSkill, float] = field(default_factory=dict)
    
    def get_weight(self, skill: TradingSkill) -> float:
        """Get weight for a skill, defaulting to 1.0."""
        return self.skill_weights.get(skill, 1.0)


@dataclass
class EntropyTargets:
    """
    Entropy targets for exploration management.
    
    Prevents policy collapse (too low entropy) or random behavior (too high).
    """
    min_entropy: float = 0.1
    max_entropy: float = 0.8
    
    # Penalty coefficient when outside range
    low_entropy_penalty_scale: float = 0.1
    high_entropy_penalty_scale: float = 0.05
    
    # Whether to use entropy in promotion criteria
    use_in_promotion: bool = True


@dataclass
class CompositeScoringConfig:
    """
    Configuration for weighted composite competence scoring.
    
    Allows nuanced evaluation rather than all-or-nothing gating.
    
    IMPORTANT: The weight keys and hard_floor keys must match the component
    names produced by compute_composite_score() in curriculum_manager.py:
        - "win_rate", "profit_factor", "drawdown", "consistency",
        - "r_multiple", "dd_breach_rate", "trade_activity", "consecutive_loss_rate"
    """
    enabled: bool = True
    
    # Weights for composite score (must sum to ~1.0)
    weights: Dict[str, float] = field(default_factory=lambda: {
        "win_rate": 0.20,
        "profit_factor": 0.20,
        "drawdown": 0.15,
        "consistency": 0.15,
        "r_multiple": 0.10,
        "dd_breach_rate": 0.10,
        "trade_activity": 0.05,
        "consecutive_loss_rate": 0.05,
    })
    
    # Hard floors: must meet regardless of composite score
    hard_floors: Dict[str, float] = field(default_factory=lambda: {
        "win_rate": 0.30,
        "max_drawdown": 0.25,
        "dd_breach_rate": 0.40,
    })
    
    # Threshold composite score must exceed for promotion
    promotion_threshold: float = 0.70
    
    # Threshold below which demotion is triggered
    demotion_threshold: float = 0.35


@dataclass
class AdaptiveThresholdConfig:
    """
    Configuration for adaptive threshold relaxation.
    
    Slightly relaxes thresholds if agent is plateaued but close to promotion.
    """
    enabled: bool = True
    
    # Plateau detection
    plateau_episodes_threshold: int = 100  # Episodes without improvement
    plateau_improvement_threshold: float = 0.01  # Min improvement to not count as plateau
    
    # Maximum relaxation allowed (as fraction)
    max_relaxation: float = 0.10  # Up to 10% relaxation
    
    # Episodes over which relaxation builds up
    relaxation_buildup_episodes: int = 500
    
    # Which metrics can be relaxed (safety metrics excluded)
    relaxable_metrics: Set[str] = field(default_factory=lambda: {
        "min_win_rate",
        "min_profit_factor",
        "min_avg_pnl",
        "min_trade_count_avg",
        "max_win_rate_std",
        "max_pnl_std",
    })
    
    # Metrics that should NEVER be relaxed (safety-critical)
    never_relax: Set[str] = field(default_factory=lambda: {
        "max_avg_drawdown",
        "max_dd_breach_rate",
        "max_consecutive_loss_rate",
    })


@dataclass
class RecoveryProtocolConfig:
    """
    Configuration for recovery protocol after repeated failures.
    """
    enabled: bool = True
    
    # Trigger after this many demotions from the same stage
    trigger_after_demotions: int = 2
    
    # Duration of recovery protocol in episodes
    recovery_duration_episodes: int = 200
    
    # Focus skill override (if None, determined from diagnosis)
    focus_skill: Optional[TradingSkill] = None
    
    # Reward modifications during recovery
    reward_modifications: Dict[str, float] = field(default_factory=dict)
    
    # Constraint modifications during recovery (stricter limits)
    constraint_modifications: Dict[str, float] = field(default_factory=dict)


@dataclass
class MixedStageSamplingConfig:
    """
    Configuration for training on mixture of stages.
    
    Prevents catastrophic forgetting by occasionally training on earlier stages.
    """
    enabled: bool = True
    
    # Weight for current stage
    current_stage_weight: float = 0.70
    
    # Weight for recent stages (1-2 stages back)
    recent_stages_weight: float = 0.20
    
    # Weight for foundation stage (always some basics)
    foundation_weight: float = 0.10
    
    # How many stages back to sample from
    recent_stage_depth: int = 2


@dataclass
class ReviewSessionConfig:
    """
    Configuration for periodic review sessions on earlier stages.
    
    Ensures agent hasn't forgotten earlier skills.
    """
    enabled: bool = True
    
    # Episodes between review sessions
    review_frequency: int = 500
    
    # Episodes per review session
    review_duration: int = 50
    
    # How many stages back to review
    review_depth: int = 2
    
    # Minimum stage to trigger reviews (no reviews in early discovery)
    min_stage_for_review: CurriculumStage = CurriculumStage.INTEGRATOR


@dataclass
class ValidationConfig:
    """
    Configuration for hold-out validation before promotion.
    """
    enabled: bool = False  # Disabled by default (requires separate validation data)
    
    # Number of validation episodes
    validation_episodes: int = 100
    
    # Minimum performance ratio (validation / training)
    min_performance_ratio: float = 0.85
    
    # Maximum acceptable performance drop
    max_performance_drop: float = 0.15
    
    # Required regimes to validate on
    required_regimes: List[MarketRegime] = field(default_factory=lambda: [
        MarketRegime.TRENDING_UP,
        MarketRegime.TRENDING_DOWN,
        MarketRegime.RANGING,
    ])
    
    # Minimum episodes per regime
    min_episodes_per_regime: int = 20


@dataclass
class CurriculumStageConfig:
    stage: CurriculumStage
    name: str
    description: str

    execution: ExecutionDifficulty
    rewards: RewardShaping
    constraints: TradingConstraints
    competence: CompetenceThresholds

    max_steps_per_episode: int = 2000

    include_memory_features: bool = True
    include_world_model_features: bool = True
    include_expert_signals: bool = True
    expert_signal_dropout: float = 0.0  # Probability of dropping expert signals (0.0 = never, 1.0 = always)

    allow_demotion: bool = False
    is_terminal: bool = False
    
    data_difficulty: DataDifficulty = field(default_factory=DataDifficulty)
    transition: TransitionSettings = field(default_factory=TransitionSettings)
    
    # New v2.0 configurations
    skill_requirements: SkillRequirements = field(default_factory=SkillRequirements)
    entropy_targets: EntropyTargets = field(default_factory=EntropyTargets)
    composite_scoring: CompositeScoringConfig = field(default_factory=CompositeScoringConfig)
    adaptive_thresholds: AdaptiveThresholdConfig = field(default_factory=AdaptiveThresholdConfig)
    recovery_protocol: RecoveryProtocolConfig = field(default_factory=RecoveryProtocolConfig)
    mixed_stage_sampling: MixedStageSamplingConfig = field(default_factory=MixedStageSamplingConfig)
    review_session: ReviewSessionConfig = field(default_factory=ReviewSessionConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)


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
        ),
        rewards=RewardShaping(
            # ============================================================================
            # PHASE 0 DISCOVERY: Pure exploration, NO penalties
            # ============================================================================
            reward_scale=3.0,                 # Low scale - outcomes don't matter much
            loss_multiplier=1.0,
            
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
            
            # ANTI-CHURN: DISABLED - let agent trade freely
            anti_churn_enabled=False,
            daily_trade_soft_limit=100,
            churn_penalty_per_trade=0.0,
            
            hard_block_penalty=0.0,
            soft_block_penalty=0.0,
            per_step_shaping_enabled=False,
            holding_cost_per_bar=0.0,
            
            # EXPLORATION BONUS: HIGH - encourage trying everything
            exploration_bonus=0.05,
            directional_accuracy_weight=0.5,  # Light directional signal
            min_reward=-1.5,
            max_reward=1.5,
        ),
        constraints=TradingConstraints(
            max_positions=1,
            max_trades_per_day=50,            # Very permissive
            max_trades_per_session=25,
            max_consecutive_losses=20,        # Very permissive
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
            entry_quality_gate_enabled=False,
            entry_quality_threshold=0.0,
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
        ),
        rewards=RewardShaping(
            # ============================================================================
            # PHASE 0 DISCOVERY: Light outcome signals
            # ============================================================================
            reward_scale=4.0,                 # Slightly higher than Stage 0
            loss_multiplier=1.0,
            
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
            
            # ANTI-CHURN: Still disabled
            anti_churn_enabled=False,
            daily_trade_soft_limit=50,
            churn_penalty_per_trade=0.0,
            
            hard_block_penalty=0.0,
            soft_block_penalty=0.0,
            per_step_shaping_enabled=False,
            holding_cost_per_bar=0.0,
            
            # EXPLORATION BONUS: Still high
            exploration_bonus=0.04,
            directional_accuracy_weight=0.7,
            min_reward=-2.0,
            max_reward=2.0,
        ),
        constraints=TradingConstraints(
            max_positions=1,
            max_trades_per_day=40,
            max_trades_per_session=20,
            max_consecutive_losses=15,
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
            entry_quality_gate_enabled=False,
            entry_quality_threshold=0.0,
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
        ),
        rewards=RewardShaping(
            # ============================================================================
            # PHASE 1 FOUNDATION: Learn trend alignment
            # ============================================================================
            reward_scale=5.0,
            loss_multiplier=1.0,
            
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
            
            exit_quality_enabled=False,
            trailing_stop_bonus=0.0,
            agent_close_bonus=0.0,
            hard_stop_penalty=0.0,
            risk_liquidation_penalty=0.0,
            
            truncation_winner_discount=0.20,
            truncation_loser_extra_penalty=0.10,
            
            entry_quality_integration=False,
            entry_quality_weight=0.0,
            
            # DD SHAPING: Light
            dd_shaping_enabled=True,
            dd_threshold=0.10,
            dd_penalty_scale=0.3,
            dd_severity_exponent=1.1,
            dd_severity_cap=0.5,
            
            streak_modifier_enabled=False,
            win_streak_bonus_per_win=0.0,
            loss_streak_penalty_per_loss=0.0,
            
            # ANTI-CHURN: Light
            anti_churn_enabled=True,
            daily_trade_soft_limit=3,
            churn_penalty_per_trade=0.03,
            
            hard_block_penalty=0.01,
            soft_block_penalty=0.005,
            per_step_shaping_enabled=False,
            holding_cost_per_bar=0.0,
            
            # EXPLORATION: Still moderate
            exploration_bonus=0.03,
            directional_accuracy_weight=1.2,  # INCREASED - reward trend alignment
            min_reward=-2.5,
            max_reward=2.5,
        ),
        constraints=TradingConstraints(
            max_positions=1,
            max_trades_per_day=25,
            max_trades_per_session=12,
            max_consecutive_losses=10,
            enforce_session_windows=False,
            enforce_no_new_trades_window=False,
            enforce_weekend_block=False,
            enforce_hard_close=False,
            min_minutes_between_entries=1,
            min_minutes_after_loss=2,
            daily_drawdown_limit=0.25,
            max_drawdown_limit=0.30,
            daily_dd_safety_buffer=0.0,
            max_dd_safety_buffer=0.0,
            emergency_close_threshold=0.25,
            entry_quality_gate_enabled=False,
            entry_quality_threshold=0.0,
            hard_stop_loss_eur=600.0,
            soft_stop_loss_eur=400.0,
            trailing_activation_eur=120.0,
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
        ),
        rewards=RewardShaping(
            reward_scale=5.5,
            loss_multiplier=1.0,
            
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
            # AUDIT FIX: Stop rewarding premature agent_close exits
            # Strongly prefer trailing stops, penalize discretionary exits and hard stops
            exit_quality_enabled=True,
            trailing_stop_bonus=0.08,    # Strong reward for trailing stop exits
            agent_close_bonus=0.00,      # NEUTRAL - don't reward "I got scared"
            hard_stop_penalty=0.05,      # Meaningful penalty for hitting stop loss
            risk_liquidation_penalty=0.10,  # Strong penalty for risk liquidation
            
            truncation_winner_discount=0.20,
            truncation_loser_extra_penalty=0.10,
            
            entry_quality_integration=False,
            entry_quality_weight=0.0,
            
            dd_shaping_enabled=True,
            dd_threshold=0.08,
            dd_penalty_scale=0.4,
            dd_severity_exponent=1.2,
            dd_severity_cap=0.6,
            
            streak_modifier_enabled=False,
            win_streak_bonus_per_win=0.0,
            loss_streak_penalty_per_loss=0.0,
            
            anti_churn_enabled=True,
            daily_trade_soft_limit=6,    # AUDIT FIX: 2 was too aggressive, causing constant penalty
            churn_penalty_per_trade=0.06,  # Keep pressure on overtrading
            
            hard_block_penalty=0.015,
            soft_block_penalty=0.008,
            per_step_shaping_enabled=False,
            holding_cost_per_bar=0.0,
            
            exploration_bonus=0.025,
            directional_accuracy_weight=1.15,
            min_reward=-2.8,
            max_reward=2.8,
        ),
        constraints=TradingConstraints(
            max_positions=1,
            max_trades_per_day=20,
            max_trades_per_session=10,
            max_consecutive_losses=8,
            enforce_session_windows=True,      # KEY: Session awareness
            enforce_no_new_trades_window=True,
            enforce_weekend_block=True,
            enforce_hard_close=False,
            min_minutes_between_entries=30,   # AUDIT FIX: 2 bars minimum (was 2 min = <1 bar)
            min_minutes_after_loss=45,         # AUDIT FIX: 3 bars after loss (was 3 min = <1 bar)
            daily_drawdown_limit=0.20,
            max_drawdown_limit=0.25,
            daily_dd_safety_buffer=0.0,
            max_dd_safety_buffer=0.0,
            emergency_close_threshold=0.22,
            entry_quality_gate_enabled=False,
            entry_quality_threshold=0.0,
            hard_stop_loss_eur=500.0,
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
        ),
        rewards=RewardShaping(
            reward_scale=6.0,
            loss_multiplier=1.0,
            
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
            trailing_stop_bonus=0.05,
            agent_close_bonus=0.02,
            hard_stop_penalty=0.02,
            risk_liquidation_penalty=0.04,
            
            truncation_winner_discount=0.22,
            truncation_loser_extra_penalty=0.12,
            
            # ENTRY QUALITY: NEW
            entry_quality_integration=True,
            entry_quality_weight=0.10,
            
            dd_shaping_enabled=True,
            dd_threshold=0.06,
            dd_penalty_scale=0.5,
            dd_severity_exponent=1.3,
            dd_severity_cap=0.8,
            
            streak_modifier_enabled=False,
            win_streak_bonus_per_win=0.0,
            loss_streak_penalty_per_loss=0.0,
            
            anti_churn_enabled=True,
            daily_trade_soft_limit=2,
            churn_penalty_per_trade=0.06,
            
            hard_block_penalty=0.02,
            soft_block_penalty=0.01,
            per_step_shaping_enabled=False,
            holding_cost_per_bar=0.0,
            
            exploration_bonus=0.02,
            directional_accuracy_weight=1.1,
            min_reward=-3.0,
            max_reward=3.0,
        ),
        constraints=TradingConstraints(
            max_positions=1,
            max_trades_per_day=18,
            max_trades_per_session=9,
            max_consecutive_losses=7,
            enforce_session_windows=True,
            enforce_no_new_trades_window=True,
            enforce_weekend_block=True,
            enforce_hard_close=True,
            min_minutes_between_entries=3,
            min_minutes_after_loss=5,
            daily_drawdown_limit=0.15,
            max_drawdown_limit=0.20,
            daily_dd_safety_buffer=0.0,
            max_dd_safety_buffer=0.0,
            emergency_close_threshold=0.18,
            entry_quality_gate_enabled=True,
            entry_quality_threshold=0.25,
            hard_stop_loss_eur=400.0,
            soft_stop_loss_eur=280.0,
            trailing_activation_eur=90.0,
            trailing_retrace_pct=0.38,
            time_decay_hours=8.0,
            risk_per_trade_pct=0.005,
            max_risk_per_trade_pct=0.01,
        ),
        competence=CompetenceThresholds(
            min_episodes=300,
            min_timesteps=500_000,
            min_win_rate=0.40,
            min_profit_factor=0.90,
            max_avg_drawdown=0.15,
            min_avg_pnl=-100.0,
            min_avg_r_multiple=0.0,
            min_entropy=0.30,  # AUDIT FIX: Must match entropy_targets.min_entropy
            max_win_rate_std=0.25,
            max_pnl_std=8000.0,
            min_trade_count_avg=4.0,
            max_dd_breach_rate=0.18,
            max_consecutive_loss_rate=0.20,
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
            trailing_stop_bonus=0.08,
            agent_close_bonus=0.03,
            hard_stop_penalty=0.03,
            risk_liquidation_penalty=0.06,
            
            truncation_winner_discount=0.22,
            truncation_loser_extra_penalty=0.12,
            
            entry_quality_integration=True,
            entry_quality_weight=0.15,
            
            dd_shaping_enabled=True,
            dd_threshold=0.05,
            dd_penalty_scale=0.6,
            dd_severity_exponent=1.35,
            dd_severity_cap=0.9,
            
            streak_modifier_enabled=False,
            win_streak_bonus_per_win=0.0,
            loss_streak_penalty_per_loss=0.0,
            
            anti_churn_enabled=True,
            daily_trade_soft_limit=1,
            churn_penalty_per_trade=0.08,
            
            hard_block_penalty=0.025,
            soft_block_penalty=0.012,
            per_step_shaping_enabled=False,
            holding_cost_per_bar=0.0,
            
            exploration_bonus=0.015,
            directional_accuracy_weight=1.05,
            min_reward=-3.5,
            max_reward=3.5,
        ),
        constraints=TradingConstraints(
            max_positions=1,
            max_trades_per_day=14,
            max_trades_per_session=7,
            max_consecutive_losses=6,
            enforce_session_windows=True,
            enforce_no_new_trades_window=True,
            enforce_weekend_block=True,
            enforce_hard_close=True,
            min_minutes_between_entries=5,
            min_minutes_after_loss=8,
            daily_drawdown_limit=0.10,
            max_drawdown_limit=0.15,
            daily_dd_safety_buffer=0.005,
            max_dd_safety_buffer=0.01,
            emergency_close_threshold=0.13,
            entry_quality_gate_enabled=True,
            entry_quality_threshold=0.35,
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
            trailing_stop_bonus=0.10,
            agent_close_bonus=0.04,
            hard_stop_penalty=0.04,
            risk_liquidation_penalty=0.08,
            
            truncation_winner_discount=0.25,
            truncation_loser_extra_penalty=0.15,
            
            entry_quality_integration=True,
            entry_quality_weight=0.18,
            
            # DD SHAPING: Strong
            dd_shaping_enabled=True,
            dd_threshold=0.04,
            dd_penalty_scale=0.8,
            dd_severity_exponent=1.45,
            dd_severity_cap=1.2,
            
            # STREAKS: NEW
            streak_modifier_enabled=True,
            win_streak_bonus_per_win=0.02,
            loss_streak_penalty_per_loss=0.03,
            
            anti_churn_enabled=True,
            daily_trade_soft_limit=1,
            churn_penalty_per_trade=0.10,
            
            hard_block_penalty=0.03,
            soft_block_penalty=0.015,
            per_step_shaping_enabled=False,
            holding_cost_per_bar=0.0,
            
            exploration_bonus=0.01,
            directional_accuracy_weight=1.0,
            min_reward=-4.0,
            max_reward=4.0,
        ),
        constraints=TradingConstraints(
            max_positions=1,
            max_trades_per_day=12,
            max_trades_per_session=6,
            max_consecutive_losses=5,
            enforce_session_windows=True,
            enforce_no_new_trades_window=True,
            enforce_weekend_block=True,
            enforce_hard_close=True,
            min_minutes_between_entries=8,
            min_minutes_after_loss=12,
            daily_drawdown_limit=0.06,
            max_drawdown_limit=0.10,
            daily_dd_safety_buffer=0.006,
            max_dd_safety_buffer=0.01,
            emergency_close_threshold=0.09,
            entry_quality_gate_enabled=True,
            entry_quality_threshold=0.40,
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
            max_dd_breach_rate=0.10,
            max_consecutive_loss_rate=0.15,
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
                TradingSkill.PATIENCE: 0.55,
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
            trailing_stop_bonus=0.12,
            agent_close_bonus=0.05,
            hard_stop_penalty=0.05,
            risk_liquidation_penalty=0.10,
            
            truncation_winner_discount=0.22,
            truncation_loser_extra_penalty=0.18,
            
            entry_quality_integration=True,
            entry_quality_weight=0.20,
            
            dd_shaping_enabled=True,
            dd_threshold=0.035,
            dd_penalty_scale=0.9,
            dd_severity_exponent=1.5,
            dd_severity_cap=1.3,
            
            streak_modifier_enabled=True,
            win_streak_bonus_per_win=0.025,
            loss_streak_penalty_per_loss=0.035,
            
            anti_churn_enabled=True,
            daily_trade_soft_limit=1,
            churn_penalty_per_trade=0.12,
            
            hard_block_penalty=0.035,
            soft_block_penalty=0.018,
            per_step_shaping_enabled=False,
            holding_cost_per_bar=0.0,
            
            exploration_bonus=0.005,
            directional_accuracy_weight=1.0,
            min_reward=-4.5,
            max_reward=4.5,
        ),
        constraints=TradingConstraints(
            max_positions=1,
            max_trades_per_day=10,
            max_trades_per_session=5,
            max_consecutive_losses=4,
            enforce_session_windows=True,
            enforce_no_new_trades_window=True,
            enforce_weekend_block=True,
            enforce_hard_close=True,
            min_minutes_between_entries=10,
            min_minutes_after_loss=15,
            daily_drawdown_limit=0.055,
            max_drawdown_limit=0.095,
            daily_dd_safety_buffer=0.007,
            max_dd_safety_buffer=0.012,
            emergency_close_threshold=0.085,
            entry_quality_gate_enabled=True,
            entry_quality_threshold=0.45,
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
            max_dd_breach_rate=0.08,
            max_consecutive_loss_rate=0.12,
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
                TradingSkill.PATIENCE: 0.58,
                TradingSkill.RISK_REWARD: 0.50,
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
            trailing_stop_bonus=0.15,
            agent_close_bonus=0.06,
            hard_stop_penalty=0.06,
            risk_liquidation_penalty=0.12,
            
            truncation_winner_discount=0.20,
            truncation_loser_extra_penalty=0.20,
            
            entry_quality_integration=True,
            entry_quality_weight=0.22,
            
            dd_shaping_enabled=True,
            dd_threshold=0.030,
            dd_penalty_scale=1.0,
            dd_severity_exponent=1.55,
            dd_severity_cap=1.35,
            
            streak_modifier_enabled=True,
            win_streak_bonus_per_win=0.03,
            loss_streak_penalty_per_loss=0.04,
            
            anti_churn_enabled=True,
            daily_trade_soft_limit=1,
            churn_penalty_per_trade=0.14,
            
            hard_block_penalty=0.04,
            soft_block_penalty=0.02,
            per_step_shaping_enabled=False,
            holding_cost_per_bar=0.0,
            
            exploration_bonus=0.0,
            directional_accuracy_weight=1.0,
            min_reward=-5.0,
            max_reward=5.0,
        ),
        constraints=TradingConstraints(
            max_positions=1,
            max_trades_per_day=8,
            max_trades_per_session=4,
            max_consecutive_losses=4,
            enforce_session_windows=True,
            enforce_no_new_trades_window=True,
            enforce_weekend_block=True,
            enforce_hard_close=True,
            min_minutes_between_entries=12,
            min_minutes_after_loss=18,
            daily_drawdown_limit=0.05,
            max_drawdown_limit=0.09,
            daily_dd_safety_buffer=0.008,
            max_dd_safety_buffer=0.015,
            emergency_close_threshold=0.08,
            entry_quality_gate_enabled=True,
            entry_quality_threshold=0.50,
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
            max_pnl_std=5000.0,
            min_trade_count_avg=5.0,
            max_dd_breach_rate=0.06,
            max_consecutive_loss_rate=0.10,
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
                TradingSkill.PATIENCE: 0.62,
                TradingSkill.RISK_REWARD: 0.55,
                TradingSkill.CONSISTENCY: 0.55,
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
            trailing_stop_bonus=0.18,
            agent_close_bonus=0.08,
            hard_stop_penalty=0.08,
            risk_liquidation_penalty=0.15,
            
            truncation_winner_discount=0.18,
            truncation_loser_extra_penalty=0.22,
            
            entry_quality_integration=True,
            entry_quality_weight=0.25,
            
            dd_shaping_enabled=True,
            dd_threshold=0.025,
            dd_penalty_scale=1.1,
            dd_severity_exponent=1.6,
            dd_severity_cap=1.4,
            
            streak_modifier_enabled=True,
            win_streak_bonus_per_win=0.035,
            loss_streak_penalty_per_loss=0.045,
            
            anti_churn_enabled=True,
            daily_trade_soft_limit=1,
            churn_penalty_per_trade=0.16,
            
            hard_block_penalty=0.045,
            soft_block_penalty=0.022,
            per_step_shaping_enabled=False,
            holding_cost_per_bar=0.0,
            
            exploration_bonus=0.0,
            directional_accuracy_weight=1.0,
            min_reward=-5.0,
            max_reward=5.0,
        ),
        constraints=TradingConstraints(
            max_positions=1,
            max_trades_per_day=7,
            max_trades_per_session=4,
            max_consecutive_losses=3,
            enforce_session_windows=True,
            enforce_no_new_trades_window=True,
            enforce_weekend_block=True,
            enforce_hard_close=True,
            min_minutes_between_entries=15,
            min_minutes_after_loss=20,
            daily_drawdown_limit=0.048,
            max_drawdown_limit=0.085,
            daily_dd_safety_buffer=0.008,
            max_dd_safety_buffer=0.015,
            emergency_close_threshold=0.075,
            entry_quality_gate_enabled=True,
            entry_quality_threshold=0.55,
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
            min_entropy=0.04,
            max_win_rate_std=0.12,
            max_pnl_std=4500.0,
            min_trade_count_avg=5.5,
            max_dd_breach_rate=0.05,
            max_consecutive_loss_rate=0.08,
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
                TradingSkill.PATIENCE: 0.68,
                TradingSkill.RISK_REWARD: 0.60,
                TradingSkill.CONSISTENCY: 0.65,
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
    
    # Validate composite scoring weights - keys must match compute_composite_score() components
    cs = cfg.composite_scoring
    if cs.enabled:
        weight_sum = sum(cs.weights.values())
        if abs(weight_sum - 1.0) > 0.1:
            issues.append(f"{cfg.stage.name}: composite scoring weights sum to {weight_sum:.2f}, expected ~1.0")
        
        # Validate weight keys against allowed composite weight keys
        bad_weight_keys = set(cs.weights.keys()) - COMPOSITE_WEIGHT_KEYS
        if bad_weight_keys:
            issues.append(f"{cfg.stage.name}: composite_scoring.weights has unknown keys: {sorted(bad_weight_keys)}. Allowed: {sorted(COMPOSITE_WEIGHT_KEYS)}")
        
        # Validate hard_floors keys against allowed hard floor keys
        bad_floor_keys = set(cs.hard_floors.keys()) - COMPOSITE_HARD_FLOOR_KEYS
        if bad_floor_keys:
            issues.append(f"{cfg.stage.name}: composite_scoring.hard_floors has unknown keys: {sorted(bad_floor_keys)}. Allowed: {sorted(COMPOSITE_HARD_FLOOR_KEYS)}")
    
    # Validate adaptive thresholds - keys must be valid threshold fields
    at = cfg.adaptive_thresholds
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
    if not cfg.include_expert_signals and cfg.expert_signal_dropout > 0.0:
        issues.append(f"{cfg.stage.name}: expert_signal_dropout={cfg.expert_signal_dropout} has no effect when include_expert_signals=False (suggest setting to 0.0)")
    
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