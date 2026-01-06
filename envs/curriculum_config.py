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
    FOUNDATION = 0
    DISCIPLINE = 1
    MARKET_STRUCTURE = 2
    ECONOMIC_LOGIC = 3
    PROFESSIONAL = 4
    ADAPTIVE = 5
    SPECIALIST = 6
    LIVE_READY = 7


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
    
    # Minimum stage to trigger reviews (no reviews in foundation)
    min_stage_for_review: CurriculumStage = CurriculumStage.MARKET_STRUCTURE


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
# 3-PHASE PROP FIRM REALITY CURRICULUM
# =====================================
# 
# PHASE 1: SURVIVAL (Stages 0-2)
#   - Focus: Don't blow up. Learn risk management.
#   - Rewards: Base PnL + DD penalties + anti-churn ONLY
#   - Market: Easy (low vol, clear trends, no slippage)
#   - Promotion: DD breach rate < 10%, survive without blowing up
#
# PHASE 2: PROFITABILITY (Stages 3-5)
#   - Focus: Make money. Agent discovers what works.
#   - Rewards: ALL enabled (entry, exit, R-multiple, MAE, time)
#   - Market: Normal → Hard (progressive difficulty)
#   - Promotion: profit_factor > 1.2, positive R-multiple
#
# PHASE 3: CONSISTENCY (Stages 6-7)
#   - Focus: Stay profitable across ALL conditions.
#   - Rewards: All + variance penalties (loss_multiplier > 1.0)
#   - Market: Full difficulty + domain randomization
#   - Promotion: Low PnL std, stable across regimes
#
# =============================================================================

def get_foundation_config() -> CurriculumStageConfig:
    """
    PHASE 1 - SURVIVAL: Stage 0 (FOUNDATION)
    =========================================
    Learn to not blow up. Pure PnL + risk signal.
    
    ENABLED: Base PnL, DD shaping, light anti-churn
    DISABLED: ALL bonuses (R-multiple, MAE, exit quality, entry quality, streaks)
    MARKET: Very easy (no slippage, no randomization)
    """
    return CurriculumStageConfig(
        stage=CurriculumStage.FOUNDATION,
        name="Foundation",
        description="SURVIVAL Phase: Learn risk basics. No bonuses, pure PnL.",
        execution=ExecutionDifficulty(
            # VERY EASY - no execution friction
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
            # PHASE 1 SURVIVAL: Pure PnL + Risk Only
            # ============================================================================
            reward_scale=5.0,
            loss_multiplier=1.0,              # Symmetric
            
            # ALL BONUSES DISABLED
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
            
            truncation_winner_discount=0.30,
            truncation_loser_extra_penalty=0.15,
            
            entry_quality_integration=False,
            entry_quality_weight=0.0,
            
            # DD SHAPING: ENABLED (learning risk)
            dd_shaping_enabled=True,
            dd_threshold=0.08,
            dd_penalty_scale=0.5,
            dd_severity_exponent=1.2,
            dd_severity_cap=0.8,
            
            streak_modifier_enabled=False,
            win_streak_bonus_per_win=0.0,
            loss_streak_penalty_per_loss=0.0,
            
            # ANTI-CHURN: Moderate (teach patience from the start)
            # BUGFIX: Was too lenient (30/0.01), agent learned to overtrade
            anti_churn_enabled=True,
            daily_trade_soft_limit=15,
            churn_penalty_per_trade=0.03,
            
            hard_block_penalty=0.01,
            soft_block_penalty=0.005,
            per_step_shaping_enabled=False,
            holding_cost_per_bar=0.0,
            exploration_bonus=0.0,
            directional_accuracy_weight=1.0,
            min_reward=-2.5,
            max_reward=2.5,
        ),
        constraints=TradingConstraints(
            max_positions=1,
            max_trades_per_day=30,
            max_trades_per_session=15,
            max_consecutive_losses=10,
            enforce_session_windows=False,
            enforce_no_new_trades_window=False,
            enforce_weekend_block=False,
            enforce_hard_close=False,
            min_minutes_between_entries=1,
            min_minutes_after_loss=2,
            daily_drawdown_limit=0.30,
            max_drawdown_limit=0.30,
            daily_dd_safety_buffer=0.0,
            max_dd_safety_buffer=0.0,
            emergency_close_threshold=0.25,
            entry_quality_gate_enabled=False,
            entry_quality_threshold=0.0,
            hard_stop_loss_eur=500.0,
            soft_stop_loss_eur=300.0,
            trailing_activation_eur=100.0,
            trailing_retrace_pct=0.40,
            time_decay_hours=12.0,
            risk_per_trade_pct=0.005,
            max_risk_per_trade_pct=0.01,
        ),
        competence=CompetenceThresholds(
            # SURVIVAL PHASE: Focus on NOT blowing up
            min_episodes=150,
            min_timesteps=150_000,
            min_win_rate=0.35,              # Can be low
            min_profit_factor=0.6,          # Can lose money
            max_avg_drawdown=0.20,          # KEY: Don't blow up
            min_avg_pnl=-500.0,             # Acceptable to lose
            min_avg_r_multiple=-0.20,
            min_entropy=0.15,
            max_win_rate_std=0.25,
            max_pnl_std=6000.0,             # Permissive
            min_trade_count_avg=2.0,
            max_dd_breach_rate=0.30,        # KEY METRIC
            max_consecutive_loss_rate=0.25,
            evaluation_window=50,
        ),
        max_steps_per_episode=1500,
        include_memory_features=False,
        include_world_model_features=False,
        include_expert_signals=True,
        allow_demotion=False,
        data_difficulty=DataDifficulty(
            # EASY MARKET: Low volatility, clear trends
            volatility_percentile_range=(0.0, 0.40),
            min_trend_clarity=0.4,
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
            transition_cooldown_episodes=30,
        ),
        skill_requirements=SkillRequirements(
            required_skills={
                TradingSkill.DRAWDOWN_CONTROL: 0.40,
            },
            min_confidence=0.4,
            require_all_skills=False,
            weighted_threshold=0.35,
        ),
        entropy_targets=EntropyTargets(
            # BUGFIX: Raised min_entropy from 0.30 to 0.50 to prevent collapse
            # For 10-action space, max entropy=2.3, healthy range is 0.5-1.5
            min_entropy=0.50,
            max_entropy=1.20,
            low_entropy_penalty_scale=0.15,  # Stronger penalty for collapse
            high_entropy_penalty_scale=0.02,
            use_in_promotion=True,
        ),
        composite_scoring=CompositeScoringConfig(
            enabled=True,
            promotion_threshold=0.60,
            demotion_threshold=0.20,
            hard_floors={
                "win_rate": 0.32,
                "max_drawdown": 0.20,
                "dd_breach_rate": 0.30,
            },
        ),
        adaptive_thresholds=AdaptiveThresholdConfig(
            enabled=True,
            plateau_episodes_threshold=100,
            max_relaxation=0.15,
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


def get_discipline_config() -> CurriculumStageConfig:
    """
    PHASE 1 - SURVIVAL: Stage 1 (DISCIPLINE)
    =========================================
    Continue learning to survive. Tighter DD limits, stronger anti-churn.
    
    ENABLED: Base PnL, DD shaping (stronger), anti-churn (stronger)
    DISABLED: ALL bonuses
    MARKET: Easy with slight randomization
    """
    return CurriculumStageConfig(
        stage=CurriculumStage.DISCIPLINE,
        name="Discipline",
        description="Learn patience. Strong anti-churn, quality over quantity.",
        execution=ExecutionDifficulty(
            base_spread_points=0.08,
            spread_mult_range=(0.95, 1.10),
            max_spread_points=0.15,
            slippage_points_sigma=0.01,
            slippage_mult_range=(0.9, 1.2),
            max_slippage_points=0.05,
            commission_per_lot=0.0,
            latency_bars=0,
            enable_randomization=True,
            spread_randomization_range=(0.95, 1.08),
            slippage_randomization_range=(0.95, 1.10),
            latency_randomization_range=(0, 0),
            volatility_scale_range=(0.95, 1.08),
        ),
        rewards=RewardShaping(
            # ============================================================================
            # PHASE 1 SURVIVAL: Stage 1 - Tighter risk, stronger anti-churn
            # ============================================================================
            reward_scale=5.5,
            loss_multiplier=1.0,
            
            # ALL BONUSES DISABLED
            r_multiple_bonus_threshold=99.0,
            r_multiple_bonus_scale=0.0,
            r_multiple_bonus_cap=0.0,
            
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
            
            truncation_winner_discount=0.30,
            truncation_loser_extra_penalty=0.15,
            
            entry_quality_integration=False,
            entry_quality_weight=0.0,
            
            # DD SHAPING: Stronger than Stage 0
            dd_shaping_enabled=True,
            dd_threshold=0.06,
            dd_penalty_scale=0.7,
            dd_severity_exponent=1.3,
            dd_severity_cap=1.0,
            
            streak_modifier_enabled=False,
            win_streak_bonus_per_win=0.0,
            loss_streak_penalty_per_loss=0.0,
            
            # ANTI-CHURN: Strong (building patience habit)
            # BUGFIX: Was too lenient, raised penalty significantly
            anti_churn_enabled=True,
            daily_trade_soft_limit=12,
            churn_penalty_per_trade=0.04,
            
            hard_block_penalty=0.02,
            soft_block_penalty=0.01,
            per_step_shaping_enabled=False,
            holding_cost_per_bar=0.0,
            exploration_bonus=0.0,
            directional_accuracy_weight=1.0,
            min_reward=-3.0,
            max_reward=3.0,
        ),
        constraints=TradingConstraints(
            max_positions=1,
            max_trades_per_day=20,
            max_trades_per_session=10,
            max_consecutive_losses=8,
            enforce_session_windows=False,
            enforce_no_new_trades_window=False,
            enforce_weekend_block=False,
            enforce_hard_close=False,
            min_minutes_between_entries=2,
            min_minutes_after_loss=3,
            daily_drawdown_limit=0.20,
            max_drawdown_limit=0.25,
            daily_dd_safety_buffer=0.0,
            max_dd_safety_buffer=0.0,
            emergency_close_threshold=0.20,
            entry_quality_gate_enabled=False,
            entry_quality_threshold=0.0,
            hard_stop_loss_eur=400.0,
            soft_stop_loss_eur=250.0,
            trailing_activation_eur=100.0,
            trailing_retrace_pct=0.40,
            time_decay_hours=10.0,
            risk_per_trade_pct=0.005,
            max_risk_per_trade_pct=0.01,
        ),
        competence=CompetenceThresholds(
            # SURVIVAL PHASE: Tighter than Stage 0
            min_episodes=175,
            min_timesteps=200_000,
            min_win_rate=0.37,
            min_profit_factor=0.70,
            max_avg_drawdown=0.15,          # Tighter
            min_avg_pnl=-300.0,
            min_avg_r_multiple=-0.15,
            min_entropy=0.12,
            max_win_rate_std=0.22,
            max_pnl_std=5500.0,
            min_trade_count_avg=2.5,
            max_dd_breach_rate=0.20,        # KEY: Tighter
            max_consecutive_loss_rate=0.20,
            evaluation_window=60,
        ),
        max_steps_per_episode=1500,
        include_memory_features=False,
        include_world_model_features=False,
        include_expert_signals=True,
        allow_demotion=True,
        data_difficulty=DataDifficulty(
            # Still easy market
            volatility_percentile_range=(0.0, 0.50),
            min_trend_clarity=0.3,
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
            transition_cooldown_episodes=30,
        ),
        skill_requirements=SkillRequirements(
            required_skills={
                TradingSkill.DRAWDOWN_CONTROL: 0.45,
                TradingSkill.PATIENCE: 0.40,
            },
            min_confidence=0.4,
            require_all_skills=False,
            weighted_threshold=0.40,
        ),
        entropy_targets=EntropyTargets(
            # BUGFIX: Raised min_entropy from 0.25 to 0.40 to prevent collapse
            min_entropy=0.40,
            max_entropy=1.00,
            low_entropy_penalty_scale=0.12,  # Stronger penalty
            high_entropy_penalty_scale=0.03,
            use_in_promotion=True,
        ),
        composite_scoring=CompositeScoringConfig(
            enabled=True,
            promotion_threshold=0.62,
            demotion_threshold=0.25,
            hard_floors={
                "win_rate": 0.35,
                "max_drawdown": 0.15,
                "dd_breach_rate": 0.20,
            },
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
            current_stage_weight=0.85,
            recent_stages_weight=0.15,
            foundation_weight=0.0,
        ),
        review_session=ReviewSessionConfig(
            enabled=False,
        ),
    )


def get_market_structure_config() -> CurriculumStageConfig:
    """
    PHASE 1 - SURVIVAL: Stage 2 (MARKET_STRUCTURE)
    ==============================================
    Final survival stage. Tightest risk limits before profitability phase.
    
    ENABLED: Base PnL, DD shaping (strong), anti-churn (strong)
    DISABLED: ALL bonuses (still pure survival)
    MARKET: Medium difficulty
    """
    return CurriculumStageConfig(
        stage=CurriculumStage.MARKET_STRUCTURE,
        name="Market Structure",
        description="SURVIVAL Phase: Final risk mastery before profitability.",
        execution=ExecutionDifficulty(
            # MEDIUM difficulty
            base_spread_points=0.10,
            spread_mult_range=(0.90, 1.20),
            max_spread_points=0.30,
            slippage_points_sigma=0.03,
            slippage_mult_range=(0.85, 1.35),
            max_slippage_points=0.15,
            commission_per_lot=0.0,
            latency_bars=0,
            enable_randomization=True,
            spread_randomization_range=(0.92, 1.15),
            slippage_randomization_range=(0.90, 1.20),
            latency_randomization_range=(0, 1),
            volatility_scale_range=(0.92, 1.12),
        ),
        rewards=RewardShaping(
            # ============================================================================
            # PHASE 1 SURVIVAL: Stage 2 - Tightest risk before Phase 2
            # ============================================================================
            reward_scale=6.0,
            loss_multiplier=1.0,
            
            # ALL BONUSES STILL DISABLED
            r_multiple_bonus_threshold=99.0,
            r_multiple_bonus_scale=0.0,
            r_multiple_bonus_cap=0.0,
            
            mae_efficiency_enabled=False,
            mae_efficiency_scale=0.0,
            mae_efficiency_threshold=99.0,
            
            time_efficiency_enabled=False,
            time_efficiency_scale=0.0,
            optimal_trade_bars=10,
            max_trade_bars_for_bonus=32,
            
            exit_quality_enabled=False,
            trailing_stop_bonus=0.0,
            agent_close_bonus=0.0,
            hard_stop_penalty=0.0,
            risk_liquidation_penalty=0.0,
            
            truncation_winner_discount=0.25,
            truncation_loser_extra_penalty=0.10,
            
            # Entry quality: STILL DISABLED in SURVIVAL phase
            entry_quality_integration=False,
            entry_quality_weight=0.0,
            
            # DD shaping: STRONG - tightest risk before Phase 2
            dd_shaping_enabled=True,
            dd_threshold=0.03,              # Tighter than Stage 1
            dd_penalty_scale=0.7,           # Stronger penalty
            dd_severity_exponent=1.4,
            dd_severity_cap=1.0,
            
            # Streaks: DISABLED in SURVIVAL phase
            streak_modifier_enabled=False,
            win_streak_bonus_per_win=0.0,
            loss_streak_penalty_per_loss=0.0,
            
            # Anti-churn: VERY STRONG - gate to Phase 2
            # BUGFIX: Raised penalty to enforce patience before Phase 2
            anti_churn_enabled=True,
            daily_trade_soft_limit=8,
            churn_penalty_per_trade=0.05,  # Strong penalty for overtrading
            
            hard_block_penalty=0.03,
            soft_block_penalty=0.015,
            per_step_shaping_enabled=False,
            holding_cost_per_bar=0.0,
            opportunity_bonus_scale=0.0,
            min_reward=-3.5,
            max_reward=3.5,
        ),
        constraints=TradingConstraints(
            max_positions=1,
            max_trades_per_day=15,
            max_trades_per_session=8,
            max_consecutive_losses=5,
            enforce_session_windows=True,
            enforce_no_new_trades_window=True,
            enforce_weekend_block=True,
            enforce_hard_close=True,
            min_minutes_between_entries=5,
            min_minutes_after_loss=10,
            daily_drawdown_limit=0.06,      # Tighter - gate to Phase 2
            max_drawdown_limit=0.12,        # Tighter - gate to Phase 2
            daily_dd_safety_buffer=0.006,
            max_dd_safety_buffer=0.012,
            emergency_close_threshold=0.11,
            entry_quality_gate_enabled=False,  # No entry gate in SURVIVAL
            entry_quality_threshold=0.0,
            hard_stop_loss_eur=280.0,
            soft_stop_loss_eur=180.0,
            trailing_activation_eur=100.0,
            trailing_retrace_pct=0.35,
            time_decay_hours=6.0,
            risk_per_trade_pct=0.0035,
            max_risk_per_trade_pct=0.007,
        ),
        competence=CompetenceThresholds(
            # GATE TO PHASE 2 - Must prove survival skills
            min_episodes=200,
            min_timesteps=400_000,
            min_win_rate=0.40,              # Win rate not critical in SURVIVAL
            min_profit_factor=0.95,         # Near break-even is fine
            max_avg_drawdown=0.08,          # CRITICAL: Low DD required for Phase 2
            min_avg_pnl=0.0,                # PnL doesn't matter in SURVIVAL
            min_avg_r_multiple=0.0,         # R-multiple doesn't matter in SURVIVAL
            min_entropy=0.20,               # Must explore
            max_win_rate_std=0.20,          # Relaxed in SURVIVAL
            max_pnl_std=15000.0,            # Relaxed in SURVIVAL
            min_trade_count_avg=4.0,
            max_dd_breach_rate=0.10,        # CRITICAL: Gate to Phase 2
            max_consecutive_loss_rate=0.15,
            evaluation_window=100,
        ),
        max_steps_per_episode=2000,
        include_memory_features=True,
        include_world_model_features=True,
        include_expert_signals=True,
        allow_demotion=True,
        data_difficulty=DataDifficulty(
            # MEDIUM difficulty - still learning
            volatility_percentile_range=(0.0, 0.70),  # Expanded range
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
            lr_warmup_steps=8_000,
            reward_blend_enabled=True,
            reward_blend_episodes=15,
            checkpoint_on_transition=True,
            transition_cooldown_episodes=40,
        ),
        skill_requirements=SkillRequirements(
            # SURVIVAL: Only DD control and patience matter
            required_skills={
                TradingSkill.DRAWDOWN_CONTROL: 0.55,  # CRITICAL for Phase 2
                TradingSkill.PATIENCE: 0.50,          # CRITICAL for Phase 2
            },
            min_confidence=0.45,
            require_all_skills=True,  # MUST have both survival skills
            weighted_threshold=0.50,
        ),
        entropy_targets=EntropyTargets(
            # BUGFIX: Raised min_entropy from 0.20 to 0.35 to prevent collapse
            min_entropy=0.35,
            max_entropy=0.85,
            low_entropy_penalty_scale=0.12,
            high_entropy_penalty_scale=0.04,
            use_in_promotion=True,
        ),
        composite_scoring=CompositeScoringConfig(
            enabled=True,
            promotion_threshold=0.65,  # Gate to Phase 2
            demotion_threshold=0.28,
            hard_floors={
                "max_drawdown": 0.08,       # CRITICAL: Gate to Phase 2
                "dd_breach_rate": 0.10,     # CRITICAL: Gate to Phase 2
            },
        ),
        adaptive_thresholds=AdaptiveThresholdConfig(
            enabled=True,
            plateau_episodes_threshold=80,
            max_relaxation=0.10,
        ),
        recovery_protocol=RecoveryProtocolConfig(
            enabled=True,
            trigger_after_demotions=2,
            recovery_duration_episodes=150,
        ),
        mixed_stage_sampling=MixedStageSamplingConfig(
            enabled=True,
            current_stage_weight=0.80,
            recent_stages_weight=0.15,
            foundation_weight=0.05,
        ),
        review_session=ReviewSessionConfig(
            enabled=False,  # No review in SURVIVAL phase
        ),
    )


def get_economic_logic_config() -> CurriculumStageConfig:
    """
    PHASE 2 - PROFITABILITY: Stage 3 (ECONOMIC_LOGIC)
    ==================================================
    First profitability stage. ALL rewards enabled for the first time.
    Agent learns to make money now that survival is mastered.
    
    ENABLED: ALL bonuses (r-multiple, MAE, time, exit, entry, streaks)
    MARKET: Normal difficulty (full volatility range)
    FOCUS: Let winners run via trailing stops
    """
    return CurriculumStageConfig(
        stage=CurriculumStage.ECONOMIC_LOGIC,
        name="Economic Logic",
        description="PROFITABILITY Phase: ALL rewards enabled. Learn to make money.",
        execution=ExecutionDifficulty(
            # NORMAL difficulty - real market conditions
            base_spread_points=0.15,
            spread_mult_range=(0.85, 1.40),
            max_spread_points=0.60,
            slippage_points_sigma=0.04,
            slippage_mult_range=(0.70, 1.60),
            max_slippage_points=0.30,
            commission_per_lot=0.0,
            latency_bars=1,
            enable_randomization=True,
            spread_randomization_range=(0.88, 1.30),
            slippage_randomization_range=(0.80, 1.45),
            latency_randomization_range=(0, 2),
            volatility_scale_range=(0.88, 1.18),
        ),
        rewards=RewardShaping(
            # ============================================================================
            # PHASE 2 PROFITABILITY: Stage 3 - ALL REWARDS ENABLED
            # ============================================================================
            # This is the BIG UNLOCK. Agent now gets rewarded for:
            # - Good R-multiples (profit/risk)
            # - Efficient entries (low MAE)
            # - Proper exits (trailing stops)
            # - Time efficiency
            # - Win streaks
            # ============================================================================
            reward_scale=8.0,
            loss_multiplier=1.0,
            
            # R-multiple bonus: ENABLED - reward good risk/reward
            r_multiple_bonus_threshold=1.5,
            r_multiple_bonus_scale=0.06,
            r_multiple_bonus_cap=0.10,
            
            # MAE efficiency: ENABLED - reward efficient entries
            mae_efficiency_enabled=True,
            mae_efficiency_scale=0.05,
            mae_efficiency_threshold=1.8,
            
            # Time efficiency: ENABLED (light)
            time_efficiency_enabled=True,
            time_efficiency_scale=0.02,
            optimal_trade_bars=10,
            max_trade_bars_for_bonus=28,
            
            # Exit quality: ENABLED - trailing stops strongly rewarded
            exit_quality_enabled=True,
            trailing_stop_bonus=0.25,         # HIGH - reward letting winners run
            agent_close_bonus=0.0,            # ZERO - never reward cutting winners
            hard_stop_penalty=0.08,
            risk_liquidation_penalty=0.15,
            
            truncation_winner_discount=0.30,
            truncation_loser_extra_penalty=0.12,
            
            # Entry quality: ENABLED
            entry_quality_integration=True,
            entry_quality_weight=0.20,
            
            # DD shaping: Maintained from SURVIVAL
            dd_shaping_enabled=True,
            dd_threshold=0.03,
            dd_penalty_scale=0.8,
            dd_severity_exponent=1.4,
            dd_severity_cap=1.2,
            
            # Streaks: ENABLED - reward consistency
            streak_modifier_enabled=True,
            win_streak_bonus_per_win=0.02,
            loss_streak_penalty_per_loss=0.03,
            
            # Anti-churn: Maintained from SURVIVAL
            anti_churn_enabled=True,
            daily_trade_soft_limit=12,
            churn_penalty_per_trade=0.02,
            
            hard_block_penalty=0.03,
            soft_block_penalty=0.015,
            per_step_shaping_enabled=False,
            holding_cost_per_bar=0.0,
            opportunity_bonus_scale=0.0,
            min_reward=-4.0,
            max_reward=4.0,
        ),
        constraints=TradingConstraints(
            max_positions=1,
            max_trades_per_day=12,
            max_trades_per_session=6,
            max_consecutive_losses=4,
            enforce_session_windows=True,
            enforce_no_new_trades_window=True,
            enforce_weekend_block=True,
            enforce_hard_close=True,
            min_minutes_between_entries=8,
            min_minutes_after_loss=15,
            daily_drawdown_limit=0.06,
            max_drawdown_limit=0.12,
            daily_dd_safety_buffer=0.006,
            max_dd_safety_buffer=0.012,
            emergency_close_threshold=0.11,
            entry_quality_gate_enabled=True,
            entry_quality_threshold=0.35,
            hard_stop_loss_eur=250.0,
            soft_stop_loss_eur=160.0,
            trailing_activation_eur=80.0,
            trailing_retrace_pct=0.32,
            time_decay_hours=5.0,
            risk_per_trade_pct=0.0032,
            max_risk_per_trade_pct=0.0065,
        ),
        competence=CompetenceThresholds(
            # PROFITABILITY: Must demonstrate actual profitability
            min_episodes=250,
            min_timesteps=600_000,
            min_win_rate=0.48,
            min_profit_factor=1.15,         # Must be profitable
            max_avg_drawdown=0.07,
            min_avg_pnl=100.0,              # Must make money
            min_avg_r_multiple=0.05,        # Positive expectancy
            min_entropy=0.10,
            max_win_rate_std=0.12,
            max_pnl_std=9000.0,
            min_trade_count_avg=4.5,
            max_dd_breach_rate=0.10,
            max_consecutive_loss_rate=0.10,
            evaluation_window=100,
        ),
        max_steps_per_episode=2000,
        include_memory_features=True,
        include_world_model_features=True,
        include_expert_signals=True,
        allow_demotion=True,
        data_difficulty=DataDifficulty(
            # NORMAL difficulty - full volatility range
            volatility_percentile_range=(0.0, 1.0),
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
            lr_warmup_steps=12_000,
            reward_blend_enabled=True,
            reward_blend_episodes=25,
            checkpoint_on_transition=True,
            transition_cooldown_episodes=60,
        ),
        skill_requirements=SkillRequirements(
            # PROFITABILITY: Now care about trading skills
            required_skills={
                TradingSkill.DRAWDOWN_CONTROL: 0.60,
                TradingSkill.PATIENCE: 0.55,
                TradingSkill.EXIT_QUALITY: 0.50,
                TradingSkill.RISK_REWARD: 0.45,
            },
            min_confidence=0.50,
            require_all_skills=False,
            weighted_threshold=0.52,
        ),
        entropy_targets=EntropyTargets(
            min_entropy=0.12,
            max_entropy=0.50,
            low_entropy_penalty_scale=0.08,
            high_entropy_penalty_scale=0.05,
            use_in_promotion=True,
        ),
        composite_scoring=CompositeScoringConfig(
            enabled=True,
            promotion_threshold=0.70,
            demotion_threshold=0.35,
            hard_floors={
                "win_rate": 0.45,
                "max_drawdown": 0.08,
                "dd_breach_rate": 0.12,
                "profit_factor": 1.05,
            },
        ),
        adaptive_thresholds=AdaptiveThresholdConfig(
            enabled=True,
            plateau_episodes_threshold=60,
            max_relaxation=0.08,
        ),
        recovery_protocol=RecoveryProtocolConfig(
            enabled=True,
            trigger_after_demotions=2,
            recovery_duration_episodes=200,
        ),
        mixed_stage_sampling=MixedStageSamplingConfig(
            enabled=True,
            current_stage_weight=0.75,
            recent_stages_weight=0.18,
            foundation_weight=0.07,
        ),
        review_session=ReviewSessionConfig(
            enabled=True,
            review_frequency=400,
            review_duration=50,
            review_depth=2,
        ),
        validation=ValidationConfig(
            enabled=True,  # Enable validation from Phase 2
            validation_episodes=80,
            min_performance_ratio=0.80,  # Allow 20% drop
            max_performance_drop=0.20,
            required_regimes=[
                MarketRegime.TRENDING_UP,
                MarketRegime.TRENDING_DOWN,
                MarketRegime.RANGING,
            ],
            min_episodes_per_regime=15,
        ),
    )


def get_professional_config() -> CurriculumStageConfig:
    """
    PHASE 2 - PROFITABILITY: Stage 4 (PROFESSIONAL)
    ================================================
    Harder market conditions. Agent must maintain profitability
    under more difficult execution.
    
    ENABLED: ALL bonuses (same as Stage 3)
    MARKET: Hard difficulty
    FOCUS: Maintain profitability under adversity
    """
    return CurriculumStageConfig(
        stage=CurriculumStage.PROFESSIONAL,
        name="Professional Trading",
        description="PROFITABILITY Phase: Harder market, maintain profits.",
        execution=ExecutionDifficulty(
            # HARD difficulty - challenging execution
            base_spread_points=0.18,
            spread_mult_range=(0.80, 1.50),
            max_spread_points=0.80,
            slippage_points_sigma=0.05,
            slippage_mult_range=(0.65, 1.70),
            max_slippage_points=0.35,
            commission_per_lot=2.0,       # Introduce commission: $2/lot
            latency_bars=1,
            enable_randomization=True,
            spread_randomization_range=(0.82, 1.40),
            slippage_randomization_range=(0.70, 1.55),
            latency_randomization_range=(0, 2),
            volatility_scale_range=(0.85, 1.22),
        ),
        rewards=RewardShaping(
            # ============================================================================
            # PHASE 2 PROFITABILITY: Stage 4 - HARDER MARKET
            # ============================================================================
            # Same rewards as Stage 3, but market is harder.
            # Agent must prove profitability is robust, not luck.
            # ============================================================================
            reward_scale=8.5,
            loss_multiplier=1.0,
            
            # ALL BONUSES ENABLED (same as Stage 3)
            r_multiple_bonus_threshold=1.5,
            r_multiple_bonus_scale=0.08,
            r_multiple_bonus_cap=0.12,
            
            mae_efficiency_enabled=True,
            mae_efficiency_scale=0.06,
            mae_efficiency_threshold=1.7,
            
            time_efficiency_enabled=True,
            time_efficiency_scale=0.02,
            optimal_trade_bars=9,
            max_trade_bars_for_bonus=26,
            
            exit_quality_enabled=True,
            trailing_stop_bonus=0.25,
            agent_close_bonus=0.0,            # ALWAYS ZERO
            hard_stop_penalty=0.10,
            risk_liquidation_penalty=0.18,
            
            truncation_winner_discount=0.30,
            truncation_loser_extra_penalty=0.14,
            
            entry_quality_integration=True,
            entry_quality_weight=0.22,
            
            dd_shaping_enabled=True,
            dd_threshold=0.025,
            dd_penalty_scale=0.9,
            dd_severity_exponent=1.45,
            dd_severity_cap=1.25,
            
            streak_modifier_enabled=True,
            win_streak_bonus_per_win=0.025,
            loss_streak_penalty_per_loss=0.035,
            
            anti_churn_enabled=True,
            daily_trade_soft_limit=10,
            churn_penalty_per_trade=0.022,
            
            hard_block_penalty=0.035,
            soft_block_penalty=0.018,
            per_step_shaping_enabled=False,
            holding_cost_per_bar=0.0,
            opportunity_bonus_scale=0.0,
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
            min_minutes_after_loss=18,
            daily_drawdown_limit=0.055,
            max_drawdown_limit=0.11,
            daily_dd_safety_buffer=0.007,
            max_dd_safety_buffer=0.014,
            emergency_close_threshold=0.10,
            entry_quality_gate_enabled=True,
            entry_quality_threshold=0.40,
            hard_stop_loss_eur=240.0,
            soft_stop_loss_eur=155.0,
            trailing_activation_eur=85.0,
            trailing_retrace_pct=0.32,
            time_decay_hours=5.0,
            risk_per_trade_pct=0.003,
            max_risk_per_trade_pct=0.006,
        ),
        competence=CompetenceThresholds(
            # PROFITABILITY: Higher bar with harder market
            min_episodes=300,
            min_timesteps=800_000,
            min_win_rate=0.50,
            min_profit_factor=1.25,
            max_avg_drawdown=0.065,
            min_avg_pnl=150.0,
            min_avg_r_multiple=0.08,
            min_entropy=0.08,
            max_win_rate_std=0.11,
            max_pnl_std=8500.0,
            min_trade_count_avg=5.0,
            max_dd_breach_rate=0.08,
            max_consecutive_loss_rate=0.08,
            evaluation_window=120,
        ),
        max_steps_per_episode=2000,
        include_memory_features=True,
        include_world_model_features=True,
        include_expert_signals=True,
        allow_demotion=True,
        data_difficulty=DataDifficulty(
            # FULL range - including difficult periods
            volatility_percentile_range=(0.0, 1.0),
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
            lr_warmup_steps=12_000,
            reward_blend_enabled=True,
            reward_blend_episodes=25,
            checkpoint_on_transition=True,
            transition_cooldown_episodes=70,
        ),
        skill_requirements=SkillRequirements(
            required_skills={
                TradingSkill.DRAWDOWN_CONTROL: 0.65,
                TradingSkill.PATIENCE: 0.60,
                TradingSkill.EXIT_QUALITY: 0.55,
                TradingSkill.RISK_REWARD: 0.50,
                TradingSkill.CONSISTENCY: 0.45,
            },
            min_confidence=0.55,
            require_all_skills=False,
            weighted_threshold=0.55,
        ),
        entropy_targets=EntropyTargets(
            min_entropy=0.10,
            max_entropy=0.45,
            low_entropy_penalty_scale=0.07,
            high_entropy_penalty_scale=0.06,
            use_in_promotion=True,
        ),
        composite_scoring=CompositeScoringConfig(
            enabled=True,
            promotion_threshold=0.72,
            demotion_threshold=0.38,
            hard_floors={
                "win_rate": 0.47,
                "max_drawdown": 0.07,
                "dd_breach_rate": 0.10,
                "profit_factor": 1.15,
            },
        ),
        adaptive_thresholds=AdaptiveThresholdConfig(
            enabled=True,
            plateau_episodes_threshold=65,
            max_relaxation=0.08,
        ),
        recovery_protocol=RecoveryProtocolConfig(
            enabled=True,
            trigger_after_demotions=2,
            recovery_duration_episodes=220,
        ),
        mixed_stage_sampling=MixedStageSamplingConfig(
            enabled=True,
            current_stage_weight=0.72,
            recent_stages_weight=0.20,
            foundation_weight=0.08,
        ),
        review_session=ReviewSessionConfig(
            enabled=True,
            review_frequency=380,
            review_duration=55,
            review_depth=2,
        ),
        validation=ValidationConfig(
            enabled=True,
            validation_episodes=90,
            min_performance_ratio=0.82,
            max_performance_drop=0.18,
            required_regimes=[
                MarketRegime.TRENDING_UP,
                MarketRegime.TRENDING_DOWN,
                MarketRegime.RANGING,
            ],
            min_episodes_per_regime=18,
        ),
    )


def get_adaptive_config() -> CurriculumStageConfig:
    """
    PHASE 2 - PROFITABILITY: Stage 5 (ADAPTIVE) - Final Gate to Phase 3
    ====================================================================
    Hardest Phase 2 stage. Must prove consistent profitability
    before moving to consistency phase.
    
    ENABLED: ALL bonuses (same as Stage 3-4)
    MARKET: Hardest Phase 2 difficulty
    FOCUS: Gate to Phase 3 - must be consistently profitable
    """
    return CurriculumStageConfig(
        stage=CurriculumStage.ADAPTIVE,
        name="Adaptive Intelligence",
        description="PROFITABILITY Phase: Gate to consistency. Prove robust profits.",
        execution=ExecutionDifficulty(
            # HARD difficulty - toughest in Phase 2
            base_spread_points=0.20,
            spread_mult_range=(0.78, 1.55),
            max_spread_points=1.00,
            slippage_points_sigma=0.05,
            slippage_mult_range=(0.60, 1.80),
            max_slippage_points=0.40,
            commission_per_lot=3.0,       # Commission: $3/lot
            latency_bars=1,
            enable_randomization=True,
            spread_randomization_range=(0.80, 1.45),
            slippage_randomization_range=(0.68, 1.65),
            latency_randomization_range=(0, 2),
            volatility_scale_range=(0.82, 1.28),
        ),
        rewards=RewardShaping(
            # ============================================================================
            # PHASE 2 PROFITABILITY: Stage 5 - GATE TO PHASE 3
            # ============================================================================
            # Same rewards as Stage 3-4, but must prove consistency.
            # This is the final test before variance penalties kick in.
            # ============================================================================
            reward_scale=9.0,
            loss_multiplier=1.0,
            
            # ALL BONUSES ENABLED (same as Stage 3-4)
            r_multiple_bonus_threshold=1.4,
            r_multiple_bonus_scale=0.10,
            r_multiple_bonus_cap=0.14,
            
            mae_efficiency_enabled=True,
            mae_efficiency_scale=0.07,
            mae_efficiency_threshold=1.6,
            
            time_efficiency_enabled=True,
            time_efficiency_scale=0.025,
            optimal_trade_bars=8,
            max_trade_bars_for_bonus=24,
            
            exit_quality_enabled=True,
            trailing_stop_bonus=0.25,
            agent_close_bonus=0.0,            # ALWAYS ZERO
            hard_stop_penalty=0.12,
            risk_liquidation_penalty=0.20,
            
            truncation_winner_discount=0.30,
            truncation_loser_extra_penalty=0.15,
            
            entry_quality_integration=True,
            entry_quality_weight=0.25,
            
            dd_shaping_enabled=True,
            dd_threshold=0.022,
            dd_penalty_scale=1.0,
            dd_severity_exponent=1.5,
            dd_severity_cap=1.3,
            
            streak_modifier_enabled=True,
            win_streak_bonus_per_win=0.03,
            loss_streak_penalty_per_loss=0.04,
            
            anti_churn_enabled=True,
            daily_trade_soft_limit=9,
            churn_penalty_per_trade=0.025,
            
            hard_block_penalty=0.04,
            soft_block_penalty=0.02,
            per_step_shaping_enabled=False,
            holding_cost_per_bar=0.0,
            opportunity_bonus_scale=0.0,
            min_reward=-4.5,
            max_reward=4.5,
        ),
        constraints=TradingConstraints(
            max_positions=1,
            max_trades_per_day=9,
            max_trades_per_session=5,
            max_consecutive_losses=4,
            enforce_session_windows=True,
            enforce_no_new_trades_window=True,
            enforce_weekend_block=True,
            enforce_hard_close=True,
            min_minutes_between_entries=12,
            min_minutes_after_loss=20,
            daily_drawdown_limit=0.05,
            max_drawdown_limit=0.10,
            daily_dd_safety_buffer=0.008,
            max_dd_safety_buffer=0.015,
            emergency_close_threshold=0.09,
            entry_quality_gate_enabled=True,
            entry_quality_threshold=0.42,
            hard_stop_loss_eur=230.0,
            soft_stop_loss_eur=150.0,
            trailing_activation_eur=90.0,
            trailing_retrace_pct=0.30,
            time_decay_hours=4.5,
            risk_per_trade_pct=0.003,
            max_risk_per_trade_pct=0.006,
        ),
        competence=CompetenceThresholds(
            # GATE TO PHASE 3: Must be consistently profitable
            min_episodes=350,
            min_timesteps=1_000_000,
            min_win_rate=0.52,
            min_profit_factor=1.30,
            max_avg_drawdown=0.055,
            min_avg_pnl=200.0,
            min_avg_r_multiple=0.10,
            min_entropy=0.06,
            max_win_rate_std=0.10,          # Starting to care about variance
            max_pnl_std=8000.0,              # Starting to care about variance
            min_trade_count_avg=5.0,
            max_dd_breach_rate=0.06,
            max_consecutive_loss_rate=0.06,
            evaluation_window=150,
        ),
        max_steps_per_episode=2000,
        include_memory_features=True,
        include_world_model_features=True,
        include_expert_signals=True,
        allow_demotion=True,
        data_difficulty=DataDifficulty(
            # FULL range + domain randomization
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
            lr_warmup_factor=0.30,
            lr_warmup_steps=15_000,
            reward_blend_enabled=True,
            reward_blend_episodes=30,
            checkpoint_on_transition=True,
            transition_cooldown_episodes=80,
        ),
        skill_requirements=SkillRequirements(
            required_skills={
                TradingSkill.DRAWDOWN_CONTROL: 0.70,
                TradingSkill.PATIENCE: 0.65,
                TradingSkill.EXIT_QUALITY: 0.60,
                TradingSkill.RISK_REWARD: 0.55,
                TradingSkill.CONSISTENCY: 0.50,
            },
            min_confidence=0.58,
            require_all_skills=False,
            weighted_threshold=0.58,
        ),
        entropy_targets=EntropyTargets(
            min_entropy=0.08,
            max_entropy=0.40,
            low_entropy_penalty_scale=0.06,
            high_entropy_penalty_scale=0.07,
            use_in_promotion=True,
        ),
        composite_scoring=CompositeScoringConfig(
            enabled=True,
            promotion_threshold=0.75,
            demotion_threshold=0.40,
            hard_floors={
                "win_rate": 0.50,
                "max_drawdown": 0.06,
                "dd_breach_rate": 0.08,
                "profit_factor": 1.20,
            },
        ),
        adaptive_thresholds=AdaptiveThresholdConfig(
            enabled=True,
            plateau_episodes_threshold=70,
            max_relaxation=0.07,
        ),
        recovery_protocol=RecoveryProtocolConfig(
            enabled=True,
            trigger_after_demotions=2,
            recovery_duration_episodes=250,
        ),
        mixed_stage_sampling=MixedStageSamplingConfig(
            enabled=True,
            current_stage_weight=0.70,
            recent_stages_weight=0.22,
            foundation_weight=0.08,
        ),
        review_session=ReviewSessionConfig(
            enabled=True,
            review_frequency=350,
            review_duration=60,
            review_depth=2,
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


def get_specialist_config() -> CurriculumStageConfig:
    """
    PHASE 3 - CONSISTENCY: Stage 6 (SPECIALIST)
    ============================================
    First consistency stage. Variance penalties kick in.
    losses_multiplier > 1.0 for the first time.
    
    ENABLED: ALL bonuses (same as Phase 2)
    NEW: loss_multiplier > 1.0 (losses hurt MORE than wins help)
    MARKET: Full domain randomization
    FOCUS: Consistent performance across all regimes
    """
    return CurriculumStageConfig(
        stage=CurriculumStage.SPECIALIST,
        name="Specialization",
        description="CONSISTENCY Phase: Variance penalties. Losses hurt more.",
        execution=ExecutionDifficulty(
            # FULL domain randomization
            base_spread_points=0.20,
            spread_mult_range=(0.75, 1.60),
            max_spread_points=1.10,
            slippage_points_sigma=0.06,
            slippage_mult_range=(0.55, 1.90),
            max_slippage_points=0.45,
            commission_per_lot=4.0,       # Realistic commission: $4/lot
            latency_bars=1,
            enable_randomization=True,
            spread_randomization_range=(0.78, 1.50),
            slippage_randomization_range=(0.62, 1.75),
            latency_randomization_range=(0, 3),
            volatility_scale_range=(0.78, 1.35),
            spread_shock_enabled=True,      # Enable spread shocks for robustness
            spread_shock_probability=0.015,  # 1.5% of steps
            spread_shock_multiplier=2.5,
        ),
        rewards=RewardShaping(
            # ============================================================================
            # PHASE 3 CONSISTENCY: Stage 6 - VARIANCE PENALTIES BEGIN
            # ============================================================================
            # Same rewards as Phase 2, BUT:
            # - loss_multiplier > 1.0 (losses hurt MORE than wins help)
            # - Tighter PnL variance requirements
            # - Must be consistent, not just profitable
            # ============================================================================
            reward_scale=9.5,
            loss_multiplier=1.05,           # LOSSES HURT 5% MORE - variance penalty
            
            # ALL BONUSES ENABLED (same as Phase 2)
            r_multiple_bonus_threshold=1.3,
            r_multiple_bonus_scale=0.10,
            r_multiple_bonus_cap=0.15,
            
            mae_efficiency_enabled=True,
            mae_efficiency_scale=0.08,
            mae_efficiency_threshold=1.5,
            
            time_efficiency_enabled=True,
            time_efficiency_scale=0.025,
            optimal_trade_bars=8,
            max_trade_bars_for_bonus=22,
            
            exit_quality_enabled=True,
            trailing_stop_bonus=0.25,
            agent_close_bonus=0.0,            # ALWAYS ZERO
            hard_stop_penalty=0.12,
            risk_liquidation_penalty=0.22,
            
            truncation_winner_discount=0.30,
            truncation_loser_extra_penalty=0.15,
            
            entry_quality_integration=True,
            entry_quality_weight=0.25,
            
            dd_shaping_enabled=True,
            dd_threshold=0.020,
            dd_penalty_scale=1.1,
            dd_severity_exponent=1.55,
            dd_severity_cap=1.4,
            
            # Streaks: STRONGER penalties for losing streaks (consistency)
            streak_modifier_enabled=True,
            win_streak_bonus_per_win=0.03,
            loss_streak_penalty_per_loss=0.05,   # Stronger - variance penalty
            
            anti_churn_enabled=True,
            daily_trade_soft_limit=8,
            churn_penalty_per_trade=0.028,
            
            hard_block_penalty=0.045,
            soft_block_penalty=0.022,
            per_step_shaping_enabled=False,
            holding_cost_per_bar=0.0,
            opportunity_bonus_scale=0.0,
            min_reward=-5.0,
            max_reward=5.0,
        ),
        constraints=TradingConstraints(
            max_positions=1,
            max_trades_per_day=8,
            max_trades_per_session=4,
            max_consecutive_losses=3,
            enforce_session_windows=True,
            enforce_no_new_trades_window=True,
            enforce_weekend_block=True,
            enforce_hard_close=True,
            min_minutes_between_entries=15,
            min_minutes_after_loss=25,
            daily_drawdown_limit=0.05,
            max_drawdown_limit=0.10,
            daily_dd_safety_buffer=0.008,
            max_dd_safety_buffer=0.015,
            emergency_close_threshold=0.09,
            entry_quality_gate_enabled=True,
            entry_quality_threshold=0.45,
            hard_stop_loss_eur=220.0,
            soft_stop_loss_eur=145.0,
            trailing_activation_eur=95.0,
            trailing_retrace_pct=0.28,
            time_decay_hours=4.0,
            risk_per_trade_pct=0.003,
            max_risk_per_trade_pct=0.006,
        ),
        competence=CompetenceThresholds(
            # CONSISTENCY: Variance requirements tighten (but achievable from Stage 5)
            min_episodes=350,             # Reduced from 400
            min_timesteps=1_000_000,      # Reduced from 1.2M
            min_win_rate=0.51,            # Softened from 0.53
            min_profit_factor=1.30,       # Softened from 1.35
            max_avg_drawdown=0.055,       # Softened from 0.05
            min_avg_pnl=200.0,            # Softened from 250
            min_avg_r_multiple=0.10,      # Softened from 0.12
            min_entropy=0.05,
            max_win_rate_std=0.09,        # Softened from 0.08 - more achievable
            max_pnl_std=7500.0,           # Softened from 7000 - more achievable
            min_trade_count_avg=5.0,      # Softened from 5.5
            max_dd_breach_rate=0.06,      # Softened from 0.05
            max_consecutive_loss_rate=0.06, # Softened from 0.05
            evaluation_window=150,        # Reduced from 180
        ),
        max_steps_per_episode=2000,
        include_memory_features=True,
        include_world_model_features=True,
        include_expert_signals=False,   # DISABLED: Live-robustness - agent must work without expert signals
        expert_signal_dropout=0.0,      # No dropout needed when signals are disabled
        allow_demotion=True,
        data_difficulty=DataDifficulty(
            # FULL domain randomization
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
            lr_warmup_steps=18_000,
            reward_blend_enabled=True,
            reward_blend_episodes=35,
            checkpoint_on_transition=True,
            transition_cooldown_episodes=100,
        ),
        skill_requirements=SkillRequirements(
            required_skills={
                TradingSkill.DRAWDOWN_CONTROL: 0.75,
                TradingSkill.PATIENCE: 0.70,
                TradingSkill.EXIT_QUALITY: 0.65,
                TradingSkill.RISK_REWARD: 0.60,
                TradingSkill.CONSISTENCY: 0.60,
            },
            min_confidence=0.62,
            require_all_skills=False,
            weighted_threshold=0.62,
        ),
        entropy_targets=EntropyTargets(
            min_entropy=0.05,
            max_entropy=0.35,
            low_entropy_penalty_scale=0.05,
            high_entropy_penalty_scale=0.08,
            use_in_promotion=True,
        ),
        composite_scoring=CompositeScoringConfig(
            enabled=True,
            promotion_threshold=0.78,
            demotion_threshold=0.42,
            hard_floors={
                "win_rate": 0.52,
                "max_drawdown": 0.055,
                "dd_breach_rate": 0.06,
                "profit_factor": 1.28,
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
            recovery_duration_episodes=300,
        ),
        mixed_stage_sampling=MixedStageSamplingConfig(
            enabled=True,
            current_stage_weight=0.68,
            recent_stages_weight=0.24,
            foundation_weight=0.08,
        ),
        review_session=ReviewSessionConfig(
            enabled=True,
            review_frequency=320,
            review_duration=65,
            review_depth=3,
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
            min_episodes_per_regime=20,
        ),
    )


def get_live_ready_config() -> CurriculumStageConfig:
    """
    PHASE 3 - CONSISTENCY: Stage 7 (LIVE_READY) - TERMINAL
    =======================================================
    Final stage. Strictest variance penalties.
    Must demonstrate consistent profitability for live deployment.
    
    ENABLED: ALL bonuses (same as Phase 2)
    STRONGEST: loss_multiplier = 1.08 (losses hurt 8% MORE)
    MARKET: Full domain randomization
    FOCUS: Prove live-ready consistency
    """
    return CurriculumStageConfig(
        stage=CurriculumStage.LIVE_READY,
        name="Live Ready",
        description="CONSISTENCY Phase: Terminal stage. Prove live-ready.",
        execution=ExecutionDifficulty(
            # FULL domain randomization - worst case scenarios
            base_spread_points=0.22,
            spread_mult_range=(0.70, 1.70),
            max_spread_points=1.25,
            slippage_points_sigma=0.06,
            slippage_mult_range=(0.50, 2.00),
            max_slippage_points=0.50,
            commission_per_lot=5.0,       # Realistic commission: $5/lot (worst case)
            latency_bars=1,
            enable_randomization=True,
            spread_randomization_range=(0.75, 1.60),
            slippage_randomization_range=(0.55, 1.85),
            latency_randomization_range=(0, 3),
            volatility_scale_range=(0.75, 1.40),
            spread_shock_enabled=True,      # Enable spread shocks for robustness
            spread_shock_probability=0.02,   # 2% of steps
            spread_shock_multiplier=3.0,     # 3x spread during news/open/close
        ),
        rewards=RewardShaping(
            # ============================================================================
            # PHASE 3 CONSISTENCY: Stage 7 - TERMINAL STAGE
            # ============================================================================
            # Strictest variance penalty. Losses hurt 8% MORE than wins help.
            # Agent must demonstrate CONSISTENT profitability.
            # This is the final test before live deployment.
            # ============================================================================
            reward_scale=10.0,
            loss_multiplier=1.08,           # LOSSES HURT 8% MORE - strictest variance
            
            # ALL BONUSES ENABLED at full strength
            r_multiple_bonus_threshold=1.2,
            r_multiple_bonus_scale=0.12,
            r_multiple_bonus_cap=0.18,
            
            mae_efficiency_enabled=True,
            mae_efficiency_scale=0.10,
            mae_efficiency_threshold=1.4,
            
            time_efficiency_enabled=True,
            time_efficiency_scale=0.03,
            optimal_trade_bars=7,
            max_trade_bars_for_bonus=20,
            
            exit_quality_enabled=True,
            trailing_stop_bonus=0.25,
            agent_close_bonus=0.0,            # ALWAYS ZERO
            hard_stop_penalty=0.15,
            risk_liquidation_penalty=0.25,
            
            truncation_winner_discount=0.35,
            truncation_loser_extra_penalty=0.18,
            
            entry_quality_integration=True,
            entry_quality_weight=0.28,
            
            dd_shaping_enabled=True,
            dd_threshold=0.018,
            dd_penalty_scale=1.2,
            dd_severity_exponent=1.6,
            dd_severity_cap=1.5,
            
            # Streaks: STRONGEST penalties for losing streaks
            streak_modifier_enabled=True,
            win_streak_bonus_per_win=0.035,
            loss_streak_penalty_per_loss=0.06,   # STRONGEST variance penalty
            
            anti_churn_enabled=True,
            daily_trade_soft_limit=7,
            churn_penalty_per_trade=0.03,
            
            hard_block_penalty=0.05,
            soft_block_penalty=0.025,
            per_step_shaping_enabled=False,
            holding_cost_per_bar=0.0,
            opportunity_bonus_scale=0.0,
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
            min_minutes_between_entries=18,
            min_minutes_after_loss=30,
            daily_drawdown_limit=0.05,
            max_drawdown_limit=0.10,
            daily_dd_safety_buffer=0.008,
            max_dd_safety_buffer=0.015,
            emergency_close_threshold=0.09,
            entry_quality_gate_enabled=True,
            entry_quality_threshold=0.48,
            hard_stop_loss_eur=210.0,
            soft_stop_loss_eur=140.0,
            trailing_activation_eur=100.0,
            trailing_retrace_pct=0.28,
            time_decay_hours=4.0,
            risk_per_trade_pct=0.003,
            max_risk_per_trade_pct=0.006,
        ),
        competence=CompetenceThresholds(
            # TERMINAL: Strict consistency (but achievable from Stage 6)
            min_episodes=400,             # Reduced from 500
            min_timesteps=1_200_000,      # Reduced from 1.5M
            min_win_rate=0.52,            # Softened from 0.54
            min_profit_factor=1.35,       # Softened from 1.40
            max_avg_drawdown=0.05,        # Softened from 0.045
            min_avg_pnl=250.0,            # Softened from 300
            min_avg_r_multiple=0.12,      # Softened from 0.15
            min_entropy=0.04,
            max_win_rate_std=0.08,        # Softened from 0.06 - realistic
            max_pnl_std=7000.0,           # Softened from 6000 - realistic
            min_trade_count_avg=5.0,      # Softened from 5.5
            max_dd_breach_rate=0.05,      # Softened from 0.04
            max_consecutive_loss_rate=0.05, # Softened from 0.04
            evaluation_window=160,        # Reduced from 200
        ),
        max_steps_per_episode=2000,
        include_memory_features=True,
        include_world_model_features=True,
        include_expert_signals=False,   # DISABLED: Live-robustness - no expert signals in terminal stage
        expert_signal_dropout=0.0,      # No dropout needed when signals are disabled
        allow_demotion=True,
        is_terminal=True,
        data_difficulty=DataDifficulty(
            # FULL domain randomization
            volatility_percentile_range=(0.0, 1.0),
            min_trend_clarity=0.0,
            include_asian_session=True,
            include_london_session=True,
            include_ny_session=True,
            include_overlap_sessions=True,
            exclude_high_impact_news=False,
            exclude_market_open_close=False,
            prefer_recent_data=True,
            recent_data_weight=1.5,
        ),
        transition=TransitionSettings(
            lr_warmup_enabled=True,
            lr_warmup_factor=0.20,
            lr_warmup_steps=20_000,
            reward_blend_enabled=True,
            reward_blend_episodes=40,
            checkpoint_on_transition=True,
            transition_cooldown_episodes=120,
        ),
        skill_requirements=SkillRequirements(
            required_skills={
                TradingSkill.DRAWDOWN_CONTROL: 0.80,
                TradingSkill.PATIENCE: 0.75,
                TradingSkill.EXIT_QUALITY: 0.70,
                TradingSkill.RISK_REWARD: 0.65,
                TradingSkill.CONSISTENCY: 0.70,
            },
            min_confidence=0.68,
            require_all_skills=False,
            weighted_threshold=0.68,
        ),
        entropy_targets=EntropyTargets(
            min_entropy=0.04,
            max_entropy=0.30,
            low_entropy_penalty_scale=0.04,
            high_entropy_penalty_scale=0.10,
            use_in_promotion=False,  # Terminal stage
        ),
        composite_scoring=CompositeScoringConfig(
            enabled=True,
            promotion_threshold=0.82,  # Terminal stage
            demotion_threshold=0.45,
            hard_floors={
                "win_rate": 0.53,
                "max_drawdown": 0.05,
                "dd_breach_rate": 0.05,
                "profit_factor": 1.35,
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
            recovery_duration_episodes=350,
        ),
        mixed_stage_sampling=MixedStageSamplingConfig(
            enabled=True,
            current_stage_weight=0.65,
            recent_stages_weight=0.27,
            foundation_weight=0.08,
        ),
        review_session=ReviewSessionConfig(
            enabled=True,
            review_frequency=280,
            review_duration=75,
            review_depth=4,
        ),
        validation=ValidationConfig(
            enabled=True,
            validation_episodes=150,
            min_performance_ratio=0.90,
            max_performance_drop=0.10,
            required_regimes=[
                MarketRegime.TRENDING_UP,
                MarketRegime.TRENDING_DOWN,
                MarketRegime.RANGING,
                MarketRegime.HIGH_VOLATILITY,
                MarketRegime.LOW_VOLATILITY,
            ],
            min_episodes_per_regime=25,
        ),
    )


# =============================================================================
# Registry and Helpers
# =============================================================================

CURRICULUM_CONFIGS: Dict[CurriculumStage, Callable[[], CurriculumStageConfig]] = {
    CurriculumStage.FOUNDATION: get_foundation_config,
    CurriculumStage.DISCIPLINE: get_discipline_config,
    CurriculumStage.MARKET_STRUCTURE: get_market_structure_config,
    CurriculumStage.ECONOMIC_LOGIC: get_economic_logic_config,
    CurriculumStage.PROFESSIONAL: get_professional_config,
    CurriculumStage.ADAPTIVE: get_adaptive_config,
    CurriculumStage.SPECIALIST: get_specialist_config,
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
    print("3-PHASE PROP FIRM CURRICULUM SUMMARY")
    print("=" * 80)
    
    phases = {
        "PHASE 1 - SURVIVAL": [CurriculumStage.FOUNDATION, CurriculumStage.DISCIPLINE, CurriculumStage.MARKET_STRUCTURE],
        "PHASE 2 - PROFITABILITY": [CurriculumStage.ECONOMIC_LOGIC, CurriculumStage.PROFESSIONAL, CurriculumStage.ADAPTIVE],
        "PHASE 3 - CONSISTENCY": [CurriculumStage.SPECIALIST, CurriculumStage.LIVE_READY],
    }
    
    for phase_name, stages in phases.items():
        print(f"\n{phase_name}")
        print("-" * 40)
        for stage in stages:
            cfg = get_stage_config(stage)
            c = cfg.competence
            e = cfg.execution
            print(f"  Stage {stage.value}: {cfg.name}")
            print(f"    WR >= {c.min_win_rate:.0%}, PF >= {c.min_profit_factor:.2f}, DD <= {c.max_avg_drawdown:.1%}")
            print(f"    Commission: ${e.commission_per_lot}/lot, Expert Signals: {cfg.include_expert_signals}")
    
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