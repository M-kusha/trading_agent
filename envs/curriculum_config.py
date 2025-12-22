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

    allow_demotion: bool = False
    is_terminal: bool = False
    
    data_difficulty: DataDifficulty = None  # type: ignore[assignment]
    transition: TransitionSettings = None  # type: ignore[assignment]
    
    # New v2.0 configurations
    skill_requirements: SkillRequirements = None  # type: ignore[assignment]
    entropy_targets: EntropyTargets = None  # type: ignore[assignment]
    composite_scoring: CompositeScoringConfig = None  # type: ignore[assignment]
    adaptive_thresholds: AdaptiveThresholdConfig = None  # type: ignore[assignment]
    recovery_protocol: RecoveryProtocolConfig = None  # type: ignore[assignment]
    mixed_stage_sampling: MixedStageSamplingConfig = None  # type: ignore[assignment]
    review_session: ReviewSessionConfig = None  # type: ignore[assignment]
    validation: ValidationConfig = None  # type: ignore[assignment]
    
    def __post_init__(self) -> None:
        if self.data_difficulty is None:
            self.data_difficulty = DataDifficulty()
        if self.transition is None:
            self.transition = TransitionSettings()
        if self.skill_requirements is None:
            self.skill_requirements = SkillRequirements()
        if self.entropy_targets is None:
            self.entropy_targets = EntropyTargets()
        if self.composite_scoring is None:
            self.composite_scoring = CompositeScoringConfig()
        if self.adaptive_thresholds is None:
            self.adaptive_thresholds = AdaptiveThresholdConfig()
        if self.recovery_protocol is None:
            self.recovery_protocol = RecoveryProtocolConfig()
        if self.mixed_stage_sampling is None:
            self.mixed_stage_sampling = MixedStageSamplingConfig()
        if self.review_session is None:
            self.review_session = ReviewSessionConfig()
        if self.validation is None:
            self.validation = ValidationConfig()


# =============================================================================
# Stage Factory Functions
# =============================================================================

def get_foundation_config() -> CurriculumStageConfig:
    return CurriculumStageConfig(
        stage=CurriculumStage.FOUNDATION,
        name="Foundation",
        description="Learn trading mechanics without punishment. Build intuition for price movement.",
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
            # FOUNDATION: PURE PnL LEARNING - MINIMAL BONUSES
            # ============================================================================
            # Goal: Agent learns that PnL matters. Bonuses kept near-zero to ensure
            # the base reward signal dominates. Breakeven should be ~47-50% win rate.
            # 
            # Math for $300 trade (0.3% of 100k):
            #   base_reward = 0.003 * 5.0 = 0.015
            #   max_bonus   = 0.003 (20% of base) 
            #   Breakeven with bonuses: ~47% win rate
            # ============================================================================
            reward_scale=5.0,           # Moderate scale
            loss_multiplier=0.95,       # 95% penalty - slight leniency for learning
            
            # R-multiple bonus: DISABLED in Foundation - keep it simple
            r_multiple_bonus_threshold=99.0,  # Effectively disabled
            r_multiple_bonus_scale=0.0,
            r_multiple_bonus_cap=0.0,
            
            # MAE efficiency: DISABLED
            mae_efficiency_enabled=False,
            mae_efficiency_scale=0.0,
            mae_efficiency_threshold=99.0,
            
            # Time efficiency: DISABLED
            time_efficiency_enabled=False,
            time_efficiency_scale=0.0,
            optimal_trade_bars=12,
            max_trade_bars_for_bonus=48,
            
            # Exit quality: DISABLED (learn to trade first)
            exit_quality_enabled=False,
            trailing_stop_bonus=0.0,
            agent_close_bonus=0.0,
            hard_stop_penalty=0.0,
            risk_liquidation_penalty=0.0,
            
            # Truncation handling: Keep
            truncation_winner_discount=0.30,
            truncation_loser_extra_penalty=0.15,
            
            # Entry quality: DISABLED
            entry_quality_integration=False,
            entry_quality_weight=0.0,
            
            # DD shaping: Light touch
            dd_shaping_enabled=True,
            dd_threshold=0.10,          # Only start at 10% DD
            dd_penalty_scale=0.25,      # Light penalty
            dd_severity_exponent=1.2,
            dd_severity_cap=0.5,
            
            # Streaks: DISABLED
            streak_modifier_enabled=False,
            win_streak_bonus_per_win=0.0,
            loss_streak_penalty_per_loss=0.0,
            
            # Anti-churn: DISABLED
            anti_churn_enabled=False,
            daily_trade_soft_limit=100,
            churn_penalty_per_trade=0.0,
            
            hard_block_penalty=0.01,
            soft_block_penalty=0.005,
            per_step_shaping_enabled=False,
            exploration_bonus=0.002,    # Tiny exploration bonus
            directional_accuracy_weight=1.0,
            min_reward=-2.5,
            max_reward=2.5,
        ),
        constraints=TradingConstraints(
            max_positions=1,
            max_trades_per_day=100,
            max_trades_per_session=50,
            max_consecutive_losses=20,
            enforce_session_windows=False,
            enforce_no_new_trades_window=False,
            enforce_weekend_block=False,
            enforce_hard_close=False,
            min_minutes_between_entries=0,
            min_minutes_after_loss=0,
            daily_drawdown_limit=0.50,
            max_drawdown_limit=0.50,
            daily_dd_safety_buffer=0.0,
            max_dd_safety_buffer=0.0,
            emergency_close_threshold=0.45,
            entry_quality_gate_enabled=False,
            entry_quality_threshold=0.0,
            hard_stop_loss_eur=500.0,
            soft_stop_loss_eur=300.0,
            trailing_activation_eur=200.0,
            trailing_retrace_pct=0.50,
            time_decay_hours=12.0,
            risk_per_trade_pct=0.005,
            max_risk_per_trade_pct=0.01,
        ),
        competence=CompetenceThresholds(
            min_episodes=250,
            min_timesteps=200_000,
            min_win_rate=0.38,  # Raised from 0.35 - need better than random
            min_profit_factor=0.75,  # Raised from 0.6 - need near-breakeven
            max_avg_drawdown=0.25,  # Tightened from 0.30
            min_avg_pnl=-150.0,  # CRITICAL: Raised from -500 - can't hemorrhage money
            min_avg_r_multiple=-0.15,  # CRITICAL: Raised from 0.0 - must show risk awareness
            min_entropy=0.12,  # Slightly reduced
            max_win_rate_std=0.22,  # Tightened from 0.25
            max_pnl_std=3000.0,  # Tightened from 5000
            min_trade_count_avg=2.0,
            max_dd_breach_rate=0.40,
            max_consecutive_loss_rate=0.25,
            evaluation_window=75,
        ),
        max_steps_per_episode=1500,
        include_memory_features=False,
        include_world_model_features=False,
        include_expert_signals=True,
        allow_demotion=False,
        data_difficulty=DataDifficulty(
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
            lr_warmup_enabled=False,
            lr_warmup_factor=1.0,
            lr_warmup_steps=0,
            reward_blend_enabled=False,
            reward_blend_episodes=0,
            checkpoint_on_transition=True,
            transition_cooldown_episodes=30,
        ),
        # Skill requirements for Foundation - very basic
        skill_requirements=SkillRequirements(
            required_skills={
                TradingSkill.ENTRY_TIMING: 0.30,
                TradingSkill.DRAWDOWN_CONTROL: 0.40,
            },
            min_confidence=0.4,
            require_all_skills=False,
            weighted_threshold=0.35,
        ),
        # High entropy targets - encourage exploration
        entropy_targets=EntropyTargets(
            min_entropy=0.30,
            max_entropy=0.80,
            low_entropy_penalty_scale=0.15,
            high_entropy_penalty_scale=0.02,
            use_in_promotion=True,
        ),
        composite_scoring=CompositeScoringConfig(
            enabled=True,
            promotion_threshold=0.68,  # Raised from 0.65 - needs real competence
            demotion_threshold=0.25,
            # Hard floors must match or exceed traditional thresholds
            hard_floors={
                "win_rate": 0.38,       # Raised from 0.35
                "max_drawdown": 0.25,   # Tightened from 0.30
                "dd_breach_rate": 0.35, # Tightened from 0.40
                "profit_factor": 0.70,  # NEW: Must show some profitability sense
            },
        ),
        adaptive_thresholds=AdaptiveThresholdConfig(
            enabled=True,
            plateau_episodes_threshold=150,
            max_relaxation=0.15,
        ),
        recovery_protocol=RecoveryProtocolConfig(
            enabled=False,  # No recovery in foundation
        ),
        mixed_stage_sampling=MixedStageSamplingConfig(
            enabled=False,  # No mixing in foundation
        ),
        review_session=ReviewSessionConfig(
            enabled=False,  # No reviews in foundation
        ),
    )


def get_discipline_config() -> CurriculumStageConfig:
    return CurriculumStageConfig(
        stage=CurriculumStage.DISCIPLINE,
        name="Discipline",
        description="Learn that randomness loses money. Develop accuracy and precision.",
        execution=ExecutionDifficulty(
            base_spread_points=0.08,
            spread_mult_range=(0.95, 1.15),
            max_spread_points=0.25,
            slippage_points_sigma=0.02,
            slippage_mult_range=(0.8, 1.3),
            max_slippage_points=0.10,
            commission_per_lot=0.0,
            latency_bars=0,
            enable_randomization=True,
            spread_randomization_range=(0.95, 1.10),
            slippage_randomization_range=(0.90, 1.15),
            latency_randomization_range=(0, 0),
            volatility_scale_range=(0.95, 1.10),
        ),
        rewards=RewardShaping(
            # ============================================================================
            # DISCIPLINE: INTRODUCE SMALL BONUSES - BASE PnL STILL DOMINANT
            # ============================================================================
            # Goal: Agent learns that QUALITY wins matter. Small bonuses introduced
            # but capped at ~25% of base reward. Breakeven: ~45-48% win rate.
            #
            # Math for $300 trade (0.3% of 100k):
            #   base_reward = 0.003 * 6.0 = 0.018
            #   max_bonus   = 0.0045 (25% of base)
            #   Win with bonus: 0.0225, Loss: -0.018
            #   Breakeven: 0.018 / (0.018 + 0.0225) = 44.4%
            # ============================================================================
            reward_scale=6.0,
            loss_multiplier=1.0,        # SYMMETRIC - losses hurt equally
            
            # R-multiple bonus: SMALL - reward exceptional R trades
            r_multiple_bonus_threshold=2.0,   # Need R > 2.0 to get any bonus
            r_multiple_bonus_scale=0.02,      # Very small scale
            r_multiple_bonus_cap=0.003,       # Cap at 0.003 (~17% of base)
            
            # MAE efficiency: Introduce lightly
            mae_efficiency_enabled=True,
            mae_efficiency_scale=0.001,       # Tiny - max ~6% of base
            mae_efficiency_threshold=3.0,     # High threshold
            
            # Time efficiency: Light
            time_efficiency_enabled=True,
            time_efficiency_scale=0.001,      # Tiny
            optimal_trade_bars=12,
            max_trade_bars_for_bonus=36,
            
            # Exit quality: Introduce with SMALL bonuses
            exit_quality_enabled=True,
            trailing_stop_bonus=0.001,        # Tiny bonus for trailing stop
            agent_close_bonus=0.0005,         # Even smaller for manual close
            hard_stop_penalty=0.002,          # Small penalty for hard stop
            risk_liquidation_penalty=0.005,   # Larger for emergency close
            
            truncation_winner_discount=0.30,
            truncation_loser_extra_penalty=0.15,
            
            # Entry quality: Light integration
            entry_quality_integration=True,
            entry_quality_weight=0.10,        # Reduced from 0.15
            
            # DD shaping: Active
            dd_shaping_enabled=True,
            dd_threshold=0.04,                # Start penalty at 4% DD
            dd_penalty_scale=0.5,
            dd_severity_exponent=1.3,
            dd_severity_cap=1.0,
            
            # Streaks: SMALL modifiers
            streak_modifier_enabled=True,
            win_streak_bonus_per_win=0.0005,  # Tiny: 5 wins = 0.0025 bonus
            loss_streak_penalty_per_loss=0.001, # Small: 3 losses = 0.003 penalty
            
            # Anti-churn: Active
            anti_churn_enabled=True,
            daily_trade_soft_limit=25,
            churn_penalty_per_trade=0.002,
            
            hard_block_penalty=0.02,
            soft_block_penalty=0.01,
            per_step_shaping_enabled=False,
            exploration_bonus=0.0,
            directional_accuracy_weight=1.0,
            min_reward=-3.0,
            max_reward=3.0,
        ),
        constraints=TradingConstraints(
            max_positions=1,
            max_trades_per_day=30,
            max_trades_per_session=15,
            max_consecutive_losses=8,
            enforce_session_windows=False,
            enforce_no_new_trades_window=False,
            enforce_weekend_block=False,
            enforce_hard_close=False,
            min_minutes_between_entries=3,
            min_minutes_after_loss=5,
            daily_drawdown_limit=0.15,
            max_drawdown_limit=0.25,
            daily_dd_safety_buffer=0.01,
            max_dd_safety_buffer=0.02,
            emergency_close_threshold=0.22,
            entry_quality_gate_enabled=True,
            entry_quality_threshold=0.25,
            hard_stop_loss_eur=350.0,
            soft_stop_loss_eur=200.0,
            trailing_activation_eur=150.0,
            trailing_retrace_pct=0.40,
            time_decay_hours=8.0,
            risk_per_trade_pct=0.004,
            max_risk_per_trade_pct=0.008,
        ),
        competence=CompetenceThresholds(
            min_episodes=200,  # Raised from 150 - need more experience
            min_timesteps=300_000,  # Raised from 200_000
            min_win_rate=0.42,  # RAISED from 0.38 - need consistent winning
            min_profit_factor=0.90,  # RAISED from 0.75 - need near-breakeven minimum
            max_avg_drawdown=0.12,  # TIGHTENED from 0.18
            min_avg_pnl=-50.0,  # CRITICAL: Raised from -200 - stop promoting losers!
            min_avg_r_multiple=0.05,  # RAISED from 0.02 - must have positive expectancy
            min_entropy=0.10,  # Slightly reduced
            max_win_rate_std=0.18,  # TIGHTENED from 0.22 - need consistency
            max_pnl_std=2500.0,  # TIGHTENED from 5000
            min_trade_count_avg=3.5,  # Raised from 2.5 - need trading activity
            max_dd_breach_rate=0.20,  # TIGHTENED from 0.30
            max_consecutive_loss_rate=0.15,  # TIGHTENED from 0.20
            evaluation_window=100,  # Raised from 75 - need longer track record
        ),
        max_steps_per_episode=1800,
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
            exclude_high_impact_news=True,
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
            required_skills={
                TradingSkill.ENTRY_TIMING: 0.40,
                TradingSkill.DRAWDOWN_CONTROL: 0.50,
                TradingSkill.PATIENCE: 0.45,
                TradingSkill.LOSS_MANAGEMENT: 0.40,
            },
            min_confidence=0.5,
            require_all_skills=False,
            weighted_threshold=0.45,
        ),
        entropy_targets=EntropyTargets(
            min_entropy=0.20,
            max_entropy=0.60,
            low_entropy_penalty_scale=0.12,
            high_entropy_penalty_scale=0.04,
            use_in_promotion=True,
        ),
        composite_scoring=CompositeScoringConfig(
            enabled=True,
            promotion_threshold=0.70,  # RAISED from 0.65 - need real competence
            demotion_threshold=0.35,  # Raised from 0.30 - quicker to demote poor performance
            hard_floors={
                "win_rate": 0.42,       # Match min_win_rate (raised)
                "max_drawdown": 0.12,   # Match max_avg_drawdown (tightened)
                "dd_breach_rate": 0.20, # Match max_dd_breach_rate (tightened)
                "profit_factor": 0.85,  # NEW: Must be near-profitable
                "r_multiple": -0.05,    # NEW: R-multiple can't be too negative
            },
        ),
        adaptive_thresholds=AdaptiveThresholdConfig(
            enabled=True,
            plateau_episodes_threshold=120,
            max_relaxation=0.12,
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
            enabled=False,
        ),
    )


def get_market_structure_config() -> CurriculumStageConfig:
    return CurriculumStageConfig(
        stage=CurriculumStage.MARKET_STRUCTURE,
        name="Market Structure",
        description="Learn adversarial markets. Demonstrate consistency across regimes.",
        execution=ExecutionDifficulty(
            base_spread_points=0.15,
            spread_mult_range=(0.90, 1.35),
            max_spread_points=0.60,
            slippage_points_sigma=0.04,
            slippage_mult_range=(0.70, 1.60),
            max_slippage_points=0.25,
            commission_per_lot=0.0,
            latency_bars=1,
            enable_randomization=True,
            spread_randomization_range=(0.90, 1.25),
            slippage_randomization_range=(0.85, 1.40),
            latency_randomization_range=(0, 1),
            volatility_scale_range=(0.90, 1.15),
        ),
        rewards=RewardShaping(
            # ============================================================================
            # MARKET_STRUCTURE: MODERATE BONUSES - QUALITY STARTS TO MATTER
            # ============================================================================
            # Goal: Agent learns to adapt to different market conditions.
            # Bonuses now ~30% of base max. Must be actually profitable to advance.
            #
            # Math for $300 trade (0.3% of 100k):
            #   base_reward = 0.003 * 7.0 = 0.021
            #   max_bonus   = 0.006 (30% of base)
            #   Breakeven: ~44%
            # ============================================================================
            reward_scale=7.0,
            loss_multiplier=1.0,        # Symmetric
            
            # R-multiple bonus: Growing
            r_multiple_bonus_threshold=1.8,
            r_multiple_bonus_scale=0.025,
            r_multiple_bonus_cap=0.004,       # ~19% of base
            
            # MAE efficiency
            mae_efficiency_enabled=True,
            mae_efficiency_scale=0.0015,      # ~7% of base
            mae_efficiency_threshold=2.5,
            
            # Time efficiency
            time_efficiency_enabled=True,
            time_efficiency_scale=0.0015,
            optimal_trade_bars=10,
            max_trade_bars_for_bonus=32,
            
            # Exit quality: Growing importance
            exit_quality_enabled=True,
            trailing_stop_bonus=0.0015,
            agent_close_bonus=0.001,
            hard_stop_penalty=0.003,
            risk_liquidation_penalty=0.006,
            
            truncation_winner_discount=0.25,
            truncation_loser_extra_penalty=0.10,
            
            entry_quality_integration=True,
            entry_quality_weight=0.12,
            
            dd_shaping_enabled=True,
            dd_threshold=0.035,
            dd_penalty_scale=0.6,
            dd_severity_exponent=1.3,
            dd_severity_cap=1.0,
            
            streak_modifier_enabled=True,
            win_streak_bonus_per_win=0.0008,
            loss_streak_penalty_per_loss=0.0015,
            
            anti_churn_enabled=True,
            daily_trade_soft_limit=20,
            churn_penalty_per_trade=0.003,
            
            hard_block_penalty=0.025,
            soft_block_penalty=0.012,
            per_step_shaping_enabled=False,
            min_reward=-3.5,
            max_reward=3.5,
        ),
        constraints=TradingConstraints(
            max_positions=1,
            max_trades_per_day=20,
            max_trades_per_session=10,
            max_consecutive_losses=5,
            enforce_session_windows=True,
            enforce_no_new_trades_window=True,
            enforce_weekend_block=True,
            enforce_hard_close=True,
            min_minutes_between_entries=5,
            min_minutes_after_loss=10,
            daily_drawdown_limit=0.08,
            max_drawdown_limit=0.15,
            daily_dd_safety_buffer=0.005,
            max_dd_safety_buffer=0.01,
            emergency_close_threshold=0.13,
            entry_quality_gate_enabled=True,
            entry_quality_threshold=0.35,
            hard_stop_loss_eur=280.0,
            soft_stop_loss_eur=180.0,
            trailing_activation_eur=120.0,
            trailing_retrace_pct=0.35,
            time_decay_hours=6.0,
            risk_per_trade_pct=0.0035,
            max_risk_per_trade_pct=0.007,
        ),
        competence=CompetenceThresholds(
            min_episodes=200,
            min_timesteps=400_000,
            min_win_rate=0.47,
            min_profit_factor=1.0,
            max_avg_drawdown=0.10,
            min_avg_pnl=50.0,
            min_avg_r_multiple=0.05,
            min_entropy=0.08,
            max_win_rate_std=0.12,
            max_pnl_std=1000.0,
            min_trade_count_avg=4.0,
            max_dd_breach_rate=0.15,
            max_consecutive_loss_rate=0.10,
            evaluation_window=100,
        ),
        max_steps_per_episode=2000,
        include_memory_features=True,
        include_world_model_features=True,
        include_expert_signals=True,
        allow_demotion=True,
        data_difficulty=DataDifficulty(
            volatility_percentile_range=(0.05, 0.80),
            min_trend_clarity=0.05,
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
            lr_warmup_steps=10_000,
            reward_blend_enabled=True,
            reward_blend_episodes=20,
            checkpoint_on_transition=True,
            transition_cooldown_episodes=50,
        ),
        skill_requirements=SkillRequirements(
            required_skills={
                TradingSkill.ENTRY_TIMING: 0.50,
                TradingSkill.EXIT_QUALITY: 0.45,
                TradingSkill.DRAWDOWN_CONTROL: 0.60,
                TradingSkill.PATIENCE: 0.55,
                TradingSkill.CONSISTENCY: 0.50,
                TradingSkill.RISK_REWARD: 0.45,
            },
            min_confidence=0.5,
            require_all_skills=False,
            weighted_threshold=0.52,
        ),
        entropy_targets=EntropyTargets(
            min_entropy=0.15,
            max_entropy=0.50,
            low_entropy_penalty_scale=0.10,
            high_entropy_penalty_scale=0.05,
            use_in_promotion=True,
        ),
        composite_scoring=CompositeScoringConfig(
            enabled=True,
            promotion_threshold=0.70,  # Raised for quality
            demotion_threshold=0.35,
            hard_floors={
                "win_rate": 0.47,       # Match min_win_rate
                "max_drawdown": 0.10,   # Match max_avg_drawdown
                "dd_breach_rate": 0.15, # Match max_dd_breach_rate
                "profit_factor": 0.95,  # ADDED: Must be near-profitable
                "r_multiple": 0.02,     # ADDED: Positive expectancy required
            },
        ),
        adaptive_thresholds=AdaptiveThresholdConfig(
            enabled=True,
            plateau_episodes_threshold=100,
            max_relaxation=0.10,
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
            review_frequency=500,
            review_duration=50,
            review_depth=2,
            min_stage_for_review=CurriculumStage.MARKET_STRUCTURE,
        ),
    )


def get_economic_logic_config() -> CurriculumStageConfig:
    return CurriculumStageConfig(
        stage=CurriculumStage.ECONOMIC_LOGIC,
        name="Economic Logic",
        description="Exploit market structure. Build a worldview, not a lookup table.",
        execution=ExecutionDifficulty(
            base_spread_points=0.18,
            spread_mult_range=(0.88, 1.40),
            max_spread_points=0.80,
            slippage_points_sigma=0.05,
            slippage_mult_range=(0.65, 1.70),
            max_slippage_points=0.35,
            commission_per_lot=0.0,
            latency_bars=1,
            enable_randomization=True,
            spread_randomization_range=(0.88, 1.35),
            slippage_randomization_range=(0.80, 1.50),
            latency_randomization_range=(0, 2),
            volatility_scale_range=(0.88, 1.20),
        ),
        rewards=RewardShaping(
            # ============================================================================
            # ECONOMIC_LOGIC: GROWING BONUSES - EFFICIENCY MATTERS MORE
            # ============================================================================
            # Goal: Agent learns economic reasoning. Must exploit market structure.
            # Bonuses ~35% of base max. Agent must show positive edge.
            #
            # Math for $300 trade (0.3% of 100k):
            #   base_reward = 0.003 * 8.0 = 0.024
            #   max_bonus   = 0.0084 (35% of base)
            #   Breakeven: ~42%
            # ============================================================================
            reward_scale=8.0,
            loss_multiplier=1.0,
            
            r_multiple_bonus_threshold=1.6,
            r_multiple_bonus_scale=0.03,
            r_multiple_bonus_cap=0.005,       # ~21% of base
            
            mae_efficiency_enabled=True,
            mae_efficiency_scale=0.002,       # ~8% of base
            mae_efficiency_threshold=2.2,
            
            time_efficiency_enabled=True,
            time_efficiency_scale=0.002,
            optimal_trade_bars=10,
            max_trade_bars_for_bonus=28,
            
            exit_quality_enabled=True,
            trailing_stop_bonus=0.002,
            agent_close_bonus=0.001,
            hard_stop_penalty=0.004,
            risk_liquidation_penalty=0.008,
            
            truncation_winner_discount=0.30,
            truncation_loser_extra_penalty=0.15,
            
            entry_quality_integration=True,
            entry_quality_weight=0.15,
            
            dd_shaping_enabled=True,
            dd_threshold=0.03,
            dd_penalty_scale=0.8,
            dd_severity_exponent=1.4,
            dd_severity_cap=1.2,
            
            streak_modifier_enabled=True,
            win_streak_bonus_per_win=0.001,
            loss_streak_penalty_per_loss=0.002,
            
            anti_churn_enabled=True,
            daily_trade_soft_limit=16,
            churn_penalty_per_trade=0.004,
            
            hard_block_penalty=0.03,
            soft_block_penalty=0.015,
            per_step_shaping_enabled=True,
            holding_cost_per_bar=0.0002,
            opportunity_bonus_scale=0.003,
            min_reward=-4.0,
            max_reward=4.0,
        ),
        constraints=TradingConstraints(
            max_positions=1,
            max_trades_per_day=15,
            max_trades_per_session=8,
            max_consecutive_losses=4,
            enforce_session_windows=True,
            enforce_no_new_trades_window=True,
            enforce_weekend_block=True,
            enforce_hard_close=True,
            min_minutes_between_entries=5,
            min_minutes_after_loss=12,
            daily_drawdown_limit=0.06,
            max_drawdown_limit=0.12,
            daily_dd_safety_buffer=0.006,
            max_dd_safety_buffer=0.012,
            emergency_close_threshold=0.11,
            entry_quality_gate_enabled=True,
            entry_quality_threshold=0.40,
            hard_stop_loss_eur=250.0,
            soft_stop_loss_eur=160.0,
            trailing_activation_eur=110.0,
            trailing_retrace_pct=0.32,
            time_decay_hours=5.0,
            risk_per_trade_pct=0.0032,
            max_risk_per_trade_pct=0.0065,
        ),
        competence=CompetenceThresholds(
            min_episodes=250,
            min_timesteps=600_000,
            min_win_rate=0.50,
            min_profit_factor=1.2,
            max_avg_drawdown=0.07,
            min_avg_pnl=150.0,
            min_avg_r_multiple=0.08,
            min_entropy=0.05,
            max_win_rate_std=0.10,
            max_pnl_std=600.0,
            min_trade_count_avg=4.5,
            max_dd_breach_rate=0.10,
            max_consecutive_loss_rate=0.08,
            evaluation_window=100,
        ),
        max_steps_per_episode=2000,
        include_memory_features=True,
        include_world_model_features=True,
        include_expert_signals=True,
        allow_demotion=True,
        data_difficulty=DataDifficulty(
            volatility_percentile_range=(0.1, 0.95),
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
            lr_warmup_factor=0.3,
            lr_warmup_steps=12_000,
            reward_blend_enabled=True,
            reward_blend_episodes=25,
            checkpoint_on_transition=True,
            transition_cooldown_episodes=60,
        ),
        skill_requirements=SkillRequirements(
            required_skills={
                TradingSkill.ENTRY_TIMING: 0.55,
                TradingSkill.EXIT_QUALITY: 0.55,
                TradingSkill.DRAWDOWN_CONTROL: 0.65,
                TradingSkill.PATIENCE: 0.60,
                TradingSkill.TREND_ALIGNMENT: 0.50,
                TradingSkill.RISK_REWARD: 0.55,
                TradingSkill.CONSISTENCY: 0.55,
            },
            min_confidence=0.55,
            require_all_skills=False,
            weighted_threshold=0.57,
        ),
        entropy_targets=EntropyTargets(
            min_entropy=0.10,
            max_entropy=0.40,
            low_entropy_penalty_scale=0.08,
            high_entropy_penalty_scale=0.06,
            use_in_promotion=True,
        ),
        composite_scoring=CompositeScoringConfig(
            enabled=True,
            promotion_threshold=0.72,  # Raised for quality
            demotion_threshold=0.38,
            hard_floors={
                "win_rate": 0.50,       # Match min_win_rate
                "max_drawdown": 0.07,   # Match max_avg_drawdown
                "dd_breach_rate": 0.10, # Match max_dd_breach_rate
                "profit_factor": 1.1,   # ADDED: Must be profitable
                "r_multiple": 0.05,     # ADDED: Positive expectancy required
            },
        ),
        adaptive_thresholds=AdaptiveThresholdConfig(
            enabled=True,
            plateau_episodes_threshold=100,
            max_relaxation=0.08,
        ),
        recovery_protocol=RecoveryProtocolConfig(
            enabled=True,
            trigger_after_demotions=2,
            recovery_duration_episodes=200,
        ),
        mixed_stage_sampling=MixedStageSamplingConfig(
            enabled=True,
            current_stage_weight=0.72,
            recent_stages_weight=0.20,
            foundation_weight=0.08,
        ),
        review_session=ReviewSessionConfig(
            enabled=True,
            review_frequency=450,
            review_duration=50,
            review_depth=2,
        ),
    )


def get_professional_config() -> CurriculumStageConfig:
    return CurriculumStageConfig(
        stage=CurriculumStage.PROFESSIONAL,
        name="Professional Trading",
        description="Full prop-firm reality. Discipline is survival.",
        execution=ExecutionDifficulty(
            base_spread_points=0.20,
            spread_mult_range=(0.85, 1.45),
            max_spread_points=1.00,
            slippage_points_sigma=0.05,
            slippage_mult_range=(0.60, 1.80),
            max_slippage_points=0.40,
            commission_per_lot=0.0,
            latency_bars=1,
            enable_randomization=True,
            spread_randomization_range=(0.85, 1.40),
            slippage_randomization_range=(0.75, 1.60),
            latency_randomization_range=(0, 2),
            volatility_scale_range=(0.85, 1.25),
        ),
        rewards=RewardShaping(
            # ============================================================================
            # PROFESSIONAL: SUBSTANTIAL BONUSES - EXCELLENCE REWARDED
            # ============================================================================
            # Goal: Real prop firm conditions. Agent must show professional discipline.
            # Bonuses ~40% of base max. Only consistently profitable agents advance.
            #
            # Math for $300 trade (0.3% of 100k):
            #   base_reward = 0.003 * 9.0 = 0.027
            #   max_bonus   = 0.0108 (40% of base)
            #   Breakeven: ~41%
            # ============================================================================
            reward_scale=9.0,
            loss_multiplier=1.0,
            
            r_multiple_bonus_threshold=1.5,
            r_multiple_bonus_scale=0.035,
            r_multiple_bonus_cap=0.006,       # ~22% of base
            
            mae_efficiency_enabled=True,
            mae_efficiency_scale=0.0025,      # ~9% of base
            mae_efficiency_threshold=2.0,
            
            time_efficiency_enabled=True,
            time_efficiency_scale=0.0025,
            optimal_trade_bars=8,
            max_trade_bars_for_bonus=24,
            
            exit_quality_enabled=True,
            trailing_stop_bonus=0.0025,
            agent_close_bonus=0.0012,
            hard_stop_penalty=0.005,
            risk_liquidation_penalty=0.01,
            
            truncation_winner_discount=0.30,
            truncation_loser_extra_penalty=0.15,
            
            entry_quality_integration=True,
            entry_quality_weight=0.18,
            
            dd_shaping_enabled=True,
            dd_threshold=0.025,
            dd_penalty_scale=1.0,
            dd_severity_exponent=1.5,
            dd_severity_cap=1.3,
            
            streak_modifier_enabled=True,
            win_streak_bonus_per_win=0.0012,
            loss_streak_penalty_per_loss=0.0025,
            
            anti_churn_enabled=True,
            daily_trade_soft_limit=12,
            churn_penalty_per_trade=0.005,
            
            hard_block_penalty=0.035,
            soft_block_penalty=0.018,
            per_step_shaping_enabled=True,
            holding_cost_per_bar=0.0003,
            opportunity_bonus_scale=0.006,
            min_reward=-4.5,
            max_reward=4.5,
        ),
        constraints=TradingConstraints(
            max_positions=1,
            max_trades_per_day=12,
            max_trades_per_session=6,
            max_consecutive_losses=3,
            enforce_session_windows=True,
            enforce_no_new_trades_window=True,
            enforce_weekend_block=True,
            enforce_hard_close=True,
            min_minutes_between_entries=5,
            min_minutes_after_loss=15,
            daily_drawdown_limit=0.05,
            max_drawdown_limit=0.10,
            daily_dd_safety_buffer=0.008,
            max_dd_safety_buffer=0.015,
            emergency_close_threshold=0.09,
            entry_quality_gate_enabled=True,
            entry_quality_threshold=0.45,
            hard_stop_loss_eur=220.0,
            soft_stop_loss_eur=140.0,
            trailing_activation_eur=100.0,
            trailing_retrace_pct=0.30,
            time_decay_hours=4.0,
            risk_per_trade_pct=0.003,
            max_risk_per_trade_pct=0.006,
        ),
        competence=CompetenceThresholds(
            min_episodes=300,
            min_timesteps=1_000_000,
            min_win_rate=0.52,
            min_profit_factor=1.3,
            max_avg_drawdown=0.05,
            min_avg_pnl=250.0,
            min_avg_r_multiple=0.12,
            min_entropy=0.03,
            max_win_rate_std=0.08,
            max_pnl_std=400.0,
            min_trade_count_avg=5.0,
            max_dd_breach_rate=0.05,
            max_consecutive_loss_rate=0.05,
            evaluation_window=150,
        ),
        max_steps_per_episode=2000,
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
            lr_warmup_factor=0.3,
            lr_warmup_steps=15_000,
            reward_blend_enabled=True,
            reward_blend_episodes=25,
            checkpoint_on_transition=True,
            transition_cooldown_episodes=75,
        ),
        skill_requirements=SkillRequirements(
            required_skills={
                TradingSkill.ENTRY_TIMING: 0.60,
                TradingSkill.EXIT_QUALITY: 0.60,
                TradingSkill.DRAWDOWN_CONTROL: 0.70,
                TradingSkill.PATIENCE: 0.65,
                TradingSkill.TREND_ALIGNMENT: 0.55,
                TradingSkill.RISK_REWARD: 0.60,
                TradingSkill.CONSISTENCY: 0.60,
                TradingSkill.LOSS_MANAGEMENT: 0.60,
            },
            min_confidence=0.6,
            require_all_skills=False,
            weighted_threshold=0.62,
        ),
        entropy_targets=EntropyTargets(
            min_entropy=0.05,
            max_entropy=0.30,
            low_entropy_penalty_scale=0.06,
            high_entropy_penalty_scale=0.08,
            use_in_promotion=True,
        ),
        composite_scoring=CompositeScoringConfig(
            enabled=True,
            promotion_threshold=0.76,  # Raised for quality
            demotion_threshold=0.40,
            hard_floors={
                "win_rate": 0.52,       # Match min_win_rate
                "max_drawdown": 0.05,   # Match max_avg_drawdown
                "dd_breach_rate": 0.05, # Match max_dd_breach_rate
                "profit_factor": 1.2,   # ADDED: Must be clearly profitable
                "r_multiple": 0.08,     # ADDED: Good risk/reward required
            },
        ),
        adaptive_thresholds=AdaptiveThresholdConfig(
            enabled=True,
            plateau_episodes_threshold=100,
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
            review_frequency=400,
            review_duration=60,
            review_depth=2,
        ),
    )


def get_adaptive_config() -> CurriculumStageConfig:
    return CurriculumStageConfig(
        stage=CurriculumStage.ADAPTIVE,
        name="Adaptive Intelligence",
        description="Earn complexity. Detect market regime changes.",
        execution=ExecutionDifficulty(
            base_spread_points=0.20,
            spread_mult_range=(0.85, 1.50),
            max_spread_points=1.20,
            slippage_points_sigma=0.05,
            slippage_mult_range=(0.55, 1.90),
            max_slippage_points=0.45,
            commission_per_lot=0.0,
            latency_bars=1,
            enable_randomization=True,
            spread_randomization_range=(0.82, 1.45),
            slippage_randomization_range=(0.70, 1.70),
            latency_randomization_range=(0, 3),
            volatility_scale_range=(0.82, 1.30),
        ),
        rewards=RewardShaping(
            # ============================================================================
            # ADAPTIVE: HIGH BONUSES - EXCELLENCE STRONGLY REWARDED
            # ============================================================================
            # Goal: Agent learns to adapt to regime changes. Quality matters most.
            # Bonuses ~45% of base max. Loss multiplier > 1.0 (losses hurt MORE).
            #
            # Math for $300 trade (0.3% of 100k):
            #   base_reward = 0.003 * 9.5 = 0.0285
            #   max_bonus   = 0.0128 (45% of base)
            #   Loss penalty = 0.003 * 9.5 * 1.02 = 0.029 (slightly worse)
            #   Breakeven: ~42%
            # ============================================================================
            reward_scale=9.5,
            loss_multiplier=1.02,           # Losses hurt slightly MORE
            
            r_multiple_bonus_threshold=1.4,
            r_multiple_bonus_scale=0.04,
            r_multiple_bonus_cap=0.007,       # ~25% of base
            
            mae_efficiency_enabled=True,
            mae_efficiency_scale=0.003,       # ~11% of base
            mae_efficiency_threshold=1.8,
            
            time_efficiency_enabled=True,
            time_efficiency_scale=0.003,
            optimal_trade_bars=8,
            max_trade_bars_for_bonus=22,
            
            exit_quality_enabled=True,
            trailing_stop_bonus=0.003,
            agent_close_bonus=0.0015,
            hard_stop_penalty=0.006,
            risk_liquidation_penalty=0.012,
            
            truncation_winner_discount=0.30,
            truncation_loser_extra_penalty=0.15,
            
            entry_quality_integration=True,
            entry_quality_weight=0.20,
            
            dd_shaping_enabled=True,
            dd_threshold=0.02,
            dd_penalty_scale=1.2,
            dd_severity_exponent=1.5,
            dd_severity_cap=1.4,
            
            streak_modifier_enabled=True,
            win_streak_bonus_per_win=0.0015,
            loss_streak_penalty_per_loss=0.003,
            
            anti_churn_enabled=True,
            daily_trade_soft_limit=10,
            churn_penalty_per_trade=0.006,
            
            hard_block_penalty=0.04,
            soft_block_penalty=0.02,
            per_step_shaping_enabled=True,
            holding_cost_per_bar=0.0003,
            opportunity_bonus_scale=0.008,
            min_reward=-5.0,
            max_reward=5.0,
        ),
        constraints=TradingConstraints(
            max_positions=1,
            max_trades_per_day=12,
            max_trades_per_session=6,
            max_consecutive_losses=3,
            enforce_session_windows=True,
            enforce_no_new_trades_window=True,
            enforce_weekend_block=True,
            enforce_hard_close=True,
            min_minutes_between_entries=5,
            min_minutes_after_loss=15,
            daily_drawdown_limit=0.05,
            max_drawdown_limit=0.10,
            daily_dd_safety_buffer=0.008,
            max_dd_safety_buffer=0.015,
            emergency_close_threshold=0.09,
            entry_quality_gate_enabled=True,
            entry_quality_threshold=0.50,
            hard_stop_loss_eur=220.0,
            soft_stop_loss_eur=140.0,
            trailing_activation_eur=100.0,
            trailing_retrace_pct=0.30,
            time_decay_hours=4.0,
            risk_per_trade_pct=0.003,
            max_risk_per_trade_pct=0.006,
        ),
        competence=CompetenceThresholds(
            min_episodes=400,
            min_timesteps=2_000_000,
            min_win_rate=0.55,
            min_profit_factor=1.4,
            max_avg_drawdown=0.04,
            min_avg_pnl=350.0,
            min_avg_r_multiple=0.15,
            min_entropy=0.03,
            max_win_rate_std=0.06,
            max_pnl_std=300.0,
            min_trade_count_avg=5.5,
            max_dd_breach_rate=0.03,
            max_consecutive_loss_rate=0.03,
            evaluation_window=200,
        ),
        max_steps_per_episode=2000,
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
            recent_data_weight=1.3,
        ),
        transition=TransitionSettings(
            lr_warmup_enabled=True,
            lr_warmup_factor=0.25,
            lr_warmup_steps=15_000,
            reward_blend_enabled=True,
            reward_blend_episodes=30,
            checkpoint_on_transition=True,
            transition_cooldown_episodes=100,
        ),
        skill_requirements=SkillRequirements(
            required_skills={
                TradingSkill.ENTRY_TIMING: 0.65,
                TradingSkill.EXIT_QUALITY: 0.65,
                TradingSkill.DRAWDOWN_CONTROL: 0.75,
                TradingSkill.PATIENCE: 0.70,
                TradingSkill.TREND_ALIGNMENT: 0.60,
                TradingSkill.RISK_REWARD: 0.65,
                TradingSkill.CONSISTENCY: 0.65,
                TradingSkill.ADAPTATION: 0.55,
            },
            min_confidence=0.6,
            require_all_skills=False,
            weighted_threshold=0.65,
        ),
        entropy_targets=EntropyTargets(
            min_entropy=0.05,
            max_entropy=0.25,
            low_entropy_penalty_scale=0.05,
            high_entropy_penalty_scale=0.08,
            use_in_promotion=True,
        ),
        composite_scoring=CompositeScoringConfig(
            enabled=True,
            promotion_threshold=0.78,  # Raised for quality
            demotion_threshold=0.42,
            hard_floors={
                "win_rate": 0.55,       # Match min_win_rate
                "max_drawdown": 0.04,   # Match max_avg_drawdown
                "dd_breach_rate": 0.03, # Match max_dd_breach_rate
                "profit_factor": 1.3,   # ADDED: Strong profitability required
                "r_multiple": 0.10,     # ADDED: Strong risk/reward required
            },
        ),
        adaptive_thresholds=AdaptiveThresholdConfig(
            enabled=True,
            plateau_episodes_threshold=100,
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
            review_frequency=350,
            review_duration=60,
            review_depth=3,
        ),
    )


def get_specialist_config() -> CurriculumStageConfig:
    return CurriculumStageConfig(
        stage=CurriculumStage.SPECIALIST,
        name="Specialization",
        description="Develop niche advantage. Master specific market conditions.",
        execution=ExecutionDifficulty(
            base_spread_points=0.20,
            spread_mult_range=(0.85, 1.50),
            max_spread_points=1.30,
            slippage_points_sigma=0.05,
            slippage_mult_range=(0.50, 2.00),
            max_slippage_points=0.50,
            commission_per_lot=0.0,
            latency_bars=1,
            enable_randomization=True,
            spread_randomization_range=(0.80, 1.50),
            slippage_randomization_range=(0.65, 1.80),
            latency_randomization_range=(0, 3),
            volatility_scale_range=(0.80, 1.35),
        ),
        rewards=RewardShaping(
            # ============================================================================
            # SPECIALIST: MAXIMUM BONUSES - ELITE PERFORMANCE REWARDED
            # ============================================================================
            # Goal: Agent develops specialized edge in specific conditions.
            # Bonuses ~50% of base max. Losses penalized more than gains rewarded.
            #
            # Math for $300 trade (0.3% of 100k):
            #   base_reward = 0.003 * 10.0 = 0.030
            #   max_bonus   = 0.015 (50% of base)
            #   Loss penalty = 0.003 * 10.0 * 1.03 = 0.031 (losses hurt MORE)
            #   Breakeven: ~42%
            # ============================================================================
            reward_scale=10.0,
            loss_multiplier=1.03,           # Losses hurt 3% more
            
            r_multiple_bonus_threshold=1.3,
            r_multiple_bonus_scale=0.045,
            r_multiple_bonus_cap=0.008,       # ~27% of base
            
            mae_efficiency_enabled=True,
            mae_efficiency_scale=0.0035,      # ~12% of base
            mae_efficiency_threshold=1.7,
            
            time_efficiency_enabled=True,
            time_efficiency_scale=0.0035,
            optimal_trade_bars=7,
            max_trade_bars_for_bonus=20,
            
            exit_quality_enabled=True,
            trailing_stop_bonus=0.0035,
            agent_close_bonus=0.0018,
            hard_stop_penalty=0.007,
            risk_liquidation_penalty=0.014,
            
            truncation_winner_discount=0.30,
            truncation_loser_extra_penalty=0.15,
            
            entry_quality_integration=True,
            entry_quality_weight=0.22,
            
            dd_shaping_enabled=True,
            dd_threshold=0.018,
            dd_penalty_scale=1.3,
            dd_severity_exponent=1.5,
            dd_severity_cap=1.5,
            
            streak_modifier_enabled=True,
            win_streak_bonus_per_win=0.0018,
            loss_streak_penalty_per_loss=0.0035,
            
            anti_churn_enabled=True,
            daily_trade_soft_limit=8,
            churn_penalty_per_trade=0.007,
            
            hard_block_penalty=0.045,
            soft_block_penalty=0.022,
            per_step_shaping_enabled=True,
            holding_cost_per_bar=0.0004,
            opportunity_bonus_scale=0.01,
            min_reward=-5.0,
            max_reward=5.0,
        ),
        constraints=TradingConstraints(
            max_positions=1,
            max_trades_per_day=10,
            max_trades_per_session=5,
            max_consecutive_losses=3,
            enforce_session_windows=True,
            enforce_no_new_trades_window=True,
            enforce_weekend_block=True,
            enforce_hard_close=True,
            min_minutes_between_entries=5,
            min_minutes_after_loss=15,
            daily_drawdown_limit=0.05,
            max_drawdown_limit=0.10,
            daily_dd_safety_buffer=0.008,
            max_dd_safety_buffer=0.015,
            emergency_close_threshold=0.09,
            entry_quality_gate_enabled=True,
            entry_quality_threshold=0.55,
            hard_stop_loss_eur=220.0,
            soft_stop_loss_eur=140.0,
            trailing_activation_eur=100.0,
            trailing_retrace_pct=0.30,
            time_decay_hours=4.0,
            risk_per_trade_pct=0.003,
            max_risk_per_trade_pct=0.006,
        ),
        competence=CompetenceThresholds(
            min_episodes=500,
            min_timesteps=3_000_000,
            min_win_rate=0.57,
            min_profit_factor=1.5,
            max_avg_drawdown=0.035,
            min_avg_pnl=450.0,
            min_avg_r_multiple=0.18,
            min_entropy=0.02,
            max_win_rate_std=0.05,
            max_pnl_std=250.0,
            min_trade_count_avg=5.5,
            max_dd_breach_rate=0.02,
            max_consecutive_loss_rate=0.02,
            evaluation_window=250,
        ),
        max_steps_per_episode=2000,
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
            recent_data_weight=1.5,
        ),
        transition=TransitionSettings(
            lr_warmup_enabled=True,
            lr_warmup_factor=0.2,
            lr_warmup_steps=20_000,
            reward_blend_enabled=True,
            reward_blend_episodes=35,
            checkpoint_on_transition=True,
            transition_cooldown_episodes=150,
        ),
        skill_requirements=SkillRequirements(
            required_skills={
                TradingSkill.ENTRY_TIMING: 0.70,
                TradingSkill.EXIT_QUALITY: 0.70,
                TradingSkill.DRAWDOWN_CONTROL: 0.80,
                TradingSkill.PATIENCE: 0.75,
                TradingSkill.TREND_ALIGNMENT: 0.65,
                TradingSkill.RISK_REWARD: 0.70,
                TradingSkill.CONSISTENCY: 0.70,
                TradingSkill.ADAPTATION: 0.60,
                TradingSkill.LOSS_MANAGEMENT: 0.70,
            },
            min_confidence=0.65,
            require_all_skills=False,
            weighted_threshold=0.70,
        ),
        entropy_targets=EntropyTargets(
            min_entropy=0.03,
            max_entropy=0.20,
            low_entropy_penalty_scale=0.04,
            high_entropy_penalty_scale=0.10,
            use_in_promotion=True,
        ),
        composite_scoring=CompositeScoringConfig(
            enabled=True,
            promotion_threshold=0.82,  # Raised for quality
            demotion_threshold=0.45,
            hard_floors={
                "win_rate": 0.57,       # Match min_win_rate
                "max_drawdown": 0.035,  # Match max_avg_drawdown
                "dd_breach_rate": 0.02, # Match max_dd_breach_rate
                "profit_factor": 1.4,   # ADDED: Excellent profitability required
                "r_multiple": 0.12,     # ADDED: Excellent risk/reward required
            },
        ),
        adaptive_thresholds=AdaptiveThresholdConfig(
            enabled=True,
            plateau_episodes_threshold=100,
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
            review_frequency=300,
            review_duration=70,
            review_depth=3,
        ),
    )


def get_live_ready_config() -> CurriculumStageConfig:
    return CurriculumStageConfig(
        stage=CurriculumStage.LIVE_READY,
        name="Live Ready",
        description="Real-time constraints. Think like a business, not a student.",
        execution=ExecutionDifficulty(
            base_spread_points=0.20,
            spread_mult_range=(0.85, 1.60),
            max_spread_points=1.50,
            slippage_points_sigma=0.05,
            slippage_mult_range=(0.50, 2.00),
            max_slippage_points=0.50,
            commission_per_lot=0.0,
            latency_bars=1,
            enable_randomization=True,
            spread_randomization_range=(0.78, 1.55),
            slippage_randomization_range=(0.60, 1.90),
            latency_randomization_range=(1, 3),
            volatility_scale_range=(0.78, 1.40),
        ),
        rewards=RewardShaping(
            # ============================================================================
            # LIVE_READY: ELITE CONFIGURATION - PROFESSIONAL EXCELLENCE
            # ============================================================================
            # Goal: Agent is ready for live trading. No excuses.
            # Bonuses ~50% of base max. Losses penalized 5% more than gains rewarded.
            # This is the harshest reward structure - only truly skilled agents survive.
            #
            # Math for $300 trade (0.3% of 100k):
            #   base_reward = 0.003 * 10.0 = 0.030
            #   max_bonus   = 0.015 (50% of base)
            #   Loss penalty = 0.003 * 10.0 * 1.05 = 0.0315 (losses hurt MORE)
            #   Breakeven: ~43%
            # ============================================================================
            reward_scale=10.0,
            loss_multiplier=1.05,           # Losses hurt 5% more than gains help
            
            r_multiple_bonus_threshold=1.2,   # Lower threshold for elite agents
            r_multiple_bonus_scale=0.05,
            r_multiple_bonus_cap=0.01,        # ~33% of base
            
            mae_efficiency_enabled=True,
            mae_efficiency_scale=0.004,       # ~13% of base
            mae_efficiency_threshold=1.5,     # Elite efficiency required
            
            time_efficiency_enabled=True,
            time_efficiency_scale=0.004,
            optimal_trade_bars=6,             # Quick execution rewarded
            max_trade_bars_for_bonus=18,
            
            exit_quality_enabled=True,
            trailing_stop_bonus=0.004,
            agent_close_bonus=0.002,
            hard_stop_penalty=0.008,
            risk_liquidation_penalty=0.016,
            
            truncation_winner_discount=0.30,
            truncation_loser_extra_penalty=0.15,
            
            entry_quality_integration=True,
            entry_quality_weight=0.25,        # High importance
            
            dd_shaping_enabled=True,
            dd_threshold=0.015,               # Very tight DD sensitivity
            dd_penalty_scale=1.5,             # Harsh DD penalty
            dd_severity_exponent=1.6,
            dd_severity_cap=1.5,
            
            streak_modifier_enabled=True,
            win_streak_bonus_per_win=0.002,
            loss_streak_penalty_per_loss=0.004,
            
            anti_churn_enabled=True,
            daily_trade_soft_limit=8,
            churn_penalty_per_trade=0.008,
            
            hard_block_penalty=0.05,
            soft_block_penalty=0.025,
            per_step_shaping_enabled=True,
            holding_cost_per_bar=0.0005,
            opportunity_bonus_scale=0.012,
            min_reward=-5.0,
            max_reward=5.0,
        ),
        constraints=TradingConstraints(
            max_positions=1,
            max_trades_per_day=10,
            max_trades_per_session=5,
            max_consecutive_losses=3,
            enforce_session_windows=True,
            enforce_no_new_trades_window=True,
            enforce_weekend_block=True,
            enforce_hard_close=True,
            min_minutes_between_entries=5,
            min_minutes_after_loss=15,
            daily_drawdown_limit=0.05,
            max_drawdown_limit=0.10,
            daily_dd_safety_buffer=0.008,
            max_dd_safety_buffer=0.015,
            emergency_close_threshold=0.09,
            entry_quality_gate_enabled=True,
            entry_quality_threshold=0.55,
            hard_stop_loss_eur=220.0,
            soft_stop_loss_eur=140.0,
            trailing_activation_eur=100.0,
            trailing_retrace_pct=0.30,
            time_decay_hours=4.0,
            risk_per_trade_pct=0.003,
            max_risk_per_trade_pct=0.006,
        ),
        competence=CompetenceThresholds(
            min_episodes=1000,
            min_timesteps=5_000_000,
            min_win_rate=0.58,
            min_profit_factor=1.6,
            max_avg_drawdown=0.03,
            min_avg_pnl=550.0,
            min_avg_r_multiple=0.20,
            min_entropy=0.02,
            max_win_rate_std=0.04,
            max_pnl_std=200.0,
            min_trade_count_avg=5.5,
            max_dd_breach_rate=0.01,
            max_consecutive_loss_rate=0.01,
            evaluation_window=300,
        ),
        max_steps_per_episode=2000,
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
            recent_data_weight=2.0,
        ),
        transition=TransitionSettings(
            lr_warmup_enabled=True,
            lr_warmup_factor=0.15,
            lr_warmup_steps=25_000,
            reward_blend_enabled=True,
            reward_blend_episodes=40,
            checkpoint_on_transition=True,
            transition_cooldown_episodes=200,
        ),
        skill_requirements=SkillRequirements(
            required_skills={
                TradingSkill.ENTRY_TIMING: 0.75,
                TradingSkill.EXIT_QUALITY: 0.75,
                TradingSkill.DRAWDOWN_CONTROL: 0.85,
                TradingSkill.PATIENCE: 0.80,
                TradingSkill.TREND_ALIGNMENT: 0.70,
                TradingSkill.RISK_REWARD: 0.75,
                TradingSkill.CONSISTENCY: 0.75,
                TradingSkill.ADAPTATION: 0.65,
                TradingSkill.LOSS_MANAGEMENT: 0.75,
                TradingSkill.POSITION_SIZING: 0.70,
            },
            min_confidence=0.7,
            require_all_skills=False,
            weighted_threshold=0.75,
        ),
        entropy_targets=EntropyTargets(
            min_entropy=0.02,
            max_entropy=0.15,
            low_entropy_penalty_scale=0.03,
            high_entropy_penalty_scale=0.12,
            use_in_promotion=False,  # Terminal stage
        ),
        composite_scoring=CompositeScoringConfig(
            enabled=True,
            promotion_threshold=0.88,  # Very high bar for terminal stage
            demotion_threshold=0.50,
            hard_floors={
                "win_rate": 0.58,       # Match min_win_rate - FINAL STAGE
                "max_drawdown": 0.03,   # Match max_avg_drawdown - STRICT
                "dd_breach_rate": 0.01, # Match max_dd_breach_rate - VERY STRICT
                "profit_factor": 1.5,   # ADDED: Outstanding profitability required
                "r_multiple": 0.15,     # ADDED: Outstanding risk/reward required
            },
        ),
        adaptive_thresholds=AdaptiveThresholdConfig(
            enabled=True,
            plateau_episodes_threshold=150,
            max_relaxation=0.04,
        ),
        recovery_protocol=RecoveryProtocolConfig(
            enabled=True,
            trigger_after_demotions=2,
            recovery_duration_episodes=400,
        ),
        mixed_stage_sampling=MixedStageSamplingConfig(
            enabled=True,
            current_stage_weight=0.60,
            recent_stages_weight=0.30,
            foundation_weight=0.10,
        ),
        review_session=ReviewSessionConfig(
            enabled=True,
            review_frequency=250,
            review_duration=80,
            review_depth=4,
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
    
    # Validate composite scoring weights
    cs = cfg.composite_scoring
    if cs.enabled:
        weight_sum = sum(cs.weights.values())
        if abs(weight_sum - 1.0) > 0.1:
            issues.append(f"{cfg.stage.name}: composite scoring weights sum to {weight_sum:.2f}, expected ~1.0")
    
    # Validate skill requirements
    sr = cfg.skill_requirements
    for skill, threshold in sr.required_skills.items():
        if not (0.0 <= threshold <= 1.0):
            issues.append(f"{cfg.stage.name}: skill {skill.value} threshold {threshold} out of [0,1]")
    
    return issues


def validate_all_configs() -> Dict[CurriculumStage, List[str]]:
    """Validate all stage configurations."""
    return {stage: validate_stage_config(get_stage_config(stage)) for stage in CurriculumStage}