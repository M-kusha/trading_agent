# envs/curriculum_config.py
"""
Curriculum Configuration for Trading RL Agent
==============================================

Defines curriculum progression from early learning to live-ready discipline.

Enhancements in this version:
- Minor typing cleanups
- Added lightweight validation helpers (optional; no runtime impact unless called)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Dict, List, Optional, Tuple


class CurriculumStage(IntEnum):
    FOUNDATION = 0
    DISCIPLINE = 1
    MARKET_STRUCTURE = 2
    ECONOMIC_LOGIC = 3
    PROFESSIONAL = 4
    ADAPTIVE = 5
    SPECIALIST = 6
    LIVE_READY = 7


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
    # Volatility filtering: percentile range [0, 1] to include
    # (0.0, 0.5) = only low volatility periods; (0.0, 1.0) = all
    volatility_percentile_range: Tuple[float, float] = (0.0, 1.0)
    
    # Trend clarity: filter for periods with clearer directional movement
    # Higher threshold = only clearer trends included
    min_trend_clarity: float = 0.0  # [0, 1] where 0=any, 1=strongest trends only
    
    # Session filtering
    include_asian_session: bool = True
    include_london_session: bool = True
    include_ny_session: bool = True
    include_overlap_sessions: bool = True
    
    # News/event filtering
    exclude_high_impact_news: bool = False  # Avoid high-volatility news periods
    exclude_market_open_close: bool = False  # Avoid first/last 30 min
    
    # Date range weighting
    prefer_recent_data: bool = False  # Weight recent data more heavily
    recent_data_weight: float = 1.0  # 1.0 = no preference, 2.0 = 2x weight to recent


@dataclass
class TransitionSettings:
    """
    Settings for smooth stage transitions.
    
    Prevents sudden destabilization when moving to harder stages.
    """
    # Learning rate adjustment on promotion
    lr_warmup_enabled: bool = True
    lr_warmup_factor: float = 0.3  # Start at 30% of normal LR
    lr_warmup_steps: int = 10_000  # Steps to reach full LR
    
    # Reward smoothing between stages
    reward_blend_enabled: bool = True
    reward_blend_episodes: int = 20  # Episodes to blend old/new reward config
    
    # Checkpoint on promotion/demotion
    checkpoint_on_transition: bool = True
    
    # Cooldown before next transition evaluation
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
    
    # R-Multiple threshold: require good risk-adjusted returns
    min_avg_r_multiple: float = 0.0  # 0 = disabled, 0.3+ = positive expectancy
    
    # Entropy floor: prevent premature policy collapse
    min_entropy: float = 0.0  # 0 = disabled, 0.1+ = require exploration

    max_win_rate_std: float = 0.30
    max_pnl_std: float = 10000.0
    min_trade_count_avg: float = 1.0

    max_dd_breach_rate: float = 0.50
    max_consecutive_loss_rate: float = 0.30

    evaluation_window: int = 50


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
    
    # Data difficulty for this stage (filter training data)
    data_difficulty: DataDifficulty = None  # type: ignore[assignment]
    
    # Transition settings for this stage
    transition: TransitionSettings = None  # type: ignore[assignment]
    
    def __post_init__(self) -> None:
        if self.data_difficulty is None:
            self.data_difficulty = DataDifficulty()
        if self.transition is None:
            self.transition = TransitionSettings()


# --- Stage factories (UNCHANGED semantics; kept as provided) ---

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
            reward_scale=5.0,
            loss_multiplier=0.5,
            r_multiple_bonus_threshold=1.0,
            r_multiple_bonus_scale=0.2,
            r_multiple_bonus_cap=0.4,
            mae_efficiency_enabled=False,
            time_efficiency_enabled=False,
            exit_quality_enabled=False,
            entry_quality_integration=False,
            dd_shaping_enabled=False,
            streak_modifier_enabled=False,
            anti_churn_enabled=False,
            hard_block_penalty=0.0,
            soft_block_penalty=0.0,
            per_step_shaping_enabled=False,
            exploration_bonus=0.01,
            directional_accuracy_weight=1.5,
            min_reward=-2.0,
            max_reward=3.0,
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
            min_episodes=250,  # Extended from 100 for stronger foundation
            min_timesteps=200_000,  # Extended from 100_000
            min_win_rate=0.35,
            min_profit_factor=0.6,
            max_avg_drawdown=0.30,
            min_avg_pnl=-500.0,
            min_avg_r_multiple=0.0,  # No R-multiple requirement in foundation
            min_entropy=0.15,  # Require some exploration to prevent collapse
            max_win_rate_std=0.25,
            max_pnl_std=5000.0,
            min_trade_count_avg=2.0,
            max_dd_breach_rate=0.40,
            max_consecutive_loss_rate=0.25,
            evaluation_window=75,  # Larger window for more stable evaluation
        ),
        max_steps_per_episode=1500,
        include_memory_features=False,
        include_world_model_features=False,
        include_expert_signals=True,
        allow_demotion=False,
        # Easy data: low volatility, clear trends, avoid news
        data_difficulty=DataDifficulty(
            volatility_percentile_range=(0.0, 0.50),  # Only low volatility (tightened from 0.6)
            min_trend_clarity=0.3,  # Prefer clearer directional moves
            include_asian_session=True,
            include_london_session=True,
            include_ny_session=True,
            include_overlap_sessions=True,
            exclude_high_impact_news=True,  # Avoid news spikes
            exclude_market_open_close=True,  # Avoid chaotic open/close
            prefer_recent_data=False,
            recent_data_weight=1.0,
        ),
        # No LR warmup needed for first stage
        transition=TransitionSettings(
            lr_warmup_enabled=False,
            lr_warmup_factor=1.0,
            lr_warmup_steps=0,
            reward_blend_enabled=False,
            reward_blend_episodes=0,
            checkpoint_on_transition=True,
            transition_cooldown_episodes=30,
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
            reward_scale=8.0,
            loss_multiplier=0.8,
            r_multiple_bonus_threshold=1.5,
            r_multiple_bonus_scale=0.25,
            r_multiple_bonus_cap=0.5,
            mae_efficiency_enabled=True,
            mae_efficiency_scale=0.15,
            mae_efficiency_threshold=2.0,
            time_efficiency_enabled=True,
            time_efficiency_scale=0.10,
            optimal_trade_bars=10,
            max_trade_bars_for_bonus=30,
            exit_quality_enabled=False,
            entry_quality_integration=False,
            dd_shaping_enabled=True,
            dd_threshold=0.05,
            dd_penalty_scale=0.5,
            dd_severity_exponent=1.2,
            dd_severity_cap=1.0,
            streak_modifier_enabled=False,
            anti_churn_enabled=True,
            daily_trade_soft_limit=25,
            churn_penalty_per_trade=0.01,
            hard_block_penalty=0.03,
            soft_block_penalty=0.01,
            per_step_shaping_enabled=False,
            exploration_bonus=0.0,
            directional_accuracy_weight=1.0,
            min_reward=-3.0,
            max_reward=4.0,
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
            min_episodes=150,  # Increased for more stability
            min_timesteps=200_000,
            min_win_rate=0.38,
            min_profit_factor=0.75,
            max_avg_drawdown=0.18,
            min_avg_pnl=-200.0,
            min_avg_r_multiple=0.1,  # Start requiring positive R-multiple
            min_entropy=0.12,  # Allow some policy focus
            max_win_rate_std=0.22,
            max_pnl_std=3000.0,
            min_trade_count_avg=2.5,
            max_dd_breach_rate=0.30,
            max_consecutive_loss_rate=0.20,
            evaluation_window=75,
        ),
        max_steps_per_episode=1800,
        include_memory_features=True,
        include_world_model_features=False,
        include_expert_signals=True,
        allow_demotion=True,
        # Medium-easy data: gradual volatility increase
        data_difficulty=DataDifficulty(
            volatility_percentile_range=(0.0, 0.65),  # Gradual increase from 0.50
            min_trend_clarity=0.15,  # Slightly clearer moves
            include_asian_session=True,
            include_london_session=True,
            include_ny_session=True,
            include_overlap_sessions=True,
            exclude_high_impact_news=True,
            exclude_market_open_close=False,  # Allow open/close now
            prefer_recent_data=False,
            recent_data_weight=1.0,
        ),
        # LR warmup when entering this stage
        transition=TransitionSettings(
            lr_warmup_enabled=True,
            lr_warmup_factor=0.4,  # Start at 40% LR
            lr_warmup_steps=8_000,
            reward_blend_enabled=True,
            reward_blend_episodes=15,
            checkpoint_on_transition=True,
            transition_cooldown_episodes=40,
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
            reward_scale=10.0,
            loss_multiplier=1.0,
            r_multiple_bonus_threshold=1.5,
            r_multiple_bonus_scale=0.30,
            r_multiple_bonus_cap=0.6,
            mae_efficiency_enabled=True,
            mae_efficiency_scale=0.25,
            mae_efficiency_threshold=2.0,
            time_efficiency_enabled=True,
            time_efficiency_scale=0.15,
            optimal_trade_bars=8,
            max_trade_bars_for_bonus=24,
            exit_quality_enabled=True,
            trailing_stop_bonus=0.10,
            agent_close_bonus=0.03,
            hard_stop_penalty=0.10,
            risk_liquidation_penalty=0.20,
            truncation_winner_discount=0.25,
            truncation_loser_extra_penalty=0.10,
            entry_quality_integration=True,
            entry_quality_weight=0.15,
            dd_shaping_enabled=True,
            dd_threshold=0.03,
            dd_penalty_scale=0.8,
            dd_severity_exponent=1.3,
            dd_severity_cap=1.2,
            streak_modifier_enabled=True,
            win_streak_bonus_per_win=0.015,
            loss_streak_penalty_per_loss=0.02,
            anti_churn_enabled=True,
            daily_trade_soft_limit=18,
            churn_penalty_per_trade=0.015,
            hard_block_penalty=0.06,
            soft_block_penalty=0.02,
            per_step_shaping_enabled=False,
            min_reward=-4.0,
            max_reward=5.0,
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
            min_avg_r_multiple=0.3,  # Require meaningful R-multiple
            min_entropy=0.08,  # Policy can be more focused
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
        # Wider volatility range, gradual increase
        data_difficulty=DataDifficulty(
            volatility_percentile_range=(0.05, 0.80),  # Exclude extreme calm, include more vol
            min_trend_clarity=0.05,  # Accept most market conditions
            include_asian_session=True,
            include_london_session=True,
            include_ny_session=True,
            include_overlap_sessions=True,
            exclude_high_impact_news=False,  # Start handling news
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
            reward_scale=10.0,
            loss_multiplier=1.0,
            r_multiple_bonus_threshold=1.5,
            r_multiple_bonus_scale=0.30,
            r_multiple_bonus_cap=0.6,
            mae_efficiency_enabled=True,
            mae_efficiency_scale=0.25,
            mae_efficiency_threshold=2.0,
            time_efficiency_enabled=True,
            time_efficiency_scale=0.15,
            optimal_trade_bars=8,
            max_trade_bars_for_bonus=24,
            exit_quality_enabled=True,
            trailing_stop_bonus=0.15,
            agent_close_bonus=0.05,
            hard_stop_penalty=0.15,
            risk_liquidation_penalty=0.25,
            truncation_winner_discount=0.30,
            truncation_loser_extra_penalty=0.15,
            entry_quality_integration=True,
            entry_quality_weight=0.20,
            dd_shaping_enabled=True,
            dd_threshold=0.025,
            dd_penalty_scale=1.0,
            dd_severity_exponent=1.4,
            dd_severity_cap=1.3,
            streak_modifier_enabled=True,
            win_streak_bonus_per_win=0.02,
            loss_streak_penalty_per_loss=0.025,
            anti_churn_enabled=True,
            daily_trade_soft_limit=14,
            churn_penalty_per_trade=0.018,
            hard_block_penalty=0.08,
            soft_block_penalty=0.025,
            per_step_shaping_enabled=True,
            holding_cost_per_bar=0.0003,
            opportunity_bonus_scale=0.005,
            min_reward=-4.5,
            max_reward=5.0,
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
            min_avg_r_multiple=0.5,  # Strong R-multiple requirement
            min_entropy=0.05,  # Allow focused policy
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
        # Full data: all market conditions including news
        data_difficulty=DataDifficulty(
            volatility_percentile_range=(0.1, 0.95),  # Avoid extreme calm, handle most volatility
            min_trend_clarity=0.0,  # Accept all market conditions
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
            reward_scale=10.0,
            loss_multiplier=1.0,
            r_multiple_bonus_threshold=1.5,
            r_multiple_bonus_scale=0.30,
            r_multiple_bonus_cap=0.6,
            mae_efficiency_enabled=True,
            mae_efficiency_scale=0.25,
            mae_efficiency_threshold=2.0,
            time_efficiency_enabled=True,
            time_efficiency_scale=0.15,
            optimal_trade_bars=8,
            max_trade_bars_for_bonus=24,
            exit_quality_enabled=True,
            trailing_stop_bonus=0.15,
            agent_close_bonus=0.05,
            hard_stop_penalty=0.15,
            risk_liquidation_penalty=0.30,
            truncation_winner_discount=0.30,
            truncation_loser_extra_penalty=0.15,
            entry_quality_integration=True,
            entry_quality_weight=0.20,
            dd_shaping_enabled=True,
            dd_threshold=0.02,
            dd_penalty_scale=1.0,
            dd_severity_exponent=1.5,
            dd_severity_cap=1.5,
            streak_modifier_enabled=True,
            win_streak_bonus_per_win=0.02,
            loss_streak_penalty_per_loss=0.03,
            anti_churn_enabled=True,
            daily_trade_soft_limit=10,
            churn_penalty_per_trade=0.02,
            hard_block_penalty=0.10,
            soft_block_penalty=0.03,
            per_step_shaping_enabled=True,
            holding_cost_per_bar=0.0005,
            opportunity_bonus_scale=0.01,
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
            min_avg_r_multiple=0.7,  # Strong R-multiple for professional stage
            min_entropy=0.03,  # Allow very focused policy
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
        # Full real-world data
        data_difficulty=DataDifficulty(
            volatility_percentile_range=(0.0, 1.0),
            min_trend_clarity=0.0,
            include_asian_session=True,
            include_london_session=True,
            include_ny_session=True,
            include_overlap_sessions=True,
            exclude_high_impact_news=False,
            exclude_market_open_close=False,
            prefer_recent_data=True,  # Start preferring recent data
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
            reward_scale=10.0,
            loss_multiplier=1.05,
            r_multiple_bonus_threshold=1.5,
            r_multiple_bonus_scale=0.30,
            r_multiple_bonus_cap=0.6,
            mae_efficiency_enabled=True,
            mae_efficiency_scale=0.25,
            mae_efficiency_threshold=2.0,
            time_efficiency_enabled=True,
            time_efficiency_scale=0.15,
            optimal_trade_bars=8,
            max_trade_bars_for_bonus=24,
            exit_quality_enabled=True,
            trailing_stop_bonus=0.15,
            agent_close_bonus=0.05,
            hard_stop_penalty=0.15,
            risk_liquidation_penalty=0.30,
            truncation_winner_discount=0.30,
            truncation_loser_extra_penalty=0.15,
            entry_quality_integration=True,
            entry_quality_weight=0.20,
            dd_shaping_enabled=True,
            dd_threshold=0.02,
            dd_penalty_scale=1.0,
            dd_severity_exponent=1.5,
            dd_severity_cap=1.5,
            streak_modifier_enabled=True,
            win_streak_bonus_per_win=0.02,
            loss_streak_penalty_per_loss=0.03,
            anti_churn_enabled=True,
            daily_trade_soft_limit=10,
            churn_penalty_per_trade=0.02,
            hard_block_penalty=0.10,
            soft_block_penalty=0.03,
            per_step_shaping_enabled=True,
            holding_cost_per_bar=0.0005,
            opportunity_bonus_scale=0.01,
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
        # Full data with recent preference
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
            reward_scale=10.0,
            loss_multiplier=1.0,
            r_multiple_bonus_threshold=1.5,
            r_multiple_bonus_scale=0.30,
            r_multiple_bonus_cap=0.6,
            mae_efficiency_enabled=True,
            mae_efficiency_scale=0.25,
            mae_efficiency_threshold=2.0,
            time_efficiency_enabled=True,
            time_efficiency_scale=0.15,
            optimal_trade_bars=8,
            max_trade_bars_for_bonus=24,
            exit_quality_enabled=True,
            trailing_stop_bonus=0.15,
            agent_close_bonus=0.05,
            hard_stop_penalty=0.15,
            risk_liquidation_penalty=0.30,
            truncation_winner_discount=0.30,
            truncation_loser_extra_penalty=0.15,
            entry_quality_integration=True,
            entry_quality_weight=0.20,
            dd_shaping_enabled=True,
            dd_threshold=0.02,
            dd_penalty_scale=1.0,
            dd_severity_exponent=1.5,
            dd_severity_cap=1.5,
            streak_modifier_enabled=True,
            win_streak_bonus_per_win=0.02,
            loss_streak_penalty_per_loss=0.03,
            anti_churn_enabled=True,
            daily_trade_soft_limit=10,
            churn_penalty_per_trade=0.02,
            hard_block_penalty=0.10,
            soft_block_penalty=0.03,
            per_step_shaping_enabled=True,
            holding_cost_per_bar=0.0005,
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
        # Full data with strong recent focus
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
            recent_data_weight=1.5,  # Strong preference for recent data
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
            reward_scale=10.0,
            loss_multiplier=1.0,
            r_multiple_bonus_threshold=1.5,
            r_multiple_bonus_scale=0.30,
            r_multiple_bonus_cap=0.6,
            mae_efficiency_enabled=True,
            mae_efficiency_scale=0.25,
            mae_efficiency_threshold=2.0,
            time_efficiency_enabled=True,
            time_efficiency_scale=0.15,
            optimal_trade_bars=8,
            max_trade_bars_for_bonus=24,
            exit_quality_enabled=True,
            trailing_stop_bonus=0.15,
            agent_close_bonus=0.05,
            hard_stop_penalty=0.15,
            risk_liquidation_penalty=0.30,
            truncation_winner_discount=0.30,
            truncation_loser_extra_penalty=0.15,
            entry_quality_integration=True,
            entry_quality_weight=0.20,
            dd_shaping_enabled=True,
            dd_threshold=0.02,
            dd_penalty_scale=1.0,
            dd_severity_exponent=1.5,
            dd_severity_cap=1.5,
            streak_modifier_enabled=True,
            win_streak_bonus_per_win=0.02,
            loss_streak_penalty_per_loss=0.03,
            anti_churn_enabled=True,
            daily_trade_soft_limit=10,
            churn_penalty_per_trade=0.02,
            hard_block_penalty=0.10,
            soft_block_penalty=0.03,
            per_step_shaping_enabled=True,
            holding_cost_per_bar=0.0005,
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
            min_episodes=1000,
            min_timesteps=5_000_000,
            min_win_rate=0.58,
            min_profit_factor=1.6,
            max_avg_drawdown=0.03,
            min_avg_pnl=550.0,
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
        # Live-ready: full data, maximum recent focus
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
            recent_data_weight=2.0,  # Heavy emphasis on recent market conditions
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
    )


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
    Optional: call this in tests/CI to flag misconfigured thresholds.
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
    return issues
