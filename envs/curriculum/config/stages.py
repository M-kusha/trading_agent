

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from envs.curriculum.config.constraints import RewardShaping, TradingConstraints
    from envs.curriculum.config.execution import DataDifficulty, ExecutionDifficulty, TransitionSettings
    from envs.curriculum.config.protocols import (
        MixedStageSamplingConfig,
        RecoveryProtocolConfig,
        ReviewSessionConfig,
        ValidationConfig,
    )
    from envs.curriculum.config.thresholds import (
        AdaptiveThresholdConfig,
        CompetenceThresholds,
        CompositeScoringConfig,
        EntropyTargets,
        SkillRequirements,
    )


class CurriculumStage(IntEnum):
    EXPLORER = 0
    EXPERIMENTER = 1

    TREND_STUDENT = 2
    SESSION_STUDENT = 3
    TIMING_STUDENT = 4

    INTEGRATOR = 5
    RISK_MANAGER = 6
    STRATEGIST = 7

    PROFESSIONAL = 8
    LIVE_READY = 9


class TradingSkill(Enum):
    ENTRY_TIMING = "entry_timing"
    EXIT_QUALITY = "exit_quality"
    DRAWDOWN_CONTROL = "drawdown_control"
    POSITION_SIZING = "position_sizing"
    PATIENCE = "patience"
    SELECTIVITY = "selectivity"
    CERTAINTY = "certainty"
    DISCIPLINE = "discipline"
    SETUP_QUALITY = "setup_quality"
    TREND_ALIGNMENT = "trend_alignment"
    RISK_REWARD = "risk_reward"
    CONSISTENCY = "consistency"
    LOSS_MANAGEMENT = "loss_management"
    ADAPTATION = "adaptation"


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class CurriculumStageConfig:
    stage: CurriculumStage
    name: str
    description: str

    execution: "ExecutionDifficulty"
    rewards: "RewardShaping"
    constraints: "TradingConstraints"
    competence: "CompetenceThresholds"

    max_steps_per_episode: int = 2000
    include_expert_signals: bool = True
    expert_signal_dropout: float = 0.0

    allow_demotion: bool = False
    is_terminal: bool = False

    data_difficulty: "DataDifficulty" = field(default_factory=lambda: _default_data_difficulty())
    transition: "TransitionSettings" = field(default_factory=lambda: _default_transition_settings())


    skill_requirements: "SkillRequirements" = field(default_factory=lambda: _default_skill_requirements())
    entropy_targets: "EntropyTargets" = field(default_factory=lambda: _default_entropy_targets())
    composite_scoring: "CompositeScoringConfig" = field(default_factory=lambda: _default_composite_scoring())
    adaptive_thresholds: "AdaptiveThresholdConfig" = field(default_factory=lambda: _default_adaptive_thresholds())
    recovery_protocol: "RecoveryProtocolConfig" = field(default_factory=lambda: _default_recovery_protocol())
    mixed_stage_sampling: "MixedStageSamplingConfig" = field(default_factory=lambda: _default_mixed_stage_sampling())
    review_session: "ReviewSessionConfig" = field(default_factory=lambda: _default_review_session())
    validation: "ValidationConfig" = field(default_factory=lambda: _default_validation())


    env_overrides: Optional[Dict[str, Any]] = None
    reward_overrides: Optional[Dict[str, Any]] = None
    execution_overrides: Optional[Dict[str, Any]] = None


    overrides: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:

        if self.env_overrides is None:
            self.env_overrides = self._compute_env_overrides()
        if self.reward_overrides is None:
            self.reward_overrides = self._compute_reward_overrides()
        if self.execution_overrides is None:
            self.execution_overrides = self._compute_execution_overrides()

    def _compute_env_overrides(self) -> Dict[str, Any]:
        c = self.constraints
        e = self.execution
        out: Dict[str, Any] = {}


        try:
            max_steps = int(self.max_steps_per_episode)
            if max_steps > 0:
                out["max_steps_per_episode"] = max_steps
        except Exception:
            pass


        for k in (
            "max_positions",
            "max_trades_per_day",
            "max_trades_per_session",
            "max_consecutive_losses",
            "loss_layer_stop",

            "session_loss_limit_pct",
            "session_consecutive_loss_limit",
            "enforce_no_new_trades_window",
            "enforce_weekend_block",
            "enforce_hard_close",
            "min_bars_between_entries",
            "min_bars_after_loss",
            "min_minutes_between_entries",
            "min_minutes_after_loss",
            "daily_drawdown_limit",
            "max_drawdown_limit",
            "daily_dd_safety_buffer",
            "max_dd_safety_buffer",
            "emergency_close_threshold",
            "entry_quality_gate_enabled",
            "entry_quality_threshold",
            "hard_stop_loss_eur",
            "soft_stop_loss_eur",
            "trailing_activation_eur",
            "trailing_retrace_pct",
            "time_decay_hours",
            "risk_per_trade_pct",
            "max_risk_per_trade_pct",
        ):
            if hasattr(c, k):
                out[k] = getattr(c, k)


        if getattr(c, "enforce_session_windows", False):
            out.setdefault("enforce_no_new_trades_window", True)
            out.setdefault("enforce_hard_close", True)


        if hasattr(e, "enable_randomization"):
            out["domain_randomization_enabled"] = bool(e.enable_randomization)


        if hasattr(e, "spread_mult_range"):
            out["spread_mult_range"] = tuple(e.spread_mult_range)
        if hasattr(e, "slippage_mult_range"):
            out["slippage_mult_range"] = tuple(e.slippage_mult_range)
        if hasattr(e, "latency_randomization_range"):
            out["latency_bars_range"] = tuple(e.latency_randomization_range)
        if hasattr(e, "volatility_scale_range"):
            out["volatility_scale_range"] = tuple(e.volatility_scale_range)

        return out

    def _compute_reward_overrides(self) -> Dict[str, Any]:
        r = self.rewards

        keys = [

            "reward_scale", "loss_multiplier",

            "pnl_scale_factor",
            "max_shaping_to_pnl_ratio",

            "execution_cost_visibility_enabled",
            "execution_cost_reward_scale",

            "good_loss_cut_enabled",
            "good_loss_cut_bonus",
            "good_loss_cut_efficiency_threshold",
            "good_loss_cut_max_bonus",

            "cost_erosion_penalty_enabled",
            "cost_erosion_threshold",
            "cost_erosion_penalty_scale",
            "cost_erosion_penalty_cap",

            "r_multiple_bonus_threshold", "r_multiple_bonus_scale", "r_multiple_bonus_cap",
            "mae_efficiency_enabled", "mae_efficiency_scale", "mae_efficiency_threshold",
            "time_efficiency_enabled", "time_efficiency_scale", "optimal_trade_bars", "max_trade_bars_for_bonus",

            "exit_quality_enabled", "trailing_stop_bonus", "agent_close_bonus", "hard_stop_penalty", "risk_liquidation_penalty",
            "premature_close_capture_threshold", "premature_close_penalty_scale", "premature_close_penalty_cap",

            "truncation_winner_discount", "truncation_loser_extra_penalty",

            "entry_quality_integration", "entry_quality_weight",

            "setup_quality_enabled", "setup_quality_threshold", "setup_quality_bonus_scale", "hasty_entry_penalty",
            "certainty_threshold", "entry_certainty_bonus", "low_certainty_penalty",

            "session_timing_enabled", "off_hours_trade_penalty", "prime_hours_trade_bonus", "time_of_day_quality",

            "market_structure_enabled", "sr_proximity_bonus", "sr_proximity_penalty", "structure_alignment_bonus",
            "bos_alignment_bonus", "order_block_entry_bonus",
            "divergence_awareness_enabled", "divergence_contra_penalty", "divergence_aligned_bonus",
            "overbought_long_penalty", "oversold_short_penalty",
            "regime_awareness_enabled", "risk_off_aggressive_penalty", "high_vol_size_penalty",

            "dd_shaping_enabled", "dd_threshold", "dd_penalty_scale", "dd_severity_exponent", "dd_severity_cap",
            "streak_modifier_enabled", "win_streak_bonus_per_win", "loss_streak_penalty_per_loss",
            "anti_churn_enabled", "daily_trade_soft_limit", "churn_penalty_per_trade", "churn_action_cost",

            "activity_consistency_enabled", "target_trades_per_1k_steps", "stage_activity_targets",
            "activity_deviation_penalty_scale", "activity_deviation_penalty_cap", "min_trades_penalty",

            "hard_block_penalty", "soft_block_penalty",
            "per_step_shaping_enabled", "holding_cost_per_bar", "opportunity_bonus_scale",
            "patience_shaping_enabled", "patience_bonus_per_bar", "patience_quality_threshold",
            "dynamic_patience_enabled", "patience_bonus_base", "patience_bonus_multiplier",
            "observation_period_required", "min_bars_observation_before_entry",
            "observation_completion_bonus", "premature_entry_penalty",
            "strategic_patience_enabled", "setup_rejection_bonus", "max_setup_rejections_for_bonus",
            "deliberation_time_tracking", "min_deliberation_bars", "optimal_deliberation_range",
            "too_fast_penalty", "deliberation_quality_bonus",
            "win_rate_preservation_enabled", "current_win_rate_threshold",
            "selectivity_bonus", "win_rate_decay_penalty",
            "psychological_factors_enabled", "fear_of_missing_out_penalty",
            "revenge_trading_penalty", "overconfidence_penalty", "overconfidence_streak_threshold",
            "compounding_success_enabled", "consecutive_quality_trades_bonus",
            "quality_trade_r_multiple", "quality_trade_entry_quality", "quality_trade_exit_type",
            "per_step_min", "per_step_max",

            "min_reward", "max_reward",

            "exploration_bonus",
            "directional_accuracy_weight",
        ]
        out: Dict[str, Any] = {}
        for k in keys:
            if hasattr(r, k):
                out[k] = getattr(r, k)
        return out

    def _compute_execution_overrides(self) -> Dict[str, Any]:
        e = self.execution
        out: Dict[str, Any] = {}
        for k in (
            "base_spread_points",
            "max_spread_points",
            "slippage_points_sigma",
            "max_slippage_points",
            "commission_per_lot",
            "latency_bars",

            "spread_shock_enabled",
            "spread_shock_probability",
            "spread_shock_multiplier",

            "use_data_spread",
            "data_spread_scale",
        ):
            if hasattr(e, k):
                out[k] = getattr(e, k)
        return out


def _default_data_difficulty():
    from envs.curriculum.config.execution import DataDifficulty
    return DataDifficulty()

def _default_transition_settings():
    from envs.curriculum.config.execution import TransitionSettings
    return TransitionSettings()

def _default_skill_requirements():
    from envs.curriculum.config.thresholds import SkillRequirements
    return SkillRequirements()

def _default_entropy_targets():
    from envs.curriculum.config.thresholds import EntropyTargets
    return EntropyTargets()

def _default_composite_scoring():
    from envs.curriculum.config.thresholds import CompositeScoringConfig
    return CompositeScoringConfig()

def _default_adaptive_thresholds():
    from envs.curriculum.config.thresholds import AdaptiveThresholdConfig
    return AdaptiveThresholdConfig()

def _default_recovery_protocol():
    from envs.curriculum.config.protocols import RecoveryProtocolConfig
    return RecoveryProtocolConfig()

def _default_mixed_stage_sampling():
    from envs.curriculum.config.protocols import MixedStageSamplingConfig
    return MixedStageSamplingConfig()

def _default_review_session():
    from envs.curriculum.config.protocols import ReviewSessionConfig
    return ReviewSessionConfig()

def _default_validation():
    from envs.curriculum.config.protocols import ValidationConfig
    return ValidationConfig()


__all__ = [
    "CurriculumStage",
    "CurriculumStageConfig",
    "MarketRegime",
    "TradingSkill",
]
