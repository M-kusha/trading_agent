

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class RewardShaping:

    reward_scale: float = 10.0
    loss_multiplier: float = 1.0


    pnl_scale_factor: float = 200.0
    max_shaping_to_pnl_ratio: float = 0.5


    execution_cost_visibility_enabled: bool = True
    execution_cost_reward_scale: float = 0.5


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


    agent_close_bonus: float = -0.05

    hard_stop_penalty: float = 0.15
    risk_liquidation_penalty: float = 0.30


    premature_close_capture_threshold: float = 0.7
    premature_close_penalty_scale: float = 0.25
    premature_close_penalty_cap: float = 0.15


    good_loss_cut_enabled: bool = True
    good_loss_cut_bonus: float = 0.08
    good_loss_cut_efficiency_threshold: float = 0.3
    good_loss_cut_max_bonus: float = 0.15


    truncation_winner_discount: float = 0.30
    truncation_loser_extra_penalty: float = 0.15


    entry_quality_integration: bool = False
    entry_quality_weight: float = 0.2


    setup_quality_enabled: bool = False
    setup_quality_threshold: float = 0.55
    setup_quality_bonus_scale: float = 0.15
    hasty_entry_penalty: float = 0.08


    certainty_threshold: float = 0.70
    entry_certainty_bonus: Dict[str, float] = field(default_factory=dict)
    low_certainty_penalty: float = 0.15


    session_timing_enabled: bool = False
    off_hours_trade_penalty: float = 0.15
    prime_hours_trade_bonus: float = 0.05
    time_of_day_quality: Dict[str, float] = field(default_factory=dict)


    market_structure_enabled: bool = False
    sr_proximity_bonus: float = 0.10
    sr_proximity_penalty: float = 0.08
    structure_alignment_bonus: float = 0.12
    bos_alignment_bonus: float = 0.08
    order_block_entry_bonus: float = 0.06


    divergence_awareness_enabled: bool = False
    divergence_contra_penalty: float = 0.15
    divergence_aligned_bonus: float = 0.10
    overbought_long_penalty: float = 0.12
    oversold_short_penalty: float = 0.12


    regime_awareness_enabled: bool = False
    risk_off_aggressive_penalty: float = 0.10
    high_vol_size_penalty: float = 0.08


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


    churn_action_cost: float = 0.0


    cost_erosion_penalty_enabled: bool = True
    cost_erosion_threshold: float = 0.5
    cost_erosion_penalty_scale: float = 0.15
    cost_erosion_penalty_cap: float = 0.30

    activity_consistency_enabled: bool = False
    target_trades_per_1k_steps: float = 10.0
    stage_activity_targets: Optional[Dict[int, float]] = None
    activity_deviation_penalty_scale: float = 0.2
    activity_deviation_penalty_cap: float = 2.0
    min_trades_penalty: float = 0.3


    hard_block_penalty: float = 0.02
    soft_block_penalty: float = 0.01


    per_step_shaping_enabled: bool = False
    holding_cost_per_bar: float = 0.0
    opportunity_bonus_scale: float = 0.0


    patience_shaping_enabled: bool = False
    patience_bonus_per_bar: float = 0.0
    patience_quality_threshold: float = 0.35
    dynamic_patience_enabled: bool = False
    patience_bonus_base: float = 0.001
    patience_bonus_multiplier: Dict[str, float] = field(default_factory=dict)


    observation_period_required: bool = False
    min_bars_observation_before_entry: int = 0
    observation_completion_bonus: float = 0.0
    premature_entry_penalty: float = 0.0


    strategic_patience_enabled: bool = False
    setup_rejection_bonus: float = 0.0
    max_setup_rejections_for_bonus: int = 0


    deliberation_time_tracking: bool = False
    min_deliberation_bars: int = 0
    optimal_deliberation_range: Tuple[int, int] = (0, 0)
    too_fast_penalty: float = 0.0
    deliberation_quality_bonus: float = 0.0


    win_rate_preservation_enabled: bool = False
    current_win_rate_threshold: float = 0.45
    selectivity_bonus: float = 0.0
    win_rate_decay_penalty: float = 0.0


    psychological_factors_enabled: bool = False
    fear_of_missing_out_penalty: float = 0.0
    revenge_trading_penalty: float = 0.0
    overconfidence_penalty: float = 0.0
    overconfidence_streak_threshold: int = 3


    compounding_success_enabled: bool = False
    consecutive_quality_trades_bonus: List[float] = field(default_factory=list)
    quality_trade_r_multiple: float = 1.0
    quality_trade_entry_quality: float = 0.6
    quality_trade_exit_type: str = "trailing_stop"


    loss_streak_caution_enabled: bool = True
    loss_streak_caution_base: float = 0.03
    loss_streak_caution_cap: float = 0.25


    per_step_min: float = -0.05
    per_step_max: float = 0.05


    min_reward: float = -50.0
    max_reward: float = 50.0


    exploration_bonus: float = 0.0
@dataclass
class TradingConstraints:
    max_positions: int = 1

    max_trades_per_day: int = 100
    max_trades_per_session: int = 50
    max_trades_per_episode: int = 0
    max_consecutive_losses: int = 10


    loss_layer_stop: int = 5


    session_loss_limit_pct: float = 0.99
    session_consecutive_loss_limit: int = 99


    enforce_session_windows: bool = False

    enforce_no_new_trades_window: bool = False
    enforce_weekend_block: bool = False
    enforce_hard_close: bool = False

    observation_period_required: bool = False
    min_bars_observation_before_entry: int = 0


    min_bars_between_entries: int = 0
    min_bars_after_loss: int = 0
    min_minutes_between_entries: int = 0
    min_minutes_after_loss: int = 0

    daily_drawdown_limit: float = 1.0
    max_drawdown_limit: float = 1.0
    daily_dd_safety_buffer: float = 0.0
    max_dd_safety_buffer: float = 0.0
    emergency_close_threshold: float = 1.0

    entry_quality_gate_enabled: bool = False
    entry_quality_threshold: float = 0.0
    min_setup_quality_for_entry: float = 0.0

    hard_stop_loss_eur: float = 10000.0
    soft_stop_loss_eur: float = 10000.0
    trailing_activation_eur: float = 10000.0
    trailing_retrace_pct: float = 0.50
    time_decay_hours: float = 24.0

    risk_per_trade_pct: float = 0.01
    max_risk_per_trade_pct: float = 0.02

    def __post_init__(self) -> None:

        if self.enforce_session_windows:
            if not self.enforce_no_new_trades_window:
                self.enforce_no_new_trades_window = True
            if not self.enforce_hard_close:
                self.enforce_hard_close = True


__all__ = [
    "RewardShaping",
    "TradingConstraints",
]
