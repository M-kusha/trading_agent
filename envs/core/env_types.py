

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from datetime import time as dtime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from envs.core.execution_model import ExecutionConfig


class CloseReason(str, Enum):

    TRAILING_STOP = "trailing_stop"
    AGENT_CLOSE = "agent_close"


    TIME_DECAY = "time_decay"
    HARD_CLOSE = "hard_close"
    WEEKEND_FLATTEN = "weekend_flatten"
    DAILY_LIMIT_SAFETY = "daily_limit_safety"


    HARD_STOP = "hard_stop"
    EMERGENCY_CLOSE = "emergency_close"
    RISK_LIQUIDATION = "risk_liquidation"


    EPISODE_TRUNCATE = "episode_truncate_flatten"

    @property
    def quality_score(self) -> float:
        scores = {
            CloseReason.TRAILING_STOP: 1.0,
            CloseReason.AGENT_CLOSE: 0.85,
            CloseReason.TIME_DECAY: 0.5,
            CloseReason.HARD_CLOSE: 0.5,
            CloseReason.WEEKEND_FLATTEN: 0.5,
            CloseReason.DAILY_LIMIT_SAFETY: 0.3,
            CloseReason.HARD_STOP: 0.15,
            CloseReason.EMERGENCY_CLOSE: 0.1,
            CloseReason.RISK_LIQUIDATION: 0.0,
            CloseReason.EPISODE_TRUNCATE: 0.25,
        }
        return float(scores.get(self, 0.5))

    @property
    def close_priority(self) -> int:
        priorities = {
            CloseReason.RISK_LIQUIDATION: 100,
            CloseReason.EMERGENCY_CLOSE: 95,
            CloseReason.DAILY_LIMIT_SAFETY: 90,
            CloseReason.HARD_STOP: 85,
            CloseReason.TRAILING_STOP: 70,
            CloseReason.HARD_CLOSE: 60,
            CloseReason.WEEKEND_FLATTEN: 60,
            CloseReason.TIME_DECAY: 50,
            CloseReason.EPISODE_TRUNCATE: 40,
            CloseReason.AGENT_CLOSE: 10,
        }
        return int(priorities.get(self, 40))


@dataclass
class RewardConfig:


    reward_scale: float = 10.0


    loss_multiplier: float = 1.0


    pnl_scale_factor: float = 200.0


    max_shaping_to_pnl_ratio: float = 0.5


    execution_cost_visibility_enabled: bool = True
    execution_cost_reward_scale: float = 0.5


    r_multiple_bonus_threshold: float = 1.5
    r_multiple_bonus_scale: float = 0.3
    r_multiple_bonus_cap: float = 0.6


    mae_efficiency_enabled: bool = True
    mae_efficiency_scale: float = 0.25
    mae_efficiency_threshold: float = 2.0


    time_efficiency_enabled: bool = True
    time_efficiency_scale: float = 0.15
    optimal_trade_bars: int = 8
    max_trade_bars_for_bonus: int = 24


    exit_quality_enabled: bool = True
    trailing_stop_bonus: float = 0.25
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


    entry_quality_integration: bool = True
    entry_quality_weight: float = 0.2


    setup_quality_enabled: bool = False
    setup_quality_threshold: float = 0.55
    setup_quality_bonus_scale: float = 0.15
    hasty_entry_penalty: float = 0.08


    certainty_threshold: float = 0.70
    entry_certainty_bonus: Dict[str, float] = field(default_factory=dict)
    low_certainty_penalty: float = 0.15


    session_timing_enabled: bool = True
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


    # Fixed deficit left after a drawdown breach forfeits the episode's
    # earnings. Sized against observed episode rewards (mean 37, max 140)
    # so no run of wins makes a breach worth risking - a real breach ends
    # the account and zeroes all future profit.
    dd_breach_penalty: float = 25.0

    dd_shaping_enabled: bool = True
    dd_threshold: float = 0.02
    dd_penalty_scale: float = 1.0
    dd_severity_exponent: float = 1.5
    dd_severity_cap: float = 1.5


    streak_modifier_enabled: bool = True
    win_streak_bonus_per_win: float = 0.02
    loss_streak_penalty_per_loss: float = 0.03


    anti_churn_enabled: bool = True


    daily_trade_soft_limit: int = 8
    churn_penalty_per_trade: float = 0.05


    churn_action_cost: float = 0.0


    cost_erosion_penalty_enabled: bool = True
    cost_erosion_threshold: float = 0.5
    cost_erosion_penalty_scale: float = 0.15
    cost_erosion_penalty_cap: float = 0.30


    activity_consistency_enabled: bool = True
    target_trades_per_1k_steps: float = 10.0
    stage_activity_targets: Optional[Dict[int, float]] = None
    activity_deviation_penalty_scale: float = 0.2
    # Every stage sets this (10.0 -> 20.0 -> 10.0 across the curriculum) but the
    # field was missing here, so _apply_overrides_to_object dropped it silently
    # and the env fell back to the getattr default of 2.0. The over-trading
    # penalty saturated at -2.0 per episode regardless of how badly the stage
    # target was exceeded.
    activity_deviation_penalty_cap: float = 2.0
    min_trades_penalty: float = 0.3


    hard_block_penalty: float = 0.10
    soft_block_penalty: float = 0.03


    per_step_shaping_enabled: bool = False
    holding_cost_per_bar: float = 0.0005


    opportunity_bonus_scale: float = 0.01


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


    exploration_bonus: float = 0.0

    per_step_min: float = -0.05
    per_step_max: float = 0.05


    min_reward: float = -50.0
    max_reward: float = 50.0


def load_risk_policy() -> Dict[str, Any]:

    config_path = Path(__file__).resolve().parent.parent.parent / "config" / "risk_policy.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


@dataclass
class PropFirmConfig:


    tz: str = "Europe/Berlin"
    primary_timeframe: str = "M15"
    instruments: List[str] = field(default_factory=lambda: ["XAUUSD"])


    initial_balance: float = 100_000.0


    daily_drawdown_limit: float = 0.05
    max_drawdown_limit: float = 0.10
    trailing_drawdown: bool = False

    # Probability that an episode runs on a price-mirrored copy of the data.
    # XAUUSD rose 149.8% across this dataset (1,735 -> 4,332), so a random long
    # held 96 bars earns +241 points and a random short loses the same. Long
    # P&L was +26,553 against short -17,591 purely from that drift, and the
    # holdout is also a bull market (+32.8%), so nothing anywhere in the
    # pipeline punishes a permanently-long policy. Mirroring half the episodes
    # makes direction symmetric, so edge has to come from structure rather than
    # from knowing gold went up. 0.0 disables.
    mirror_augmentation_prob: float = 0.5


    live_mode: bool = False
    debug: bool = False


    max_position_pct: float = 0.05
    max_total_exposure: float = 1.0

    daily_dd_safety_buffer: float = 0.008
    max_dd_safety_buffer: float = 0.015
    emergency_close_threshold: float = 0.09


    # Stop placed at a fixed ATR multiple rather than a fixed euro amount, so
    # risk stays constant in probability terms as volatility changes. 1.5 ATR
    # is the level the reachability study used: P(hit 2R before stop) 0.307
    # over a 96-bar horizon.
    # Share of episodes that start inside the top volatility quintile. The
    # post-war regime is ~10% of the merged dataset, so uniform sampling gives
    # the agent 10% exposure to the market it now has to trade.
    high_vol_oversample_prob: float = 0.40

    # Share of the drawdown budget that may be spent before new entries are
    # refused by the action mask. 0.75 leaves a quarter of the limit as
    # reserve to trade out of an existing position. 0.0 disables the veto.
    dd_entry_veto_fraction: float = 0.75

    atr_stop_enabled: bool = True
    atr_stop_multiplier: float = 1.5

    risk_per_trade_pct: float = 0.003
    max_risk_per_trade_pct: float = 0.007
    # Anti-martingale, the mirror of loss_layer_risk_mult: press while the
    # account is proving itself. Gated on drawdown headroom in
    # _win_streak_risk_multiplier - a winning streak is not evidence when
    # the account is already halfway to its limit. max_risk_per_trade_pct
    # still binds, so this raises typical risk without raising the cap.
    win_streak_risk_enabled: bool = True
    win_streak_risk_min_wins: int = 2
    win_streak_risk_mult: Tuple[float, ...] = (1.0, 1.15, 1.30, 1.45)
    max_positions: int = 1


    hard_stop_loss_eur: float = 220.0
    soft_stop_loss_eur: float = 140.0
    trailing_activation_eur: float = 100.0
    trailing_retrace_pct: float = 0.30
    time_decay_hours: float = 4.0


    max_trades_per_day: int = 20
    max_trades_per_session: int = 10
    max_trades_per_episode: int = 0
    max_consecutive_losses: int = 3
    loss_layer_stop: int = 5


    session_loss_limit_pct: float = 0.99
    session_consecutive_loss_limit: int = 99


    observation_period_required: bool = False
    min_bars_observation_before_entry: int = 0
    min_bars_between_entries: int = 0
    min_bars_after_loss: int = 0
    min_minutes_between_entries: int = 5
    min_minutes_after_loss: int = 15


    prime_start: dtime = dtime(14, 0)
    prime_end: dtime = dtime(17, 0)
    no_new_trades_start: dtime = dtime(18, 0)
    no_new_trades_end: dtime = dtime(9, 0)
    hard_close_time: dtime = dtime(22, 0)
    final_exit_window_minutes: int = 60
    allow_weekend_holding: bool = False


    enforce_no_new_trades_window: bool = True
    enforce_weekend_block: bool = True
    enforce_hard_close: bool = True


    entry_quality_gate_enabled: bool = True
    entry_quality_threshold: float = 0.35
    min_setup_quality_for_entry: float = 0.0


    observation_size: int = 90
    max_steps_per_episode: int = 2000
    gamma: float = 0.95


    reward: RewardConfig = field(default_factory=RewardConfig)


    reward_scale: float = 10.0
    loss_multiplier: float = 1.0


    execution: ExecutionConfig = field(default_factory=ExecutionConfig)


    domain_randomization_enabled: bool = True
    spread_mult_range: Tuple[float, float] = (0.90, 1.45)
    slippage_mult_range: Tuple[float, float] = (0.90, 1.60)
    latency_bars_range: Tuple[int, int] = (0, 2)
    volatility_scale_range: Tuple[float, float] = (0.90, 1.20)


    size_buckets: Tuple[float, ...] = (0.35, 0.60, 0.85, 1.10)


    def sync_legacy_from_reward(self) -> None:
        try:
            self.reward_scale = float(self.reward.reward_scale)
            self.loss_multiplier = float(self.reward.loss_multiplier)
        except Exception:

            pass

    def sync_reward_from_legacy(self) -> None:
        try:
            self.reward.reward_scale = float(self.reward_scale)
            self.reward.loss_multiplier = float(self.loss_multiplier)
        except Exception:
            pass

    def __post_init__(self) -> None:

        self.sync_legacy_from_reward()

        try:
            policy = load_risk_policy()
            if not policy:
                return

            pf = policy.get("prop_firm", {})
            self.daily_drawdown_limit = pf.get("daily_drawdown_limit", self.daily_drawdown_limit)
            self.max_drawdown_limit = pf.get("max_drawdown_limit", self.max_drawdown_limit)
            self.trailing_drawdown = pf.get("trailing_drawdown", self.trailing_drawdown)
            self.daily_dd_safety_buffer = pf.get("daily_dd_safety_buffer", self.daily_dd_safety_buffer)
            self.max_dd_safety_buffer = pf.get("max_dd_safety_buffer", self.max_dd_safety_buffer)
            self.emergency_close_threshold = pf.get("emergency_close_all_threshold", self.emergency_close_threshold)

            ls = policy.get("lot_sizing", {})
            self.risk_per_trade_pct = ls.get("risk_per_trade_pct", self.risk_per_trade_pct)
            self.max_risk_per_trade_pct = ls.get("max_risk_per_trade_pct", self.max_risk_per_trade_pct)
            self.max_positions = ls.get("max_positions", self.max_positions)

            ex = policy.get("exit_strategies", {})
            self.hard_stop_loss_eur = ex.get("hard_stop_loss_eur", self.hard_stop_loss_eur)
            self.soft_stop_loss_eur = ex.get("soft_stop_loss_eur", self.soft_stop_loss_eur)
            self.trailing_activation_eur = ex.get("trailing_activation_eur", self.trailing_activation_eur)
            self.trailing_retrace_pct = ex.get("trailing_retrace_pct", self.trailing_retrace_pct)
            self.time_decay_hours = ex.get("time_decay_hours", self.time_decay_hours)

            tl = policy.get("trade_limits", {})
            self.max_trades_per_day = tl.get("max_trades_per_day", self.max_trades_per_day)

            pm = policy.get("position_manager", {})
            self.max_consecutive_losses = pm.get("max_consecutive_losses", self.max_consecutive_losses)

            timing = policy.get("timing_policy", {})
            self.min_bars_between_entries = timing.get("min_bars_between_entries", self.min_bars_between_entries)
            self.min_bars_after_loss = timing.get("min_bars_after_loss", self.min_bars_after_loss)
            self.min_minutes_between_entries = timing.get("min_minutes_between_entries", self.min_minutes_between_entries)
            self.min_minutes_after_loss = timing.get("min_minutes_after_loss", self.min_minutes_after_loss)
            self.max_trades_per_session = timing.get("max_trades_per_session", self.max_trades_per_session)

            sess = policy.get("session_management", {})
            if isinstance(sess, dict):
                self.allow_weekend_holding = sess.get("allow_weekend_holding", self.allow_weekend_holding)
                self.final_exit_window_minutes = sess.get("final_exit_window_minutes", self.final_exit_window_minutes)


            self.sync_legacy_from_reward()

        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"sync_from_yaml() failed: {type(e).__name__}: {e}")


@dataclass
class PropPosition:
    instrument: str
    direction: str
    entry_price: float
    entry_dt: Optional[datetime]
    entry_bar: int
    lot_size: float
    initial_risk_eur: float
    peak_pnl: float = 0.0
    lowest_pnl: float = 0.0
    entry_fee_eur: float = 0.0
    entry_quality: float = 0.5
    entry_certainty: float = 0.5
    setup_quality: float = 0.5
    confluence_count: int = 0
    bars_since_setup: int = 0
    deliberation_bars: int = 0
    is_fomo_entry: bool = False
    is_revenge_entry: bool = False
    entry_context: Optional[Dict[str, Any]] = None


@dataclass
class TradeResult:
    net_pnl: float
    initial_risk_eur: float
    mae: float
    mfe: float
    bars_held: int
    close_reason: CloseReason
    entry_quality: float
    direction: str
    lot_size: float
    entry_bar: int = 0
    entry_certainty: float = 0.5
    setup_quality: float = 0.5
    confluence_count: int = 0
    bars_since_setup: int = 0
    deliberation_bars: int = 0
    is_fomo_entry: bool = False
    is_revenge_entry: bool = False
    total_fees: float = 0.0
    entry_dt: Optional[datetime] = None
    entry_context: Optional[Dict[str, Any]] = None


__all__ = [
    "CloseReason",
    "PropFirmConfig",
    "PropPosition",
    "RewardConfig",
    "TradeResult",
    "load_risk_policy",
]
