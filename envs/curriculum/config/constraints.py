# envs/curriculum/config/constraints.py
"""
Trading constraints and reward shaping configuration.

Contains:
- TradingConstraints: Position limits, drawdown limits, session rules
- RewardShaping: Reward configuration for each curriculum stage

Upgrades (Jan 2026):
- RewardShaping expanded to stay feature-parallel with env_types.RewardConfig
  (premature close, activity consistency, patience shaping, bounded per-step shaping).
- Fixed default sign bug: agent_close_bonus should discourage manual closes by default.
- TradingConstraints keeps enforce_session_windows for backward compatibility, but
  auto-maps it to the newer fine-grained flags when used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class RewardShaping:
    """Reward shaping configuration for a curriculum stage."""
    # Core scaling
    reward_scale: float = 10.0
    loss_multiplier: float = 1.0

    # --------------------
    # PnL Dominance Scaling (v6.0 - CRITICAL FOR PROFITABILITY)
    # --------------------
    # Multiplier to make base_pnl reward numerically competitive with shaping.
    # Without this, a €100 profit = 0.001 * reward_scale = 0.006, while shaped
    # bonuses are ~0.1-0.2. This causes reward optimization to decouple from profitability.
    pnl_scale_factor: float = 200.0
    max_shaping_to_pnl_ratio: float = 0.5  # Cap shaping to 50% of |base_pnl|

    # --------------------
    # Execution Cost Visibility (v6.0)
    # --------------------
    execution_cost_visibility_enabled: bool = True
    execution_cost_reward_scale: float = 0.5

    # R-multiple bonuses
    r_multiple_bonus_threshold: float = 1.5
    r_multiple_bonus_scale: float = 0.3
    r_multiple_bonus_cap: float = 0.6

    # MAE efficiency
    mae_efficiency_enabled: bool = False
    mae_efficiency_scale: float = 0.25
    mae_efficiency_threshold: float = 2.0

    # Time efficiency
    time_efficiency_enabled: bool = False
    time_efficiency_scale: float = 0.15
    optimal_trade_bars: int = 8
    max_trade_bars_for_bonus: int = 24

    # Exit quality
    exit_quality_enabled: bool = False
    trailing_stop_bonus: float = 0.15

    # FIX: Default should DISCOURAGE manual closes (aligns with env_types RewardConfig intent)
    agent_close_bonus: float = -0.05

    hard_stop_penalty: float = 0.15
    risk_liquidation_penalty: float = 0.30

    # Premature close penalty (agent_close leaving profit on table)
    premature_close_capture_threshold: float = 0.7
    premature_close_penalty_scale: float = 0.25
    premature_close_penalty_cap: float = 0.15

    # --------------------
    # Good Loss Cut Rewards (v6.0 - CRITICAL FOR AGENT CONTROL)
    # --------------------
    good_loss_cut_enabled: bool = True
    good_loss_cut_bonus: float = 0.08
    good_loss_cut_efficiency_threshold: float = 0.3
    good_loss_cut_max_bonus: float = 0.15

    # Truncation handling
    truncation_winner_discount: float = 0.30
    truncation_loser_extra_penalty: float = 0.15

    # Entry quality integration
    entry_quality_integration: bool = False
    entry_quality_weight: float = 0.2

    # Session timing rewards (teach trading hours)
    session_timing_enabled: bool = False
    off_hours_trade_penalty: float = 0.15
    prime_hours_trade_bonus: float = 0.05

    # Market structure rewards (teach WHERE to trade - v5.3)
    market_structure_enabled: bool = False
    sr_proximity_bonus: float = 0.10
    sr_proximity_penalty: float = 0.08
    structure_alignment_bonus: float = 0.12
    bos_alignment_bonus: float = 0.08
    order_block_entry_bonus: float = 0.06

    # Divergence/momentum rewards (teach reversal awareness - v5.3)
    divergence_awareness_enabled: bool = False
    divergence_contra_penalty: float = 0.15
    divergence_aligned_bonus: float = 0.10
    overbought_long_penalty: float = 0.12
    oversold_short_penalty: float = 0.12

    # Regime awareness rewards (teach context sensitivity - v5.3)
    regime_awareness_enabled: bool = False
    risk_off_aggressive_penalty: float = 0.10
    high_vol_size_penalty: float = 0.08

    # Drawdown shaping
    dd_shaping_enabled: bool = False
    dd_threshold: float = 0.02
    dd_penalty_scale: float = 1.0
    dd_severity_exponent: float = 1.5
    dd_severity_cap: float = 1.5

    # Streak modifiers
    streak_modifier_enabled: bool = False
    win_streak_bonus_per_win: float = 0.02
    loss_streak_penalty_per_loss: float = 0.03

    # Anti-churn
    anti_churn_enabled: bool = False
    daily_trade_soft_limit: int = 20
    churn_penalty_per_trade: float = 0.02

    # Optional: small cost for “button mashing” while flat (only meaningful if per-step shaping enabled)
    churn_action_cost: float = 0.0
    # --------------------
    # Cost-Aware Anti-Churn (v6.0)
    # --------------------
    cost_erosion_penalty_enabled: bool = True
    cost_erosion_threshold: float = 0.5
    cost_erosion_penalty_scale: float = 0.15
    cost_erosion_penalty_cap: float = 0.30
    # Trade activity consistency (episode-end)
    activity_consistency_enabled: bool = False
    target_trades_per_1k_steps: float = 10.0
    stage_activity_targets: Optional[Dict[int, float]] = None
    activity_deviation_penalty_scale: float = 0.2
    activity_deviation_penalty_cap: float = 2.0  # FIX: Max penalty for episode-level overtrading
    min_trades_penalty: float = 0.3

    # Blocked action penalties
    hard_block_penalty: float = 0.02
    soft_block_penalty: float = 0.01

    # Per-step shaping
    per_step_shaping_enabled: bool = False
    holding_cost_per_bar: float = 0.0
    opportunity_bonus_scale: float = 0.0

    # Optional patience shaping (reward waiting when setups are weak)
    patience_shaping_enabled: bool = False
    patience_bonus_per_bar: float = 0.0
    patience_quality_threshold: float = 0.35

    # Loss streak caution (penalize entry attempts while tilted)
    loss_streak_caution_enabled: bool = True
    loss_streak_caution_base: float = 0.03
    loss_streak_caution_cap: float = 0.25

    # Bounds for per-step shaping so it never dominates trade-close reward
    per_step_min: float = -0.05
    per_step_max: float = 0.05

    # Global clipping
    min_reward: float = -5.0
    max_reward: float = 5.0

    # Legacy / external curriculum fields (not necessarily used by env)
    exploration_bonus: float = 0.0
    directional_accuracy_weight: float = 1.0


@dataclass
class TradingConstraints:
    """Trading constraints for a curriculum stage."""
    max_positions: int = 1

    max_trades_per_day: int = 100
    max_trades_per_session: int = 50
    max_consecutive_losses: int = 10
    
    # Loss-layer governor: hard stop-trading mode after this many consecutive losses
    # Early stages should be permissive (10+) to allow exploration
    # Later stages tighten to enforce loss discipline
    loss_layer_stop: int = 5
    
    # Session budget constraints (v5.5 - governor/budget observation)
    # These are ALWAYS computed but limits start "effectively infinite" in early stages.
    # This ensures observation features are meaningful from day one (no distribution shift).
    # session_loss_limit_pct: max % loss within a session before entries blocked
    # session_consecutive_loss_limit: max consecutive losses within a session before blocked
    session_loss_limit_pct: float = 0.99  # Default: effectively infinite (99% loss allowed)
    session_consecutive_loss_limit: int = 99  # Default: effectively infinite

    # Backward compat: was a coarse switch, newer code uses the three flags below.
    # If True, we will auto-enable enforce_no_new_trades_window and enforce_hard_close unless explicitly set.
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

    def __post_init__(self) -> None:
        # Auto-map old coarse flag to the newer fine-grained flags (avoids silent “no enforcement” bugs).
        if self.enforce_session_windows:
            if not self.enforce_no_new_trades_window:
                self.enforce_no_new_trades_window = True
            if not self.enforce_hard_close:
                self.enforce_hard_close = True


__all__ = [
    "RewardShaping",
    "TradingConstraints",
]
