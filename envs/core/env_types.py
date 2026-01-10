# envs/env_types.py
"""
Environment type definitions for prop firm trading.

Contains dataclasses, enums, and configuration types used by PropFirmTradingEnv.

Upgrades (Jan 2026):
- Added per-step shaping bounds + patience/anti-churn knobs used by RewardShapingMixin
- Added legacy reward fields + sync helpers (sync_legacy_from_reward / sync_reward_from_legacy)
  to prevent silent drift when older code expects config.reward_scale etc.
- Kept full backward compatibility with existing attributes and YAML overrides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Import ExecutionConfig for PropFirmConfig
from envs.core.execution_model import ExecutionConfig


class CloseReason(str, Enum):
    """Enumeration of all possible trade close reasons with quality ranking."""
    # Good exits (agent demonstrated skill)
    TRAILING_STOP = "trailing_stop"           # Best: locked in profits
    AGENT_CLOSE = "agent_close"               # Good: voluntary exit

    # Neutral exits (circumstantial)
    TIME_DECAY = "time_decay"                 # Neutral: time limit reached
    HARD_CLOSE = "hard_close"                 # Neutral: session end
    WEEKEND_FLATTEN = "weekend_flatten"       # Neutral: weekend policy
    DAILY_LIMIT_SAFETY = "daily_limit_safety" # Neutral-bad: approaching limit

    # Bad exits (risk management triggered)
    HARD_STOP = "hard_stop"                   # Bad: max loss hit
    EMERGENCY_CLOSE = "emergency_close"       # Bad: emergency threshold
    RISK_LIQUIDATION = "risk_liquidation"     # Worst: DD breach

    # Truncation (episode boundary)
    EPISODE_TRUNCATE = "episode_truncate_flatten"  # Context-dependent

    @property
    def quality_score(self) -> float:
        """Returns exit quality score [0, 1] where 1 is best."""
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
        """
        Returns close priority for override logic.
        Higher priority = more urgent close reason.
        Used to ensure forced closes override agent closes.
        """
        priorities = {
            CloseReason.RISK_LIQUIDATION: 100,     # Highest: DD breach
            CloseReason.EMERGENCY_CLOSE: 95,       # Emergency threshold
            CloseReason.DAILY_LIMIT_SAFETY: 90,    # Near daily limit
            CloseReason.HARD_STOP: 85,             # Max loss hit
            CloseReason.TRAILING_STOP: 70,         # Profit protection
            CloseReason.HARD_CLOSE: 60,            # Session end
            CloseReason.WEEKEND_FLATTEN: 60,       # Weekend policy
            CloseReason.TIME_DECAY: 50,            # Time limit
            CloseReason.EPISODE_TRUNCATE: 40,      # Episode boundary
            CloseReason.AGENT_CLOSE: 10,           # Lowest: voluntary
        }
        return int(priorities.get(self, 40))


@dataclass
class RewardConfig:
    """
    Comprehensive reward shaping configuration.

    Design principles:
    1) Trade-close-centric: main learning signal comes from completed trades
    2) Risk-adjusted: rewards consider actual risk taken (MAE), not just outcome
    3) Behavior shaping: encourage habits (quick profits, proper exits)
    4) Anti-gaming: prevent reward hacking (truncation abuse, churn)
    5) Bounded: components have caps to prevent gradient explosion
    """

    # --------------------
    # Core scaling / shape
    # --------------------
    reward_scale: float = 10.0

    # Asymmetry control: <1.0 = risk-seeking, 1.0 = neutral, >1.0 = risk-averse
    loss_multiplier: float = 1.0

    # --------------------
    # PnL Dominance Scaling (v6.0 - CRITICAL FOR PROFITABILITY)
    # --------------------
    # Multiplier to make base_pnl reward numerically competitive with shaping.
    # Without this, a €100 profit = 0.001 * reward_scale = 0.006, while shaped
    # bonuses are ~0.1-0.2. This causes reward optimization to decouple from profitability.
    # Recommended: 150-300 so €100 profit → 0.15-0.30 base reward (dominant over shaping)
    pnl_scale_factor: float = 200.0

    # Cap shaping rewards relative to base_pnl magnitude (prevents shaping from
    # overwhelming PnL signal). Set to 0.0 to disable capping.
    # E.g., 0.5 means total shaping cannot exceed 50% of |base_pnl|
    max_shaping_to_pnl_ratio: float = 0.5

    # --------------------
    # Execution Cost Visibility (v6.0)
    # --------------------
    # Make execution costs (spread + slippage + commission) visible in reward.
    # This teaches the agent that frequent trading has a real cost.
    execution_cost_visibility_enabled: bool = True
    execution_cost_reward_scale: float = 0.5  # Multiplier for cost penalty in reward

    # --------------------
    # R-multiple bonuses
    # --------------------
    r_multiple_bonus_threshold: float = 1.5
    r_multiple_bonus_scale: float = 0.3
    r_multiple_bonus_cap: float = 0.6

    # --------------------
    # MAE efficiency bonus
    # --------------------
    mae_efficiency_enabled: bool = True
    mae_efficiency_scale: float = 0.25
    mae_efficiency_threshold: float = 2.0

    # --------------------
    # Time efficiency
    # --------------------
    time_efficiency_enabled: bool = True
    time_efficiency_scale: float = 0.15
    optimal_trade_bars: int = 8
    max_trade_bars_for_bonus: int = 24

    # --------------------
    # Exit quality modifiers
    # --------------------
    exit_quality_enabled: bool = True
    trailing_stop_bonus: float = 0.25   # Reward letting winners run to trailing stop
    agent_close_bonus: float = -0.05    # Slight penalty for manual closes (often premature)
    hard_stop_penalty: float = 0.15
    risk_liquidation_penalty: float = 0.30

    # Premature close penalty (penalize agent_close that leaves profit on table)
    premature_close_capture_threshold: float = 0.7  # If captured < 70% of MFE, penalize
    premature_close_penalty_scale: float = 0.25
    premature_close_penalty_cap: float = 0.15

    # --------------------
    # Good Loss Cut Rewards (v6.0 - CRITICAL FOR AGENT CONTROL)
    # --------------------
    # Reward agent for voluntarily cutting losses BEFORE they hit hard stop.
    # This teaches the agent that controlled exits are better than letting stops hit.
    # Without this, agent learns to "let the environment handle exits" = reward hacking.
    good_loss_cut_enabled: bool = True
    good_loss_cut_bonus: float = 0.08           # Base bonus for cutting a loser early
    good_loss_cut_efficiency_threshold: float = 0.3  # Min efficiency to qualify (1 - |pnl|/mae)
    good_loss_cut_max_bonus: float = 0.15       # Cap on loss cut bonus

    # --------------------
    # Truncation handling
    # --------------------
    truncation_winner_discount: float = 0.30
    truncation_loser_extra_penalty: float = 0.15

    # --------------------
    # Entry quality integration
    # --------------------
    entry_quality_integration: bool = True
    entry_quality_weight: float = 0.2

    # --------------------
    # Session timing rewards
    # --------------------
    session_timing_enabled: bool = True
    off_hours_trade_penalty: float = 0.15
    prime_hours_trade_bonus: float = 0.05

    # --------------------
    # Market structure rewards (v5.3)
    # --------------------
    market_structure_enabled: bool = False
    sr_proximity_bonus: float = 0.10
    sr_proximity_penalty: float = 0.08
    structure_alignment_bonus: float = 0.12
    bos_alignment_bonus: float = 0.08
    order_block_entry_bonus: float = 0.06

    # --------------------
    # Divergence awareness (v5.3)
    # --------------------
    divergence_awareness_enabled: bool = False
    divergence_contra_penalty: float = 0.15
    divergence_aligned_bonus: float = 0.10
    overbought_long_penalty: float = 0.12
    oversold_short_penalty: float = 0.12

    # --------------------
    # Regime awareness (v5.3)
    # --------------------
    regime_awareness_enabled: bool = False
    risk_off_aggressive_penalty: float = 0.10
    high_vol_size_penalty: float = 0.08

    # --------------------
    # Drawdown shaping
    # --------------------
    dd_shaping_enabled: bool = True
    dd_threshold: float = 0.02
    dd_penalty_scale: float = 1.0
    dd_severity_exponent: float = 1.5
    dd_severity_cap: float = 1.5

    # --------------------
    # Streak modifiers
    # --------------------
    streak_modifier_enabled: bool = True
    win_streak_bonus_per_win: float = 0.02
    loss_streak_penalty_per_loss: float = 0.03

    # --------------------
    # Anti-churn (episode-level + per-step optional)
    # --------------------
    anti_churn_enabled: bool = True

    # Episode/day-level soft limits (already used elsewhere in your reward code)
    daily_trade_soft_limit: int = 8
    churn_penalty_per_trade: float = 0.05

    # NEW (optional, used by upgraded RewardShapingMixin if per_step_shaping_enabled=True):
    # Small cost for taking actions while flat to discourage “button mashing”.
    # Keep tiny; shaping is bounded anyway.
    churn_action_cost: float = 0.0  # recommended 0.001–0.003 when enabled
    # --------------------
    # Cost-Aware Anti-Churn (v6.0)
    # --------------------
    # Penalize when execution costs exceed a threshold of gross profits.
    # This directly teaches "overtrading erodes edge" rather than just counting trades.
    cost_erosion_penalty_enabled: bool = True
    cost_erosion_threshold: float = 0.5        # Trigger when costs > 50% of gross profit
    cost_erosion_penalty_scale: float = 0.15   # Penalty multiplier
    cost_erosion_penalty_cap: float = 0.30     # Maximum penalty
    # --------------------
    # Trade activity consistency (episode-end)
    # --------------------
    activity_consistency_enabled: bool = True
    target_trades_per_1k_steps: float = 10.0
    stage_activity_targets: Optional[Dict[int, float]] = None
    activity_deviation_penalty_scale: float = 0.2
    min_trades_penalty: float = 0.3

    # --------------------
    # Blocked action penalties
    # --------------------
    hard_block_penalty: float = 0.10
    soft_block_penalty: float = 0.03

    # --------------------
    # Per-step shaping (optional)
    # --------------------
    per_step_shaping_enabled: bool = False
    holding_cost_per_bar: float = 0.0005

    # Legacy placeholder (kept for compatibility; not all code uses it)
    opportunity_bonus_scale: float = 0.01

    # NEW (optional, used by upgraded RewardShapingMixin):
    patience_shaping_enabled: bool = False
    patience_bonus_per_bar: float = 0.0          # recommended 0.0005–0.002 when enabled
    patience_quality_threshold: float = 0.35     # if best(q_long,q_short) < threshold, reward waiting    
    # C4 FIX: exploration_bonus must be in RewardConfig for shaping to read it
    # Used in early curriculum stages to encourage trade attempts
    exploration_bonus: float = 0.0
    # NEW: explicit bounds for per-step shaping so it can’t dominate trade-close reward
    per_step_min: float = -0.05
    per_step_max: float = 0.05

    # --------------------
    # Reward clipping (global)
    # --------------------
    min_reward: float = -5.0
    max_reward: float = 5.0


def load_risk_policy() -> Dict[str, Any]:
    """Load risk policy from config file."""
    # C8 FIX: Correct path to repo root config/risk_policy.yaml (was envs/config/)
    config_path = Path(__file__).resolve().parent.parent.parent / "config" / "risk_policy.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


@dataclass
class PropFirmConfig:
    """Configuration for PropFirmTradingEnv."""
    # --------------------
    # Time semantics
    # --------------------
    tz: str = "Europe/Berlin"
    primary_timeframe: str = "M15"
    instruments: List[str] = field(default_factory=lambda: ["XAUUSD"])

    # --------------------
    # Account
    # --------------------
    initial_balance: float = 100_000.0

    # --------------------
    # Prop firm limits
    # --------------------
    daily_drawdown_limit: float = 0.05
    max_drawdown_limit: float = 0.10
    trailing_drawdown: bool = False

    daily_dd_safety_buffer: float = 0.008
    max_dd_safety_buffer: float = 0.015
    emergency_close_threshold: float = 0.09

    # --------------------
    # Position sizing / limits
    # --------------------
    risk_per_trade_pct: float = 0.003
    max_risk_per_trade_pct: float = 0.007
    max_positions: int = 1

    # --------------------
    # Exits
    # --------------------
    hard_stop_loss_eur: float = 220.0
    soft_stop_loss_eur: float = 140.0
    trailing_activation_eur: float = 100.0
    trailing_retrace_pct: float = 0.30
    time_decay_hours: float = 4.0

    # --------------------
    # Trade limits
    # --------------------
    max_trades_per_day: int = 20
    max_trades_per_session: int = 10
    max_consecutive_losses: int = 3

    # --------------------
    # Timing policy
    # --------------------
    min_minutes_between_entries: int = 5
    min_minutes_after_loss: int = 15

    # --------------------
    # Session management
    # --------------------
    prime_start: dtime = dtime(14, 0)
    prime_end: dtime = dtime(17, 0)
    no_new_trades_start: dtime = dtime(18, 0)
    no_new_trades_end: dtime = dtime(9, 0)
    hard_close_time: dtime = dtime(22, 0)
    final_exit_window_minutes: int = 60
    allow_weekend_holding: bool = False

    # Constraint enforcement flags (controlled by curriculum stages)
    enforce_no_new_trades_window: bool = True
    enforce_weekend_block: bool = True
    enforce_hard_close: bool = True

    # --------------------
    # Entry-quality gate
    # --------------------
    entry_quality_gate_enabled: bool = True
    entry_quality_threshold: float = 0.35

    # --------------------
    # PPO env
    # --------------------
    observation_size: int = 64
    max_steps_per_episode: int = 2000
    gamma: float = 0.95

    # --------------------
    # Reward configuration
    # --------------------
    reward: RewardConfig = field(default_factory=RewardConfig)

    # LEGACY reward fields (for older codepaths / dashboards)
    # These are synchronized from RewardConfig by sync_legacy_from_reward().
    reward_scale: float = 10.0
    loss_multiplier: float = 1.0

    # --------------------
    # Execution anti-cheat
    # --------------------
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    # --------------------
    # Domain randomization
    # --------------------
    domain_randomization_enabled: bool = True
    spread_mult_range: Tuple[float, float] = (0.90, 1.45)
    slippage_mult_range: Tuple[float, float] = (0.90, 1.60)
    latency_bars_range: Tuple[int, int] = (0, 2)
    volatility_scale_range: Tuple[float, float] = (0.90, 1.20)

    # --------------------
    # Discrete action space (size buckets)
    # --------------------
    size_buckets: Tuple[float, ...] = (0.35, 0.60, 0.85, 1.10)

    # --------------------
    # Sync helpers
    # --------------------
    def sync_legacy_from_reward(self) -> None:
        """
        Copy RewardConfig -> legacy fields.
        Safe to call repeatedly; does not overwrite RewardConfig.
        """
        try:
            self.reward_scale = float(self.reward.reward_scale)
            self.loss_multiplier = float(self.reward.loss_multiplier)
        except Exception:
            # stay silent; legacy fields are non-critical
            pass

    def sync_reward_from_legacy(self) -> None:
        """
        Copy legacy fields -> RewardConfig.
        Use only if you intentionally set legacy fields directly.
        """
        try:
            self.reward.reward_scale = float(self.reward_scale)
            self.reward.loss_multiplier = float(self.loss_multiplier)
        except Exception:
            pass

    def __post_init__(self) -> None:
        # Ensure legacy fields start consistent with RewardConfig defaults
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
            self.min_minutes_between_entries = timing.get("min_minutes_between_entries", self.min_minutes_between_entries)
            self.min_minutes_after_loss = timing.get("min_minutes_after_loss", self.min_minutes_after_loss)
            self.max_trades_per_session = timing.get("max_trades_per_session", self.max_trades_per_session)

            sess = policy.get("session_management", {})
            if isinstance(sess, dict):
                self.allow_weekend_holding = sess.get("allow_weekend_holding", self.allow_weekend_holding)
                self.final_exit_window_minutes = sess.get("final_exit_window_minutes", self.final_exit_window_minutes)

            # IMPORTANT: do NOT call sync_reward_from_legacy() here.
            # RewardConfig is intended to be stage-controlled by curriculum overrides.
            self.sync_legacy_from_reward()

        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"sync_from_yaml() failed: {type(e).__name__}: {e}")


@dataclass
class PropPosition:
    """Represents an open trading position."""
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
    entry_context: Optional[Dict[str, Any]] = None  # Market structure context at entry (v5.3)


@dataclass
class TradeResult:
    """Encapsulates all information about a completed trade for reward calculation."""
    net_pnl: float
    initial_risk_eur: float
    mae: float
    mfe: float
    bars_held: int
    close_reason: CloseReason
    entry_quality: float
    direction: str
    lot_size: float
    total_fees: float = 0.0
    entry_dt: Optional[datetime] = None
    entry_context: Optional[Dict[str, Any]] = None  # v5.3


__all__ = [
    "CloseReason",
    "RewardConfig",
    "PropFirmConfig",
    "PropPosition",
    "TradeResult",
    "load_risk_policy",
]
