# envs/prop_firm_env.py
"""
Prop Firm Trading Environment - Unified Observation + Anti-Cheat Execution
==========================================================================

Enhanced Reward System v2.0:
- MAE-aware trade quality rewards (uses actual risk experienced)
- Time-efficiency bonuses for quick profitable trades
- Proper asymmetric reward scaling with configurable risk aversion
- Exit-quality differentiation (trailing stop > agent close > time decay > hard stop)
- Truncation penalties for BOTH winners and losers
- Progressive drawdown shaping with soft caps
- Entry quality integration into trade outcome rewards
- Opportunity cost signals for missed good setups
- Anti-churn mechanisms to prevent overtrading
- Consecutive win/loss streak modifiers
- Risk-adjusted return component (profit/MAE ratio)

Upgrades implemented:
- Discrete action space (single Discrete action id)
- Action masking for hard-rule legality via env.action_masks() (sb3-contrib MaskablePPO)
- Domain randomization per episode (spread/slippage/latency/vol scaling)
- Reward: trade-close-centric (prevents per-step penalty accumulation collapse)
- Time semantics: Europe/Berlin local (naive datetimes treated as local)
- Risk: DD breach liquidation BEFORE termination (agent experiences realized outcome)

Fixes applied (reward/system correctness):
- Mask/step time alignment: action_masks evaluates legality on the SAME bar that step() uses
- Per-step quote caching: quote() called once per step; mark-to-market, forced-exit checks,
  and observation all reuse the same bid/ask to avoid intra-step stochastic drift.
- Termination penalty does not stack on top of a just-executed risk_liquidation close reward.
- Forced-close logic uses a single unrealized pnl value per step (no repeated re-quotes).
- ALWAYS flatten open position on TRUNCATION (prevents "free option" at episode end).
- Trade-close reward uses NET trade outcome (includes entry fee + exit fee).
- Close outcomes / win-rate / consecutive-losses use net trade outcome (fee-aware).
- Fixed double-penalty bug for risk_liquidation losses
- Fixed asymmetric scaling encouraging excessive risk-taking
- Fixed truncation winners receiving full reward
- Fixed R-multiple bonus being size-independent
- Fixed unbounded drawdown severity explosion

Additional hardening (Dec 2025):
- CONSISTENT fill timing: entries and exits both enforce a minimum 1-bar delay (+ latency)
- Seed MAE/MFE immediately on entry mark-to-market (prevents truncation “free option” leak)
- Clamp drawdowns to [0, +inf) to avoid negative DD artifacts while in-profit
- Session counting for cross-midnight windows uses an anchor date (prevents per-midnight reset exploits)
- dt=None fallback enforces spacing/cooldown via bar counts (prevents overtrade on timestamp-less data)
- Aggregated HTF market_data includes open+volume keys (builder contract robustness)

CURRICULUM INTEGRATION (this version):
- Optional CurriculumManager wiring (stage/epoch metadata + episode recording)
- Stage transition is applied on NEXT reset (clean epoch boundaries)
- Robust override application from CurriculumStageConfig when available:
  - env_overrides -> PropFirmConfig
  - reward_overrides -> RewardConfig
  - execution_overrides -> ExecutionConfig
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, date, timedelta
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import yaml
from zoneinfo import ZoneInfo

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------
# Optional curriculum support
# ---------------------------
if TYPE_CHECKING:
    from envs.curriculum_manager import CurriculumManager

try:
    from envs.curriculum_manager import CurriculumManager as _CurriculumManager
    from envs.curriculum_config import CurriculumStage, CurriculumStageConfig, DataDifficulty
    CURRICULUM_AVAILABLE = True
except Exception:
    _CurriculumManager = None  # type: ignore
    CurriculumStage = None  # type: ignore
    CurriculumStageConfig = None  # type: ignore
    DataDifficulty = None  # type: ignore
    CURRICULUM_AVAILABLE = False

try:
    from modules.meta.ppo_observation_builder import PPOObservationBuilder, PPO_OBS_SIZE, PPO_OBS_VERSION
    OBS_BUILDER_AVAILABLE = True
except Exception:
    PPOObservationBuilder = None  # type: ignore
    PPO_OBS_SIZE = 64
    PPO_OBS_VERSION = "5.0"
    OBS_BUILDER_AVAILABLE = False


def validate_observation_version(saved_version: str, saved_size: int) -> None:
    """
    Validate that saved model's observation version matches current builder.
    
    AUDIT FIX (CRIT-4): Prevents silent policy corruption when observation
    layout changes between training and inference.
    
    Args:
        saved_version: PPO_OBS_VERSION from saved model metadata
        saved_size: Observation size from saved model's observation space
        
    Raises:
        ValueError: If version or size mismatch detected
    """
    if saved_size != PPO_OBS_SIZE:
        raise ValueError(
            f"Observation SIZE MISMATCH! Model trained with {saved_size}-dim observations, "
            f"but current PPOObservationBuilder produces {PPO_OBS_SIZE}-dim. "
            f"This will cause silent policy corruption. Retrain model or rollback builder."
        )
    
    # Compare major.minor version (ignore patch)
    def parse_version(v: str) -> tuple:
        parts = v.split(".")
        return tuple(int(p) for p in parts[:2])
    
    try:
        saved_major_minor = parse_version(saved_version)
        current_major_minor = parse_version(PPO_OBS_VERSION)
        
        if saved_major_minor != current_major_minor:
            raise ValueError(
                f"Observation VERSION MISMATCH! Model trained with v{saved_version}, "
                f"current builder is v{PPO_OBS_VERSION}. Feature layout may have changed. "
                f"Retrain model or downgrade builder."
            )
    except Exception as e:
        if "MISMATCH" in str(e):
            raise
        # If version parsing fails, just warn
        import warnings
        warnings.warn(
            f"Could not parse observation versions (saved={saved_version}, current={PPO_OBS_VERSION}). "
            f"Proceeding but policy may be corrupted if layout changed.",
            UserWarning
        )


import logging
# MED-3 FIX: Use standardized envs logging pattern
from envs.shared_utils import (
    get_envs_logger,
    safe_float as _safe_float,
    safe_int as _safe_int,
    clamp as _clamp,
    direction_sign,
    TIMEFRAME_MINUTES,
    DEFAULT_PRIMARY_TIMEFRAME,
    timeframe_to_minutes,
)

logger = get_envs_logger("prop_firm_env")

from envs.execution_model import ExecutionConfig, ExecutionModel


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
        return scores.get(self, 0.5)

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
        return priorities.get(self, 40)


@dataclass
class RewardConfig:
    """
    Comprehensive reward shaping configuration.

    Design principles:
    1. Trade-close-centric: Main learning signal comes from completed trades
    2. Risk-adjusted: Rewards consider actual risk taken (MAE), not just outcome
    3. Behavior shaping: Encourage good habits (quick profits, proper exits)
    4. Anti-gaming: Prevent reward hacking (truncation abuse, churn)
    5. Bounded: All components have soft caps to prevent gradient explosion
    """
    # Base reward scaling
    reward_scale: float = 10.0

    # Asymmetry control: <1.0 = risk-seeking, 1.0 = neutral, >1.0 = risk-averse
    loss_multiplier: float = 1.0

    # R-multiple bonuses
    r_multiple_bonus_threshold: float = 1.5
    r_multiple_bonus_scale: float = 0.3
    r_multiple_bonus_cap: float = 0.6

    # MAE efficiency bonus
    mae_efficiency_enabled: bool = True
    mae_efficiency_scale: float = 0.25
    mae_efficiency_threshold: float = 2.0

    # Time efficiency
    time_efficiency_enabled: bool = True
    time_efficiency_scale: float = 0.15
    optimal_trade_bars: int = 8
    max_trade_bars_for_bonus: int = 24

    # Exit quality modifiers - TEACH PATIENCE
    # The key insight: trailing_stop = agent let the trade run and got stopped at profit
    # agent_close = agent chickened out and closed manually (often leaving profit on table)
    exit_quality_enabled: bool = True
    trailing_stop_bonus: float = 0.25  # STRONG: Reward letting winners run to trailing stop
    agent_close_bonus: float = -0.05   # PENALTY: Discourage manually cutting winners
    hard_stop_penalty: float = 0.15
    risk_liquidation_penalty: float = 0.30
    
    # Premature close penalty (penalize agent_close that leaves profit on table)
    premature_close_capture_threshold: float = 0.7  # If captured < 70% of MFE, penalize
    premature_close_penalty_scale: float = 0.25  # Scale factor for penalty
    premature_close_penalty_cap: float = 0.15  # Maximum penalty

    # Truncation handling
    truncation_winner_discount: float = 0.30
    truncation_loser_extra_penalty: float = 0.15

    # Entry quality integration
    entry_quality_integration: bool = True
    entry_quality_weight: float = 0.2

    # Session timing penalties (direct signal for learning trading hours)
    session_timing_enabled: bool = True
    off_hours_trade_penalty: float = 0.15  # Penalty for trading in no-new-trades window
    prime_hours_trade_bonus: float = 0.05  # Bonus for trading in prime hours (14:00-17:00)

    # Market structure rewards (v5.3 - teach WHERE to trade)
    market_structure_enabled: bool = False
    sr_proximity_bonus: float = 0.10       # Bonus for entering near S/R
    sr_proximity_penalty: float = 0.08     # Penalty for entering far from S/R
    structure_alignment_bonus: float = 0.12 # Bonus for trading with structure (HH/HL or LL/LH)
    bos_alignment_bonus: float = 0.08      # Bonus for trading after BOS confirmation
    order_block_entry_bonus: float = 0.06  # Bonus for entering at order block levels
    
    # Divergence/momentum rewards (v5.3 - teach reversal awareness)
    divergence_awareness_enabled: bool = False
    divergence_contra_penalty: float = 0.15 # Penalty for trading against divergence
    divergence_aligned_bonus: float = 0.10  # Bonus for trading with divergence
    overbought_long_penalty: float = 0.12   # Penalty for going long when overbought
    oversold_short_penalty: float = 0.12    # Penalty for going short when oversold
    
    # Regime awareness rewards (v5.3 - teach context sensitivity)
    regime_awareness_enabled: bool = False
    risk_off_aggressive_penalty: float = 0.10  # Penalty for aggressive trades in risk-off
    high_vol_size_penalty: float = 0.08        # Penalty for large positions in high volatility

    # Drawdown shaping
    dd_shaping_enabled: bool = True
    dd_threshold: float = 0.02
    dd_penalty_scale: float = 1.0
    dd_severity_exponent: float = 1.5
    dd_severity_cap: float = 1.5

    # Streak modifiers
    streak_modifier_enabled: bool = True
    win_streak_bonus_per_win: float = 0.02
    loss_streak_penalty_per_loss: float = 0.03

    # Anti-churn - REASONABLE LIMITS
    # daily_trade_soft_limit = trades per DAY in market data, NOT per episode!
    # With M15 bars, 2400-step episode ≈ 25 trading days
    # So limit=8 means 200 trades/episode is the soft limit
    anti_churn_enabled: bool = True
    daily_trade_soft_limit: int = 8  # FIXED: 5 was too restrictive for learning
    churn_penalty_per_trade: float = 0.05  # FIXED: Softer penalty - exponential growth handles escalation

    # Trade activity consistency (NEW)
    # Encourages consistent trade counts across episodes
    activity_consistency_enabled: bool = True
    target_trades_per_1k_steps: float = 10.0  # ~20 trades per 2000-step episode (was 40→80!)
    # Stage-specific targets: early stages allow more exploration, later enforce discipline
    # Key = stage index (0=Foundation, 7=LiveReady), Value = trades per 1k steps
    stage_activity_targets: Optional[Dict[int, float]] = None  # If None, use target_trades_per_1k_steps
    activity_deviation_penalty_scale: float = 0.2  # Stronger penalty for deviation (was 0.1)
    min_trades_penalty: float = 0.3  # Penalty if < 20% of target trades

    # Blocked action penalties
    hard_block_penalty: float = 0.10
    soft_block_penalty: float = 0.03

    # Per-step shaping
    per_step_shaping_enabled: bool = False
    holding_cost_per_bar: float = 0.0005
    opportunity_bonus_scale: float = 0.01

    # Reward clipping
    min_reward: float = -5.0
    max_reward: float = 5.0


def load_risk_policy() -> Dict[str, Any]:
    config_path = Path(__file__).parent.parent / "config" / "risk_policy.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


@dataclass
class PropFirmConfig:
    # Time semantics
    tz: str = "Europe/Berlin"
    primary_timeframe: str = "M15"
    instruments: List[str] = field(default_factory=lambda: ["XAUUSD"])

    # Account
    initial_balance: float = 100_000.0

    # Prop firm limits
    daily_drawdown_limit: float = 0.05
    max_drawdown_limit: float = 0.10
    trailing_drawdown: bool = False

    daily_dd_safety_buffer: float = 0.008
    max_dd_safety_buffer: float = 0.015
    emergency_close_threshold: float = 0.09

    # Position sizing / limits
    risk_per_trade_pct: float = 0.003
    max_risk_per_trade_pct: float = 0.007
    max_positions: int = 1

    # Exits
    hard_stop_loss_eur: float = 220.0
    soft_stop_loss_eur: float = 140.0
    trailing_activation_eur: float = 100.0
    trailing_retrace_pct: float = 0.30
    time_decay_hours: float = 4.0

    # Trade limits
    max_trades_per_day: int = 20
    max_trades_per_session: int = 10
    max_consecutive_losses: int = 3

    # Timing policy
    min_minutes_between_entries: int = 5
    min_minutes_after_loss: int = 15

    # Session management
    prime_start: dtime = dtime(14, 0)
    prime_end: dtime = dtime(17, 0)
    no_new_trades_start: dtime = dtime(18, 0)
    no_new_trades_end: dtime = dtime(9, 0)
    hard_close_time: dtime = dtime(22, 0)
    final_exit_window_minutes: int = 60
    allow_weekend_holding: bool = False

    # F-7 FIX: Constraint enforcement flags (controlled by curriculum stages)
    # When False, the corresponding timing restriction is relaxed
    # NOTE: enforce_session_windows was removed (was defined but never used)
    # Session blocking is controlled by: enforce_no_new_trades_window, enforce_hard_close, enforce_weekend_block
    enforce_no_new_trades_window: bool = True  # No new trades after 18:00
    enforce_weekend_block: bool = True  # No weekend holding
    enforce_hard_close: bool = True  # Hard close at 22:00

    # Entry-quality gate
    entry_quality_gate_enabled: bool = True
    entry_quality_threshold: float = 0.35

    # PPO env
    observation_size: int = 64
    max_steps_per_episode: int = 2000
    gamma: float = 0.95

    # Reward configuration (PRIMARY - use this)
    reward: RewardConfig = field(default_factory=RewardConfig)

    # DEPRECATED Legacy reward params - use `reward.xxx` instead
    # These are kept for backward compatibility and will be removed in v2.0
    # ARCH-3 FIX: Added deprecation notice
    reward_scale: float = 10.0  # DEPRECATED: use reward.reward_scale
    risk_penalty_scale: float = 2.0  # DEPRECATED: use reward.dd_penalty_scale
    quality_bonus_scale: float = 0.5  # DEPRECATED: unused
    blocked_action_penalty: float = 0.02  # DEPRECATED: use reward.soft_block_penalty
    hard_block_penalty: float = 0.02  # DEPRECATED: use reward.hard_block_penalty

    # Execution anti-cheat
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    # Domain randomization
    domain_randomization_enabled: bool = True
    spread_mult_range: Tuple[float, float] = (0.90, 1.45)
    slippage_mult_range: Tuple[float, float] = (0.90, 1.60)
    latency_bars_range: Tuple[int, int] = (0, 2)
    volatility_scale_range: Tuple[float, float] = (0.90, 1.20)

    # Discrete action space (size buckets)
    size_buckets: Tuple[float, ...] = (0.35, 0.60, 0.85, 1.10)

    def sync_reward_from_legacy(self) -> None:
        """
        DEPRECATED: Legacy sync from old scalar fields to RewardConfig.
        
        WARNING: This overwrites any curriculum/stage reward overrides.
        Only call at init/load time, NEVER after applying curriculum config.
        Use sync_legacy_from_reward() after curriculum updates instead.
        
        This method will be removed in v2.0. Access reward fields directly:
        - config.reward.reward_scale instead of config.reward_scale
        - config.reward.dd_penalty_scale instead of config.risk_penalty_scale
        """
        import warnings
        warnings.warn(
            "sync_reward_from_legacy() is deprecated. "
            "Use config.reward.xxx fields directly instead of legacy scalars.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.reward.reward_scale = float(self.reward_scale)
        self.reward.dd_penalty_scale = float(self.risk_penalty_scale)
        self.reward.hard_block_penalty = float(self.hard_block_penalty)
        self.reward.soft_block_penalty = float(self.blocked_action_penalty)

    def sync_legacy_from_reward(self) -> None:
        """
        Sync legacy scalar fields FROM RewardConfig (reverse direction).
        
        Call this after applying curriculum reward overrides to keep
        legacy fields consistent for external inspection/serialization.
        """
        self.reward_scale = float(self.reward.reward_scale)
        self.risk_penalty_scale = float(self.reward.dd_penalty_scale)
        self.hard_block_penalty = float(self.reward.hard_block_penalty)
        self.blocked_action_penalty = float(self.reward.soft_block_penalty)

    def __post_init__(self) -> None:
        # AUDIT FIX (HIGH-1): REMOVED automatic sync_reward_from_legacy() call.
        # This was overwriting curriculum reward settings.
        # Legacy fields are now synced FROM RewardConfig via sync_legacy_from_reward()
        # after curriculum applies stage config.
        # 
        # If you need legacy field -> RewardConfig sync, call sync_reward_from_legacy()
        # explicitly BEFORE curriculum configuration.

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

            # AUDIT FIX (HIGH-1): REMOVED sync_reward_from_legacy() call.
            # Curriculum stages now control reward settings exclusively.
        except Exception as e:
            # AUDIT FIX (HIGH-1): Log config sync failures
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
    total_fees: float = 0.0  # Entry + exit fees for consistent MFE/net_pnl comparisons
    entry_dt: Optional[datetime] = None  # Entry timestamp for session timing rewards
    entry_context: Optional[Dict[str, Any]] = None  # Market structure context at entry (v5.3)


class PropFirmTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_dict: Dict[str, Dict[str, pd.DataFrame]],
        config: Optional[PropFirmConfig] = None,
        *,
        curriculum_manager: Optional[CurriculumManager] = None,
        apply_curriculum_overrides: bool = True,
    ):
        super().__init__()
        self.config = config or PropFirmConfig()
        self.tz = ZoneInfo(self.config.tz)

        # Curriculum (Any type at runtime for compatibility)
        self.curriculum: Any = None
        self._apply_curriculum_overrides = bool(apply_curriculum_overrides)
        self._pending_stage_apply: bool = False
        self._last_stage_name: str = ""
        self._last_stage_epoch: int = 0
        self._curriculum_stage_idx: int = 0  # Stage index for stage-specific reward settings
        if curriculum_manager is not None and CURRICULUM_AVAILABLE:
            self.curriculum = curriculum_manager
            self._last_stage_name = getattr(self.curriculum.current_stage, "name", "")
            self._last_stage_epoch = int(getattr(self.curriculum, "current_stage_epoch", 0))
            # Initialize stage index
            current_stage = getattr(self.curriculum, "current_stage", None)
            if current_stage is not None:
                self._curriculum_stage_idx = getattr(current_stage, "value", 0)

        self.data = data_dict
        self.instruments = [i for i in self.config.instruments if i in self.data]
        if not self.instruments:
            self.instruments = list(self.data.keys())[:1]
        if not self.instruments:
            raise ValueError("No valid instruments found in data")

        # Keep obs size consistent with builder contract
        try:
            self.config.observation_size = int(PPO_OBS_SIZE)
        except Exception as e:
            # AUDIT FIX (HIGH-1): Log observation size sync failures
            import logging
            logging.getLogger(__name__).debug(f"Could not sync observation_size: {e}")

        self._K = int(len(self.config.size_buckets))
        self._ACTION_HOLD = 0
        self._ACTION_LONG_START = 1
        self._ACTION_SHORT_START = 1 + self._K
        self._ACTION_CLOSE = 1 + 2 * self._K
        self._N_ACTIONS = 2 * self._K + 2

        self.action_space = spaces.Discrete(self._N_ACTIONS)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(int(self.config.observation_size),), dtype=np.float32
        )

        self.obs_builder: Optional["PPOObservationBuilder"] = None  # type: ignore
        if OBS_BUILDER_AVAILABLE and PPOObservationBuilder is not None:
            self.obs_builder = PPOObservationBuilder()
            logger.info(f"[OBS] Using PPOObservationBuilder v{PPO_OBS_VERSION}")
        else:
            logger.warning("[OBS] PPOObservationBuilder not available -> fallback observation")

        # FIXED: Use primary timeframe length for episode boundaries, not min across all TFs
        # Higher timeframes (D1, H4) are only used for context, not for episode progression
        self._primary_data_len = self._get_primary_data_length()
        self._min_data_len = self._primary_data_len  # Backward compat alias

        # Runtime / account
        self.balance = float(self.config.initial_balance)
        self.equity = float(self.config.initial_balance)
        self.initial_balance = float(self.config.initial_balance)
        self.day_start_balance = float(self.config.initial_balance)
        self.peak_balance = float(self.config.initial_balance)

        # Position & orders
        self.position: Optional[PropPosition] = None
        self.pending_entry: Optional[Dict[str, Any]] = None
        self.pending_exit: Optional[Dict[str, Any]] = None

        # Stats
        self.total_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.max_consecutive_losses_reached = 0  # Peak consecutive losses during episode

        # Session tracking
        self._current_day: Optional[date] = None
        self._current_session_key: Optional[Tuple[date, str]] = None
        self._session_trades = 0

        # Timing
        self._last_entry_dt: Optional[datetime] = None
        self._last_loss_dt: Optional[datetime] = None

        # dt=None fallback timing (bar-based)
        self._last_entry_step: Optional[int] = None
        self._last_loss_step: Optional[int] = None

        # Episode tracking
        self.current_step = 0
        self.episode_step = 0
        self.episode_bars = 0
        self._episode_return = 0.0  # curriculum needs total return, not last-step reward

        # Execution anti-cheat (seeded in reset)
        self._exec: Optional[ExecutionModel] = None
        self._episode_execution_cfg: Optional[ExecutionConfig] = None

        # Per-step caches
        self._ohlcv_cache_key: Optional[Tuple[str, int]] = None
        self._ohlcv_cache: Dict[int, Dict[str, Any]] = {}

        # Quote cache (critical for deterministic intra-step equity/DD)
        self._quote_cache_step: Optional[int] = None
        self._quote_cache_inst: Optional[str] = None
        self._quote_cache_mid: float = 0.0
        self._quote_cache_vol: float = 0.0
        self._quote_cache_bid: float = 0.0
        self._quote_cache_ask: float = 0.0

        # Reward tracking state
        # AUDIT FIX: Removed _opportunity_history and _action_history (dead code - never used)
        self._avg_vol: Optional[float] = None
        self._episode_trade_results: List[TradeResult] = []
        self._last_reward_components: Dict[str, float] = {}
        
        # Aggregate reward component tracking for dashboard
        self._episode_reward_components: Dict[str, float] = {}
        self._episode_reward_component_counts: Dict[str, int] = {}

        # Episode randomization state
        self._episode_spread_mult = 1.0
        self._episode_slip_mult = 1.0
        self._episode_latency_bars = 0
        self._episode_vol_scale = 1.0

        # Data difficulty settings (curriculum-based filtering)
        self._data_difficulty: Optional[Any] = None  # DataDifficulty when set
        self._valid_start_indices: Optional[np.ndarray] = None  # Pre-computed valid episode starts
        self._volatility_percentiles: Optional[np.ndarray] = None  # Rolling volatility percentile per bar

    # ---------------------------
    # Curriculum wiring
    # ---------------------------

    def _apply_overrides_to_object(self, target: Any, overrides: Dict[str, Any]) -> None:
        if not isinstance(overrides, dict):
            return
        for k, v in overrides.items():
            if not isinstance(k, str):
                continue
            # support dot paths: "reward.reward_scale", etc.
            if "." in k:
                head, rest = k.split(".", 1)
                if hasattr(target, head):
                    sub = getattr(target, head)
                    try:
                        self._apply_overrides_to_object(sub, {rest: v})
                    except Exception as e:
                        logger.debug(f"Override {k}={v} failed: {e}")
                continue
            if hasattr(target, k):
                try:
                    setattr(target, k, v)
                except Exception as e:
                    logger.debug(f"Override {k}={v} failed: {e}")

    def _sync_curriculum_stage_overrides(self) -> None:
        """
        Applies overrides from CurriculumStageConfig (when available) into env config.
        This is intentionally defensive: unknown keys are ignored.
        """
        if not (self.curriculum and CURRICULUM_AVAILABLE and self._apply_curriculum_overrides):
            return

        try:
            stage_cfg = getattr(self.curriculum, "stage_config", None)
            if stage_cfg is None:
                return

            # Track current stage index for stage-specific reward calculations
            current_stage = getattr(self.curriculum, "current_stage", None)
            if current_stage is not None:
                # Use stage value as index (0=Foundation, 7=LiveReady)
                self._curriculum_stage_idx = getattr(current_stage, "value", 0)
            else:
                self._curriculum_stage_idx = 0

            # Common patterns:
            # - env_overrides: { "risk_per_trade_pct": 0.001, "domain_randomization_enabled": False, ...}
            # - reward_overrides: { "reward_scale": 6.0, ... }
            # - execution_overrides: { "latency_bars": 1, ... }
            env_overrides = getattr(stage_cfg, "env_overrides", None)
            reward_overrides = getattr(stage_cfg, "reward_overrides", None)
            execution_overrides = getattr(stage_cfg, "execution_overrides", None)

            # Also support a generic "overrides" dict if you use that name
            generic = getattr(stage_cfg, "overrides", None)

            if isinstance(generic, dict):
                self._apply_overrides_to_object(self.config, generic)

            if isinstance(env_overrides, dict):
                self._apply_overrides_to_object(self.config, env_overrides)

            if isinstance(reward_overrides, dict):
                self._apply_overrides_to_object(self.config.reward, reward_overrides)

            if isinstance(execution_overrides, dict):
                self._apply_overrides_to_object(self.config.execution, execution_overrides)

            # Sync legacy fields FROM RewardConfig (not the other way around!)
            # This ensures curriculum reward overrides are NOT overwritten.
            self.config.sync_legacy_from_reward()
        except Exception as e:
            # AUDIT FIX (HIGH-1): Log curriculum override failures
            import logging
            logging.getLogger(__name__).warning(f"_sync_curriculum_stage_overrides() failed: {type(e).__name__}: {e}")
            return

    # ---------------------------
    # Setters for Curriculum Integration
    # ---------------------------
    
    def set_execution_params(self, difficulty: Any) -> None:
        """
        Set execution difficulty parameters from curriculum stage.
        
        Args:
            difficulty: ExecutionDifficulty dataclass or dict with execution params
        """
        if difficulty is None:
            return
        
        try:
            # Map from curriculum ExecutionDifficulty to env's ExecutionConfig
            exec_cfg = self.config.execution
            
            if hasattr(difficulty, "base_spread_points"):
                exec_cfg.base_spread_points = float(difficulty.base_spread_points)
            if hasattr(difficulty, "max_spread_points"):
                exec_cfg.max_spread_points = float(difficulty.max_spread_points)
            if hasattr(difficulty, "slippage_points_sigma"):
                exec_cfg.slippage_points_sigma = float(difficulty.slippage_points_sigma)
            if hasattr(difficulty, "max_slippage_points"):
                exec_cfg.max_slippage_points = float(difficulty.max_slippage_points)
            if hasattr(difficulty, "commission_per_lot"):
                exec_cfg.commission_per_lot = float(difficulty.commission_per_lot)
            if hasattr(difficulty, "latency_bars"):
                exec_cfg.latency_bars = int(difficulty.latency_bars)
            
            # Domain randomization ranges
            if hasattr(difficulty, "spread_mult_range"):
                self.config.spread_mult_range = tuple(difficulty.spread_mult_range)
            if hasattr(difficulty, "slippage_mult_range"):
                self.config.slippage_mult_range = tuple(difficulty.slippage_mult_range)
            if hasattr(difficulty, "latency_randomization_range"):
                self.config.latency_bars_range = tuple(difficulty.latency_randomization_range)
            if hasattr(difficulty, "volatility_scale_range"):
                self.config.volatility_scale_range = tuple(difficulty.volatility_scale_range)
            if hasattr(difficulty, "enable_randomization"):
                self.config.domain_randomization_enabled = bool(difficulty.enable_randomization)
                
        except Exception as e:
            logger.warning(f"Failed to apply execution params: {e}")
    
    def set_constraints(self, constraints: Any) -> None:
        """
        Set trading constraints from curriculum stage.
        
        Args:
            constraints: TradingConstraints dataclass or dict with constraint params
        """
        if constraints is None:
            return
        
        try:
            cfg = self.config
            
            # Position limits
            if hasattr(constraints, "max_positions"):
                cfg.max_positions = int(constraints.max_positions)
            
            # Trade limits
            if hasattr(constraints, "max_trades_per_day"):
                cfg.max_trades_per_day = int(constraints.max_trades_per_day)
            if hasattr(constraints, "max_trades_per_session"):
                cfg.max_trades_per_session = int(constraints.max_trades_per_session)
            if hasattr(constraints, "max_consecutive_losses"):
                cfg.max_consecutive_losses = int(constraints.max_consecutive_losses)
            
            # Timing
            if hasattr(constraints, "min_minutes_between_entries"):
                cfg.min_minutes_between_entries = int(constraints.min_minutes_between_entries)
            if hasattr(constraints, "min_minutes_after_loss"):
                cfg.min_minutes_after_loss = int(constraints.min_minutes_after_loss)
            
            # F-7 FIX: Apply constraint enforcement flags
            # These control whether timing windows are enforced or relaxed
            # NOTE: enforce_session_windows was removed (was defined but never used)
            if hasattr(constraints, "enforce_no_new_trades_window"):
                cfg.enforce_no_new_trades_window = bool(constraints.enforce_no_new_trades_window)
            if hasattr(constraints, "enforce_weekend_block"):
                cfg.enforce_weekend_block = bool(constraints.enforce_weekend_block)
            if hasattr(constraints, "enforce_hard_close"):
                cfg.enforce_hard_close = bool(constraints.enforce_hard_close)
            
            # Drawdown limits
            if hasattr(constraints, "daily_drawdown_limit"):
                cfg.daily_drawdown_limit = float(constraints.daily_drawdown_limit)
            if hasattr(constraints, "max_drawdown_limit"):
                cfg.max_drawdown_limit = float(constraints.max_drawdown_limit)
            if hasattr(constraints, "daily_dd_safety_buffer"):
                cfg.daily_dd_safety_buffer = float(constraints.daily_dd_safety_buffer)
            if hasattr(constraints, "max_dd_safety_buffer"):
                cfg.max_dd_safety_buffer = float(constraints.max_dd_safety_buffer)
            if hasattr(constraints, "emergency_close_threshold"):
                cfg.emergency_close_threshold = float(constraints.emergency_close_threshold)
            
            # Entry quality gate
            if hasattr(constraints, "entry_quality_gate_enabled"):
                cfg.entry_quality_gate_enabled = bool(constraints.entry_quality_gate_enabled)
            if hasattr(constraints, "entry_quality_threshold"):
                cfg.entry_quality_threshold = float(constraints.entry_quality_threshold)
            
            # Stop loss / trailing
            if hasattr(constraints, "hard_stop_loss_eur"):
                cfg.hard_stop_loss_eur = float(constraints.hard_stop_loss_eur)
            if hasattr(constraints, "soft_stop_loss_eur"):
                cfg.soft_stop_loss_eur = float(constraints.soft_stop_loss_eur)
            if hasattr(constraints, "trailing_activation_eur"):
                cfg.trailing_activation_eur = float(constraints.trailing_activation_eur)
            if hasattr(constraints, "trailing_retrace_pct"):
                cfg.trailing_retrace_pct = float(constraints.trailing_retrace_pct)
            if hasattr(constraints, "time_decay_hours"):
                cfg.time_decay_hours = float(constraints.time_decay_hours)
            
            # Risk per trade
            if hasattr(constraints, "risk_per_trade_pct"):
                cfg.risk_per_trade_pct = float(constraints.risk_per_trade_pct)
            if hasattr(constraints, "max_risk_per_trade_pct"):
                cfg.max_risk_per_trade_pct = float(constraints.max_risk_per_trade_pct)
                
        except Exception as e:
            logger.warning(f"Failed to apply constraints: {e}")
    
    def set_reward_config(self, reward_shaping: Any) -> None:
        """
        Set reward configuration from curriculum stage.
        
        CRITICAL: This syncs the curriculum's RewardShaping to the env's RewardConfig.
        This must be called when curriculum stage changes to ensure proper reward signals.
        
        Args:
            reward_shaping: RewardShaping dataclass from curriculum config
        """
        if reward_shaping is None:
            return
        
        try:
            rcfg = self.config.reward
            
            # Core reward scaling - CRITICAL for proper learning signal
            if hasattr(reward_shaping, "reward_scale"):
                rcfg.reward_scale = float(reward_shaping.reward_scale)
                self.config.reward_scale = float(reward_shaping.reward_scale)  # Legacy sync
            if hasattr(reward_shaping, "loss_multiplier"):
                rcfg.loss_multiplier = float(reward_shaping.loss_multiplier)
            
            # R-multiple bonuses
            if hasattr(reward_shaping, "r_multiple_bonus_threshold"):
                rcfg.r_multiple_bonus_threshold = float(reward_shaping.r_multiple_bonus_threshold)
            if hasattr(reward_shaping, "r_multiple_bonus_scale"):
                rcfg.r_multiple_bonus_scale = float(reward_shaping.r_multiple_bonus_scale)
            if hasattr(reward_shaping, "r_multiple_bonus_cap"):
                rcfg.r_multiple_bonus_cap = float(reward_shaping.r_multiple_bonus_cap)
            
            # MAE efficiency
            if hasattr(reward_shaping, "mae_efficiency_enabled"):
                rcfg.mae_efficiency_enabled = bool(reward_shaping.mae_efficiency_enabled)
            if hasattr(reward_shaping, "mae_efficiency_scale"):
                rcfg.mae_efficiency_scale = float(reward_shaping.mae_efficiency_scale)
            if hasattr(reward_shaping, "mae_efficiency_threshold"):
                rcfg.mae_efficiency_threshold = float(reward_shaping.mae_efficiency_threshold)
            
            # Time efficiency
            if hasattr(reward_shaping, "time_efficiency_enabled"):
                rcfg.time_efficiency_enabled = bool(reward_shaping.time_efficiency_enabled)
            if hasattr(reward_shaping, "time_efficiency_scale"):
                rcfg.time_efficiency_scale = float(reward_shaping.time_efficiency_scale)
            if hasattr(reward_shaping, "optimal_trade_bars"):
                rcfg.optimal_trade_bars = int(reward_shaping.optimal_trade_bars)
            if hasattr(reward_shaping, "max_trade_bars_for_bonus"):
                rcfg.max_trade_bars_for_bonus = int(reward_shaping.max_trade_bars_for_bonus)
            
            # Exit quality
            if hasattr(reward_shaping, "exit_quality_enabled"):
                rcfg.exit_quality_enabled = bool(reward_shaping.exit_quality_enabled)
            if hasattr(reward_shaping, "trailing_stop_bonus"):
                rcfg.trailing_stop_bonus = float(reward_shaping.trailing_stop_bonus)
            if hasattr(reward_shaping, "agent_close_bonus"):
                rcfg.agent_close_bonus = float(reward_shaping.agent_close_bonus)
            if hasattr(reward_shaping, "hard_stop_penalty"):
                rcfg.hard_stop_penalty = float(reward_shaping.hard_stop_penalty)
            if hasattr(reward_shaping, "risk_liquidation_penalty"):
                rcfg.risk_liquidation_penalty = float(reward_shaping.risk_liquidation_penalty)
            
            # Entry quality
            if hasattr(reward_shaping, "entry_quality_integration"):
                rcfg.entry_quality_integration = bool(reward_shaping.entry_quality_integration)
            if hasattr(reward_shaping, "entry_quality_weight"):
                rcfg.entry_quality_weight = float(reward_shaping.entry_quality_weight)
            
            # Session timing (teaching trading hours)
            if hasattr(reward_shaping, "session_timing_enabled"):
                rcfg.session_timing_enabled = bool(reward_shaping.session_timing_enabled)
            if hasattr(reward_shaping, "off_hours_trade_penalty"):
                rcfg.off_hours_trade_penalty = float(reward_shaping.off_hours_trade_penalty)
            if hasattr(reward_shaping, "prime_hours_trade_bonus"):
                rcfg.prime_hours_trade_bonus = float(reward_shaping.prime_hours_trade_bonus)
            
            # Market structure rewards (v5.3)
            if hasattr(reward_shaping, "market_structure_enabled"):
                rcfg.market_structure_enabled = bool(reward_shaping.market_structure_enabled)
            if hasattr(reward_shaping, "sr_proximity_bonus"):
                rcfg.sr_proximity_bonus = float(reward_shaping.sr_proximity_bonus)
            if hasattr(reward_shaping, "sr_proximity_penalty"):
                rcfg.sr_proximity_penalty = float(reward_shaping.sr_proximity_penalty)
            if hasattr(reward_shaping, "structure_alignment_bonus"):
                rcfg.structure_alignment_bonus = float(reward_shaping.structure_alignment_bonus)
            if hasattr(reward_shaping, "bos_alignment_bonus"):
                rcfg.bos_alignment_bonus = float(reward_shaping.bos_alignment_bonus)
            if hasattr(reward_shaping, "order_block_entry_bonus"):
                rcfg.order_block_entry_bonus = float(reward_shaping.order_block_entry_bonus)
            
            # Divergence awareness (v5.3)
            if hasattr(reward_shaping, "divergence_awareness_enabled"):
                rcfg.divergence_awareness_enabled = bool(reward_shaping.divergence_awareness_enabled)
            if hasattr(reward_shaping, "divergence_contra_penalty"):
                rcfg.divergence_contra_penalty = float(reward_shaping.divergence_contra_penalty)
            if hasattr(reward_shaping, "divergence_aligned_bonus"):
                rcfg.divergence_aligned_bonus = float(reward_shaping.divergence_aligned_bonus)
            if hasattr(reward_shaping, "overbought_long_penalty"):
                rcfg.overbought_long_penalty = float(reward_shaping.overbought_long_penalty)
            if hasattr(reward_shaping, "oversold_short_penalty"):
                rcfg.oversold_short_penalty = float(reward_shaping.oversold_short_penalty)
            
            # Regime awareness (v5.3)
            if hasattr(reward_shaping, "regime_awareness_enabled"):
                rcfg.regime_awareness_enabled = bool(reward_shaping.regime_awareness_enabled)
            if hasattr(reward_shaping, "risk_off_aggressive_penalty"):
                rcfg.risk_off_aggressive_penalty = float(reward_shaping.risk_off_aggressive_penalty)
            if hasattr(reward_shaping, "high_vol_size_penalty"):
                rcfg.high_vol_size_penalty = float(reward_shaping.high_vol_size_penalty)
            
            # Truncation handling
            if hasattr(reward_shaping, "truncation_winner_discount"):
                rcfg.truncation_winner_discount = float(reward_shaping.truncation_winner_discount)
            if hasattr(reward_shaping, "truncation_loser_extra_penalty"):
                rcfg.truncation_loser_extra_penalty = float(reward_shaping.truncation_loser_extra_penalty)
            
            # Drawdown shaping
            if hasattr(reward_shaping, "dd_shaping_enabled"):
                rcfg.dd_shaping_enabled = bool(reward_shaping.dd_shaping_enabled)
            if hasattr(reward_shaping, "dd_threshold"):
                rcfg.dd_threshold = float(reward_shaping.dd_threshold)
            if hasattr(reward_shaping, "dd_penalty_scale"):
                rcfg.dd_penalty_scale = float(reward_shaping.dd_penalty_scale)
            if hasattr(reward_shaping, "dd_severity_exponent"):
                rcfg.dd_severity_exponent = float(reward_shaping.dd_severity_exponent)
            if hasattr(reward_shaping, "dd_severity_cap"):
                rcfg.dd_severity_cap = float(reward_shaping.dd_severity_cap)
            
            # Streak modifiers
            if hasattr(reward_shaping, "streak_modifier_enabled"):
                rcfg.streak_modifier_enabled = bool(reward_shaping.streak_modifier_enabled)
            if hasattr(reward_shaping, "win_streak_bonus_per_win"):
                rcfg.win_streak_bonus_per_win = float(reward_shaping.win_streak_bonus_per_win)
            if hasattr(reward_shaping, "loss_streak_penalty_per_loss"):
                rcfg.loss_streak_penalty_per_loss = float(reward_shaping.loss_streak_penalty_per_loss)
            
            # Anti-churn
            if hasattr(reward_shaping, "anti_churn_enabled"):
                rcfg.anti_churn_enabled = bool(reward_shaping.anti_churn_enabled)
            if hasattr(reward_shaping, "daily_trade_soft_limit"):
                rcfg.daily_trade_soft_limit = int(reward_shaping.daily_trade_soft_limit)
            if hasattr(reward_shaping, "churn_penalty_per_trade"):
                rcfg.churn_penalty_per_trade = float(reward_shaping.churn_penalty_per_trade)
            
            # Block penalties
            if hasattr(reward_shaping, "hard_block_penalty"):
                rcfg.hard_block_penalty = float(reward_shaping.hard_block_penalty)
            if hasattr(reward_shaping, "soft_block_penalty"):
                rcfg.soft_block_penalty = float(reward_shaping.soft_block_penalty)
            
            # Per-step shaping
            if hasattr(reward_shaping, "per_step_shaping_enabled"):
                rcfg.per_step_shaping_enabled = bool(reward_shaping.per_step_shaping_enabled)
            if hasattr(reward_shaping, "holding_cost_per_bar"):
                rcfg.holding_cost_per_bar = float(reward_shaping.holding_cost_per_bar)
            
            # Reward clipping
            if hasattr(reward_shaping, "min_reward"):
                rcfg.min_reward = float(reward_shaping.min_reward)
            if hasattr(reward_shaping, "max_reward"):
                rcfg.max_reward = float(reward_shaping.max_reward)
            
            # Sync legacy fields FROM RewardConfig (not the reverse!)
            # This ensures curriculum reward overrides are preserved.
            self.config.sync_legacy_from_reward()
            
            logger.debug(f"Applied reward config: scale={rcfg.reward_scale}, loss_mult={rcfg.loss_multiplier}")
            
        except Exception as e:
            logger.warning(f"Failed to apply reward config: {e}")

    # ---------------------------
    # Data Difficulty (Curriculum)
    # ---------------------------

    def set_data_difficulty(self, difficulty: Any) -> None:
        """
        Set data difficulty filtering for curriculum-based learning.
        
        Args:
            difficulty: DataDifficulty config specifying which market conditions to train on.
                        Early stages use easier conditions (clear trends, lower volatility).
        """
        self._data_difficulty = difficulty
        self._valid_start_indices = None  # Force recomputation
        self._volatility_percentiles = None
        self._difficulty_cache_hash: Optional[int] = None
        
        if difficulty is not None:
            self._precompute_data_difficulty_indices()

    def _precompute_data_difficulty_indices(self) -> None:
        """
        Pre-compute valid episode starting indices based on data difficulty settings.
        
        This avoids expensive per-reset filtering by caching valid positions.
        Uses hash-based caching to skip redundant computation when settings unchanged.
        """
        # Compute config hash to detect if recomputation needed
        if self._data_difficulty is not None:
            d = self._data_difficulty
            config_hash = hash((
                getattr(d, 'volatility_percentile_range', (0.0, 1.0)),
                getattr(d, 'min_trend_clarity', 0.0),
                getattr(d, 'include_asian_session', True),
                getattr(d, 'include_london_session', True),
                getattr(d, 'include_ny_session', True),
                getattr(d, 'avoid_session_boundaries', False),
            ))
            
            # Skip if already computed with same settings
            if self._difficulty_cache_hash == config_hash and self._valid_start_indices is not None:
                return
            
            self._difficulty_cache_hash = config_hash
        if self._data_difficulty is None:
            self._valid_start_indices = None
            return

        difficulty = self._data_difficulty
        inst = self.instruments[0]
        primary_tf = self.config.primary_timeframe
        df = self.data.get(inst, {}).get(primary_tf)
        
        if df is None or len(df) < 200:
            self._valid_start_indices = None
            return

        n_bars = len(df)
        buffer = 120
        max_end = n_bars - self.config.max_steps_per_episode - buffer
        
        if max_end <= buffer:
            self._valid_start_indices = None
            return

        # Initialize all indices as valid
        valid_mask = np.ones(n_bars, dtype=bool)

        # Apply volatility filter
        vol_range = getattr(difficulty, "volatility_percentile_range", (0.0, 1.0))
        if vol_range != (0.0, 1.0):
            self._compute_volatility_percentiles(df)
            if self._volatility_percentiles is not None:
                valid_mask &= (self._volatility_percentiles >= vol_range[0])
                valid_mask &= (self._volatility_percentiles <= vol_range[1])

        # Apply trend clarity filter
        min_trend = getattr(difficulty, "min_trend_clarity", 0.0)
        if min_trend > 0.0:
            trend_clarity = self._compute_trend_clarity(df)
            valid_mask &= (trend_clarity >= min_trend)

        # Apply session filters
        if isinstance(df.index, pd.DatetimeIndex):
            try:
                hours = np.asarray(df.index.hour, dtype=np.int32)
                if hours is not None:
                    session_mask = np.zeros(n_bars, dtype=bool)
                    
                    # Session hours (approximate, Europe/Berlin perspective)
                    # Asian: 00:00 - 08:00
                    # London: 08:00 - 16:00
                    # NY: 14:00 - 22:00
                    # Overlap (London/NY): 14:00 - 16:00
                    
                    if getattr(difficulty, "include_asian_session", True):
                        session_mask |= (hours < 8)
                    if getattr(difficulty, "include_london_session", True):
                        session_mask |= ((hours >= 8) & (hours < 16))
                    if getattr(difficulty, "include_ny_session", True):
                        session_mask |= ((hours >= 14) & (hours < 22))
                    if getattr(difficulty, "include_overlap_sessions", True):
                        session_mask |= ((hours >= 14) & (hours < 16))
                    
                    # If at least one session enabled, apply filter
                    if session_mask.any():
                        valid_mask &= session_mask
            except Exception as e:
                logger.debug(f"Skip session filtering: {e}")

        # Apply market open/close filter
        if getattr(difficulty, "exclude_market_open_close", False):
            try:
                if isinstance(df.index, pd.DatetimeIndex):
                    hours = np.asarray(df.index.hour, dtype=np.int32)
                    # Exclude first/last hour of major sessions
                    open_close_mask = ~(
                        (hours == 0) | (hours == 8) | (hours == 14) |  # Opens
                        (hours == 7) | (hours == 15) | (hours == 21)   # Closes
                    )
                    valid_mask &= open_close_mask
            except Exception as e:
                logger.debug(f"Skip market open/close filter: {e}")

        # Restrict to valid start range
        range_mask = np.zeros(n_bars, dtype=bool)
        range_mask[buffer:max_end] = True
        valid_mask &= range_mask

        # Get valid indices
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            # Fallback: use all indices in valid range
            logger.warning(
                f"DataDifficulty filter found 0 valid indices with settings: "
                f"volatility_range={self._data_difficulty.volatility_percentile_range}, "
                f"min_trend_clarity={self._data_difficulty.min_trend_clarity}, "
                f"sessions=(asia={getattr(self._data_difficulty, 'include_asian_session', True)}, "
                f"london={getattr(self._data_difficulty, 'include_london_session', True)}, "
                f"ny={getattr(self._data_difficulty, 'include_ny_session', True)}). "
                f"Falling back to full dataset ({max_end - buffer} bars)."
            )
            self._valid_start_indices = np.arange(buffer, max_end)
        else:
            self._valid_start_indices = valid_indices
            logger.debug(f"DataDifficulty filter: {len(valid_indices)} valid start indices out of {max_end - buffer}")

    def _resolve_column(self, df: pd.DataFrame, col_lower: str) -> str:
        """
        Resolve column name to handle both lowercase and uppercase OHLC columns.
        
        Checks for lowercase first (more common), then titlecase/uppercase.
        Returns the resolved column name or raises KeyError if not found.
        """
        if col_lower in df.columns:
            return col_lower
        col_title = col_lower.capitalize()  # e.g., "close" -> "Close"
        if col_title in df.columns:
            return col_title
        col_upper = col_lower.upper()  # e.g., "close" -> "CLOSE"
        if col_upper in df.columns:
            return col_upper
        raise KeyError(f"Column '{col_lower}' not found in any case variant (tried: {col_lower}, {col_title}, {col_upper})")

    def _compute_volatility_percentiles(self, df: pd.DataFrame) -> None:
        """
        Compute volatility percentile for each bar.
        
        Uses O(n log n) rank-based algorithm instead of O(n²) expanding window.
        For data difficulty filtering, global percentile ranking is statistically
        equivalent and much faster than exact expanding window percentile.
        """
        try:
            if "close" not in df.columns and "Close" not in df.columns:
                self._volatility_percentiles = None
                return
            
            close_col = "close" if "close" in df.columns else "Close"
            close = np.asarray(df[close_col].values, dtype=np.float64)
            
            # Rolling ATR-like volatility (20-bar)
            window = 20
            if len(close) < window + 1:
                self._volatility_percentiles = None
                return
            
            returns = np.abs(np.diff(close) / (close[:-1] + 1e-10))
            
            # HIGH-1 FIX: Use pandas rolling + rank instead of manual loop + scipy.rankdata
            # This removes the SciPy dependency that can crash training boxes without scipy
            vol_series = pd.Series(returns).rolling(window=window, min_periods=1).std().fillna(0.0)
            vol = np.concatenate([[0.0], vol_series.to_numpy(dtype=np.float64)])
            
            # Percentile ranks in [0,1] using pandas (no SciPy required)
            percentiles = pd.Series(vol).rank(pct=True, method="average").to_numpy(dtype=np.float64)
            
            # First bar gets default 0.5 (median assumption for unknown)
            if len(percentiles) > 0:
                percentiles[0] = 0.5
            
            self._volatility_percentiles = percentiles
        except Exception:
            self._volatility_percentiles = None

    def _compute_trend_clarity(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute trend clarity for each bar.
        
        Uses a simple measure: abs(SMA slope) normalized by volatility.
        High values = clear trend, low values = choppy/ranging.
        """
        try:
            close_col = "close" if "close" in df.columns else "Close"
            if close_col not in df.columns:
                return np.ones(len(df))  # Default: all clear
            
            close = np.asarray(df[close_col].values, dtype=np.float64)
            n = len(close)
            clarity = np.zeros(n)
            
            window = 20
            for i in range(window, n):
                segment = close[i-window:i]
                seg_mean = float(np.mean(segment))
                slope = (segment[-1] - segment[0]) / (window * (seg_mean + 1e-10))
                vol = float(np.std(np.diff(segment))) / (seg_mean + 1e-10)
                
                # Clarity = trend strength / noise
                clarity[i] = min(1.0, abs(slope) / (vol + 1e-10))
            
            # First bars get median clarity
            clarity[:window] = np.median(clarity[window:]) if n > window else 0.5
            
            return clarity
        except Exception:
            return np.ones(len(df))

    def _sample_episode_start_with_difficulty(self, buffer: int, max_start: int) -> int:
        """
        Sample episode starting position respecting data difficulty settings.
        
        Args:
            buffer: Minimum starting index (lookback buffer)
            max_start: Maximum starting index
            
        Returns:
            Starting bar index for this episode
        """
        if self._valid_start_indices is None or len(self._valid_start_indices) == 0:
            # No difficulty filtering - use uniform random
            if max_start > buffer:
                return int(self.np_random.integers(buffer, max_start))
            return min(buffer, max(self._min_data_len - 2, 0))

        # Filter to valid range
        valid_in_range = self._valid_start_indices[
            (self._valid_start_indices >= buffer) & 
            (self._valid_start_indices < max_start)
        ]
        
        if len(valid_in_range) == 0:
            # Fallback to any valid index
            if len(self._valid_start_indices) > 0:
                return int(self.np_random.choice(self._valid_start_indices))
            if max_start > buffer:
                return int(self.np_random.integers(buffer, max_start))
            return min(buffer, max(self._min_data_len - 2, 0))

        # Apply recency weighting if configured
        if (self._data_difficulty is not None and 
            getattr(self._data_difficulty, "prefer_recent_data", False)):
            
            weight = getattr(self._data_difficulty, "recent_data_weight", 1.0)
            if weight > 1.0:
                # Exponential weighting toward recent data
                positions = np.arange(len(valid_in_range))
                weights = np.exp(weight * positions / len(positions))
                weights /= weights.sum()
                idx = self.np_random.choice(len(valid_in_range), p=weights)
                return int(valid_in_range[idx])

        # Uniform random from valid indices
        return int(self.np_random.choice(valid_in_range))

    def _curriculum_step_metadata(self) -> Dict[str, Any]:
        if not self.curriculum:
            return {}
        try:
            return {
                "stage": getattr(self.curriculum.current_stage, "name", ""),
                "stage_epoch": int(getattr(self.curriculum, "current_stage_epoch", 0)),
            }
        except Exception:
            return {}

    def _curriculum_on_episode_end(self, info: Dict[str, Any]) -> None:
        """
        Augments episode info with curriculum metadata.
        
        AUDIT FIX (CRIT-3): Episode recording is now ONLY done by CurriculumEnvWrapper.
        This method only adds metadata - it does NOT record episodes to the manager.
        This prevents double episode counting that was causing 2x promotion speed.
        
        The wrapper calls manager.record_episode_from_info() in its _on_episode_end().
        """
        if not self.curriculum:
            return

        try:
            # Ensure the schema curriculum manager expects exists
            if "episode_stats" not in info:
                info["episode_stats"] = self.get_episode_stats()

            # REMOVED: Episode recording moved to CurriculumEnvWrapper
            # The wrapper is the single source of truth for episode recording.
            # self.curriculum.record_episode_from_info(...)  # REMOVED - causes double counting
            # changed, new_stage = self.curriculum.update()  # REMOVED - wrapper handles this

            # Only add metadata for dashboard (no recording/transition logic)
            info["curriculum"] = self.curriculum.get_progress_report()
            
            # Track if stage changed (wrapper will handle actual transition)
            old_stage = getattr(self.curriculum.current_stage, "name", "")
            old_epoch = int(getattr(self.curriculum, "current_stage_epoch", 0))
            info["curriculum_metadata"] = {
                "stage": old_stage,
                "epoch": old_epoch,
            }
        except Exception:
            return

    # ---------------------------
    # Action decoding / masking
    # ---------------------------

    def _get_min_data_length(self) -> int:
        """
        DEPRECATED: Get minimum data length across ALL timeframes.
        
        WARNING: This method is DEPRECATED and should NOT be used.
        It returns the minimum bar count across ALL timeframes, which means
        if you have D1 (252 bars) and M15 (50,000 bars), it returns 252.
        This causes 99.5% of training data to be UNUSED.
        
        Use _get_primary_data_length() instead, which correctly returns
        the primary timeframe bar count.
        
        AUDIT FIX (CRIT-2): This method now emits DeprecationWarning.
        """
        import warnings
        warnings.warn(
            "_get_min_data_length() is DEPRECATED. Use _get_primary_data_length() instead. "
            "This method returns min across ALL TFs, which can cause 99%+ of data to be unused.",
            DeprecationWarning,
            stacklevel=2
        )
        m = float("inf")
        for inst in self.instruments:
            for _, df in self.data.get(inst, {}).items():
                m = min(m, len(df))
        return int(m) if m != float("inf") else 0

    def _get_primary_data_length(self) -> int:
        """
        Get data length for PRIMARY timeframe only.
        
        CRITICAL: Episode boundaries should be based on the primary trading timeframe,
        not the minimum across all timeframes. Higher TFs (D1, H4) are only used for
        context features and have far fewer bars - using their length would cause
        99%+ of episodes to terminate instantly when the episode start index
        (sampled from primary TF range) exceeds the higher TF bar count.
        """
        inst = self.instruments[0] if self.instruments else None
        if not inst:
            return self._get_min_data_length()
        
        primary_tf = self.config.primary_timeframe
        df = self.data.get(inst, {}).get(primary_tf)
        
        if df is not None and len(df) > 0:
            return len(df)
        
        # Fallback: use min across all TFs for this instrument
        inst_data = self.data.get(inst, {})
        if isinstance(inst_data, dict) and inst_data:
            return min(len(df) for df in inst_data.values())
        
        return self._get_min_data_length()

    def _decode_action(self, action_id: int) -> Tuple[str, float]:
        """
        Returns (intent, size_mult)
          intent in {"hold","close","long","short"}
          size_mult scales risk_per_trade_pct within safe bounds.
        """
        a = int(action_id)
        if a == self._ACTION_HOLD:
            return "hold", 0.0
        if a == self._ACTION_CLOSE:
            return "close", 0.0
        if self._ACTION_LONG_START <= a < self._ACTION_LONG_START + self._K:
            i = a - self._ACTION_LONG_START
            return "long", float(self.config.size_buckets[i])
        if self._ACTION_SHORT_START <= a < self._ACTION_SHORT_START + self._K:
            i = a - self._ACTION_SHORT_START
            return "short", float(self.config.size_buckets[i])
        return "hold", 0.0

    # ---------------------------
    # Execution timing helpers (Bug fixes from audit)
    # ---------------------------

    def _exec_latency(self) -> int:
        """Get execution latency in bars from episode config."""
        return int(getattr(self._episode_execution_cfg, "latency_bars", 0) or 0)

    def _can_fill_before_end(self, *, step_idx: int, latency: int, require_bars_after_fill: int) -> bool:
        """
        Check if an order can fill before episode end.
        
        Args:
            step_idx: Current step index
            latency: Latency in bars (fill happens at step_idx + latency)
            require_bars_after_fill: Minimum bars required after fill (e.g., 1 for entries)
        
        Returns:
            True if fill can complete with required post-fill bars
        """
        last_idx = int(self._min_data_len - 1)
        remaining = last_idx - int(step_idx)
        # Fill happens at step_idx + latency; ensure at least require_bars_after_fill bars after that
        return remaining >= int(latency + require_bars_after_fill)

    def _get_close_priority(self, reason: str) -> int:
        """Get close priority for a reason string."""
        try:
            return CloseReason(reason).close_priority
        except ValueError:
            return 40  # Default middle priority

    def _set_or_override_pending_exit(self, *, reason: str, fill_step: int) -> None:
        """
        Set pending exit, or override existing one if new reason has higher priority.
        
        Bug fix: Ensures forced closes (HARD_STOP, etc.) always override agent closes.
        Also takes the earlier fill_step when overriding.
        """
        if self.pending_exit is None:
            self.pending_exit = {"fill_step": int(fill_step), "reason": str(reason)}
            return

        old_reason = str(self.pending_exit.get("reason", CloseReason.AGENT_CLOSE.value))
        new_priority = self._get_close_priority(reason)
        old_priority = self._get_close_priority(old_reason)
        
        if new_priority >= old_priority:
            # Override with higher priority reason, use earlier fill_step
            self.pending_exit["reason"] = str(reason)
            self.pending_exit["fill_step"] = min(
                int(self.pending_exit.get("fill_step", fill_step)), 
                int(fill_step)
            )

    def action_masks(self) -> np.ndarray:
        """
        Action mask for sb3-contrib MaskablePPO.

        RELAXED for training: Only mask actions that are physically impossible.
        Soft rules are penalized in step() to allow learning.
        """
        mask = np.ones(self._N_ACTIONS, dtype=np.bool_)

        inst = self.instruments[0]
        next_step = int(self.current_step + 1)

        # BUG FIX: fill_delay should be latency only, not 1 + latency
        # The +1 was causing a hidden 2-bar delay because step() already advances time
        latency = max(0, self._exec_latency())

        last_idx = (self._min_data_len - 1)
        remaining_bars_next = last_idx - next_step
        
        # Entry requires at least 1 bar AFTER fill so position isn't instantly truncated
        can_enter_fill = remaining_bars_next >= (latency + 1)
        # Exit can fill right at the end (no post-fill bar required)
        can_exit_fill = remaining_bars_next >= latency

        has_position_or_pending = (self.position is not None) or (self.pending_entry is not None)
        can_enter = (not has_position_or_pending) and bool(can_enter_fill)

        if not can_enter:
            mask[self._ACTION_LONG_START: self._ACTION_LONG_START + self._K] = False
            mask[self._ACTION_SHORT_START: self._ACTION_SHORT_START + self._K] = False

        if self.position is None and self.pending_exit is None:
            mask[self._ACTION_CLOSE] = False
        elif self.pending_exit is not None:
            mask[self._ACTION_CLOSE] = True
        else:
            mask[self._ACTION_CLOSE] = bool(can_exit_fill)

        mask[self._ACTION_HOLD] = True
        return mask

    def get_action_mask(self) -> np.ndarray:
        """Backward-compatible alias."""
        return self.action_masks()

    # ---------------------------
    # Time helpers (Europe/Berlin)
    # ---------------------------

    def _tf_minutes(self) -> int:
        """Get minutes per bar for primary timeframe. MED-2 FIX: Use shared constants."""
        tf = (self.config.primary_timeframe or DEFAULT_PRIMARY_TIMEFRAME).upper().strip()
        return timeframe_to_minutes(tf)

    def _get_bar_dt(self, instrument: str) -> Optional[datetime]:
        return self._get_bar_dt_at(instrument, self.current_step)

    def _get_bar_dt_at(self, instrument: str, step_idx: int) -> Optional[datetime]:
        tf = self.config.primary_timeframe
        df = self.data.get(instrument, {}).get(tf)
        if df is None or df.empty:
            return None

        idx = int(np.clip(step_idx, 0, len(df) - 1))

        ts: Any = None
        if isinstance(df.index, pd.DatetimeIndex):
            ts = df.index[idx]
        else:
            for col in ("time", "timestamp", "datetime", "date"):
                if col in df.columns:
                    ts = df[col].iloc[idx]
                    break

        if ts is None:
            return None

        t = pd.to_datetime(ts, errors="coerce")
        if pd.isna(t):
            return None

        if getattr(t, "tzinfo", None) is None:
            t = t.tz_localize(self.tz)
        else:
            t = t.tz_convert(self.tz)

        return t.to_pydatetime()

    def _is_weekend(self, dt: datetime) -> bool:
        return dt.weekday() >= 5

    def _in_no_new_trades_window(self, dt: datetime) -> bool:
        start = self.config.no_new_trades_start
        end = self.config.no_new_trades_end
        # If start == end, the window is disabled (no blocked time)
        if start == end:
            return False
        t = dt.timetz().replace(tzinfo=None)
        return (t >= start) or (t < end)

    def _in_prime_window(self, dt: datetime) -> bool:
        t = dt.timetz().replace(tzinfo=None)
        return self.config.prime_start <= t < self.config.prime_end

    def _in_final_exit_window(self, dt: datetime) -> bool:
        hc = datetime.combine(dt.date(), self.config.hard_close_time, tzinfo=self.tz)
        start = hc - timedelta(minutes=int(self.config.final_exit_window_minutes))
        return start <= dt < hc

    def _at_or_after_hard_close(self, dt: datetime) -> bool:
        hc = datetime.combine(dt.date(), self.config.hard_close_time, tzinfo=self.tz)
        return dt >= hc

    def _session_name(self, dt: datetime) -> str:
        if self._in_prime_window(dt):
            return "prime"
        if self._in_no_new_trades_window(dt):
            return "no_new_trades"
        return "regular"

    def _session_anchor_date(self, dt: datetime) -> date:
        """
        For windows that cross midnight (e.g. 18:00 -> 09:00),
        anchor the after-midnight portion to the previous calendar date so that
        max_trades_per_session cannot reset at midnight.
        """
        t = dt.timetz().replace(tzinfo=None)
        start = self.config.no_new_trades_start
        end = self.config.no_new_trades_end
        crosses = start > end
        if crosses and (t < end):
            return (dt - timedelta(days=1)).date()
        return dt.date()

    def _bars_per_day(self) -> int:
        # MED-2 FIX: Use shared_utils instead of duplicate mapping
        from envs.shared_utils import bars_per_day_for_timeframe
        tf = self.config.primary_timeframe or "M15"
        return bars_per_day_for_timeframe(tf)

    def _maybe_roll_day_session(self, dt: Optional[datetime]) -> None:
        if dt is None:
            bpd = max(1, self._bars_per_day())
            day_idx = int(self.current_step // bpd)
            cur_day = (datetime(2000, 1, 1, tzinfo=self.tz) + timedelta(days=day_idx)).date()

            session = "regular"
            key = (cur_day, f"bar_day_{day_idx}_{session}")

            if self._current_session_key != key:
                self._current_session_key = key
                self._session_trades = 0

            if self._current_day != cur_day:
                self._current_day = cur_day
                # CRIT FIX: Use equity (includes unrealized PnL), not balance (cash only)
                # This prevents daily DD "reset" exploit when carrying losing positions over midnight
                self.day_start_balance = self.equity
                self.daily_trades = 0
                self.daily_pnl = 0.0
            return

        cur_day = dt.date()
        if self._current_day != cur_day:
            self._current_day = cur_day
            # CRIT FIX: Use equity (includes unrealized PnL), not balance (cash only)
            self.day_start_balance = self.equity
            self.daily_trades = 0
            self.daily_pnl = 0.0

        sname = self._session_name(dt)
        anchor = self._session_anchor_date(dt)
        key = (anchor, sname)
        if self._current_session_key != key:
            self._current_session_key = key
            self._session_trades = 0

    # ---------------------------
    # Market helpers
    # ---------------------------

    def _get_price_mid(self, instrument: str) -> float:
        tf = self.config.primary_timeframe
        df = self.data.get(instrument, {}).get(tf)
        if df is None or df.empty:
            return 0.0
        idx = int(np.clip(self.current_step, 0, len(df) - 1))
        try:
            close_col = self._resolve_column(df, "close")
            return float(df[close_col].iloc[idx])
        except KeyError:
            return 0.0

    def _get_ohlcv(self, instrument: str, lookback: int = 120) -> Dict[str, Any]:
        key = (instrument, int(self.current_step))
        if self._ohlcv_cache_key != key:
            self._ohlcv_cache_key = key
            self._ohlcv_cache = {}

        lb = int(max(1, lookback))
        if lb in self._ohlcv_cache:
            return self._ohlcv_cache[lb]

        tf = self.config.primary_timeframe
        df = self.data.get(instrument, {}).get(tf)
        if df is None or df.empty:
            self._ohlcv_cache[lb] = {}
            return {}

        end = min(self.current_step + 1, len(df))
        start = max(0, end - lb)

        # Resolve column names to handle both lowercase and uppercase variants
        try:
            open_col = self._resolve_column(df, "open")
            high_col = self._resolve_column(df, "high")
            low_col = self._resolve_column(df, "low")
            close_col = self._resolve_column(df, "close")
        except KeyError as e:
            logger.warning(f"Missing OHLC column: {e}")
            self._ohlcv_cache[lb] = {}
            return {}

        out = {
            "open": df[open_col].iloc[start:end].to_numpy(dtype=np.float64, copy=False),
            "high": df[high_col].iloc[start:end].to_numpy(dtype=np.float64, copy=False),
            "low": df[low_col].iloc[start:end].to_numpy(dtype=np.float64, copy=False),
            "close": df[close_col].iloc[start:end].to_numpy(dtype=np.float64, copy=False),
        }
        # Check volume with case variants
        vol_col = None
        for v in ["volume", "Volume", "VOLUME"]:
            if v in df.columns:
                vol_col = v
                break
        if vol_col:
            out["volume"] = df[vol_col].iloc[start:end].to_numpy(dtype=np.float64, copy=False)
        else:
            out["volume"] = np.ones(end - start, dtype=np.float64)

        self._ohlcv_cache[lb] = out
        return out

    def _get_pip_value(self, instrument: str) -> Tuple[float, float]:
        inst = instrument.upper().replace("_", "").replace("/", "")
        if "XAU" in inst or "GOLD" in inst:
            return 100.0, 1.0
        if "XAG" in inst or "SILVER" in inst:
            return 50.0, 1.0
        return 10.0, 10000.0

    def _atr_vol_proxy(self, instrument: str) -> float:
        o = self._get_ohlcv(instrument, lookback=40)
        if not o or len(o.get("close", [])) < 20:
            return 0.3
        close = np.asarray(o["close"], dtype=np.float64)
        returns = np.diff(close[-20:]) / np.maximum(close[-20:-1], 1e-8)
        vol = float(np.std(returns))
        vol = float(np.clip(vol * 100.0, 0.0, 1.0))
        return float(np.clip(vol * self._episode_vol_scale, 0.0, 2.0))

    def _get_step_bid_ask(self, instrument: str, mid: float, vol_proxy: float) -> Tuple[float, float]:
        """Quote caching: call quote() at most once per step."""
        assert self._exec is not None

        if self._quote_cache_step == int(self.current_step) and self._quote_cache_inst == instrument:
            return float(self._quote_cache_bid), float(self._quote_cache_ask)

        bid, ask, _ = self._exec.quote(mid, vol_proxy)
        self._quote_cache_step = int(self.current_step)
        self._quote_cache_inst = instrument
        self._quote_cache_mid = float(mid)
        self._quote_cache_vol = float(vol_proxy)
        self._quote_cache_bid = float(bid)
        self._quote_cache_ask = float(ask)
        return float(bid), float(ask)

    def _mark_unrealized_pnl_from_bid_ask(self, pos: PropPosition, bid: float, ask: float) -> float:
        mark = float(bid) if pos.direction == "long" else float(ask)
        pip_value, multiplier = self._get_pip_value(pos.instrument)
        diff = (mark - pos.entry_price) * multiplier
        if pos.direction == "short":
            diff = -diff
        return float(diff * pip_value * pos.lot_size)

    def _realize_pnl_on_exit(self, pos: PropPosition, exit_fill: float, exit_fee: float) -> float:
        """Net realized pnl from price move, minus EXIT fee only."""
        pip_value, multiplier = self._get_pip_value(pos.instrument)
        diff = (exit_fill - pos.entry_price) * multiplier
        if pos.direction == "short":
            diff = -diff
        gross = diff * pip_value * pos.lot_size
        return float(gross - float(exit_fee))

    def _calculate_lot_size(self, size_mult: float) -> Tuple[float, float]:
        base_risk = self.balance * self.config.risk_per_trade_pct
        risk_eur = base_risk * float(np.clip(size_mult, 0.25, 1.25))
        risk_eur = min(risk_eur, self.balance * self.config.max_risk_per_trade_pct)

        lot = risk_eur / max(self.config.hard_stop_loss_eur, 100.0)
        lot = float(np.clip(lot, 0.01, 10.0))
        initial_risk = float(lot * self.config.hard_stop_loss_eur)
        return lot, initial_risk

    def _update_peak_balance(self) -> None:
        if bool(self.config.trailing_drawdown):
            self.peak_balance = max(self.peak_balance, float(self.equity))
        else:
            self.peak_balance = max(self.peak_balance, float(self.balance))

    def _calc_dds(self) -> Tuple[float, float]:
        """Calculate current and daily drawdowns.
        
        For FTMO-style prop firms:
        - Static DD (trailing_drawdown=False): vs INITIAL balance (never increases)
        - Trailing DD (trailing_drawdown=True): vs peak equity (ratchets up)
        - Daily DD: always vs day_start_balance
        """
        if bool(self.config.trailing_drawdown):
            # Trailing: drawdown measured from peak equity
            current_dd = (self.peak_balance - self.equity) / max(self.peak_balance, 1.0)
        else:
            # Static: drawdown measured from INITIAL balance (FTMO standard)
            current_dd = (self.config.initial_balance - self.equity) / max(self.config.initial_balance, 1.0)
        
        current_daily_dd = (self.day_start_balance - self.equity) / max(self.day_start_balance, 1.0)
        current_dd = max(0.0, float(current_dd))
        current_daily_dd = max(0.0, float(current_daily_dd))
        return current_dd, current_daily_dd

    # ---------------------------
    # Enhanced Reward System
    # ---------------------------

    def _compute_trade_reward(self, result: TradeResult, current_dd: float) -> float:
        cfg = self.config.reward

        net_pnl = float(result.net_pnl)
        risk_eur = max(float(result.initial_risk_eur), 1e-6)
        mae = abs(float(result.mae))
        mfe = float(result.mfe)
        total_fees = float(result.total_fees)
        # Gross PnL (before fees) for consistent MFE comparison
        gross_pnl = net_pnl + total_fees
        bars_held = int(result.bars_held)
        close_reason = result.close_reason
        entry_quality = float(result.entry_quality)

        pnl_pct = net_pnl / max(float(self.config.initial_balance), 1.0)
        r_multiple = net_pnl / risk_eur

        reward = 0.0
        reward_components: Dict[str, float] = {}

        # 1) Base PnL reward (asymmetric optional)
        if net_pnl > 0:
            base_reward = pnl_pct * cfg.reward_scale
            reward_components["base_pnl"] = base_reward
            reward += base_reward
        else:
            base_penalty = abs(pnl_pct) * cfg.reward_scale * cfg.loss_multiplier
            reward_components["base_pnl"] = -base_penalty
            reward -= base_penalty

        # 2) R-multiple bonus (winners only)
        if net_pnl > 0 and r_multiple >= cfg.r_multiple_bonus_threshold:
            excess_r = r_multiple - cfg.r_multiple_bonus_threshold
            r_bonus = min(excess_r * cfg.r_multiple_bonus_scale, cfg.r_multiple_bonus_cap)
            reward_components["r_multiple_bonus"] = r_bonus
            reward += r_bonus

        # 3) MAE efficiency
        if cfg.mae_efficiency_enabled and net_pnl > 0 and mae > 0:
            efficiency_ratio = net_pnl / mae
            if efficiency_ratio >= cfg.mae_efficiency_threshold:
                eff_bonus = min((efficiency_ratio - cfg.mae_efficiency_threshold) * 0.1, cfg.mae_efficiency_scale)
                reward_components["mae_efficiency"] = eff_bonus
                reward += eff_bonus
        elif cfg.mae_efficiency_enabled and net_pnl < 0 and mfe > 0:
            if mfe > abs(net_pnl) * 0.5:
                missed_profit_penalty = min(mfe / risk_eur * 0.05, 0.15)
                reward_components["missed_profit_penalty"] = -missed_profit_penalty
                reward -= missed_profit_penalty

        # 4) Time efficiency
        if cfg.time_efficiency_enabled and net_pnl > 0:
            if bars_held <= cfg.optimal_trade_bars:
                time_bonus = cfg.time_efficiency_scale * (1.0 - (bars_held / max(cfg.optimal_trade_bars, 1)) * 0.5)
                reward_components["time_efficiency"] = time_bonus
                reward += time_bonus
            elif bars_held <= cfg.max_trade_bars_for_bonus:
                denom = max(cfg.max_trade_bars_for_bonus - cfg.optimal_trade_bars, 1)
                duration_factor = 1.0 - (bars_held - cfg.optimal_trade_bars) / denom
                time_bonus = cfg.time_efficiency_scale * 0.3 * duration_factor
                reward_components["time_efficiency"] = time_bonus
                reward += time_bonus
        elif cfg.time_efficiency_enabled and net_pnl < 0 and bars_held > cfg.max_trade_bars_for_bonus:
            time_penalty = 0.05 * min((bars_held - cfg.max_trade_bars_for_bonus) / max(cfg.max_trade_bars_for_bonus, 1), 1.0)
            reward_components["time_penalty"] = -time_penalty
            reward -= time_penalty

        # 5) Exit quality modifier
        if cfg.exit_quality_enabled:
            exit_modifier = 0.0
            if close_reason == CloseReason.TRAILING_STOP:
                exit_modifier = cfg.trailing_stop_bonus
            elif close_reason == CloseReason.AGENT_CLOSE:
                # Agent close: Apply bonus/penalty regardless of PnL
                # With agent_close_bonus < 0, this PENALIZES manual closes even on winners
                # This teaches the agent to let trailing stops do their job
                exit_modifier = cfg.agent_close_bonus
                if net_pnl > 0 and mfe > 0:
                    # Capture ratio uses GROSS values for consistency:
                    # MFE is gross (peak unrealized PnL before fees)
                    # gross_pnl is net_pnl + total_fees (to compare apples to apples)
                    capture_ratio = gross_pnl / mfe if mfe > 0 else 1.0
                    # If capture_ratio < threshold, we closed way too early
                    if capture_ratio < cfg.premature_close_capture_threshold:
                        # Penalty scales with how much profit we left behind
                        left_on_table = 1.0 - capture_ratio
                        premature_close_penalty = min(
                            left_on_table * cfg.premature_close_penalty_scale, 
                            cfg.premature_close_penalty_cap
                        )
                        reward_components["premature_close_penalty"] = -premature_close_penalty
                        reward -= premature_close_penalty
            elif close_reason == CloseReason.HARD_STOP:
                exit_modifier = -cfg.hard_stop_penalty
            elif close_reason == CloseReason.RISK_LIQUIDATION:
                exit_modifier = -cfg.risk_liquidation_penalty
            elif close_reason == CloseReason.EMERGENCY_CLOSE:
                exit_modifier = -cfg.hard_stop_penalty * 1.2

            if exit_modifier != 0.0:
                reward_components["exit_quality"] = exit_modifier
                reward += exit_modifier

        # 6) Truncation handling
        if close_reason == CloseReason.EPISODE_TRUNCATE:
            if net_pnl > 0:
                trunc_discount = reward_components.get("base_pnl", 0.0) * cfg.truncation_winner_discount
                reward_components["truncation_discount"] = -trunc_discount
                reward -= trunc_discount
            else:
                trunc_pen = cfg.truncation_loser_extra_penalty
                reward_components["truncation_penalty"] = -trunc_pen
                reward -= trunc_pen

        # 7) Entry quality integration
        if cfg.entry_quality_integration:
            qdev = entry_quality - 0.5
            base_mag = abs(reward_components.get("base_pnl", 0.0))
            if net_pnl > 0 and qdev > 0:
                q_bonus = qdev * cfg.entry_quality_weight * base_mag
                reward_components["entry_quality_bonus"] = q_bonus
                reward += q_bonus
            elif net_pnl < 0 and qdev < 0:
                q_pen = abs(qdev) * cfg.entry_quality_weight * base_mag
                reward_components["entry_quality_penalty"] = -q_pen
                reward -= q_pen

        # 7b) Session timing reward/penalty (DIRECT signal for learning trading hours)
        if cfg.session_timing_enabled:
            # Get the entry time from the trade result (if available)
            entry_dt = getattr(result, 'entry_dt', None)
            if entry_dt is None:
                # Fallback: use current bar time
                entry_dt = self._get_bar_dt(self.instruments[0])
            
            if entry_dt is not None:
                if self._in_no_new_trades_window(entry_dt):
                    # Traded during off-hours = penalty (even if profitable!)
                    off_hours_pen = cfg.off_hours_trade_penalty
                    reward_components["off_hours_penalty"] = -off_hours_pen
                    reward -= off_hours_pen
                elif self._in_prime_window(entry_dt):
                    # Traded during prime hours = bonus
                    prime_bonus = cfg.prime_hours_trade_bonus
                    reward_components["prime_hours_bonus"] = prime_bonus
                    reward += prime_bonus

        # 7c) Market structure rewards (v5.3 - teach WHERE to trade)
        # IMPROVED: Only reward S/R entries that actually WORKED (profitable or small loss)
        if cfg.market_structure_enabled:
            # Get entry context from stored state
            entry_context = getattr(result, 'entry_context', {}) or {}
            trade_direction = result.direction  # "long" or "short"
            trade_pnl = result.net_pnl  # Actual P&L of the trade
            
            # S/R proximity reward - ONLY if trade was successful or near-breakeven
            # This teaches: enter at S/R AND the level must hold
            near_support = float(entry_context.get("near_support", 0.0))
            near_resistance = float(entry_context.get("near_resistance", 0.0))
            
            # Trade must be at least not a disaster (> -50 EUR) to get S/R bonus
            trade_worked = trade_pnl > -50.0
            
            if trade_direction == "long" and near_support > 0.5:
                if trade_worked:
                    # Support held! Reward the good entry
                    sr_bonus = near_support * cfg.sr_proximity_bonus
                    reward_components["sr_support_bonus"] = sr_bonus
                    reward += sr_bonus
                else:
                    # Entered at support but support BROKE - double penalty
                    sr_pen = near_support * cfg.sr_proximity_penalty * 1.5
                    reward_components["sr_support_failed_penalty"] = -sr_pen
                    reward -= sr_pen
                    
            elif trade_direction == "short" and near_resistance > 0.5:
                if trade_worked:
                    # Resistance held! Reward the good entry
                    sr_bonus = near_resistance * cfg.sr_proximity_bonus
                    reward_components["sr_resistance_bonus"] = sr_bonus
                    reward += sr_bonus
                else:
                    # Entered at resistance but it BROKE - double penalty
                    sr_pen = near_resistance * cfg.sr_proximity_penalty * 1.5
                    reward_components["sr_resistance_failed_penalty"] = -sr_pen
                    reward -= sr_pen
                    
            elif trade_direction == "long" and near_resistance > 0.5:
                # Going long at resistance - bad idea
                sr_pen = near_resistance * cfg.sr_proximity_penalty
                reward_components["sr_bad_entry_penalty"] = -sr_pen
                reward -= sr_pen
            elif trade_direction == "short" and near_support > 0.5:
                # Going short at support - bad idea
                sr_pen = near_support * cfg.sr_proximity_penalty
                reward_components["sr_bad_entry_penalty"] = -sr_pen
                reward -= sr_pen
            
            # Structure alignment bonus (HH/HL for longs, LL/LH for shorts)
            structure_trend = float(entry_context.get("structure_trend", 0.0))
            if (trade_direction == "long" and structure_trend > 0.3) or \
               (trade_direction == "short" and structure_trend < -0.3):
                struct_bonus = abs(structure_trend) * cfg.structure_alignment_bonus
                reward_components["structure_alignment_bonus"] = struct_bonus
                reward += struct_bonus
            
            # BOS alignment bonus
            bos_signal = float(entry_context.get("bos_signal", 0.0))
            if cfg.bos_alignment_bonus > 0:
                if (trade_direction == "long" and bos_signal > 0.3) or \
                   (trade_direction == "short" and bos_signal < -0.3):
                    bos_bonus = abs(bos_signal) * cfg.bos_alignment_bonus
                    reward_components["bos_alignment_bonus"] = bos_bonus
                    reward += bos_bonus
            
            # Order block entry bonus
            ob_bull = float(entry_context.get("order_block_bull", 0.0))
            ob_bear = float(entry_context.get("order_block_bear", 0.0))
            if cfg.order_block_entry_bonus > 0:
                if trade_direction == "long" and ob_bull > 0.5:
                    ob_bonus = ob_bull * cfg.order_block_entry_bonus
                    reward_components["order_block_bonus"] = ob_bonus
                    reward += ob_bonus
                elif trade_direction == "short" and ob_bear > 0.5:
                    ob_bonus = ob_bear * cfg.order_block_entry_bonus
                    reward_components["order_block_bonus"] = ob_bonus
                    reward += ob_bonus

        # 7d) Divergence awareness rewards (v5.3 - teach reversal awareness)
        if cfg.divergence_awareness_enabled:
            entry_context = getattr(result, 'entry_context', {}) or {}
            trade_direction = result.direction
            
            divergence = entry_context.get("divergence_signal")
            overbought = float(entry_context.get("overbought", 0.0))
            oversold = float(entry_context.get("oversold", 0.0))
            
            # Penalty for trading against divergence
            if divergence == "bullish" and trade_direction == "short":
                div_pen = cfg.divergence_contra_penalty
                reward_components["divergence_contra_penalty"] = -div_pen
                reward -= div_pen
            elif divergence == "bearish" and trade_direction == "long":
                div_pen = cfg.divergence_contra_penalty
                reward_components["divergence_contra_penalty"] = -div_pen
                reward -= div_pen
            
            # Bonus for trading with divergence
            if divergence == "bullish" and trade_direction == "long":
                div_bonus = cfg.divergence_aligned_bonus
                reward_components["divergence_aligned_bonus"] = div_bonus
                reward += div_bonus
            elif divergence == "bearish" and trade_direction == "short":
                div_bonus = cfg.divergence_aligned_bonus
                reward_components["divergence_aligned_bonus"] = div_bonus
                reward += div_bonus
            
            # Overbought/oversold penalties
            if overbought > 0.3 and trade_direction == "long":
                ob_pen = overbought * cfg.overbought_long_penalty
                reward_components["overbought_long_penalty"] = -ob_pen
                reward -= ob_pen
            if oversold > 0.3 and trade_direction == "short":
                os_pen = oversold * cfg.oversold_short_penalty
                reward_components["oversold_short_penalty"] = -os_pen
                reward -= os_pen

        # 7e) Regime awareness rewards (v5.3 - teach context sensitivity)
        if cfg.regime_awareness_enabled:
            entry_context = getattr(result, 'entry_context', {}) or {}
            
            risk_regime = entry_context.get("risk_regime", "neutral")
            vol_regime = entry_context.get("volatility_regime", "normal")
            
            # Penalty for aggressive trades in risk-off environment
            if risk_regime == "risk_off" and net_pnl < 0:
                risk_pen = cfg.risk_off_aggressive_penalty
                reward_components["risk_off_penalty"] = -risk_pen
                reward -= risk_pen
            
            # Penalty for large losses in high volatility
            if vol_regime == "high" and net_pnl < -50:
                vol_pen = cfg.high_vol_size_penalty
                reward_components["high_vol_penalty"] = -vol_pen
                reward -= vol_pen

        # 8) Streak modifiers
        if cfg.streak_modifier_enabled:
            if net_pnl > 0:
                streak_bonus = min(self.consecutive_wins, 5) * cfg.win_streak_bonus_per_win
                if streak_bonus > 0:
                    reward_components["win_streak_bonus"] = streak_bonus
                    reward += streak_bonus
            else:
                streak_pen = min(self.consecutive_losses, 3) * cfg.loss_streak_penalty_per_loss
                if streak_pen > 0:
                    reward_components["loss_streak_penalty"] = -streak_pen
                    reward -= streak_pen

        # 9) Anti-churn penalty (SOFTER EXPONENTIAL to discourage overtrading)
        # Using base 1.5 instead of 2.0 to prevent gradient explosion while still penalizing
        # 6 excess trades: 1.5^6 ≈ 11x vs 2^6 = 64x (much softer)
        if cfg.anti_churn_enabled and self.daily_trades > cfg.daily_trade_soft_limit:
            excess = self.daily_trades - cfg.daily_trade_soft_limit
            # Softer exponential growth: 1.5^excess - 1
            # Cap at 8 excess trades: 1.5^8 ≈ 25x (was 63x with base 2)
            excess_factor = min(1.5 ** min(excess, 8) - 1, 25.0)
            churn_pen = excess_factor * cfg.churn_penalty_per_trade
            reward_components["churn_penalty"] = -churn_pen
            reward -= churn_pen

        # 10) Drawdown shaping (guard denominator)
        if cfg.dd_shaping_enabled and current_dd > cfg.dd_threshold:
            denom = float(self.config.max_drawdown_limit - cfg.dd_threshold)
            if denom > 1e-9:
                dd_ratio = (current_dd - cfg.dd_threshold) / denom
                dd_ratio = float(np.clip(dd_ratio, 0.0, cfg.dd_severity_cap))
                dd_severity = dd_ratio ** cfg.dd_severity_exponent
                dd_pen = dd_severity * cfg.dd_penalty_scale * 0.5
                reward_components["dd_shaping"] = -dd_pen
                reward -= dd_pen

        reward = float(np.clip(reward, cfg.min_reward, cfg.max_reward))
        self._last_reward_components = reward_components
        
        # Aggregate reward components for dashboard tracking
        for key, value in reward_components.items():
            if key not in self._episode_reward_components:
                self._episode_reward_components[key] = 0.0
                self._episode_reward_component_counts[key] = 0
            self._episode_reward_components[key] += value
            self._episode_reward_component_counts[key] += 1
        
        return reward

    def _compute_blocked_action_penalty(self, block_reason: str, entry_quality: float, is_hard_block: bool) -> float:
        cfg = self.config.reward
        if is_hard_block:
            return cfg.hard_block_penalty
        return cfg.soft_block_penalty * (1.0 - entry_quality * 0.5)

    def _compute_per_step_shaping(
        self,
        has_position: bool,
        bars_in_position: int,
        entry_quality_long: float,
        entry_quality_short: float,
        took_action: bool
    ) -> float:
        cfg = self.config.reward
        if not cfg.per_step_shaping_enabled:
            return 0.0
        shaping = 0.0
        if has_position and bars_in_position > 0:
            shaping -= cfg.holding_cost_per_bar
        _ = (entry_quality_long, entry_quality_short, took_action)
        return shaping

    def _close_position_now(
        self,
        *,
        reason: str,
        dt: Optional[datetime],
        mid: float,
        vol_proxy: float,
    ) -> TradeResult:
        assert self.position is not None
        assert self._exec is not None

        pos = self.position
        initial_risk = float(pos.initial_risk_eur)
        entry_fee = float(pos.entry_fee_eur)
        # MAE: Maximum Adverse Excursion - always stored as positive (absolute worst drawdown)
        # lowest_pnl is negative when position went against us
        mae = abs(float(pos.lowest_pnl))
        # MFE: Maximum Favorable Excursion - best unrealized profit (always positive or zero)
        mfe = max(0.0, float(pos.peak_pnl))
        bars_held = self.episode_bars - pos.entry_bar
        entry_quality = float(pos.entry_quality)

        exit_fill, exit_fee, _ = self._exec.fill_exit(mid, pos.direction, pos.lot_size, vol_proxy)
        realized_pnl = float(self._realize_pnl_on_exit(pos, exit_fill, exit_fee))
        net_trade_pnl = float(realized_pnl - entry_fee)
        total_fees = float(entry_fee + exit_fee)

        # Accounting
        self.balance += realized_pnl
        self.equity = self.balance
        self.total_pnl += realized_pnl
        self.daily_pnl += realized_pnl

        self.total_trades += 1

        # Stats update (fee-aware)
        if net_trade_pnl > 0:
            self.winning_trades += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            # Track peak consecutive losses during episode
            if self.consecutive_losses > self.max_consecutive_losses_reached:
                self.max_consecutive_losses_reached = self.consecutive_losses
            if dt is not None:
                self._last_loss_dt = dt
            self._last_loss_step = int(self.current_step)

        try:
            close_reason_enum = CloseReason(reason)
        except ValueError:
            close_reason_enum = CloseReason.AGENT_CLOSE

        result = TradeResult(
            net_pnl=net_trade_pnl,
            initial_risk_eur=initial_risk,
            mae=mae,
            mfe=mfe,
            bars_held=bars_held,
            close_reason=close_reason_enum,
            entry_quality=entry_quality,
            direction=pos.direction,
            lot_size=pos.lot_size,
            total_fees=total_fees,
            entry_dt=pos.entry_dt if hasattr(pos, 'entry_dt') else None,
            entry_context=pos.entry_context if hasattr(pos, 'entry_context') else None,  # v5.3
        )

        self._episode_trade_results.append(result)

        self.position = None
        self.pending_exit = None
        self._update_peak_balance()

        return result

    # ---------------------------
    # Indicators / quality
    # ---------------------------

    def _compute_rsi(self, close: np.ndarray, period: int = 14) -> float:
        if len(close) < period + 1:
            return 50.0
        deltas = np.diff(close[-period - 1:])
        gains = np.maximum(deltas, 0.0)
        losses = np.abs(np.minimum(deltas, 0.0))
        ag = float(np.mean(gains))
        al = float(np.mean(losses))
        if al < 1e-8:
            return 100.0 if ag > 0 else 50.0
        rs = ag / al
        return float(100.0 - (100.0 / (1.0 + rs)))

    @staticmethod
    def _dir_sign(d: str) -> float:
        d = (d or "").lower()
        if d in ("bullish", "long"):
            return 1.0
        if d in ("bearish", "short"):
            return -1.0
        return 0.0

    def _get_step_entry_quality(self, inst: str, target: str) -> float:
        """
        Get entry quality with per-step caching.
        Avoids redundant computation of the expensive _compute_smart_entry_quality.
        """
        cache_key = f"{inst}_{target}"
        cache = getattr(self, "_step_entry_quality_cache", None)
        if cache is not None and cache_key in cache:
            return cache[cache_key]
        
        quality = self._compute_smart_entry_quality(inst, target)
        
        if cache is not None:
            cache[cache_key] = quality
        return quality

    def _compute_smart_entry_quality(self, inst: str, target: str) -> float:
        if target not in ("long", "short"):
            return 0.5

        direction_mult = 1.0 if target == "long" else -1.0
        quality_components: List[Tuple[str, float, float]] = []

        expert_signals = self._prepare_expert_signals(inst)
        experts = expert_signals.get("experts", {}) if isinstance(expert_signals, dict) else {}
        if isinstance(experts, dict) and experts:

            def ex_score(name: str) -> float:
                sig = experts.get(name, {}) if isinstance(experts.get(name, {}), dict) else {}
                s = float(sig.get("score", 0.0) or 0.0)
                c = float(sig.get("confidence", 0.5) or 0.5)
                d = self._dir_sign(str(sig.get("direction", "neutral")))
                return d * s * c

            trend_score = ex_score("trend")
            mom_score = ex_score("momentum")
            theme_score = ex_score("theme")

            alignment = 0.40 * trend_score + 0.35 * mom_score + 0.25 * theme_score
            expert_quality = float(np.clip((alignment * direction_mult + 1.0) / 2.0, 0.0, 1.0))
            quality_components.append(("experts", expert_quality, 0.30))

        committee = self._prepare_committee_state(expert_signals)
        if isinstance(committee, dict) and committee:
            action_value = float(committee.get("action_value", 0.0) or 0.0)
            consensus = float(committee.get("consensus_score", 0.5) or 0.5)
            conf = float(committee.get("confidence", 0.5) or 0.5)
            agreement = float(committee.get("agreement", 0.5) or 0.5)

            action_alignment = float(np.clip((action_value * direction_mult + 1.0) / 2.0, 0.0, 1.0))
            committee_quality = float(np.clip(
                0.35 * consensus + 0.35 * action_alignment + 0.20 * conf + 0.10 * agreement,
                0.0, 1.0
            ))
            quality_components.append(("committee", committee_quality, 0.25))

        htf_quality = 0.5
        ohlcv = self._get_ohlcv(inst, lookback=120)
        close = np.asarray(ohlcv.get("close", np.array([])), dtype=np.float64) if ohlcv else np.array([])
        if len(close) >= 50:
            sma_20 = float(np.mean(close[-20:]))
            sma_50 = float(np.mean(close[-50:]))
            current_price = float(close[-1])

            above_sma20 = 1.0 if current_price > sma_20 else -1.0
            above_sma50 = 1.0 if current_price > sma_50 else -1.0
            sma_trend = 1.0 if sma_20 > sma_50 else -1.0

            htf_alignment = (
                0.40 * (above_sma20 * direction_mult) +
                0.35 * (above_sma50 * direction_mult) +
                0.25 * (sma_trend * direction_mult)
            )
            htf_quality = float(np.clip((htf_alignment + 1.0) / 2.0, 0.0, 1.0))
        quality_components.append(("htf_context", htf_quality, 0.20))

        risk_state = self._prepare_risk_state()
        if isinstance(risk_state, dict) and risk_state:
            dd = float(risk_state.get("current_drawdown", 0.0) or 0.0)
            ddd = float(risk_state.get("daily_drawdown", 0.0) or 0.0)

            dd_limit = max(float(self.config.max_drawdown_limit), 1e-6)
            daily_limit = max(float(self.config.daily_drawdown_limit), 1e-6)

            dd_headroom = max(0.0, (dd_limit - dd) / dd_limit)
            daily_headroom = max(0.0, (daily_limit - ddd) / daily_limit)

            risk_quality = float(np.clip(min(dd_headroom, daily_headroom), 0.0, 1.0))
            quality_components.append(("risk_state", risk_quality, 0.15))

        momentum_quality = 0.5
        if len(close) >= 14:
            rsi = self._compute_rsi(close, 14)
            if target == "long":
                if 30 <= rsi <= 50:
                    momentum_quality = 0.9
                elif 50 < rsi <= 65:
                    momentum_quality = 0.6
                elif rsi > 65:
                    momentum_quality = 0.3
                else:
                    momentum_quality = 0.7
            else:
                if 50 <= rsi <= 70:
                    momentum_quality = 0.9
                elif 35 <= rsi < 50:
                    momentum_quality = 0.6
                elif rsi < 35:
                    momentum_quality = 0.3
                else:
                    momentum_quality = 0.7
        quality_components.append(("momentum", float(momentum_quality), 0.10))

        # Timing quality
        timing_quality = 0.5
        dt = self._get_bar_dt(inst)
        if dt is not None:
            if self._in_prime_window(dt):
                timing_quality = 0.95
            elif not self._in_no_new_trades_window(dt):
                timing_quality = 0.6
            else:
                timing_quality = 0.1
        quality_components.append(("timing", float(timing_quality), 0.15))

        total_weight = sum(w for _, _, w in quality_components)
        if total_weight <= 0:
            return 0.5
        weighted_sum = sum(v * w for _, v, w in quality_components)
        return float(np.clip(weighted_sum / total_weight, 0.0, 1.0))

    # ---------------------------
    # Domain randomization
    # ---------------------------

    def _apply_domain_randomization(self) -> None:
        if not self.config.domain_randomization_enabled:
            self._episode_spread_mult = 1.0
            self._episode_slip_mult = 1.0
            self._episode_latency_bars = 0
            self._episode_vol_scale = 1.0
            return

        rng = self.np_random
        self._episode_spread_mult = float(rng.uniform(*self.config.spread_mult_range))
        self._episode_slip_mult = float(rng.uniform(*self.config.slippage_mult_range))
        self._episode_latency_bars = int(
            rng.integers(int(self.config.latency_bars_range[0]), int(self.config.latency_bars_range[1]) + 1)
        )
        self._episode_vol_scale = float(rng.uniform(*self.config.volatility_scale_range))

    def _build_episode_execution_config(self) -> ExecutionConfig:
        base = copy.deepcopy(self.config.execution)
        for name, val in (
            ("spread_mult", self._episode_spread_mult),
            ("slippage_mult", self._episode_slip_mult),
            ("latency_bars", self._episode_latency_bars),
        ):
            try:
                if hasattr(base, name):
                    setattr(base, name, val)
            except Exception:
                pass
        return base

    # ---------------------------
    # Hard-rule entry permission
    # ---------------------------

    def _hard_entry_allowed_pure(
        self,
        dt: Optional[datetime],
        *,
        step_idx: Optional[int],
        current_dd: float,
        current_daily_dd: float,
        daily_trades: int,
        session_trades: int,
    ) -> Tuple[bool, str]:
        if dt is not None:
            # F-7 FIX: Respect constraint enforcement flags from curriculum
            if self.config.enforce_weekend_block:
                if (not self.config.allow_weekend_holding) and self._is_weekend(dt):
                    return False, "weekend_block"
            if self.config.enforce_no_new_trades_window:
                if self._in_no_new_trades_window(dt):
                    return False, "no_new_trades_window"
            if self.config.enforce_hard_close:
                if self._in_final_exit_window(dt):
                    return False, "final_exit_window"
                if self._at_or_after_hard_close(dt):
                    return False, "hard_close"

        if self.consecutive_losses >= self.config.max_consecutive_losses:
            return False, "max_consecutive_losses"

        max_dd_ok = float(current_dd) < (self.config.max_drawdown_limit - self.config.max_dd_safety_buffer)
        daily_dd_ok = float(current_daily_dd) < (self.config.daily_drawdown_limit - self.config.daily_dd_safety_buffer)
        if not (max_dd_ok and daily_dd_ok):
            return False, "drawdown_headroom"

        if int(daily_trades) >= self.config.max_trades_per_day:
            return False, "max_trades_per_day"
        if int(session_trades) >= self.config.max_trades_per_session:
            return False, "max_trades_per_session"

        if dt is not None:
            if self._last_entry_dt is not None:
                mins = (dt - self._last_entry_dt).total_seconds() / 60.0
                if mins < self.config.min_minutes_between_entries:
                    return False, "min_entry_spacing"

            if self._last_loss_dt is not None:
                mins = (dt - self._last_loss_dt).total_seconds() / 60.0
                if mins < self.config.min_minutes_after_loss:
                    return False, "post_loss_cooldown"
        else:
            si = int(step_idx) if step_idx is not None else int(self.current_step)
            tfm = max(1, self._tf_minutes())
            min_entry_bars = int(np.ceil(self.config.min_minutes_between_entries / tfm))
            post_loss_bars = int(np.ceil(self.config.min_minutes_after_loss / tfm))

            if self._last_entry_step is not None:
                if (si - int(self._last_entry_step)) < max(1, min_entry_bars):
                    return False, "min_entry_spacing"

            if self._last_loss_step is not None:
                if (si - int(self._last_loss_step)) < max(1, post_loss_bars):
                    return False, "post_loss_cooldown"

        return True, ""

    def _hard_entry_allowed(self, dt: Optional[datetime]) -> Tuple[bool, str]:
        current_dd, current_daily_dd = self._calc_dds()
        return self._hard_entry_allowed_pure(
            dt,
            step_idx=int(self.current_step),
            current_dd=current_dd,
            current_daily_dd=current_daily_dd,
            daily_trades=self.daily_trades,
            session_trades=self._session_trades,
        )

    # ---------------------------
    # Gym API
    # ---------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # Apply curriculum stage overrides at clean boundary (start of episode)
        if self.curriculum and (self._pending_stage_apply or self._last_stage_name != getattr(self.curriculum.current_stage, "name", "")):
            self._sync_curriculum_stage_overrides()
            self._pending_stage_apply = False
            self._last_stage_name = getattr(self.curriculum.current_stage, "name", "")
            self._last_stage_epoch = int(getattr(self.curriculum, "current_stage_epoch", 0))

        self.balance = float(self.config.initial_balance)
        self.equity = float(self.config.initial_balance)
        self.day_start_balance = float(self.config.initial_balance)
        self.peak_balance = float(self.config.initial_balance)

        self.position = None
        self.pending_entry = None
        self.pending_exit = None

        self.total_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.consecutive_wins = 0

        self._current_day = None
        self._current_session_key = None
        self._session_trades = 0

        self._last_entry_dt = None
        self._last_loss_dt = None
        self._last_entry_step = None
        self._last_loss_step = None

        self.current_step = 0
        self.episode_step = 0
        self.episode_bars = 0
        self._episode_return = 0.0

        # AUDIT FIX: Removed _opportunity_history and _action_history (dead code - never used)
        self._avg_vol = None
        self._episode_trade_results = []
        self._last_reward_components = {}
        
        # Reset reward component aggregates
        self._episode_reward_components = {}
        self._episode_reward_component_counts = {}

        self._ohlcv_cache_key = None
        self._ohlcv_cache = {}

        self._quote_cache_step = None
        self._quote_cache_inst = None
        self._quote_cache_mid = 0.0
        self._quote_cache_vol = 0.0
        self._quote_cache_bid = 0.0
        self._quote_cache_ask = 0.0

        self._apply_domain_randomization()
        self._episode_execution_cfg = self._build_episode_execution_config()
        self._exec = ExecutionModel(self._episode_execution_cfg, self.np_random)
        
        # Apply domain randomization to execution model
        # This is CRITICAL - without this call, spread/slippage randomization is ignored!
        self._exec.set_episode_randomization(
            spread_mult=self._episode_spread_mult,
            slippage_mult=self._episode_slip_mult,
        )

        buffer = 120
        max_start = max(buffer, self._min_data_len - self.config.max_steps_per_episode - buffer)
        
        # Use data difficulty sampling if enabled, otherwise uniform random
        if self._data_difficulty is not None:
            self.current_step = self._sample_episode_start_with_difficulty(buffer, max_start)
        elif max_start > buffer:
            self.current_step = int(self.np_random.integers(buffer, max_start))
        else:
            self.current_step = min(buffer, max(self._min_data_len - 2, 0))

        inst = self.instruments[0]
        dt = self._get_bar_dt(inst)
        self._maybe_roll_day_session(dt)

        obs = self._get_observation()
        info = {"balance": self.balance, "equity": self.equity, "step": self.current_step}
        info.update(self._curriculum_step_metadata())
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Advance time FIRST
        self.current_step += 1
        self.episode_step += 1
        self.episode_bars += 1

        inst = self.instruments[0]
        dt = self._get_bar_dt(inst)
        
        # Cache entry quality at step start (computed once, reused for gating and info)
        self._step_entry_quality_cache: Dict[str, float] = {}
        
        # Cache expert signals per step (prevents recomputation and ensures consistency)
        self._step_expert_signals_cache: Optional[Dict[str, Any]] = None

        intent, size_mult = self._decode_action(int(action))

        mid = self._get_price_mid(inst)
        vol_proxy = self._atr_vol_proxy(inst)
        assert self._exec is not None

        bid, ask = self._get_step_bid_ask(inst, mid, vol_proxy)

        # BUG FIX: fill_delay should be latency only, not 1 + latency
        # The +1 was causing a hidden 2-bar delay because step() already advanced time
        latency = max(0, self._exec_latency())
        last_idx = (self._min_data_len - 1)
        remaining_bars = last_idx - int(self.current_step)
        
        # Entry requires 1 bar AFTER fill so position isn't instantly truncated
        can_entry_fill = remaining_bars >= (latency + 1)
        # Exit can fill right at the end (no post-fill bar required)
        can_exit_fill = remaining_bars >= latency

        reward = 0.0
        trade_closed = False
        close_result: Optional[TradeResult] = None

        # Mark-to-market FIRST (before day roll)
        pnl_u = 0.0
        if self.position is not None:
            pnl_u = self._mark_unrealized_pnl_from_bid_ask(self.position, bid, ask)
            self.equity = self.balance + pnl_u
            self.position.peak_pnl = max(self.position.peak_pnl, pnl_u)
            self.position.lowest_pnl = min(self.position.lowest_pnl, pnl_u)
        else:
            self.equity = self.balance

        self._update_peak_balance()
        
        # CRIT-1 FIX: Roll day/session AFTER mark-to-market so day_start_balance includes 
        # open PnL at midnight. This prevents daily drawdown "reset" exploits when carrying
        # losing positions across days.
        self._maybe_roll_day_session(dt)
        
        current_dd, current_daily_dd = self._calc_dds()

        # Forced closes
        forced_close_now = False
        close_reason_str = ""

        if self.position is not None:
            pos = self.position

            if pnl_u <= -self.config.hard_stop_loss_eur:
                forced_close_now = True
                close_reason_str = CloseReason.HARD_STOP.value

            if not forced_close_now and dt is not None and pos.entry_dt is not None:
                age_h = (dt - pos.entry_dt).total_seconds() / 3600.0
                if age_h >= self.config.time_decay_hours:
                    forced_close_now = True
                    close_reason_str = CloseReason.TIME_DECAY.value

            if not forced_close_now and pos.peak_pnl >= self.config.trailing_activation_eur:
                retrace = pos.peak_pnl - pnl_u
                if retrace > pos.peak_pnl * self.config.trailing_retrace_pct:
                    forced_close_now = True
                    close_reason_str = CloseReason.TRAILING_STOP.value

            dd_now = (self.day_start_balance - self.equity) / max(self.day_start_balance, 1.0)
            dd_now = max(0.0, float(dd_now))
            if not forced_close_now and dd_now >= (self.config.daily_drawdown_limit - self.config.daily_dd_safety_buffer):
                forced_close_now = True
                close_reason_str = CloseReason.DAILY_LIMIT_SAFETY.value

            if not forced_close_now and current_dd >= float(self.config.emergency_close_threshold):
                forced_close_now = True
                close_reason_str = CloseReason.EMERGENCY_CLOSE.value

            if not forced_close_now and dt is not None:
                if self._at_or_after_hard_close(dt):
                    forced_close_now = True
                    close_reason_str = CloseReason.HARD_CLOSE.value
                if (not self.config.allow_weekend_holding) and self._is_weekend(dt):
                    forced_close_now = True
                    close_reason_str = CloseReason.WEEKEND_FLATTEN.value

        # Agent close intent
        # BUG FIX: Use _set_or_override_pending_exit so forced closes can override
        if self.position is not None and not forced_close_now and can_exit_fill:
            if intent == "close":
                self._set_or_override_pending_exit(
                    reason=CloseReason.AGENT_CLOSE.value,
                    fill_step=self.current_step + latency
                )

        # Handle forced closes
        # BUG FIX: Split into three categories:
        # 1. IMMEDIATE (broker/risk systems): RISK_LIQUIDATION, EMERGENCY_CLOSE, DAILY_LIMIT_SAFETY, HARD_STOP
        # 2. DELAYED (realistic latency): TIME_DECAY, TRAILING_STOP, HARD_CLOSE, WEEKEND_FLATTEN
        # 3. Also use _set_or_override_pending_exit so forced closes override agent closes
        if forced_close_now and self.position is not None:
            # BUG FIX: DAILY_LIMIT_SAFETY and HARD_STOP should be immediate to prevent breaches
            immediate_reasons = {
                CloseReason.RISK_LIQUIDATION.value,
                CloseReason.EMERGENCY_CLOSE.value,
                CloseReason.DAILY_LIMIT_SAFETY.value,  # BUG FIX: Must be immediate to prevent breach
                CloseReason.HARD_STOP.value,           # BUG FIX: Server-side stop should be immediate
            }
            
            if close_reason_str in immediate_reasons:
                # Immediate execution for critical risk events
                close_result = self._close_position_now(reason=close_reason_str, dt=dt, mid=mid, vol_proxy=vol_proxy)
                trade_closed = True
                current_dd, current_daily_dd = self._calc_dds()
                reward += self._compute_trade_reward(close_result, current_dd)
            else:
                # Non-critical forced closes use delayed execution
                # BUG FIX: If can't fill before end, close immediately
                if not can_exit_fill:
                    close_result = self._close_position_now(reason=close_reason_str, dt=dt, mid=mid, vol_proxy=vol_proxy)
                    trade_closed = True
                    current_dd, current_daily_dd = self._calc_dds()
                    reward += self._compute_trade_reward(close_result, current_dd)
                else:
                    # BUG FIX: Use override helper so forced closes always win over agent closes
                    self._set_or_override_pending_exit(
                        reason=close_reason_str,
                        fill_step=self.current_step + latency
                    )
                    forced_close_now = False  # Don't execute yet

        # Execute scheduled exit (both agent closes AND delayed forced closes)
        if (not trade_closed) and self.position is not None and self.pending_exit is not None:
            if self.current_step >= int(self.pending_exit["fill_step"]):
                reason = str(self.pending_exit.get("reason", CloseReason.AGENT_CLOSE.value))
                close_result = self._close_position_now(reason=reason, dt=dt, mid=mid, vol_proxy=vol_proxy)
                trade_closed = True
                self.pending_exit = None
                current_dd, current_daily_dd = self._calc_dds()
                reward += self._compute_trade_reward(close_result, current_dd)

        current_dd, current_daily_dd = self._calc_dds()
        hard_ok, hard_block = self._hard_entry_allowed(dt)

        # Execute pending entry (fill)
        if self.position is None and self.pending_entry is not None:
            if self.current_step >= int(self.pending_entry["fill_step"]):
                hard_ok_fill, _ = self._hard_entry_allowed(dt)
                if hard_ok_fill:
                    direction = str(self.pending_entry["direction"])
                    lot = float(self.pending_entry["lot"])
                    initial_risk = float(self.pending_entry["initial_risk"])
                    entry_quality = float(self.pending_entry.get("entry_quality", 0.5))

                    entry_fill, entry_fee, _ = self._exec.fill_entry(mid, direction, lot, vol_proxy)

                    entry_fee = float(entry_fee)
                    if entry_fee > 0:
                        self.balance -= entry_fee
                        self.equity = self.balance
                        self.total_pnl -= entry_fee
                        self.daily_pnl -= entry_fee

                    pos = PropPosition(
                        instrument=inst,
                        direction=direction,
                        entry_price=entry_fill,
                        entry_dt=dt,
                        entry_bar=self.episode_bars,
                        lot_size=lot,
                        initial_risk_eur=initial_risk,
                        entry_fee_eur=float(entry_fee),
                        entry_quality=entry_quality,
                        entry_context=self._capture_entry_context(inst),  # v5.3: Market structure at entry
                    )
                    self.position = pos

                    # Immediate mark-to-market
                    u0 = self._mark_unrealized_pnl_from_bid_ask(pos, bid, ask)
                    self.equity = self.balance + float(u0)

                    # Seed MAE/MFE immediately
                    pos.peak_pnl = max(float(pos.peak_pnl), float(u0))
                    pos.lowest_pnl = min(float(pos.lowest_pnl), float(u0))

                    self.daily_trades += 1
                    self._session_trades += 1
                    if dt is not None:
                        self._last_entry_dt = dt
                    self._last_entry_step = int(self.current_step)

                self.pending_entry = None

        # Attempt new entry
        attempted_entry = (self.position is None and self.pending_entry is None and intent in ("long", "short"))
        entry_quality = self._get_step_entry_quality(inst, intent) if intent in ("long", "short") else 0.5

        entry_allowed = hard_ok
        block_reason = hard_block

        if attempted_entry and not can_entry_fill:
            entry_allowed = False
            block_reason = "insufficient_bars_for_fill"

        if attempted_entry:
            if self.config.entry_quality_gate_enabled and entry_quality < self.config.entry_quality_threshold:
                entry_allowed = False
                block_reason = "entry_quality_gate"

            if entry_allowed:
                lot, initial_risk = self._calculate_lot_size(size_mult)
                self.pending_entry = {
                    "direction": intent,
                    "lot": lot,
                    "initial_risk": initial_risk,
                    "fill_step": self.current_step + latency,  # BUG FIX: Use latency, not fill_delay
                    "entry_quality": entry_quality,
                }

        # Blocked entry penalty
        if attempted_entry and not entry_allowed:
            hard_blocks = {
                "hard_close", "drawdown_headroom", "max_trades_per_day", "max_trades_per_session",
                "post_loss_cooldown", "max_consecutive_losses", "weekend_block",
                "no_new_trades_window", "final_exit_window", "min_entry_spacing",
                "insufficient_bars_for_fill",
            }
            is_hard = block_reason in hard_blocks
            penalty = self._compute_blocked_action_penalty(block_reason, entry_quality, is_hard)
            reward -= penalty

        # DD breach liquidation
        current_dd, current_daily_dd = self._calc_dds()
        dd_breach = current_dd >= float(self.config.max_drawdown_limit)
        daily_dd_breach = current_daily_dd >= float(self.config.daily_drawdown_limit)

        did_risk_liquidate_this_step = False
        if (dd_breach or daily_dd_breach) and self.position is not None:
            close_result = self._close_position_now(
                reason=CloseReason.RISK_LIQUIDATION.value, dt=dt, mid=mid, vol_proxy=vol_proxy
            )
            did_risk_liquidate_this_step = True
            trade_closed = True
            current_dd, current_daily_dd = self._calc_dds()
            reward += self._compute_trade_reward(close_result, current_dd)

            dd_breach = current_dd >= float(self.config.max_drawdown_limit)
            daily_dd_breach = current_daily_dd >= float(self.config.daily_drawdown_limit)

        # Truncation / termination
        terminated = False
        truncated = False
        termination_reason = ""

        if self.current_step >= self._min_data_len - 1:
            truncated = True
            termination_reason = "data_exhausted"
        elif self.episode_step >= int(self.config.max_steps_per_episode):
            truncated = True
            termination_reason = "episode_length"

        # Flatten on truncation
        if truncated and not terminated and self.position is not None:
            # AUDIT FIX (HIGH-6): Re-fetch fresh market data for truncation close.
            # The mid/vol_proxy from step start may be stale after position updates.
            mid_now = self._get_price_mid(inst)
            vol_now = self._atr_vol_proxy(inst)
            close_result = self._close_position_now(
                reason=CloseReason.EPISODE_TRUNCATE.value, dt=dt, mid=mid_now, vol_proxy=vol_now
            )
            trade_closed = True
            current_dd, current_daily_dd = self._calc_dds()
            reward += self._compute_trade_reward(close_result, current_dd)

        # Cancel pending orders on truncation
        if truncated and not terminated:
            if self.pending_entry is not None:
                self.pending_entry = None
                reward -= 0.01
            if self.pending_exit is not None:
                self.pending_exit = None

        current_dd, current_daily_dd = self._calc_dds()
        dd_breach = current_dd >= float(self.config.max_drawdown_limit)
        daily_dd_breach = current_daily_dd >= float(self.config.daily_drawdown_limit)

        # Termination overrides truncation
        if dd_breach or daily_dd_breach:
            terminated = True
            truncated = False
            if dd_breach:
                termination_reason = "max_drawdown_breach"
                if (not trade_closed) and (not did_risk_liquidate_this_step):
                    reward -= 1.5
            else:
                termination_reason = "daily_limit_breach"
                if (not trade_closed) and (not did_risk_liquidate_this_step):
                    reward -= 1.0

        # Episode-end activity consistency check (NEW)
        # Penalizes extreme over/under-trading relative to episode length
        # Uses stage-specific targets if configured, otherwise falls back to default
        reward_cfg = self.config.reward
        if (terminated or truncated) and reward_cfg.activity_consistency_enabled:
            episode_steps = max(self.episode_step, 1)
            actual_trades = self.total_trades
            
            # Get stage-specific target if available
            target_per_1k = reward_cfg.target_trades_per_1k_steps  # Default
            if reward_cfg.stage_activity_targets is not None:
                stage_idx = getattr(self, '_curriculum_stage_idx', 0)
                target_per_1k = reward_cfg.stage_activity_targets.get(
                    stage_idx, reward_cfg.target_trades_per_1k_steps
                )
            
            expected_trades = (episode_steps / 1000.0) * target_per_1k
            
            # HIGH-2 FIX: Guard against penalizing short or DD-terminated episodes
            # - Skip for short episodes (expected < 2 trades)
            # - Skip for episodes < 300 steps (too short to establish pattern)
            # - Skip for DD-breach terminations (risk management is the priority, not activity)
            is_dd_termination = termination_reason in (
                "max_drawdown_breach", "daily_limit_breach", "daily_dd_breach"
            )
            should_apply_penalty = (
                expected_trades >= 2.0 and
                episode_steps >= 300 and
                not is_dd_termination
            )
            
            if should_apply_penalty:
                trade_ratio = actual_trades / expected_trades
                
                # Penalize severe under-trading (< 20% of expected)
                if trade_ratio < 0.2:
                    under_trade_penalty = reward_cfg.min_trades_penalty
                    reward -= under_trade_penalty
                # Penalize moderate deviation from target (both over and under)
                elif abs(trade_ratio - 1.0) > 0.5:  # More than 50% deviation
                    deviation = abs(trade_ratio - 1.0) - 0.5
                    deviation_penalty = min(deviation * reward_cfg.activity_deviation_penalty_scale, 0.2)
                    reward -= deviation_penalty

        # Compute qualities using step cache (computed once per step, reused)
        q_long = self._get_step_entry_quality(inst, "long")
        q_short = self._get_step_entry_quality(inst, "short")

        # Per-step shaping (if enabled)
        bars_in_pos = (self.episode_bars - self.position.entry_bar) if self.position else 0
        shaping = self._compute_per_step_shaping(
            has_position=self.position is not None,
            bars_in_position=bars_in_pos,
            entry_quality_long=q_long,
            entry_quality_short=q_short,
            took_action=intent in ("long", "short", "close"),
        )
        reward += shaping

        # Final clipping
        cfg = self.config.reward
        reward = float(np.clip(reward, cfg.min_reward, cfg.max_reward))

        # Accumulate episode return for curriculum
        self._episode_return += float(reward)

        obs = self._get_observation()
        win_rate = self.winning_trades / max(self.total_trades, 1)

        info = {
            "balance": float(self.balance),
            "equity": float(self.equity),
            "drawdown": float(current_dd),
            "daily_drawdown": float(current_daily_dd),
            "total_pnl": float(self.total_pnl),
            "trade_count": int(self.total_trades),
            "daily_trades": int(self.daily_trades),
            "session_trades": int(self._session_trades),
            "win_rate": float(win_rate),
            "has_position": bool(self.position is not None),
            "pending_entry": bool(self.pending_entry is not None),
            "pending_exit": bool(self.pending_exit is not None),
            "termination_reason": str(termination_reason),
            "close_reason": str(close_result.close_reason.value) if close_result else "",
            "entry_allowed": bool(entry_allowed),
            "block_reason": str(block_reason),
            "entry_quality": float(entry_quality),
            "entry_quality_long": float(q_long),
            "entry_quality_short": float(q_short),
            "consecutive_losses": int(self.consecutive_losses),
            "consecutive_wins": int(self.consecutive_wins),
            "last_net_trade_pnl": float(close_result.net_pnl) if close_result else 0.0,
            "last_trade_risk_eur": float(close_result.initial_risk_eur) if close_result else 0.0,
            "last_trade_mae": float(close_result.mae) if close_result else 0.0,
            "last_trade_mfe": float(close_result.mfe) if close_result else 0.0,
            "last_trade_bars": int(close_result.bars_held) if close_result else 0,
            "reward_components": dict(self._last_reward_components) if trade_closed else {},
            "episode_trade_count": len(self._episode_trade_results),
            "domain_randomization": {
                "spread_mult": float(self._episode_spread_mult),
                "slippage_mult": float(self._episode_slip_mult),
                "latency_bars": int(self._episode_latency_bars),
                "vol_scale": float(self._episode_vol_scale),
            },
        }
        info.update(self._curriculum_step_metadata())

        # Add full episode stats on episode end
        # NOTE: Curriculum recording is handled ONLY by CurriculumEnvWrapper
        # to avoid double-counting episodes
        if terminated or truncated:
            info["episode_stats"] = self.get_episode_stats()

        return obs, reward, terminated, truncated, info

    # ---------------------------
    # Observation
    # ---------------------------

    def _get_observation(self) -> np.ndarray:
        if self.obs_builder is not None:
            return self._build_observation_with_builder()
        return self._fallback_observation()

    def _build_observation_with_builder(self) -> np.ndarray:
        inst = self.instruments[0]
        market_data = self._prepare_market_data(inst)
        expert_signals = self._prepare_expert_signals(inst)
        committee_state = self._prepare_committee_state(expert_signals)
        risk_state = self._prepare_risk_state()
        memory_state = self._prepare_memory_state(inst)
        account_state = self._prepare_account_state(inst)
        trading_mode_state = self._prepare_trading_mode_state(inst)
        world_model_state = self._prepare_world_model_state(inst, expert_signals, committee_state)

        assert self.obs_builder is not None
        obs = self.obs_builder.build(
            market_data=market_data,
            expert_signals=expert_signals,
            committee_state=committee_state,
            risk_state=risk_state,
            memory_state=memory_state,
            account_state=account_state,
            world_model_state=world_model_state,
            trading_mode_state=trading_mode_state,
        )
        return obs

    def _prepare_market_data(self, instrument: str) -> Dict[str, Any]:
        o = self._get_ohlcv(instrument, lookback=120)
        if not o or len(o.get("close", [])) < 2:
            return {"M15": {"close": [0.0], "high": [0.0], "low": [0.0], "open": [0.0], "volume": [1.0]}}

        m15 = {
            "close": o["close"].tolist(),
            "high": o["high"].tolist(),
            "low": o["low"].tolist(),
            "open": o["open"].tolist(),
            "volume": o["volume"].tolist(),
        }

        def agg(ratio: int, max_bars: int) -> Dict[str, Any]:
            close = m15["close"]
            high = m15["high"]
            low = m15["low"]
            open_ = m15["open"]
            vol = m15["volume"]

            out_c, out_h, out_l, out_o, out_v = [], [], [], [], []
            for i in range(0, len(close), ratio):
                cc = close[i:i + ratio]
                hh = high[i:i + ratio]
                ll = low[i:i + ratio]
                oo = open_[i:i + ratio]
                vv = vol[i:i + ratio]
                if cc:
                    out_c.append(cc[-1])
                    out_h.append(max(hh))
                    out_l.append(min(ll))
                    out_o.append(oo[0] if oo else cc[0])
                    out_v.append(float(np.sum(vv)) if vv else 1.0)

            return {
                "close": out_c[-max_bars:] or [0.0],
                "high": out_h[-max_bars:] or [0.0],
                "low": out_l[-max_bars:] or [0.0],
                "open": out_o[-max_bars:] or [0.0],
                "volume": out_v[-max_bars:] or [1.0],
            }

        return {"M15": m15, "H1": agg(4, 50), "H4": agg(16, 30), "D1": agg(96, 20)}

    def _prepare_expert_signals(self, instrument: str) -> Dict[str, Any]:
        """
        Prepare expert signals with per-step caching.
        
        Caching prevents:
        - Redundant compute (called by entry quality, observation, entry context)
        - Subtle inconsistencies if any state changes mid-step
        """
        # Return cached result if available (set in step())
        cache = getattr(self, "_step_expert_signals_cache", None)
        if cache is not None:
            return cache
        
        o = self._get_ohlcv(instrument, lookback=60)
        if not o or len(o.get("close", [])) < 20:
            result = {"experts": {}, "market": {"regime": "unknown", "regime_strength": 0.5}}
            self._step_expert_signals_cache = result
            return result

        close = np.asarray(o["close"], dtype=np.float64)
        high = np.asarray(o.get("high", o["close"]), dtype=np.float64)
        low = np.asarray(o.get("low", o["close"]), dtype=np.float64)
        open_ = np.asarray(o.get("open", o["close"]), dtype=np.float64)

        fast = float(np.mean(close[-10:]))
        slow = float(np.mean(close[-20:]))
        trend_dir = "bullish" if fast > slow else "bearish"
        trend_strength = float(np.clip(abs(fast - slow) / max(abs(slow), 1e-8) * 50.0, 0.0, 1.0))

        # Compute BASIC market structure: S/R proximity (institutional-grade)
        near_support, near_resistance = self._compute_market_structure_signals(high, low, close)
        
        # Compute ADVANCED market structure signals (with open for better OB detection)
        advanced_structure = self._compute_advanced_market_structure(high, low, close, open_)

        rsi = self._compute_rsi(close, 14)
        if rsi > 50:
            mom_dir = "bullish"
            mom_strength = float(np.clip((rsi - 50) / 50, 0.0, 1.0))
        else:
            mom_dir = "bearish" if rsi < 50 else "neutral"
            mom_strength = float(np.clip((50 - rsi) / 50, 0.0, 1.0))
        
        # Compute MOMENTUM DIVERGENCE (RSI divergence detection)
        divergence_signal, overbought, oversold = self._compute_momentum_signals(close, rsi)

        vol_proxy = self._atr_vol_proxy(instrument)
        
        # Compute THEME/REGIME signals
        volatility_regime, risk_regime = self._compute_theme_signals(vol_proxy, trend_dir)
        
        if vol_proxy < 0.3:
            theme_dir, theme_strength = "neutral", 0.4
        elif vol_proxy > 0.7:
            theme_dir, theme_strength = "bearish", 0.7
        else:
            theme_dir, theme_strength = ("bullish", 0.5) if trend_dir == "bullish" else ("bearish", 0.5)

        result = {
            "experts": {
                "trend": {
                    "direction": trend_dir,
                    "score": trend_strength,
                    "confidence": 0.6 + 0.3 * trend_strength,
                    "proposal": {
                        "near_support": near_support,
                        "near_resistance": near_resistance,
                        # Advanced structure signals
                        "structure_trend": advanced_structure["structure_trend"],
                        "structure_strength": advanced_structure["structure_strength"],
                        "bos_signal": advanced_structure["bos_signal"],
                        "liquidity_above": advanced_structure["liquidity_above"],
                        "liquidity_below": advanced_structure["liquidity_below"],
                        "order_block_bull": advanced_structure["order_block_bull"],
                        "order_block_bear": advanced_structure["order_block_bear"],
                    },
                },
                "momentum": {
                    "direction": mom_dir,
                    "score": mom_strength,
                    "confidence": 0.5 + 0.4 * mom_strength,
                    "proposal": {
                        # Divergence: "bullish", "bearish", or None
                        "divergence_signal": divergence_signal,
                        "overbought": overbought,  # RSI > 70
                        "oversold": oversold,      # RSI < 30
                        "rsi_value": float(rsi),
                    },
                },
                "theme": {
                    "direction": theme_dir,
                    "score": float(theme_strength),
                    "confidence": 0.5,
                    "proposal": {
                        "volatility_regime": volatility_regime,  # "low", "normal", "high"
                        "risk_regime": risk_regime,              # "risk_on", "risk_off", "neutral"
                        "vol_score": float(vol_proxy),
                    },
                },
                "seasonality": {"direction": "neutral", "score": 0.0, "confidence": 0.5},
            },
            "market": {"regime": volatility_regime, "regime_strength": float(vol_proxy)},
        }
        
        # Cache for reuse within same step
        self._step_expert_signals_cache = result
        return result

    def _compute_momentum_signals(self, close: np.ndarray, rsi: float) -> Tuple[Optional[str], float, float]:
        """
        Compute momentum divergence and overbought/oversold signals.
        
        Returns:
            divergence_signal: "bullish", "bearish", or None
            overbought: 0.0-1.0 (how overbought, 1.0 = RSI at 100)
            oversold: 0.0-1.0 (how oversold, 1.0 = RSI at 0)
        """
        divergence_signal: Optional[str] = None
        overbought = 0.0
        oversold = 0.0
        
        # Overbought/Oversold
        if rsi > 70:
            overbought = float((rsi - 70) / 30)  # 0 at 70, 1 at 100
        if rsi < 30:
            oversold = float((30 - rsi) / 30)    # 0 at 30, 1 at 0
        
        # Divergence detection (simplified)
        if len(close) < 20:
            return divergence_signal, overbought, oversold
        
        # Look for divergence in last 20 bars
        lookback = 20
        recent_prices = close[-lookback:]
        
        # Compute RSI for recent bars (simplified)
        rsi_values = []
        for i in range(lookback):
            start_idx = max(0, len(close) - lookback - 14 + i)
            end_idx = len(close) - lookback + i + 1
            if end_idx > start_idx + 14:
                rsi_val = self._compute_rsi(close[start_idx:end_idx], 14)
                rsi_values.append(rsi_val)
        
        if len(rsi_values) < 10:
            return divergence_signal, overbought, oversold
        
        # Find local extremes
        price_min_idx = int(np.argmin(recent_prices[-10:]))
        price_max_idx = int(np.argmax(recent_prices[-10:]))
        
        rsi_recent = rsi_values[-10:] if len(rsi_values) >= 10 else rsi_values
        
        # Bullish divergence: price makes lower low, RSI makes higher low
        if price_min_idx > 5:  # Recent low
            prev_price_low = np.min(recent_prices[:5])
            curr_price_low = recent_prices[price_min_idx]
            if len(rsi_recent) > price_min_idx and len(rsi_recent) > 5:
                prev_rsi_low = np.min(rsi_recent[:5])
                curr_rsi_low = rsi_recent[price_min_idx] if price_min_idx < len(rsi_recent) else rsi_recent[-1]
                if curr_price_low < prev_price_low and curr_rsi_low > prev_rsi_low:
                    divergence_signal = "bullish"
        
        # Bearish divergence: price makes higher high, RSI makes lower high
        if price_max_idx > 5:  # Recent high
            prev_price_high = np.max(recent_prices[:5])
            curr_price_high = recent_prices[price_max_idx]
            if len(rsi_recent) > price_max_idx and len(rsi_recent) > 5:
                prev_rsi_high = np.max(rsi_recent[:5])
                curr_rsi_high = rsi_recent[price_max_idx] if price_max_idx < len(rsi_recent) else rsi_recent[-1]
                if curr_price_high > prev_price_high and curr_rsi_high < prev_rsi_high:
                    divergence_signal = "bearish"
        
        return divergence_signal, overbought, oversold

    def _compute_theme_signals(self, vol_proxy: float, trend_dir: str) -> Tuple[str, str]:
        """
        Compute theme/regime signals matching ThemeExpert output.
        
        Returns:
            volatility_regime: "low", "normal", "high"
            risk_regime: "risk_on", "risk_off", "neutral"
        """
        # Volatility regime
        if vol_proxy < 0.3:
            volatility_regime = "low"
        elif vol_proxy > 0.7:
            volatility_regime = "high"
        else:
            volatility_regime = "normal"
        
        # Risk regime (simplified: low vol + bullish = risk_on)
        if volatility_regime == "low" and trend_dir == "bullish":
            risk_regime = "risk_on"
        elif volatility_regime == "high" or trend_dir == "bearish":
            risk_regime = "risk_off"
        else:
            risk_regime = "neutral"
        
        return volatility_regime, risk_regime

    def _capture_entry_context(self, instrument: str) -> Dict[str, Any]:
        """
        Capture market structure context at trade entry for reward calculation.
        
        Returns dict with:
        - near_support, near_resistance: S/R proximity (0-1)
        - structure_trend: -1 (LL/LH) to +1 (HH/HL)
        - bos_signal: -1 (bearish BOS) to +1 (bullish BOS)
        - order_block_bull, order_block_bear: 0-1 proximity
        - divergence_signal: "bullish", "bearish", or None
        - overbought, oversold: 0-1 intensity
        - volatility_regime, risk_regime: string regime labels
        """
        # Get expert signals which contain all the computed values
        expert_signals = self._prepare_expert_signals(instrument)
        
        context: Dict[str, Any] = {}
        
        # Extract from trend expert (market structure)
        experts = expert_signals.get("experts", {})
        trend = experts.get("trend", {})
        trend_proposal = trend.get("proposal", {}) if isinstance(trend, dict) else {}
        
        context["near_support"] = float(trend_proposal.get("near_support", 0.0))
        context["near_resistance"] = float(trend_proposal.get("near_resistance", 0.0))
        context["structure_trend"] = float(trend_proposal.get("structure_trend", 0.0))
        context["structure_strength"] = float(trend_proposal.get("structure_strength", 0.0))
        context["bos_signal"] = float(trend_proposal.get("bos_signal", 0.0))
        context["liquidity_above"] = float(trend_proposal.get("liquidity_above", 0.0))
        context["liquidity_below"] = float(trend_proposal.get("liquidity_below", 0.0))
        context["order_block_bull"] = float(trend_proposal.get("order_block_bull", 0.0))
        context["order_block_bear"] = float(trend_proposal.get("order_block_bear", 0.0))
        
        # Extract from momentum expert (divergence, OB/OS)
        momentum = experts.get("momentum", {})
        momentum_proposal = momentum.get("proposal", {}) if isinstance(momentum, dict) else {}
        
        context["divergence_signal"] = momentum_proposal.get("divergence_signal")
        context["overbought"] = float(momentum_proposal.get("overbought", 0.0))
        context["oversold"] = float(momentum_proposal.get("oversold", 0.0))
        context["rsi_value"] = float(momentum_proposal.get("rsi_value", 50.0))
        
        # Extract from theme expert (regime)
        theme = experts.get("theme", {})
        theme_proposal = theme.get("proposal", {}) if isinstance(theme, dict) else {}
        
        context["volatility_regime"] = theme_proposal.get("volatility_regime", "normal")
        context["risk_regime"] = theme_proposal.get("risk_regime", "neutral")
        context["vol_score"] = float(theme_proposal.get("vol_score", 0.5))
        
        # Add spread percentile for regime tracking (Phase 2.2)
        # Calculate spread percentile relative to recent history
        context["spread_percentile"] = self._compute_spread_percentile(instrument)
        
        return context
    
    def _compute_spread_percentile(self, instrument: str) -> float:
        """Compute current spread as percentile of recent spread history."""
        try:
            df = self.data[instrument]["M15"]
            idx = self.current_step
            
            # Look back up to 200 bars for spread history
            lookback = min(200, idx)
            if lookback < 20:
                return 0.5  # Not enough history, assume median
            
            # Get spread history
            if "spread" in df.columns:
                spread_history = df["spread"].iloc[idx - lookback:idx].values
                current_spread = df["spread"].iloc[idx]
            else:
                # Estimate from bid/ask or use a proxy
                return 0.5
            
            # Calculate percentile
            spread_history = spread_history[~np.isnan(spread_history)]
            if len(spread_history) < 10:
                return 0.5
            
            percentile = float(np.sum(spread_history <= current_spread) / len(spread_history))
            return percentile
        except Exception:
            return 0.5  # Default to median on any error

    # ---------------------------
    # ATR + Pivots + Clustering Helpers (Institutional-Grade)
    # ---------------------------

    def _structure_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """Simple ATR for market structure calculations (robust scale measure)."""
        n = len(close)
        if n < period + 2:
            return float(max(np.mean(high - low), 1e-8))

        h = high[-(period + 1):]
        l = low[-(period + 1):]
        c = close[-(period + 1):]

        prev_c = c[:-1]
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - prev_c), np.abs(l[1:] - prev_c)))
        atr = float(np.mean(tr))
        return float(max(atr, 1e-8))

    def _find_fractal_pivots(
        self,
        high: np.ndarray,
        low: np.ndarray,
        left: int = 3,
        right: int = 3,
    ) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """
        Fractal pivots using only past data up to current bar.
        Returns lists of (index, price) in ascending index order.
        """
        n = len(high)
        if n < left + right + 3:
            return [], []

        piv_hi: List[Tuple[int, float]] = []
        piv_lo: List[Tuple[int, float]] = []

        for i in range(left, n - right):
            window_h = high[i - left : i + right + 1]
            window_l = low[i - left : i + right + 1]

            hi = high[i]
            lo = low[i]

            # Strict: require uniqueness to reduce duplicates in flat markets
            if hi == np.max(window_h) and np.sum(window_h == hi) == 1:
                piv_hi.append((i, float(hi)))
            if lo == np.min(window_l) and np.sum(window_l == lo) == 1:
                piv_lo.append((i, float(lo)))

        return piv_hi, piv_lo

    def _cluster_levels_1d(self, levels: List[float], eps: float) -> List[Tuple[float, int]]:
        """
        Cluster 1D levels. Returns list of (cluster_level_avg, count).
        """
        if not levels:
            return []
        xs = sorted(levels)
        clusters: List[Tuple[float, int]] = []

        bucket = [xs[0]]
        for x in xs[1:]:
            if abs(x - bucket[-1]) <= eps:
                bucket.append(x)
            else:
                clusters.append((float(np.mean(bucket)), len(bucket)))
                bucket = [x]
        clusters.append((float(np.mean(bucket)), len(bucket)))
        return clusters

    def _count_rejections(
        self,
        level: float,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        *,
        side: str,                 # "support" or "resistance"
        eps_touch: float,
        eps_break: float,
        move_away: float,
        fwd: int = 3,
    ) -> int:
        """
        Count validated rejections:
        - price touches near the level
        - does not break beyond eps_break
        - then moves away by move_away within fwd bars
        """
        n = len(closes)
        if n < fwd + 2:
            return 0

        rej = 0
        for i in range(0, n - fwd - 1):
            touched = (abs(highs[i] - level) <= eps_touch) or (abs(lows[i] - level) <= eps_touch)
            if not touched:
                continue

            f_hi = float(np.max(highs[i+1 : i+1+fwd]))
            f_lo = float(np.min(lows[i+1 : i+1+fwd]))
            f_cl = closes[i+1 : i+1+fwd]

            if side == "support":
                # Broken support if forward lows pierce below level - eps_break
                if f_lo < level - eps_break:
                    continue
                # Rejection if price closes above level + move_away at least once
                if np.any(f_cl > (level + move_away)):
                    rej += 1
            else:
                # Broken resistance if forward highs pierce above level + eps_break
                if f_hi > level + eps_break:
                    continue
                # Rejection if price closes below level - move_away at least once
                if np.any(f_cl < (level - move_away)):
                    rej += 1

        return rej

    # ---------------------------
    # Institutional-Grade S/R Detection
    # ---------------------------

    def _compute_market_structure_signals(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> Tuple[float, float]:
        """
        Institutional-grade S/R proximity with rejection validation:
        - Pivots via fractals (not arbitrary swings)
        - Cluster by ATR-scaled epsilon (not fixed %)
        - Validate by rejection count (actual bounce behavior)
        - Proximity score uses exp decay + strength weighting

        Returns: near_support, near_resistance in [0, 1]
        """
        if len(close) < 60:
            return 0.0, 0.0

        current_price = float(close[-1])
        if current_price <= 0:
            return 0.0, 0.0

        lookback = min(180, len(close))
        h = high[-lookback:]
        l = low[-lookback:]
        c = close[-lookback:]

        atr = self._structure_atr(h, l, c, period=14)

        # ATR-scaled epsilons (robust across regimes and instruments)
        eps_cluster = max(0.15 * atr, current_price * 0.0010)   # cluster bands
        eps_touch   = max(0.10 * atr, current_price * 0.0008)   # what counts as a "touch"
        eps_break   = max(0.20 * atr, current_price * 0.0012)   # break buffer
        move_away   = max(0.35 * atr, current_price * 0.0015)   # meaningful reaction
        eps_prox    = max(0.25 * atr, current_price * 0.0012)   # proximity decay scale

        piv_hi, piv_lo = self._find_fractal_pivots(h, l, left=3, right=3)

        # Candidate raw levels
        res_levels = [p for _, p in piv_hi]
        sup_levels = [p for _, p in piv_lo]

        res_clusters = self._cluster_levels_1d(res_levels, eps_cluster)
        sup_clusters = self._cluster_levels_1d(sup_levels, eps_cluster)

        # Validate clusters by rejection behavior
        validated_res: List[Tuple[float, float]] = []  # (level, strength)
        for level, base_ct in res_clusters:
            rej = self._count_rejections(level, h, l, c, side="resistance",
                                         eps_touch=eps_touch, eps_break=eps_break,
                                         move_away=move_away, fwd=3)
            # Strength: blend touches + rejections (rejections matter more)
            strength = 0.35 * min(base_ct / 4.0, 1.0) + 0.65 * min(rej / 3.0, 1.0)
            if rej >= 1 and (base_ct + rej) >= 3:
                validated_res.append((level, float(np.clip(strength, 0.0, 1.0))))

        validated_sup: List[Tuple[float, float]] = []
        for level, base_ct in sup_clusters:
            rej = self._count_rejections(level, h, l, c, side="support",
                                         eps_touch=eps_touch, eps_break=eps_break,
                                         move_away=move_away, fwd=3)
            strength = 0.35 * min(base_ct / 4.0, 1.0) + 0.65 * min(rej / 3.0, 1.0)
            if rej >= 1 and (base_ct + rej) >= 3:
                validated_sup.append((level, float(np.clip(strength, 0.0, 1.0))))

        # Keep strongest few
        validated_res.sort(key=lambda x: -x[1])
        validated_sup.sort(key=lambda x: -x[1])
        validated_res = validated_res[:4]
        validated_sup = validated_sup[:4]

        def prox_score(level: float, strength: float, side: str) -> float:
            dist = abs(current_price - level)
            if dist > 3.0 * eps_prox:
                return 0.0

            # Side constraint: don't call it "near support" if price is meaningfully below it, etc.
            if side == "support" and current_price < (level - eps_touch):
                return 0.0
            if side == "resistance" and current_price > (level + eps_touch):
                return 0.0

            # Smooth proximity with exponential decay
            p = float(np.exp(-dist / max(eps_prox, 1e-8)))
            return float(np.clip(p * (0.5 + 0.5 * strength), 0.0, 1.0))

        near_support = 0.0
        for level, strength in validated_sup:
            near_support = max(near_support, prox_score(level, strength, "support"))

        near_resistance = 0.0
        for level, strength in validated_res:
            near_resistance = max(near_resistance, prox_score(level, strength, "resistance"))

        return float(near_support), float(near_resistance)

    def _compute_advanced_market_structure(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        open_: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Institutional-grade advanced market structure:
        - structure_trend/strength from last pivot sequences (ATR-aware)
        - BOS/CHOCH using close breaks of last pivots
        - liquidity pools as equal-high/low clusters
        - order blocks as displacement-based, unmitigated zones
        
        Returns dict with multiple structure signals.
        """
        out = {
            "structure_trend": 0.0,
            "structure_strength": 0.0,
            "bos_signal": 0.0,
            "liquidity_above": 0.0,
            "liquidity_below": 0.0,
            "order_block_bull": 0.0,
            "order_block_bear": 0.0,
        }

        if len(close) < 80:
            return out

        current_price = float(close[-1])
        if current_price <= 0:
            return out

        lookback = min(220, len(close))
        h = np.asarray(high[-lookback:], dtype=np.float64)
        l = np.asarray(low[-lookback:], dtype=np.float64)
        c = np.asarray(close[-lookback:], dtype=np.float64)

        atr = self._structure_atr(h, l, c, period=14)

        eps_pivot_break = max(0.25 * atr, current_price * 0.0012)
        eps_liq = max(0.15 * atr, current_price * 0.0010)
        eps_ob_prox = max(0.30 * atr, current_price * 0.0015)

        piv_hi, piv_lo = self._find_fractal_pivots(h, l, left=3, right=3)

        # --- 1) Structure trend/strength from last pivots (HH/HL vs LL/LH but ATR-aware)
        if len(piv_hi) >= 2 and len(piv_lo) >= 2:
            (i_h1, h1), (i_h2, h2) = piv_hi[-2], piv_hi[-1]
            (i_l1, l1), (i_l2, l2) = piv_lo[-2], piv_lo[-1]

            # Determine classical structure with ATR buffer for noise filtering
            hh = h2 > h1 + 0.05 * atr
            hl = l2 > l1 + 0.05 * atr
            ll = l2 < l1 - 0.05 * atr
            lh = h2 < h1 - 0.05 * atr

            if hh and hl:
                trend = 1.0
            elif ll and lh:
                trend = -1.0
            else:
                trend = 0.0

            # Strength: how large the pivot changes are relative to ATR
            dh = abs(h2 - h1) / max(atr, 1e-8)
            dl = abs(l2 - l1) / max(atr, 1e-8)
            strength = float(np.clip(0.5 * (dh + dl) / 2.0, 0.0, 1.0))

            out["structure_trend"] = float(trend)
            out["structure_strength"] = float(strength)

        # --- 2) BOS / CHOCH (use CLOSE break for signal stability)
        if len(piv_hi) >= 1 and len(piv_lo) >= 1:
            last_hi = piv_hi[-1][1]
            last_lo = piv_lo[-1][1]

            # Directional break magnitude normalized by ATR
            if c[-1] > last_hi + eps_pivot_break:
                mag = (c[-1] - (last_hi + eps_pivot_break)) / max(atr, 1e-8)
                out["bos_signal"] = float(np.clip(mag, 0.0, 1.0))
            elif c[-1] < last_lo - eps_pivot_break:
                mag = ((last_lo - eps_pivot_break) - c[-1]) / max(atr, 1e-8)
                out["bos_signal"] = float(-np.clip(mag, 0.0, 1.0))

        # --- 3) Liquidity pools = equal-high/low clusters
        hi_levels = [p for _, p in piv_hi]
        lo_levels = [p for _, p in piv_lo]
        hi_clusters = self._cluster_levels_1d(hi_levels, eps_liq)
        lo_clusters = self._cluster_levels_1d(lo_levels, eps_liq)

        def liq_score(level: float, count: int) -> float:
            if count < 2:
                return 0.0
            dist = abs(level - current_price)
            # Ignore far pools
            if dist > 3.0 * atr:
                return 0.0
            prox = float(np.exp(-dist / max(1.5 * atr, 1e-8)))
            depth = float(np.clip((count - 1) / 2.0, 0.0, 1.0))
            return float(np.clip(prox * depth, 0.0, 1.0))

        best_above = 0.0
        for lvl, ct in hi_clusters:
            if lvl > current_price:
                best_above = max(best_above, liq_score(lvl, ct))
        out["liquidity_above"] = float(best_above)

        best_below = 0.0
        for lvl, ct in lo_clusters:
            if lvl < current_price:
                best_below = max(best_below, liq_score(lvl, ct))
        out["liquidity_below"] = float(best_below)

        # --- 4) Order blocks: displacement-based, unmitigated
        # Displacement: large body relative to ATR
        # Bullish OB: last bearish candle before bullish displacement
        # Bearish OB: last bullish candle before bearish displacement
        if len(c) >= 20:
            # Use provided open if available, otherwise approximate with prev close
            if open_ is not None and len(open_) >= lookback:
                approx_open = np.asarray(open_[-lookback:], dtype=np.float64)
            else:
                approx_open = np.concatenate([[c[0]], c[:-1]])

            bull_ob_scores: List[float] = []
            bear_ob_scores: List[float] = []

            body = np.abs(c - approx_open)
            disp_thresh = 0.9 * atr  # "meaningful move" threshold

            for i in range(2, len(c) - 1):
                # Bullish displacement candle
                if (c[i] > approx_open[i]) and (body[i] >= disp_thresh) and (c[i] > c[i-1]):
                    # Previous candle bearish -> bullish OB zone = previous candle range
                    if c[i-1] < approx_open[i-1]:
                        ob_low = float(l[i-1])
                        ob_high = float(h[i-1])
                        # Unmitigated check: price has NOT traded back into zone after i
                        post_low = float(np.min(l[i+1:])) if (i + 1) < len(l) else float(l[-1])
                        mitigated = post_low <= ob_high
                        if not mitigated:
                            # Proximity score
                            dist = 0.0
                            if current_price < ob_low:
                                dist = ob_low - current_price
                            elif current_price > ob_high:
                                dist = current_price - ob_high
                            prox = float(np.exp(-dist / max(eps_ob_prox, 1e-8)))
                            bull_ob_scores.append(prox)

                # Bearish displacement candle
                if (c[i] < approx_open[i]) and (body[i] >= disp_thresh) and (c[i] < c[i-1]):
                    if c[i-1] > approx_open[i-1]:
                        ob_low = float(l[i-1])
                        ob_high = float(h[i-1])
                        post_high = float(np.max(h[i+1:])) if (i + 1) < len(h) else float(h[-1])
                        mitigated = post_high >= ob_low
                        if not mitigated:
                            dist = 0.0
                            if current_price < ob_low:
                                dist = ob_low - current_price
                            elif current_price > ob_high:
                                dist = current_price - ob_high
                            prox = float(np.exp(-dist / max(eps_ob_prox, 1e-8)))
                            bear_ob_scores.append(prox)

            out["order_block_bull"] = float(np.clip(max(bull_ob_scores) if bull_ob_scores else 0.0, 0.0, 1.0))
            out["order_block_bear"] = float(np.clip(max(bear_ob_scores) if bear_ob_scores else 0.0, 0.0, 1.0))

        return out

    def _prepare_committee_state(self, expert_signals: Dict[str, Any]) -> Dict[str, Any]:
        experts = expert_signals.get("experts", {}) if isinstance(expert_signals, dict) else {}
        dirs: List[float] = []
        wts: List[float] = []
        for _, sig in (experts.items() if isinstance(experts, dict) else []):
            if not isinstance(sig, dict):
                continue
            d = self._dir_sign(str(sig.get("direction", "neutral")))
            s = float(sig.get("score", 0.0) or 0.0)
            c = float(sig.get("confidence", 0.5) or 0.5)
            dirs.append(d * s)
            wts.append(c)

        signed_score = float(sum(d * w for d, w in zip(dirs, wts)) / max(sum(wts), 1e-8)) if wts else 0.0
        action_str = "long" if signed_score > 0.1 else ("short" if signed_score < -0.1 else "flat")

        nonzero_signs = [int(np.sign(d)) for d in dirs if abs(d) > 0.1]
        # MED-1 FIX: When no meaningful signals exist, use 0.5 (uncertain), not 1.0 (full agreement)
        if not nonzero_signs:
            agreement = 0.5
        else:
            agreement = 1.0 if len(set(nonzero_signs)) == 1 else 0.0

        return {
            "consensus_score": float(np.clip(abs(signed_score), 0.0, 1.0)),
            "score": float(np.clip(signed_score, -1.0, 1.0)),
            "action": action_str,
            "action_value": float(np.clip(signed_score, -1.0, 1.0)),
            "direction": action_str,
            "confidence": float(np.mean(wts)) if wts else 0.5,
            "agreement": float(agreement),
            "fragility": float(1.0 - agreement),
        }

    def _prepare_risk_state(self) -> Dict[str, Any]:
        current_dd, daily_dd = self._calc_dds()
        return {
            "current_drawdown": float(current_dd),
            "daily_drawdown": float(daily_dd),
            "max_drawdown_limit": float(self.config.max_drawdown_limit),
            "daily_drawdown_limit": float(self.config.daily_drawdown_limit),
            "trades_today": int(self.daily_trades),
            "max_trades_per_day": int(self.config.max_trades_per_day),
            "risk_per_trade": float(self.config.risk_per_trade_pct),
        }

    def _prepare_memory_state(self, instrument: str) -> Dict[str, Any]:
        recent_pnl = float(self.total_pnl)
        recent_trades = int(self.total_trades)
        recent_losses = int(self.total_trades - self.winning_trades)

        if recent_trades == 0:
            memory_gate = 1.0
        else:
            loss_ratio = recent_losses / max(recent_trades, 1)
            memory_gate = float(np.clip(1.0 - loss_ratio * 0.5, 0.3, 1.0))

            cur_dd, _ = self._calc_dds()
            if cur_dd > 0.03:
                memory_gate *= 0.8
            if cur_dd > 0.05:
                memory_gate *= 0.7

        danger_zone_count = 0
        if self._episode_trade_results:
            recent_losses_list = [r for r in self._episode_trade_results[-10:] if r.net_pnl < 0]
            danger_zone_count = len(recent_losses_list)

        return {
            "memory_gate": float(memory_gate),
            "risk_multiplier": float(memory_gate),
            "danger_zones": {"zone_count": int(danger_zone_count), "active": bool(danger_zone_count > 0)},
            "recent_performance": {
                "win_rate": float(self.winning_trades / max(self.total_trades, 1)),
                "total_pnl": float(recent_pnl),
                "consecutive_losses": int(self.consecutive_losses),
            },
        }

    def _prepare_account_state(self, instrument: str) -> Dict[str, Any]:
        cur_dd, _ = self._calc_dds()
        state: Dict[str, Any] = {
            "balance": float(self.balance),
            "equity": float(self.equity),
            "initial_balance": float(self.config.initial_balance),
            "current_drawdown": float(cur_dd),
            "win_rate": float(self.winning_trades / max(self.total_trades, 1)),
            "trades_today": int(self.daily_trades),
            "has_position": self.position is not None,
            "position_direction": 0.0,
            "position_size": 0.0,
            "unrealized_pnl": 0.0,
            "time_in_position": 0.0,
            "on_cooldown": 0.0,
        }

        if self.position is not None:
            bid, ask = self._get_step_bid_ask(instrument, self._get_price_mid(instrument), self._atr_vol_proxy(instrument))
            u = self._mark_unrealized_pnl_from_bid_ask(self.position, bid, ask)
            state["position_direction"] = 1.0 if self.position.direction == "long" else -1.0
            state["position_size"] = float(self.position.lot_size)
            state["unrealized_pnl"] = float(u)
            state["time_in_position"] = float(self.episode_bars - self.position.entry_bar)

        inst_dt = self._get_bar_dt(instrument)
        if inst_dt is not None and self._last_loss_dt is not None:
            mins = (inst_dt - self._last_loss_dt).total_seconds() / 60.0
            if mins < self.config.min_minutes_after_loss:
                state["on_cooldown"] = 1.0
        elif inst_dt is None and self._last_loss_step is not None:
            tfm = max(1, self._tf_minutes())
            post_loss_bars = int(np.ceil(self.config.min_minutes_after_loss / tfm))
            if (int(self.current_step) - int(self._last_loss_step)) < max(1, post_loss_bars):
                state["on_cooldown"] = 1.0

        return state

    def _prepare_trading_mode_state(self, instrument: str) -> Dict[str, Any]:
        cur_dd, _ = self._calc_dds()
        if cur_dd > 0.05:
            mode = "safe"
        elif cur_dd < 0.02 and self.total_pnl > 0:
            mode = "aggressive"
        else:
            mode = "normal"

        dt = self._get_bar_dt(instrument)
        can_trade = True
        if dt is not None:
            if self._is_weekend(dt) and not self.config.allow_weekend_holding:
                can_trade = False
            if self._in_no_new_trades_window(dt):
                can_trade = False
            if self._in_final_exit_window(dt):
                can_trade = False
            if self._at_or_after_hard_close(dt):
                can_trade = False

        vol_proxy = self._atr_vol_proxy(instrument)
        vol_state = "low" if vol_proxy < 0.3 else ("high" if vol_proxy > 0.7 else "normal")
        zone_type = "good" if vol_state == "normal" else ("bad" if vol_state == "high" else "hot")

        # Use step cache if available (called from step()), fallback to direct compute
        q_long = self._get_step_entry_quality(instrument, "long")
        q_short = self._get_step_entry_quality(instrument, "short")

        return {
            "trading_mode": mode,
            "regime_stability": float(1.0 - min(vol_proxy, 1.0)),
            "theme_transition": float(min(vol_proxy, 1.0)),
            "theme_strength": float(1.0 - abs(0.5 - min(vol_proxy, 1.0)) * 2.0),
            "regime_accuracy": 0.5,
            "risk_scaling_factor": float(1.0 + 0.5 * min(vol_proxy, 1.0)),
            "liquidity_score": float(1.0 - 0.5 * min(vol_proxy, 1.0)),
            "entry_timing": {
                "entry_allowed": bool(can_trade),
                "entry_quality_long": float(q_long),
                "entry_quality_short": float(q_short),
                "zone_type": zone_type,
                "vol_state": vol_state,
                "hour_normalized": float(dt.hour / 24.0) if dt else 0.5,
                "in_prime_window": 1.0 if (dt and self._in_prime_window(dt)) else 0.0,
            },
            "mode_stats": {"mode_effectiveness": 0.5},
        }

    def _prepare_world_model_state(self, instrument: str, expert_signals: Dict[str, Any], committee_state: Dict[str, Any]) -> Dict[str, Any]:
        o = self._get_ohlcv(instrument, lookback=120)
        if not o or len(o.get("close", [])) < 60:
            return self._default_world_model_state()

        close = np.asarray(o["close"], dtype=np.float64)
        high = np.asarray(o["high"], dtype=np.float64)
        low = np.asarray(o["low"], dtype=np.float64)

        period = min(14, len(close) - 1)
        if period >= 2:
            hl = high[-period:] - low[-period:]
            hc = np.abs(high[-period:] - close[-period-1:-1])
            lc = np.abs(low[-period:] - close[-period-1:-1])
            atr = float(np.mean(np.maximum(hl, np.maximum(hc, lc))))
        else:
            atr = float(np.std(close[-20:]))

        price = float(close[-1])
        norm_atr = atr / max(price, 1.0)

        price_changes = []
        for lookback in (5, 10, 20, 40):
            if len(close) > lookback:
                ret = np.log(close[-1] / close[-lookback-1])
                normalized = (ret / (norm_atr + 1e-8)) * 0.1
                scaled = float(np.clip(normalized * 10.0, -1.0, 1.0))
            else:
                scaled = 0.0
            price_changes.append(scaled)

        vol_base = float(np.clip(norm_atr * 50.0, 0.0, 1.0))
        volatility_predictions = [vol_base, vol_base * 0.92, vol_base * 0.85, vol_base * 0.78]

        fast = float(np.mean(close[-10:]))
        slow = float(np.mean(close[-30:])) if len(close) >= 30 else fast
        trend = (fast - slow) / (atr + 1e-8)

        returns = np.diff(close[-20:]) / close[-20:-1]
        vol = float(np.std(returns))
        vol_norm = vol / 0.01

        probs = np.array([0.2, 0.2, 0.4, 0.2])

        if vol_norm > 1.5:
            probs[3] += 0.3
            probs[2] -= 0.15
            probs[0] -= 0.075
            probs[1] -= 0.075

        if abs(trend) > 0.5:
            if trend > 0:
                probs[0] += 0.25
                probs[1] -= 0.1
            else:
                probs[1] += 0.25
                probs[0] -= 0.1
            probs[2] -= 0.15

        probs = np.clip(probs, 0.05, 0.8)
        probs = probs / probs.sum()
        predicted_regime = int(np.argmax(probs))

        signs = [np.sign(pc) for pc in price_changes if abs(pc) > 0.05]
        trend_agreement = abs(sum(signs)) / len(signs) if signs else 0.5

        vol_penalty = min(norm_atr * 30, 0.3)

        comm_conf = float(committee_state.get("confidence", 0.5))
        comm_agree = float(committee_state.get("agreement", 0.5))

        rsi = self._compute_rsi(close)
        rsi_penalty = abs(rsi - 50) / 100.0

        confidence = float(np.clip(
            0.25 * trend_agreement +
            0.25 * (1.0 - vol_penalty) +
            0.20 * comm_conf +
            0.15 * comm_agree +
            0.15 * (1.0 - rsi_penalty),
            0.2, 0.9
        ))

        weights = [0.4, 0.3, 0.2, 0.1]
        weighted_mom = sum(w * pc for w, pc in zip(weights, price_changes))
        bullish_prob = float(np.clip(0.5 + weighted_mom * 0.3, 0.1, 0.9))

        if len(close) >= 60:
            short_ret = np.diff(close[-15:]) / close[-15:-1]
            long_ret = np.diff(close[-60:]) / close[-60:-1]
            short_vol = float(np.std(short_ret))
            long_vol = float(np.std(long_ret))
            vol_ratio = short_vol / (long_vol + 1e-8)

            trend_changes = np.diff(np.sign(short_ret))
            trend_consistency = 1.0 - np.sum(np.abs(trend_changes)) / (2 * len(trend_changes))

            stability = float(np.clip(
                0.5 + 0.25 * (1.0 - min(vol_ratio, 2.0) / 2.0) + 0.25 * trend_consistency,
                0.2, 0.9
            ))
        else:
            stability = 0.5

        _ = expert_signals  # reserved for future use

        return {
            "model_confidence": confidence,
            "is_trained": True,
            "stability_score": stability,
            "market_predictions": {
                "latest_predictions": {
                    "price_changes": price_changes,
                    "volatility_predictions": volatility_predictions,
                    "predicted_regime": predicted_regime,
                    "regime_probabilities": probs.tolist(),
                    "confidence": confidence,
                    "model_trained": True,
                },
            },
            "scenario_generation": {
                "bullish_probability": bullish_prob,
                "scenarios": [
                    {"outcome": 1.0, "probability": bullish_prob},
                    {"outcome": -1.0, "probability": 1.0 - bullish_prob},
                ],
            },
        }

    def _default_world_model_state(self) -> Dict[str, Any]:
        return {
            "model_confidence": 0.5,
            "is_trained": True,
            "stability_score": 0.5,
            "market_predictions": {
                "latest_predictions": {
                    "price_changes": [0.0, 0.0, 0.0, 0.0],
                    "volatility_predictions": [0.5, 0.5, 0.5, 0.5],
                    "predicted_regime": 2,
                    "regime_probabilities": [0.25, 0.25, 0.25, 0.25],
                    "confidence": 0.5,
                    "model_trained": True,
                },
            },
            "scenario_generation": {
                "bullish_probability": 0.5,
                "scenarios": [
                    {"outcome": 1.0, "probability": 0.5},
                    {"outcome": -1.0, "probability": 0.5},
                ],
            },
        }

    def _fallback_observation(self) -> np.ndarray:
        return np.zeros(int(self.config.observation_size), dtype=np.float32)

    def render(self, mode: str = "human") -> None:
        pos_str = f"{self.position.direction}@{self.position.entry_price:.2f}" if self.position else "flat"
        dd = (self.peak_balance - self.equity) / max(self.peak_balance, 1.0) * 100.0
        dd = max(0.0, float(dd))
        wr = (self.winning_trades / max(self.total_trades, 1)) * 100.0
        print(
            f"Step {self.episode_step}: Balance=€{self.balance:.2f}, Equity=€{self.equity:.2f}, "
            f"DD={dd:.2f}%, Trades={self.total_trades}, WR={wr:.0f}%, Pos={pos_str}"
        )

    def close(self) -> None:
        pass

    # ---------------------------
    # Episode Analytics
    # ---------------------------

    def get_episode_stats(self) -> Dict[str, Any]:
        if not self._episode_trade_results:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_pnl": 0.0,
                "avg_r_multiple": 0.0,
                "avg_mae": 0.0,
                "avg_bars_held": 0.0,
                "exit_quality_distribution": {},
                "consecutive_losses": int(self.consecutive_losses),
                "consecutive_wins": int(self.consecutive_wins),
                "hit_max_consecutive_losses": self.consecutive_losses >= self.config.max_consecutive_losses,
            }

        results = self._episode_trade_results
        wins = [r for r in results if r.net_pnl > 0]
        losses = [r for r in results if r.net_pnl <= 0]

        exit_dist: Dict[str, int] = {}
        for r in results:
            key = r.close_reason.value
            exit_dist[key] = exit_dist.get(key, 0) + 1

        total_net = sum(r.net_pnl for r in results)
        avg_r = sum(r.net_pnl / max(r.initial_risk_eur, 1) for r in results) / len(results) if results else 0.0
        pf = (
            (sum(r.net_pnl for r in wins) / abs(sum(r.net_pnl for r in losses)))
            if losses and sum(r.net_pnl for r in losses) != 0
            else float("inf")
        )

        # Check if episode ended due to max consecutive losses constraint
        # This is tracked for curriculum consecutive_loss_breach_rate metric
        hit_max_consec_losses = self.consecutive_losses >= self.config.max_consecutive_losses
        
        # Build reward component breakdown for dashboard
        # Separate into positive (bonuses) and negative (penalties) components
        reward_components = {}
        for key, total in self._episode_reward_components.items():
            count = self._episode_reward_component_counts.get(key, 1)
            reward_components[key] = {
                "total": float(total),
                "count": count,
                "avg": float(total / count) if count > 0 else 0.0,
            }

        return {
            "total_trades": len(results),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": len(wins) / len(results) if results else 0.0,
            "total_pnl": float(total_net),
            "avg_pnl": float(total_net / len(results)) if results else 0.0,
            "avg_win": float(sum(r.net_pnl for r in wins) / len(wins)) if wins else 0.0,
            "avg_loss": float(sum(r.net_pnl for r in losses) / len(losses)) if losses else 0.0,
            "avg_r_multiple": float(avg_r),
            "avg_mae": float(sum(abs(r.mae) for r in results) / len(results)) if results else 0.0,
            "avg_mfe": float(sum(r.mfe for r in results) / len(results)) if results else 0.0,
            "avg_bars_held": float(sum(r.bars_held for r in results) / len(results)) if results else 0.0,
            "avg_entry_quality": float(sum(r.entry_quality for r in results) / len(results)) if results else 0.5,
            "exit_quality_distribution": exit_dist,
            "profit_factor": float(pf),
            "consecutive_losses": int(self.consecutive_losses),
            "consecutive_wins": int(self.consecutive_wins),
            "max_consecutive_losses_reached": int(self.max_consecutive_losses_reached),  # Peak during episode
            "hit_max_consecutive_losses": hit_max_consec_losses,
            "reward_components": reward_components,
            # Regime-tagged trade data for skill assessment
            "trades_with_regime": self._build_trades_with_regime(results),
        }

    def _build_trades_with_regime(self, results: List[TradeResult]) -> List[Dict[str, Any]]:
        """Build regime-tagged trade list for RegimeSkillAssessment integration."""
        trades_with_regime = []
        for r in results:
            entry_context = r.entry_context or {}
            
            # Extract regime info from entry_context if available
            vol_regime = entry_context.get("volatility_regime", "medium")
            risk_regime = entry_context.get("risk_regime", "neutral")
            
            # Derive trend regime from entry context or default
            trend_strength = float(entry_context.get("structure_trend", 0.0))
            if abs(trend_strength) > 0.5:
                trend_regime = "strong_trend"
            elif abs(trend_strength) > 0.2:
                trend_regime = "weak_trend"
            else:
                trend_regime = "ranging"
            
            # Derive session from entry timestamp
            session_regime = "off_hours"
            if r.entry_dt:
                hour = r.entry_dt.hour
                if 0 <= hour < 7:
                    session_regime = "asian"
                elif 7 <= hour < 12:
                    session_regime = "london"
                elif 12 <= hour < 16:
                    session_regime = "overlap"
                elif 16 <= hour < 21:
                    session_regime = "ny"
            
            # Derive spread regime from entry context
            spread_percentile = float(entry_context.get("spread_percentile", 0.5))
            if spread_percentile < 0.3:
                spread_regime = "tight"
            elif spread_percentile > 0.7:
                spread_regime = "wide"
            else:
                spread_regime = "normal"
            
            trades_with_regime.append({
                "pnl": float(r.net_pnl),
                "r_multiple": float(r.net_pnl / max(r.initial_risk_eur, 1.0)),
                "is_winner": r.net_pnl > 0,
                "bars_held": int(r.bars_held),
                "mae": float(r.mae),
                "mfe": float(r.mfe),
                "entry_quality": float(r.entry_quality),
                "exit_type": r.close_reason.value,
                "volatility_regime": vol_regime,
                "trend_regime": trend_regime,
                "session_regime": session_regime,
                "spread_regime": spread_regime,
            })
        
        return trades_with_regime
