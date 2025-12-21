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
    from envs.curriculum_config import CurriculumStage, CurriculumStageConfig
    CURRICULUM_AVAILABLE = True
except Exception:
    _CurriculumManager = None  # type: ignore
    CurriculumStage = None  # type: ignore
    CurriculumStageConfig = None  # type: ignore
    CURRICULUM_AVAILABLE = False

try:
    from modules.meta.ppo_observation_builder import PPOObservationBuilder, PPO_OBS_SIZE, PPO_OBS_VERSION
    OBS_BUILDER_AVAILABLE = True
except Exception:
    PPOObservationBuilder = None  # type: ignore
    PPO_OBS_SIZE = 64
    PPO_OBS_VERSION = "5.0"
    OBS_BUILDER_AVAILABLE = False

import logging
logger = logging.getLogger("prop_firm_env")
logging.basicConfig(level=logging.INFO)

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

    # Exit quality modifiers
    exit_quality_enabled: bool = True
    trailing_stop_bonus: float = 0.15
    agent_close_bonus: float = 0.05
    hard_stop_penalty: float = 0.15
    risk_liquidation_penalty: float = 0.30

    # Truncation handling
    truncation_winner_discount: float = 0.30
    truncation_loser_extra_penalty: float = 0.15

    # Entry quality integration
    entry_quality_integration: bool = True
    entry_quality_weight: float = 0.2

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

    # Anti-churn
    anti_churn_enabled: bool = True
    daily_trade_soft_limit: int = 10
    churn_penalty_per_trade: float = 0.02

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

    # Entry-quality gate
    entry_quality_gate_enabled: bool = True
    entry_quality_threshold: float = 0.35

    # PPO env
    observation_size: int = 64
    max_steps_per_episode: int = 2000
    gamma: float = 0.95

    # Reward configuration
    reward: RewardConfig = field(default_factory=RewardConfig)

    # Legacy reward params
    reward_scale: float = 10.0
    risk_penalty_scale: float = 2.0
    quality_bonus_scale: float = 0.5
    blocked_action_penalty: float = 0.02
    hard_block_penalty: float = 0.02

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
        # Keep RewardConfig in sync with legacy knobs
        self.reward.reward_scale = float(self.reward_scale)
        self.reward.dd_penalty_scale = float(self.risk_penalty_scale)
        self.reward.hard_block_penalty = float(self.hard_block_penalty)
        self.reward.soft_block_penalty = float(self.blocked_action_penalty)

    def __post_init__(self) -> None:
        self.sync_reward_from_legacy()

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

            self.sync_reward_from_legacy()
        except Exception:
            pass


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
        if curriculum_manager is not None and CURRICULUM_AVAILABLE:
            self.curriculum = curriculum_manager
            self._last_stage_name = getattr(self.curriculum.current_stage, "name", "")
            self._last_stage_epoch = int(getattr(self.curriculum, "current_stage_epoch", 0))

        self.data = data_dict
        self.instruments = [i for i in self.config.instruments if i in self.data]
        if not self.instruments:
            self.instruments = list(self.data.keys())[:1]
        if not self.instruments:
            raise ValueError("No valid instruments found in data")

        # Keep obs size consistent with builder contract
        try:
            self.config.observation_size = int(PPO_OBS_SIZE)
        except Exception:
            pass

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
            print(f"[OBS] Using PPOObservationBuilder v{PPO_OBS_VERSION}")
        else:
            print("[OBS] PPOObservationBuilder not available -> fallback observation")

        self._min_data_len = self._get_min_data_length()

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
        self._opportunity_history: List[float] = []
        self._action_history: List[int] = []
        self._avg_vol: Optional[float] = None
        self._episode_trade_results: List[TradeResult] = []
        self._last_reward_components: Dict[str, float] = {}

        # Episode randomization state
        self._episode_spread_mult = 1.0
        self._episode_slip_mult = 1.0
        self._episode_latency_bars = 0
        self._episode_vol_scale = 1.0

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
                    except Exception:
                        pass
                continue
            if hasattr(target, k):
                try:
                    setattr(target, k, v)
                except Exception:
                    pass

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

            # keep legacy sync consistent
            self.config.sync_reward_from_legacy()
        except Exception:
            # Never let curriculum override application break training
            return

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
        Records the episode into CurriculumManager and triggers promote/demote decisions.
        Applies the NEW stage on the next reset (clean boundary).
        """
        if not self.curriculum:
            return

        try:
            # Ensure the schema curriculum manager expects exists
            if "episode_stats" not in info:
                info["episode_stats"] = self.get_episode_stats()

            # Record + possibly transition
            self.curriculum.record_episode_from_info(
                info=info,
                episode_reward=float(self._episode_return),
                episode_length=int(self.episode_step),
            )

            old_stage = getattr(self.curriculum.current_stage, "name", "")
            old_epoch = int(getattr(self.curriculum, "current_stage_epoch", 0))

            changed, new_stage = self.curriculum.update()

            # Update metadata for dashboard
            info["curriculum"] = self.curriculum.get_progress_report()

            if changed and new_stage is not None:
                new_stage_name = getattr(new_stage, "name", str(new_stage))
                info["curriculum_transition"] = {
                    "from_stage": old_stage,
                    "to_stage": new_stage_name,
                    "from_epoch": old_epoch,
                    "to_epoch": int(getattr(self.curriculum, "current_stage_epoch", 0)),
                }
                # Apply stage overrides on NEXT reset (avoids mixing stages within an episode)
                self._pending_stage_apply = True
        except Exception:
            return

    # ---------------------------
    # Action decoding / masking
    # ---------------------------

    def _get_min_data_length(self) -> int:
        m = float("inf")
        for inst in self.instruments:
            for _, df in self.data.get(inst, {}).items():
                m = min(m, len(df))
        return int(m) if m != float("inf") else 0

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

    def action_masks(self) -> np.ndarray:
        """
        Action mask for sb3-contrib MaskablePPO.

        RELAXED for training: Only mask actions that are physically impossible.
        Soft rules are penalized in step() to allow learning.
        """
        mask = np.ones(self._N_ACTIONS, dtype=np.bool_)

        inst = self.instruments[0]
        next_step = int(self.current_step + 1)

        latency_bars = int(getattr(self._episode_execution_cfg, "latency_bars", 0)) if self._episode_execution_cfg else 0
        fill_delay = 1 + max(0, latency_bars)

        last_idx = (self._min_data_len - 1)
        remaining_bars_next = last_idx - next_step
        can_fill_before_end = remaining_bars_next >= fill_delay

        has_position_or_pending = (self.position is not None) or (self.pending_entry is not None)
        can_enter = (not has_position_or_pending) and bool(can_fill_before_end)

        if not can_enter:
            mask[self._ACTION_LONG_START: self._ACTION_LONG_START + self._K] = False
            mask[self._ACTION_SHORT_START: self._ACTION_SHORT_START + self._K] = False

        if self.position is None and self.pending_exit is None:
            mask[self._ACTION_CLOSE] = False
        elif self.pending_exit is not None:
            mask[self._ACTION_CLOSE] = True
        else:
            mask[self._ACTION_CLOSE] = bool(can_fill_before_end)

        mask[self._ACTION_HOLD] = True
        return mask

    def get_action_mask(self) -> np.ndarray:
        """Backward-compatible alias."""
        return self.action_masks()

    # ---------------------------
    # Time helpers (Europe/Berlin)
    # ---------------------------

    def _tf_minutes(self) -> int:
        tf = (self.config.primary_timeframe or "M15").upper().strip()
        mapping = {
            "M1": 1, "M2": 2, "M3": 3, "M5": 5, "M10": 10, "M15": 15, "M30": 30,
            "H1": 60, "H2": 120, "H4": 240, "H6": 360, "H8": 480, "H12": 720,
            "D1": 1440,
        }
        return int(mapping.get(tf, 15))

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
        tf = (self.config.primary_timeframe or "M15").upper().strip()
        mapping = {
            "M1": 1440, "M2": 720, "M3": 480, "M5": 288, "M10": 144, "M15": 96, "M30": 48,
            "H1": 24, "H2": 12, "H4": 6, "H6": 4, "H8": 3, "H12": 2,
            "D1": 1,
        }
        return int(mapping.get(tf, 96))

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
                self.day_start_balance = self.balance
                self.daily_trades = 0
                self.daily_pnl = 0.0
            return

        cur_day = dt.date()
        if self._current_day != cur_day:
            self._current_day = cur_day
            self.day_start_balance = self.balance
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
        return float(df["close"].iloc[idx])

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

        out = {
            "open": df["open"].iloc[start:end].to_numpy(dtype=np.float64, copy=False),
            "high": df["high"].iloc[start:end].to_numpy(dtype=np.float64, copy=False),
            "low": df["low"].iloc[start:end].to_numpy(dtype=np.float64, copy=False),
            "close": df["close"].iloc[start:end].to_numpy(dtype=np.float64, copy=False),
        }
        if "volume" in df.columns:
            out["volume"] = df["volume"].iloc[start:end].to_numpy(dtype=np.float64, copy=False)
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
        current_dd = (self.peak_balance - self.equity) / max(self.peak_balance, 1.0)
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
                exit_modifier = cfg.agent_close_bonus if net_pnl > 0 else 0.0
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

        # 9) Anti-churn penalty
        if cfg.anti_churn_enabled and self.daily_trades > cfg.daily_trade_soft_limit:
            excess = self.daily_trades - cfg.daily_trade_soft_limit
            churn_pen = excess * cfg.churn_penalty_per_trade
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
            if (not self.config.allow_weekend_holding) and self._is_weekend(dt):
                return False, "weekend_block"
            if self._in_no_new_trades_window(dt):
                return False, "no_new_trades_window"
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

        self._opportunity_history = []
        self._action_history = []
        self._avg_vol = None
        self._episode_trade_results = []
        self._last_reward_components = {}

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

        buffer = 120
        max_start = max(buffer, self._min_data_len - self.config.max_steps_per_episode - buffer)
        if max_start > buffer:
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
        self._maybe_roll_day_session(dt)

        intent, size_mult = self._decode_action(int(action))

        mid = self._get_price_mid(inst)
        vol_proxy = self._atr_vol_proxy(inst)
        assert self._exec is not None

        bid, ask = self._get_step_bid_ask(inst, mid, vol_proxy)

        latency_bars = int(getattr(self._episode_execution_cfg, "latency_bars", 0)) if self._episode_execution_cfg else 0
        fill_delay = 1 + max(0, latency_bars)
        last_idx = (self._min_data_len - 1)
        remaining_bars = last_idx - int(self.current_step)
        can_fill_before_end = remaining_bars >= fill_delay

        reward = 0.0
        trade_closed = False
        close_result: Optional[TradeResult] = None

        # Mark-to-market
        pnl_u = 0.0
        if self.position is not None:
            pnl_u = self._mark_unrealized_pnl_from_bid_ask(self.position, bid, ask)
            self.equity = self.balance + pnl_u
            self.position.peak_pnl = max(self.position.peak_pnl, pnl_u)
            self.position.lowest_pnl = min(self.position.lowest_pnl, pnl_u)
        else:
            self.equity = self.balance

        self._update_peak_balance()
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
        if self.position is not None and not forced_close_now and can_fill_before_end:
            if intent == "close" and self.pending_exit is None:
                self.pending_exit = {"fill_step": self.current_step + fill_delay, "reason": CloseReason.AGENT_CLOSE.value}

        # Execute forced close
        if forced_close_now and self.position is not None:
            close_result = self._close_position_now(reason=close_reason_str, dt=dt, mid=mid, vol_proxy=vol_proxy)
            trade_closed = True
            current_dd, current_daily_dd = self._calc_dds()
            reward += self._compute_trade_reward(close_result, current_dd)

        # Execute scheduled exit
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
        entry_quality = self._compute_smart_entry_quality(inst, intent) if intent in ("long", "short") else 0.5

        entry_allowed = hard_ok
        block_reason = hard_block

        if attempted_entry and not can_fill_before_end:
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
                    "fill_step": self.current_step + fill_delay,
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
            close_result = self._close_position_now(
                reason=CloseReason.EPISODE_TRUNCATE.value, dt=dt, mid=mid, vol_proxy=vol_proxy
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

        # Compute qualities once (reuse)
        q_long = self._compute_smart_entry_quality(inst, "long")
        q_short = self._compute_smart_entry_quality(inst, "short")

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

        # Add full episode stats + curriculum logging on episode end
        if terminated or truncated:
            info["episode_stats"] = self.get_episode_stats()
            self._curriculum_on_episode_end(info)

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
        o = self._get_ohlcv(instrument, lookback=60)
        if not o or len(o.get("close", [])) < 20:
            return {"experts": {}, "market": {"regime": "unknown", "regime_strength": 0.5}}

        close = np.asarray(o["close"], dtype=np.float64)

        fast = float(np.mean(close[-10:]))
        slow = float(np.mean(close[-20:]))
        trend_dir = "bullish" if fast > slow else "bearish"
        trend_strength = float(np.clip(abs(fast - slow) / max(abs(slow), 1e-8) * 50.0, 0.0, 1.0))

        rsi = self._compute_rsi(close, 14)
        if rsi > 50:
            mom_dir = "bullish"
            mom_strength = float(np.clip((rsi - 50) / 50, 0.0, 1.0))
        else:
            mom_dir = "bearish" if rsi < 50 else "neutral"
            mom_strength = float(np.clip((50 - rsi) / 50, 0.0, 1.0))

        vol_proxy = self._atr_vol_proxy(instrument)
        if vol_proxy < 0.3:
            theme_dir, theme_strength = "neutral", 0.4
        elif vol_proxy > 0.7:
            theme_dir, theme_strength = "bearish", 0.7
        else:
            theme_dir, theme_strength = ("bullish", 0.5) if trend_dir == "bullish" else ("bearish", 0.5)

        return {
            "experts": {
                "trend": {"direction": trend_dir, "score": trend_strength, "confidence": 0.6 + 0.3 * trend_strength},
                "momentum": {"direction": mom_dir, "score": mom_strength, "confidence": 0.5 + 0.4 * mom_strength},
                "theme": {"direction": theme_dir, "score": float(theme_strength), "confidence": 0.5},
                "seasonality": {"direction": "neutral", "score": 0.0, "confidence": 0.5},
            },
            "market": {"regime": "unknown", "regime_strength": 0.5},
        }

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
        agreement = 1.0 if len(set(nonzero_signs)) <= 1 else 0.0

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

        q_long = self._compute_smart_entry_quality(instrument, "long")
        q_short = self._compute_smart_entry_quality(instrument, "short")

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
        }
