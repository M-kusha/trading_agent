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

JAN 2026 UPGRADE (this patch):
- Removes diamond inheritance of MarketStructureMixin (ExpertSignalsMixin already provides it)
- Fixes _get_ohlcv contract: adds timeframe param (mixin stability)
- Hardens quote caching: validates mid/vol per call
- Makes HTF aggregation ratios dynamic relative to primary_timeframe (M15 assumption removed)
- Computes episode start buffer from indicator lookbacks (prevents early-episode degenerate features)
"""

from __future__ import annotations

import copy
from collections import deque
import warnings
from datetime import datetime, date
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from pandas.api.types import is_numeric_dtype
from typing import Optional
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from zoneinfo import ZoneInfo

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------
# Optional curriculum support
# ---------------------------
if TYPE_CHECKING:
    from envs.curriculum import CurriculumManager

try:
    from envs.curriculum import CurriculumManager as _CurriculumManager
    from envs.curriculum import CurriculumStage, CurriculumStageConfig, DataDifficulty
    CURRICULUM_AVAILABLE = True
except Exception:
    _CurriculumManager = None  # type: ignore
    CurriculumStage = None  # type: ignore
    CurriculumStageConfig = None  # type: ignore
    DataDifficulty = None  # type: ignore
    CURRICULUM_AVAILABLE = False

# FAIL-LOUD: the observation builder is not optional.
#
# This import was previously wrapped in `except Exception`, which set
# PPO_OBS_SIZE = 90, OBS_BUILDER_AVAILABLE = False and let training proceed
# against `np.zeros(90)` -- a blind agent, announced only by a single log warning.
# A broken observation must never be a recoverable state, so the import is now
# unguarded and any failure stops the process at import time.
from modules.meta.ppo_observation_builder import (  # noqa: E402
    PPOObservationBuilder,
    PPO_OBS_SIZE,
    PPO_OBS_VERSION,
)

OBS_BUILDER_AVAILABLE = True  # retained for backward compatibility; always True


def validate_observation_version(saved_version: str, saved_size: int) -> None:
    """
    Validate that saved model's observation version matches current builder.

    Prevents silent policy corruption when observation layout changes between training and inference.
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
        warnings.warn(
            f"Could not parse observation versions (saved={saved_version}, current={PPO_OBS_VERSION}). "
            f"Proceeding but policy may be corrupted if layout changed.",
            UserWarning,
        )


from envs.core.shared_utils import (
    get_envs_logger,
    timeframe_to_minutes,
    DEFAULT_PRIMARY_TIMEFRAME,
)

logger = get_envs_logger("prop_firm_env")

from envs.core.execution_model import (
    ExecutionConfig,
    ExecutionModel,
    CommissionMode,
    CommissionSpec,
)

from envs.core.env_types import (
    CloseReason,
    RewardConfig,
    PropFirmConfig,
    PropPosition,
    TradeResult,
)

# Import mixins for modular functionality
from envs.prop_firm import (
    ExpertSignalsMixin,
    EntryQualityMixin,
    TradeRewardMixin,
    RewardShapingMixin,
    ObservationBuildersMixin,
    SessionTimingMixin,
    DataDifficultyMixin,
)


class PropFirmTradingEnv(
    # NOTE: ExpertSignalsMixin already inherits MarketStructureMixin.
    # Do NOT add MarketStructureMixin here (prevents diamond inheritance).
    ExpertSignalsMixin,
    EntryQualityMixin,
    TradeRewardMixin,
    RewardShapingMixin,
    ObservationBuildersMixin,
    SessionTimingMixin,
    DataDifficultyMixin,
    gym.Env,
):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_dict: Dict[str, Dict[str, pd.DataFrame]],
        config: Optional[PropFirmConfig] = None,
        *,
        curriculum_manager: Optional["CurriculumManager"] = None,
        apply_curriculum_overrides: bool = True,
    ):
        super().__init__()
        self.config = config or PropFirmConfig()
        self.tz = ZoneInfo(self.config.tz)

        # Curriculum
        self.curriculum: Any = None
        self._apply_curriculum_overrides = bool(apply_curriculum_overrides)
        self._pending_stage_apply: bool = False
        self._last_stage_name: str = ""
        self._last_stage_epoch: int = 0
        self._curriculum_stage_idx: int = 0  # Stage index for stage-specific reward settings
        if curriculum_manager is not None and CURRICULUM_AVAILABLE:
            self.curriculum = curriculum_manager
            # CRITICAL: Set pending=True so first reset() applies stage overrides
            # Without this, the stage name matches and overrides are never applied!
            self._pending_stage_apply = True
            self._last_stage_name = getattr(self.curriculum.current_stage, "name", "")
            self._last_stage_epoch = int(getattr(self.curriculum, "current_stage_epoch", 0))
            current_stage = getattr(self.curriculum, "current_stage", None)
            if current_stage is not None:
                self._curriculum_stage_idx = getattr(current_stage, "value", 0)

        self.data = data_dict
        self.instruments = [i for i in self.config.instruments if i in self.data]
        if not self.instruments:
            self.instruments = list(self.data.keys())[:1]
        if not self.instruments:
            raise ValueError("No valid instruments found in data")

        if len(self.instruments) > 1:
            logger.warning(
                "[ENV] Multiple instruments provided but env currently trades only one per episode. "
                "Using instruments[0] unless you extend sampling logic."
            )

        # Episode instrument (kept explicit for future multi-instrument upgrades)
        self._episode_instrument: str = str(self.instruments[0])

        # Keep obs size consistent with builder contract
        try:
            self.config.observation_size = int(PPO_OBS_SIZE)
        except Exception as e:
            logger.debug(f"Could not sync observation_size: {e}")

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

        # The builder is mandatory. Constructing it here (rather than behind an
        # availability flag) means a misconfigured observation fails at env
        # construction, not silently at the first step.
        self.obs_builder: "PPOObservationBuilder" = PPOObservationBuilder()
        logger.info(f"[OBS] Using PPOObservationBuilder v{PPO_OBS_VERSION} ({PPO_OBS_SIZE} dims)")

        # Episode boundaries based on primary timeframe only
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
        self.max_consecutive_losses_reached = 0

        # Session tracking
        self._current_day: Optional[date] = None
        self._current_session_key: Optional[Tuple[date, str]] = None
        self._session_trades = 0
        
        # Session budget tracking (v5.5 - governor/budget observation)
        self.session_start_balance: float = float(self.config.initial_balance)
        self.session_pnl: float = 0.0
        self.session_consecutive_losses: int = 0
        self.session_start_step: int = 0

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
        self._episode_return = 0.0
        self._mask_decision_steps = 0
        self._mask_collapse_steps = 0
        self._stop_mode_steps = 0
        self._mask_decision_steps = 0
        self._mask_collapse_steps = 0
        self._stop_mode_steps = 0

        # Execution anti-cheat (seeded in reset)
        self._exec: Optional[ExecutionModel] = None
        self._episode_execution_cfg: Optional[ExecutionConfig] = None

        # Per-step caches
        self._ohlcv_cache_key: Optional[Tuple[str, str, int]] = None  # (instrument, timeframe, step)
        self._ohlcv_cache: Dict[int, Dict[str, Any]] = {}

        # Quote cache (critical for deterministic intra-step equity/DD)
        self._quote_cache_step: Optional[int] = None
        self._quote_cache_inst: Optional[str] = None
        self._quote_cache_mid: float = 0.0
        self._quote_cache_vol: float = 0.0
        self._quote_cache_bid: float = 0.0
        self._quote_cache_ask: float = 0.0

        # Reward tracking state
        self._avg_vol: Optional[float] = None
        self._episode_trade_results: List[TradeResult] = []
        self._last_reward_components: Dict[str, float] = {}
        self._episode_max_drawdown: float = 0.0  # Peak DD during episode

        # Aggregate reward component tracking for dashboard
        self._episode_reward_components: Dict[str, float] = {}
        self._episode_reward_component_counts: Dict[str, int] = {}

        # Episode randomization state
        self._episode_spread_mult = 1.0
        self._episode_slip_mult = 1.0
        self._episode_latency_bars = 0
        self._episode_vol_scale = 1.0

        # Scenario-level execution overrides (used by validation/stress evaluators)
        self._scenario_spread_mult = 1.0
        self._scenario_slippage_mult = 1.0
        self._scenario_latency_add = 0

        # Data difficulty settings (curriculum-based filtering)
        self._data_difficulty: Optional[Any] = None
        self._valid_start_indices: Optional[np.ndarray] = None
        self._volatility_percentiles: Optional[np.ndarray] = None

        # Step-local caches (always defined to avoid attribute drift)    
        self._step_entry_quality_cache: Dict[str, float] = {}
        self._step_expert_signals_cache: Optional[Dict[str, Any]] = None

    def _track_action_mask_state_for_metrics(self) -> None:
        """
        Track decision-time periods where the agent effectively has no choice
        (valid_actions==1) and when loss-layer stop-mode is active while flat.
        """
        try:
            mask = self.action_masks()
            n_valid = int(np.sum(mask))
        except Exception:
            return

        self._mask_decision_steps = int(getattr(self, "_mask_decision_steps", 0)) + 1

        flat_no_pending = (self.position is None) and (self.pending_entry is None)
        if flat_no_pending and n_valid <= 1:
            self._mask_collapse_steps = int(getattr(self, "_mask_collapse_steps", 0)) + 1

        if flat_no_pending and self._loss_layer() >= self._loss_layer_stop():
            self._stop_mode_steps = int(getattr(self, "_stop_mode_steps", 0)) + 1

    def set_scenario_execution_overrides(
        self,
        *,
        spread_mult: float = 1.0,
        slippage_mult: float = 1.0,
        latency_add: int = 0,
    ) -> None:
        self._scenario_spread_mult = float(max(0.0, spread_mult))
        self._scenario_slippage_mult = float(max(0.0, slippage_mult))
        self._scenario_latency_add = int(latency_add)

    def clear_scenario_execution_overrides(self) -> None:
        self._scenario_spread_mult = 1.0
        self._scenario_slippage_mult = 1.0
        self._scenario_latency_add = 0

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
            # Prefer effective stage config (reward blending/recovery/selectivity)
            if hasattr(self.curriculum, "get_effective_stage_config"):
                try:
                    stage_cfg = self.curriculum.get_effective_stage_config()
                except Exception:
                    stage_cfg = getattr(self.curriculum, "stage_config", None)
            else:
                stage_cfg = getattr(self.curriculum, "stage_config", None)
            if stage_cfg is None:
                return

            current_stage = getattr(self.curriculum, "current_stage", None)
            if current_stage is not None:
                self._curriculum_stage_idx = getattr(current_stage, "value", 0)
            else:
                self._curriculum_stage_idx = 0

            env_overrides = getattr(stage_cfg, "env_overrides", None)
            reward_overrides = getattr(stage_cfg, "reward_overrides", None)
            execution_overrides = getattr(stage_cfg, "execution_overrides", None)
            generic = getattr(stage_cfg, "overrides", None)
            
            # NEW: Apply TradingConstraints from curriculum stage
            constraints = getattr(stage_cfg, "constraints", None)
            if constraints is not None:
                # Map constraint fields to PropFirmConfig fields
                constraint_mapping = {
                    "max_positions": "max_positions",
                    "max_trades_per_day": "max_trades_per_day",
                    "max_trades_per_session": "max_trades_per_session",
                    "max_trades_per_episode": "max_trades_per_episode",
                    "max_consecutive_losses": "max_consecutive_losses",
                    "loss_layer_stop": "loss_layer_stop",
                    # Session budget constraints (v5.5)
                    "session_loss_limit_pct": "session_loss_limit_pct",
                    "session_consecutive_loss_limit": "session_consecutive_loss_limit",
                    "enforce_session_windows": "enforce_session_windows",
                    "enforce_no_new_trades_window": "enforce_no_new_trades_window",
                    "enforce_weekend_block": "enforce_weekend_block",
                    "enforce_hard_close": "enforce_hard_close",
                    "observation_period_required": "observation_period_required",
                    "min_bars_observation_before_entry": "min_bars_observation_before_entry",
                    "min_bars_between_entries": "min_bars_between_entries",
                    "min_bars_after_loss": "min_bars_after_loss",
                    "min_minutes_between_entries": "min_minutes_between_entries",
                    "min_minutes_after_loss": "min_minutes_after_loss",
                    "daily_drawdown_limit": "daily_drawdown_limit",
                    "max_drawdown_limit": "max_drawdown_limit",
                    "daily_dd_safety_buffer": "daily_dd_safety_buffer",
                    "max_dd_safety_buffer": "max_dd_safety_buffer",
                    "emergency_close_threshold": "emergency_close_threshold",
                    "entry_quality_gate_enabled": "entry_quality_gate_enabled",
                    "entry_quality_threshold": "entry_quality_threshold",
                    "min_setup_quality_for_entry": "min_setup_quality_for_entry",
                    "hard_stop_loss_eur": "hard_stop_loss_eur",
                    "soft_stop_loss_eur": "soft_stop_loss_eur",
                    "trailing_activation_eur": "trailing_activation_eur",
                    "trailing_retrace_pct": "trailing_retrace_pct",
                    "time_decay_hours": "time_decay_hours",
                    "risk_per_trade_pct": "risk_per_trade_pct",
                    "max_risk_per_trade_pct": "max_risk_per_trade_pct",
                }
                applied_constraints = {}
                for constraint_key, config_key in constraint_mapping.items():
                    try:
                        value = getattr(constraints, constraint_key, None)
                        if value is not None and hasattr(self.config, config_key):
                            setattr(self.config, config_key, value)
                            applied_constraints[config_key] = value
                    except Exception:
                        pass
                
                # Log key constraints for debugging entropy issues
                stage_name = getattr(current_stage, "name", "UNKNOWN")
                loss_stop = applied_constraints.get("loss_layer_stop", "N/A")
                max_consec = applied_constraints.get("max_consecutive_losses", "N/A")
                logger.debug(
                    f"[Curriculum] Stage {stage_name}: loss_layer_stop={loss_stop}, "
                    f"max_consecutive_losses={max_consec}"
                )

            if isinstance(generic, dict):
                self._apply_overrides_to_object(self.config, generic)
            if isinstance(env_overrides, dict):
                self._apply_overrides_to_object(self.config, env_overrides)
            if isinstance(reward_overrides, dict):
                self._apply_overrides_to_object(self.config.reward, reward_overrides)
            if isinstance(execution_overrides, dict):
                self._apply_overrides_to_object(self.config.execution, execution_overrides)

                # Compatibility bridge: curriculum stages express commission as commission_per_lot,
                # but ExecutionConfig uses commission_spec (mode + rate).
                if "commission_per_lot" in execution_overrides:
                    try:
                        commission_per_lot = float(execution_overrides.get("commission_per_lot") or 0.0)
                        if commission_per_lot > 0.0:
                            self.config.execution.commission_spec = CommissionSpec(
                                mode=CommissionMode.PER_LOT_PER_SIDE,
                                commission_rate=commission_per_lot,
                            )
                        else:
                            self.config.execution.commission_spec = CommissionSpec(
                                mode=CommissionMode.NONE,
                                commission_rate=0.0,
                            )
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"_sync_curriculum_stage_overrides() failed: {type(e).__name__}: {e}")
            return

    # ---------------------------
    # Helpers: timeframes / buffers
    # ---------------------------

    def _primary_tf(self) -> str:
        tf = getattr(self.config, "primary_timeframe", None) or DEFAULT_PRIMARY_TIMEFRAME
        return str(tf)

    def _tf_minutes(self) -> int:
        try:
            return int(timeframe_to_minutes(self._primary_tf()))
        except Exception:
            return 15

    def _episode_start_buffer(self) -> int:
        """
        Compute a safe episode start buffer so expert indicators/structure have enough history.

        - Observation builder often uses 120 bars (baseline).
        - ExpertSignalsMixin uses _MIN_LOOKBACK (default 220).
        - Market structure mixin may have its own lookback (optional).
        """
        obs_lb = 120
        expert_lb = int(getattr(self, "_MIN_LOOKBACK", 220) or 220)
        struct_lb = int(getattr(self, "_STRUCTURE_LOOKBACK", 0) or 0)
        margin = 10
        return int(max(obs_lb, expert_lb, struct_lb, 50) + margin)

    # ---------------------------
    # Loss-layer governor (Jan 2026)
    # ---------------------------
    # Behavioral gates that tighten after consecutive losses.
    # More effective than exponential reward penalties (which saturate clip and lose gradient).

    def _loss_layer(self) -> int:
        """Governor layer based on consecutive losses (unclamped for curriculum flexibility)."""
        return int(max(0, getattr(self, "consecutive_losses", 0)))

    def _loss_layer_clamped(self) -> int:
        """Governor layer clamped to 0-5 for table lookups (cooldown, quality, risk)."""
        return int(np.clip(self._loss_layer(), 0, 5))

    def _loss_layer_stop(self) -> int:
        """Layer at which new entries are fully blocked (hard stop-trading mode)."""
        return int(getattr(self.config, "loss_layer_stop", 5))

    def _maybe_relax_loss_layer_on_session_roll(self, *, session_rolled: bool) -> None:
        """
        Relax loss-layer stop on session boundaries.

        Many curriculum stages set `loss_layer_stop = max_consecutive_losses - 1` to prevent a
        hard breach. Episodes can span multiple sessions; without relaxing the loss-layer at a
        natural boundary, a single loss streak can trap the agent in HOLD-only mode for the
        rest of a long episode.

        We treat a session roll as a "break" and step the loss layer down to `loss_layer_stop - 1`.
        """
        if not session_rolled:
            return

        stop = int(self._loss_layer_stop())
        if stop <= 0:
            return

        if int(getattr(self, "consecutive_losses", 0)) >= stop:
            self.consecutive_losses = max(0, stop - 1)
            self.consecutive_wins = 0

    def _loss_layer_cooldown_minutes(self, base_minutes: float) -> float:
        """
        Dynamic post-loss cooldown. Escalates with consecutive losses.

        Override via config.loss_layer_cooldown_minutes (list of 6 values for layers 0-5)
        or config.loss_layer_cooldown_multipliers.
        
        Default: layer 0-1 = base, layer 2 = 1.5x, layer 3 = 2.5x, layer 4 = 4x, layer 5 = 8x
        """
        layer = self._loss_layer_clamped()
        
        # Option A: explicit minutes per layer (preferred for curriculum control)
        mins_table = getattr(self.config, "loss_layer_cooldown_minutes", None)
        if isinstance(mins_table, (list, tuple)) and len(mins_table) >= 6:
            return float(mins_table[layer])
        
        # Option B: multiplier per layer (fallback)
        mults = getattr(self.config, "loss_layer_cooldown_multipliers", None)
        if isinstance(mults, (list, tuple)) and len(mults) >= 6:
            return float(base_minutes) * float(mults[layer])
        
        # Default escalation (reasonable, not insane)
        default_mult = [1.0, 1.0, 1.5, 2.5, 4.0, 8.0]
        return float(base_minutes) * default_mult[layer]

    def _loss_layer_entry_quality_threshold(self, base_threshold: float) -> float:
        """
        Dynamic entry-quality requirement; stricter after consecutive losses.
        
        Override via config.loss_layer_entry_q_add (list of 6 values for layers 0-5).
        
        Default: layer 0-1 = base, layer 2 = +0.05, layer 3 = +0.10, layer 4 = +0.18, layer 5 = +0.30
        """
        layer = self._loss_layer_clamped()
        
        adds = getattr(self.config, "loss_layer_entry_q_add", None)
        if isinstance(adds, (list, tuple)) and len(adds) >= 6:
            add_val = float(adds[layer])
        else:
            # Default: mild → strict, capped at 0.90
            add_val = [0.00, 0.00, 0.05, 0.10, 0.18, 0.30][layer]
        
        return float(np.clip(float(base_threshold) + add_val, 0.0, 0.90))

    def _loss_layer_risk_multiplier(self) -> float:
        """
        Reduce risk per trade after consecutive losses.
        
        Override via config.loss_layer_risk_mult (list of 6 values for layers 0-5).
        
        Default: layer 0 = 100%, layer 1 = 90%, layer 2 = 75%, layer 3 = 60%, layer 4 = 45%, layer 5 = 30%
        """
        layer = self._loss_layer_clamped()
        
        mults = getattr(self.config, "loss_layer_risk_mult", None)
        if isinstance(mults, (list, tuple)) and len(mults) >= 6:
            return float(mults[layer])
        
        # Default: progressively reduce risk; layer 5 doesn't matter (entries blocked)
        return float([1.00, 0.90, 0.75, 0.60, 0.45, 0.30][layer])

    # ---------------------------
    # Action decoding / masking
    # ---------------------------

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

    def _exec_latency(self) -> int:
        """Get execution latency in bars from episode config."""
        return int(getattr(self._episode_execution_cfg, "latency_bars", 0) or 0)

    def _get_close_priority(self, reason: str) -> int:
        """Get close priority for a reason string."""
        try:
            return CloseReason(reason).close_priority
        except ValueError:
            return 40  # Default middle priority

    def _set_or_override_pending_exit(self, *, reason: str, fill_step: int) -> None:
        """
        Set pending exit, or override existing one if new reason has higher priority.
        Ensures forced closes override agent closes. Uses earlier fill_step when overriding.
        """
        if self.pending_exit is None:
            self.pending_exit = {"fill_step": int(fill_step), "reason": str(reason)}
            return

        old_reason = str(self.pending_exit.get("reason", CloseReason.AGENT_CLOSE.value))
        new_priority = self._get_close_priority(reason)
        old_priority = self._get_close_priority(old_reason)

        if new_priority >= old_priority:
            self.pending_exit["reason"] = str(reason)
            self.pending_exit["fill_step"] = min(int(self.pending_exit.get("fill_step", fill_step)), int(fill_step))

    def action_masks(self) -> np.ndarray:
        """
        Action mask for sb3-contrib MaskablePPO.

        RELAXED for training: Only mask actions that are physically impossible.
        Soft rules are penalized in step() to allow learning.
        """
        mask = np.ones(self._N_ACTIONS, dtype=np.bool_)

        next_step = int(self.current_step + 1)

        latency = max(0, self._exec_latency())
        last_idx = (self._min_data_len - 1)
        remaining_bars_next = last_idx - next_step

        # Entry gating: account for fill ordering (fill check happens at START of step).
        # An entry created this step with fill_step = current_step + latency fills when
        # current_step >= fill_step, i.e., NEXT step at earliest = implicit +1 delay.
        # We also need 1 bar AFTER fill so position isn't instantly truncated.
        # Total: effective_entry_delay = latency + 1 (implicit) + 1 (post-fill) = latency + 2
        can_enter_fill = remaining_bars_next >= (latency + 2)
        # Exit can fill right at the end (no post-fill bar required)
        can_exit_fill = remaining_bars_next >= latency

        has_position_or_pending = (self.position is not None) or (self.pending_entry is not None)
        can_enter = (not has_position_or_pending) and bool(can_enter_fill)

        # Loss-layer governor: hard stop-trading mode after severe loss streak
        # Mask entries so agent doesn't waste exploration on blocked actions
        if self._loss_layer() >= self._loss_layer_stop():
            can_enter = False

        if not can_enter:
            mask[self._ACTION_LONG_START : self._ACTION_LONG_START + self._K] = False
            mask[self._ACTION_SHORT_START : self._ACTION_SHORT_START + self._K] = False

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
    # Market helpers
    # ---------------------------

    def _resolve_column(self, df: pd.DataFrame, col: str) -> str:
        """
        Resolve column name to handle both lowercase and uppercase variants.
        E.g., 'close' -> 'close' or 'Close' depending on what's in df.
        """
        if col in df.columns:
            return col
        upper_col = col.capitalize()
        if upper_col in df.columns:
            return upper_col
        full_upper = col.upper()
        if full_upper in df.columns:
            return full_upper
        raise KeyError(f"Column '{col}' not found (tried: {col}, {upper_col}, {full_upper})")

    def _get_price_mid(self, instrument: str) -> float:
        tf = self._primary_tf()
        df = self.data.get(instrument, {}).get(tf)
        if df is None or df.empty:
            return 0.0
        idx = int(np.clip(self.current_step, 0, len(df) - 1))
        try:
            close_col = self._resolve_column(df, "close")
            return float(df[close_col].iloc[idx])
        except KeyError:
            return 0.0



    def _df_time_ns(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Return monotonic int64 nanoseconds timestamps for df, using:
        1) DatetimeIndex if present
        2) else a time-like column if present

        Always returns a real np.ndarray[int64] (or None).
        """
        if df is None or df.empty:
            return None

        # Prefer DatetimeIndex
        if isinstance(df.index, pd.DatetimeIndex):
            idx = df.index
            try:
                if idx.tz is not None:
                    idx = idx.tz_convert(self.tz).tz_localize(None)
            except Exception:
                try:
                    idx = idx.tz_localize(None)
                except Exception:
                    pass

            if not idx.is_monotonic_increasing:
                idx = idx.sort_values()

            return np.asarray(idx.view("int64"), dtype=np.int64)

        # Fallback: time column
        for tc in ("time", "Time", "timestamp", "datetime", "Datetime"):
            if tc not in df.columns:
                continue

            s = df[tc]
            try:
                if is_numeric_dtype(s):
                    # Heuristic: seconds vs milliseconds
                    v = float(s.iloc[-1]) if len(s) else 0.0
                    unit = "ms" if v > 1e12 else "s"
                    dt = pd.to_datetime(s.astype("int64"), unit=unit, utc=True, errors="coerce")
                else:
                    dt = pd.to_datetime(s, utc=True, errors="coerce")

                # dt is a Series; convert tz -> local naive, sort, then numpy
                dt = dt.dt.tz_convert(self.tz).dt.tz_localize(None).sort_values()
                arr_dt64 = dt.to_numpy(dtype="datetime64[ns]")
                return arr_dt64.view("int64").astype(np.int64, copy=False)

            except Exception:
                return None

        return None



    def _current_primary_time_ns(self, instrument: str) -> Optional[int]:
        """Get the current primary timeframe bar timestamp (ns) for alignment."""
        primary_tf = self._primary_tf()
        primary_df = self.data.get(instrument, {}).get(primary_tf)
        if primary_df is None or primary_df.empty:
            return None

        t_ns = self._df_time_ns(primary_df)
        if t_ns is None or len(t_ns) == 0:
            return None

        idx = int(np.clip(self.current_step, 0, len(t_ns) - 1))
        return int(t_ns[idx])


    def _get_ohlcv(
        self,
        instrument: str,
        lookback: int = 120,
        timeframe: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        OHLCV slice for (instrument, timeframe) ending at the bar aligned to current primary timestamp.

        FIX (Jan 2026):
        - HTF alignment works for DatetimeIndex and for time columns.
        - Prevents "stale HTF" when HTF has fewer bars than primary.
        """
        tf = str(timeframe) if timeframe is not None else self._primary_tf()
        primary_tf = self._primary_tf()

        key = (instrument, tf, int(self.current_step))
        if self._ohlcv_cache_key != key:
            self._ohlcv_cache_key = key
            self._ohlcv_cache = {}

        lb = int(max(1, lookback))
        if lb in self._ohlcv_cache:
            return self._ohlcv_cache[lb]

        df = self.data.get(instrument, {}).get(tf)
        if df is None or df.empty:
            self._ohlcv_cache[lb] = {}
            return {}

        # Ensure stable ordering
        try:
            if isinstance(df.index, pd.DatetimeIndex) and not df.index.is_monotonic_increasing:
                df = df.sort_index()
        except Exception:
            pass

        # Decide end/start
        if tf == primary_tf:
            end = min(int(self.current_step) + 1, len(df))
            start = max(0, end - lb)
        else:
            # Align HTF bar to current primary timestamp
            cur_t = self._current_primary_time_ns(instrument)
            htf_t = self._df_time_ns(df)

            if cur_t is not None and htf_t is not None and len(htf_t) > 0:
                # Find last HTF bar with timestamp <= current primary bar timestamp
                pos = int(np.searchsorted(htf_t, cur_t, side="right") - 1)
                if pos < 0:
                    self._ohlcv_cache[lb] = {}
                    return {}
                end = min(pos + 1, len(df))
                start = max(0, end - lb)
            else:
                # LAST resort fallback: map by timeframe ratio instead of raw index
                try:
                    base_min = max(1, int(timeframe_to_minutes(primary_tf)))
                    tgt_min = max(1, int(timeframe_to_minutes(tf)))
                    ratio = max(1, int(round(tgt_min / float(base_min))))
                except Exception:
                    ratio = 1

                approx_end = int((int(self.current_step) + 1) / ratio)
                end = int(np.clip(approx_end, 1, len(df)))
                start = max(0, end - lb)

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

        vol_col = next((c for c in ("volume", "Volume", "VOLUME") if c in df.columns), None)
        out["volume"] = (
            df[vol_col].iloc[start:end].to_numpy(dtype=np.float64, copy=False)
            if vol_col
            else np.ones(end - start, dtype=np.float64)
        )

        spread_col = next((c for c in ("spread", "Spread", "SPREAD") if c in df.columns), None)
        if spread_col:
            out["spread"] = df[spread_col].iloc[start:end].to_numpy(dtype=np.float64, copy=False)

        self._ohlcv_cache[lb] = out
        return out


    def _get_pip_value(self, instrument: str) -> Tuple[float, float]:
        """
        Get pip value and pip divisor for an instrument.
        
        Returns:
            Tuple of (pip_value_per_lot, pip_divisor):
            - pip_value_per_lot: USD/EUR value per pip movement per standard lot
            - pip_divisor: Divisor to convert price to pips (e.g., 10000 for FX = 0.0001/pip)
        
        Instrument-specific values:
        - XAUUSD/GOLD: 100 USD per pip per lot, 1.0 divisor (price in USD, 1 pip = $0.01)
        - XAGUSD/SILVER: 50 USD per pip per lot, 1.0 divisor  
        - FX pairs: 10 USD per pip per lot, 10000 divisor (1 pip = 0.0001 for 4-decimal pairs)
        """
        inst = instrument.upper().replace("_", "").replace("/", "")
        if "XAU" in inst or "GOLD" in inst:
            return 100.0, 1.0  # XAUUSD: $100 per pip, 1 pip = $0.01 move
        if "XAG" in inst or "SILVER" in inst:
            return 50.0, 1.0   # XAGUSD: $50 per pip, 1 pip = $0.001 move
        return 10.0, 10000.0   # Standard FX: $10 per pip, 1 pip = 0.0001 move

    def _atr_vol_proxy(self, instrument: str) -> float:
        o = self._get_ohlcv(instrument, lookback=40, timeframe=None)
        if not o or len(o.get("close", [])) < 20:
            return 0.3
        close = np.asarray(o["close"], dtype=np.float64)
        returns = np.diff(close[-20:]) / np.maximum(close[-20:-1], 1e-8)
        vol = float(np.std(returns))
        vol = float(np.clip(vol * 100.0, 0.0, 1.0))
        return float(np.clip(vol * self._episode_vol_scale, 0.0, 2.0))

    def _get_current_data_spread(self, instrument: str) -> Optional[float]:
        """
        Get the actual spread from the data file for the current bar.
        Returns spread in PRICE UNITS (not points).
        
        UNIT CONVERSION (Jan 2026 - Issue 2 Fix):
        CSV spread column stores integer points (e.g., EURUSD=2, XAUUSD=7).
        These must be converted to price deltas before use in bid/ask calculation:
        - FX pairs: 1 point = 0.0001 (4th decimal place)
        - XAU/GOLD: 1 point = 0.01 (cents)
        - XAG/SILVER: 1 point = 0.001
        
        This is critical for realistic training with actual FTMO broker spreads
        instead of synthetic/hardcoded values.
        """
        o = self._get_ohlcv(instrument, lookback=1, timeframe=None)
        if not o or "spread" not in o:
            return None
        spread_arr = o.get("spread")
        if spread_arr is None or len(spread_arr) == 0:
            return None
        
        raw_points = float(spread_arr[-1])
        if raw_points <= 0:
            return None
        
        # Convert points to price delta based on instrument type
        # This matches the pip value conventions in _get_pip_value()
        inst_upper = instrument.upper().replace("_", "").replace("/", "")
        if "XAU" in inst_upper or "GOLD" in inst_upper:
            point_value = 0.01  # $0.01 per point for gold
        elif "XAG" in inst_upper or "SILVER" in inst_upper:
            point_value = 0.001  # $0.001 per point for silver
        else:
            point_value = 0.0001  # 1 pip = 0.0001 for FX pairs
        
        return raw_points * point_value

    def _get_effective_data_spread(self, instrument: str) -> Optional[float]:
        """Returns the spread to feed into the execution model (price units).

        Respects execution config flags:
        - config.execution.use_data_spread
        - config.execution.data_spread_scale
        """
        exec_cfg = getattr(self.config, "execution", None)
        use_data = getattr(exec_cfg, "use_data_spread", True) if exec_cfg else True
        if not bool(use_data):
            return None

        spread_scale = getattr(exec_cfg, "data_spread_scale", 1.0) if exec_cfg else 1.0
        raw_spread = self._get_current_data_spread(instrument)
        if raw_spread is None or raw_spread <= 0:
            return None
        return float(raw_spread) * float(spread_scale)

    def _get_step_bid_ask(self, instrument: str, mid: float, vol_proxy: float) -> Tuple[float, float]:
        """
        Quote caching: call quote() at most once per step, but validate mid/vol inputs.

        If mid/vol differ materially within the same step, re-quote to avoid stale bid/ask reuse.
        
        DATA SPREAD CONTROL (Jan 2026):
        - Respects config.execution.use_data_spread flag
        - Applies config.execution.data_spread_scale discount
        - Curriculum can disable data spreads for training wheels in early stages
        """
        assert self._exec is not None

        step = int(self.current_step)
        inst = str(instrument)

        if self._quote_cache_step == step and self._quote_cache_inst == inst:
            # Validate request matches cached parameters (tolerant to float noise)
            mid_tol = max(1e-9, 1e-6 * abs(float(mid)))
            # vol_proxy is computed via np.std() which has inherent float noise;
            # 1e-6 tolerance prevents spurious cache misses while catching real changes
            vol_tol = 1e-6
            if abs(float(mid) - float(self._quote_cache_mid)) <= mid_tol and abs(float(vol_proxy) - float(self._quote_cache_vol)) <= vol_tol:
                return float(self._quote_cache_bid), float(self._quote_cache_ask)

        data_spread = self._get_effective_data_spread(inst)
        bid, ask, _ = self._exec.quote(mid, vol_proxy, data_spread=data_spread)
        self._quote_cache_step = step
        self._quote_cache_inst = inst
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
        # Loss-layer governor: reduce risk after consecutive losses
        base_risk *= self._loss_layer_risk_multiplier()
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
        """
        Calculate current and daily drawdowns.
        - Static DD: vs INITIAL balance (FTMO standard)
        - Trailing DD: vs peak equity (ratchets up)
        - Daily DD: vs day_start_balance
        """
        if bool(self.config.trailing_drawdown):
            current_dd = (self.peak_balance - self.equity) / max(self.peak_balance, 1.0)
        else:
            current_dd = (self.config.initial_balance - self.equity) / max(self.config.initial_balance, 1.0)

        current_daily_dd = (self.day_start_balance - self.equity) / max(self.day_start_balance, 1.0)
        current_dd = max(0.0, float(current_dd))
        current_daily_dd = max(0.0, float(current_daily_dd))
        return current_dd, current_daily_dd

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

        mae = abs(float(pos.lowest_pnl))
        mfe = max(0.0, float(pos.peak_pnl))
        bars_held = self.episode_bars - pos.entry_bar
        entry_quality = float(pos.entry_quality)
        entry_certainty = float(getattr(pos, "entry_certainty", 0.5))
        setup_quality = float(getattr(pos, "setup_quality", 0.5))
        confluence_count = int(getattr(pos, "confluence_count", 0))
        bars_since_setup = int(getattr(pos, "bars_since_setup", 0))
        deliberation_bars = int(getattr(pos, "deliberation_bars", 0))
        is_fomo_entry = bool(getattr(pos, "is_fomo_entry", False))
        is_revenge_entry = bool(getattr(pos, "is_revenge_entry", False))
        
        data_spread = self._get_effective_data_spread(pos.instrument)
        exit_fill, exit_fee, _ = self._exec.fill_exit(mid, pos.direction, pos.lot_size, vol_proxy, data_spread=data_spread)
        realized_pnl = float(self._realize_pnl_on_exit(pos, exit_fill, exit_fee))
        net_trade_pnl = float(realized_pnl - entry_fee)
        total_fees = float(entry_fee + exit_fee)

        # Accounting
        self.balance += realized_pnl
        self.equity = self.balance
        self.total_pnl += realized_pnl
        self.daily_pnl += realized_pnl
        
        # Session budget tracking (v5.5)
        self.session_pnl += realized_pnl

        self.total_trades += 1

        # Stats update (fee-aware)
        if net_trade_pnl > 0:
            self.winning_trades += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            # Session consecutive losses reset on win
            self.session_consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            # Session consecutive losses increment
            self.session_consecutive_losses += 1
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
            entry_bar=int(pos.entry_bar),
            entry_certainty=entry_certainty,
            setup_quality=setup_quality,
            confluence_count=confluence_count,
            bars_since_setup=bars_since_setup,
            deliberation_bars=deliberation_bars,
            is_fomo_entry=is_fomo_entry,
            is_revenge_entry=is_revenge_entry,
            total_fees=total_fees,
            entry_dt=pos.entry_dt if hasattr(pos, "entry_dt") else None,
            entry_context=pos.entry_context if hasattr(pos, "entry_context") else None,  # v5.3
        )

        self._episode_trade_results.append(result)

        self.position = None
        self.pending_exit = None
        self._update_peak_balance()

        return result

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
        # Loss-layer governor: hard stop-trading mode after severe loss streak
        # Check this FIRST (prevents revenge trading spiral)
        if self._loss_layer() >= self._loss_layer_stop():
            return False, "loss_layer_stop"
        
        # Session budget constraints (v5.5)
        # Check session loss limit (% of session start balance)
        session_loss_limit = getattr(self.config, "session_loss_limit_pct", 0.99)
        if self.session_start_balance > 0:
            session_pnl_pct = self.session_pnl / self.session_start_balance
            if session_pnl_pct < -session_loss_limit:
                return False, "session_loss_limit"
        
        # Check session consecutive loss limit
        session_consec_limit = getattr(self.config, "session_consecutive_loss_limit", 99)
        if self.session_consecutive_losses >= session_consec_limit:
            return False, "session_consecutive_losses"
        
        if dt is not None:
            if getattr(self.config, "enforce_weekend_block", True):
                if (not self.config.allow_weekend_holding) and self._is_weekend(dt):
                    return False, "weekend_block"
            if getattr(self.config, "enforce_no_new_trades_window", True):
                if self._in_no_new_trades_window(dt):
                    return False, "no_new_trades_window"
            if getattr(self.config, "enforce_hard_close", True):
                if self._in_final_exit_window(dt):
                    return False, "final_exit_window"
                if self._at_or_after_hard_close(dt):
                    return False, "hard_close"

        # Observation period gating (foundation discipline)
        if getattr(self.config, "observation_period_required", False):
            min_obs_bars = int(getattr(self.config, "min_bars_observation_before_entry", 0) or 0)
            if min_obs_bars > 0 and int(getattr(self, "episode_bars", 0)) < min_obs_bars:
                return False, "observation_period"

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
                tfm = max(1, self._tf_minutes())
                min_bars = int(getattr(self.config, "min_bars_between_entries", 0) or 0)
                min_mins = float(getattr(self.config, "min_minutes_between_entries", 0) or 0.0)
                effective_min_minutes = max(min_mins, min_bars * tfm)
                if mins < effective_min_minutes:
                    return False, "min_entry_spacing"

            if self._last_loss_dt is not None:
                mins = (dt - self._last_loss_dt).total_seconds() / 60.0
                # Use governor-escalated cooldown
                tfm = max(1, self._tf_minutes())
                min_bars_after = int(getattr(self.config, "min_bars_after_loss", 0) or 0)
                min_mins_after = float(getattr(self.config, "min_minutes_after_loss", 0) or 0.0)
                base_after_loss = max(min_mins_after, min_bars_after * tfm)
                dynamic_after_loss = self._loss_layer_cooldown_minutes(base_after_loss)
                if mins < dynamic_after_loss:
                    return False, "post_loss_cooldown"
        else:
            si = int(step_idx) if step_idx is not None else int(self.current_step)
            tfm = max(1, self._tf_minutes())
            min_bars = int(getattr(self.config, "min_bars_between_entries", 0) or 0)
            min_mins = float(getattr(self.config, "min_minutes_between_entries", 0) or 0.0)
            min_entry_bars = max(1, min_bars, int(np.ceil(min_mins / tfm)))
            # Use governor-escalated cooldown
            min_bars_after = int(getattr(self.config, "min_bars_after_loss", 0) or 0)
            min_mins_after = float(getattr(self.config, "min_minutes_after_loss", 0) or 0.0)
            base_after_loss = max(min_mins_after, min_bars_after * tfm)
            dynamic_after_loss = self._loss_layer_cooldown_minutes(base_after_loss)
            post_loss_bars = max(1, min_bars_after, int(np.ceil(dynamic_after_loss / tfm)))

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

    def _get_primary_data_length(self) -> int:
        """
        Get data length for PRIMARY timeframe only.
        """
        inst = self.instruments[0] if self.instruments else None
        if not inst:
            return 0

        primary_tf = self._primary_tf()
        df = self.data.get(inst, {}).get(primary_tf)
        if df is not None and len(df) > 0:
            return len(df)

        inst_data = self.data.get(inst, {})
        if isinstance(inst_data, dict) and inst_data:
            return min(len(d) for d in inst_data.values())

        return 0

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # Apply curriculum stage overrides at clean boundary (start of episode)
        if self.curriculum and (
            self._pending_stage_apply
            or self._last_stage_name != getattr(self.curriculum.current_stage, "name", "")
        ):
            self._sync_curriculum_stage_overrides()
            self._pending_stage_apply = False
            self._last_stage_name = getattr(self.curriculum.current_stage, "name", "")
            self._last_stage_epoch = int(getattr(self.curriculum, "current_stage_epoch", 0))

        # Episode instrument (currently fixed; kept explicit for future)
        self._episode_instrument = str(self.instruments[0])

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
        self.max_consecutive_losses_reached = 0

        self._current_day = None
        self._current_session_key = None
        self._session_trades = 0
        
        # Session budget reset (v5.5)
        self.session_start_balance = float(self.config.initial_balance)
        self.session_pnl = 0.0
        self.session_consecutive_losses = 0
        self.session_start_step = 0

        self._last_entry_dt = None
        self._last_loss_dt = None
        self._last_entry_step = None
        self._last_loss_step = None

        self.current_step = 0
        self.episode_step = 0
        self.episode_bars = 0
        self._episode_return = 0.0

        self._avg_vol = None
        self._episode_trade_results = []
        self._last_reward_components = {}
        self._episode_max_drawdown = 0.0

        self._episode_reward_components = {}
        self._episode_reward_component_counts = {}
        
        # Cost erosion tracking (v6.0)
        self._episode_gross_profit = 0.0
        self._episode_total_costs = 0.0

        # Setup/discipline tracking (patience/selectivity)
        self._setup_active = False
        self._setup_active_start_bar = None
        self._setup_taken_during_active = False
        self._setup_skipped_count = 0
        self._setup_rejections_since_last_trade = 0
        self._bars_since_last_setup = 0
        self._last_setup_quality = 0.0
        self._setup_quality_history = deque(maxlen=20)
        self._setup_quality_trend = 0.0
        self._confluence_increasing = False
        self._last_trade_entry_bar = None
        self._bars_between_trades = []
        self._max_patience_bars = 0
        self._fomo_trade_count = 0
        self._revenge_trade_count = 0
        self._observation_bonus_given = False
        self._quality_trade_streak = 0

        self._ohlcv_cache_key = None
        self._ohlcv_cache = {}

        self._quote_cache_step = None
        self._quote_cache_inst = None
        self._quote_cache_mid = 0.0
        self._quote_cache_vol = 0.0
        self._quote_cache_bid = 0.0
        self._quote_cache_ask = 0.0

        self._step_entry_quality_cache = {}
        self._step_entry_certainty_cache = {}
        self._step_setup_quality_cache = {}
        self._step_expert_signals_cache = None

        self._apply_domain_randomization()
        self._episode_execution_cfg = self._build_episode_execution_config()

        # Scenario overrides (validation/stress): persist across resets by applying to the per-episode config.
        scenario_latency_add = int(getattr(self, "_scenario_latency_add", 0) or 0)
        if self._episode_execution_cfg is not None and scenario_latency_add:
            try:
                self._episode_execution_cfg.latency_bars = max(
                    0, int(getattr(self._episode_execution_cfg, "latency_bars", 0)) + scenario_latency_add
                )
            except Exception:
                pass
        try:
            self._episode_latency_bars = int(getattr(self._episode_execution_cfg, "latency_bars", self._episode_latency_bars))
        except Exception:
            pass

        self._exec = ExecutionModel(self._episode_execution_cfg, self.np_random)

        # Apply domain randomization to execution model (critical)
        effective_spread_mult = float(self._episode_spread_mult) * float(getattr(self, "_scenario_spread_mult", 1.0) or 1.0)
        effective_slip_mult = float(self._episode_slip_mult) * float(getattr(self, "_scenario_slippage_mult", 1.0) or 1.0)
        self._exec.set_episode_randomization(
            spread_mult=effective_spread_mult,
            slippage_mult=effective_slip_mult,
        )

        # Buffer derived from indicator lookbacks, but bounded so episode can still run
        raw_buffer = self._episode_start_buffer()
        latency = max(0, self._exec_latency())
        max_reasonable_buffer = max(0, int(self._min_data_len) - (int(self.config.max_steps_per_episode) + latency + 2))
        buffer = int(max(1, min(raw_buffer, max_reasonable_buffer)))

        # Ensure we can fit max_steps_per_episode + buffer margins
        max_start = int(self._min_data_len) - int(self.config.max_steps_per_episode) - buffer
        if max_start < buffer:
            max_start = buffer

        # Use data difficulty sampling if enabled, otherwise uniform random
        if self._data_difficulty is not None:
            # Assumes DataDifficultyMixin implements _sample_episode_start_with_difficulty()
            self.current_step = self._sample_episode_start_with_difficulty(buffer, max_start)
        elif max_start > buffer:
            self.current_step = int(self.np_random.integers(buffer, max_start))
        else:
            self.current_step = min(buffer, max(int(self._min_data_len) - 2, 0))

        inst = self._episode_instrument
        dt = self._get_bar_dt(inst)
        self._maybe_roll_day_session(dt)

        obs = self._get_observation()
        self._track_action_mask_state_for_metrics()
        info = {"balance": self.balance, "equity": self.equity, "step": self.current_step}
        info.update(self._curriculum_step_metadata())
        return obs, info

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

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Advance time FIRST
        self.current_step += 1
        self.episode_step += 1
        self.episode_bars += 1

        inst = self._episode_instrument
        dt = self._get_bar_dt(inst)

        # Step caches
        self._step_entry_quality_cache = {}
        self._step_entry_certainty_cache = {}
        self._step_setup_quality_cache = {}
        self._step_expert_signals_cache = None

        intent, size_mult = self._decode_action(int(action))

        mid = self._get_price_mid(inst)
        vol_proxy = self._atr_vol_proxy(inst)
        assert self._exec is not None

        bid, ask = self._get_step_bid_ask(inst, mid, vol_proxy)

        latency = max(0, self._exec_latency())
        last_idx = (self._min_data_len - 1)
        remaining_bars = last_idx - int(self.current_step)

        # Entry gating: effective_entry_delay = latency + 1 (implicit fill ordering) + 1 (post-fill bar)
        can_entry_fill = remaining_bars >= (latency + 2)
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

        # Roll day/session AFTER mark-to-market to prevent daily-DD reset exploits
        prev_session_key = getattr(self, "_current_session_key", None)
        self._maybe_roll_day_session(dt)
        session_rolled = prev_session_key != getattr(self, "_current_session_key", None)
        self._maybe_relax_loss_layer_on_session_roll(session_rolled=session_rolled)

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
        if self.position is not None and not forced_close_now and can_exit_fill:
            if intent == "close":
                self._set_or_override_pending_exit(
                    reason=CloseReason.AGENT_CLOSE.value,
                    fill_step=self.current_step + latency,
                )

        # Handle forced closes: immediate vs delayed
        if forced_close_now and self.position is not None:
            immediate_reasons = {
                CloseReason.RISK_LIQUIDATION.value,
                CloseReason.EMERGENCY_CLOSE.value,
                CloseReason.DAILY_LIMIT_SAFETY.value,
                CloseReason.HARD_STOP.value,
            }

            if close_reason_str in immediate_reasons:
                close_result = self._close_position_now(reason=close_reason_str, dt=dt, mid=mid, vol_proxy=vol_proxy)
                trade_closed = True
                current_dd, current_daily_dd = self._calc_dds()
                reward += self._compute_trade_reward(close_result, current_dd)
            else:
                if not can_exit_fill:
                    close_result = self._close_position_now(reason=close_reason_str, dt=dt, mid=mid, vol_proxy=vol_proxy)
                    trade_closed = True
                    current_dd, current_daily_dd = self._calc_dds()
                    reward += self._compute_trade_reward(close_result, current_dd)
                else:
                    self._set_or_override_pending_exit(reason=close_reason_str, fill_step=self.current_step + latency)
                    forced_close_now = False

        # Execute scheduled exit (agent closes + delayed forced closes)
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

        # Compute qualities once per step (cached in mixins)
        q_long = self._get_step_entry_quality(inst, "long")
        q_short = self._get_step_entry_quality(inst, "short")
        cert_long = self._get_step_entry_certainty(inst, "long")
        cert_short = self._get_step_entry_certainty(inst, "short")
        setup_long, confluence_long = self._get_step_setup_quality(inst, "long")
        setup_short, confluence_short = self._get_step_setup_quality(inst, "short")

        # Setup tracking (patience/selectivity)
        setup_threshold = float(getattr(self.config.reward, "setup_quality_threshold", 0.70))
        best_setup_quality = setup_long if setup_long >= setup_short else setup_short
        best_confluence = confluence_long if setup_long >= setup_short else confluence_short

        if best_setup_quality >= setup_threshold:
            if not self._setup_active:
                self._setup_active = True
                self._setup_active_start_bar = int(self.episode_bars)
                self._setup_taken_during_active = False
            self._bars_since_last_setup = 0
        else:
            if self._setup_active and not self._setup_taken_during_active:
                self._setup_skipped_count += 1
                self._setup_rejections_since_last_trade += 1
            self._setup_active = False
            self._setup_active_start_bar = None
            self._bars_since_last_setup += 1

        # Track setup quality trend for maturity metrics
        try:
            self._setup_quality_history.append(float(best_setup_quality))
            if len(self._setup_quality_history) >= 2:
                prev_mean = float(np.mean(list(self._setup_quality_history)[:-1]))
                self._setup_quality_trend = float(best_setup_quality - prev_mean)
                self._confluence_increasing = bool(best_setup_quality > self._last_setup_quality)
            else:
                self._setup_quality_trend = 0.0
                self._confluence_increasing = False
            self._last_setup_quality = float(best_setup_quality)
        except Exception:
            self._setup_quality_trend = 0.0
            self._confluence_increasing = False

        # Execute pending entry (fill)
        if self.position is None and self.pending_entry is not None:
            if self.current_step >= int(self.pending_entry["fill_step"]):
                hard_ok_fill, _ = self._hard_entry_allowed(dt)
                if hard_ok_fill:
                    direction = str(self.pending_entry["direction"])
                    lot = float(self.pending_entry["lot"])
                    initial_risk = float(self.pending_entry["initial_risk"])
                    entry_quality = float(self.pending_entry.get("entry_quality", 0.5))
                    entry_certainty = float(self.pending_entry.get("entry_certainty", 0.5))
                    setup_quality = float(self.pending_entry.get("setup_quality", 0.5))
                    confluence_count = int(self.pending_entry.get("confluence_count", 0))
                    bars_since_setup = int(self.pending_entry.get("bars_since_setup", 0))
                    deliberation_bars = int(self.pending_entry.get("deliberation_bars", 0))
                    is_fomo_entry = bool(self.pending_entry.get("is_fomo_entry", False))
                    is_revenge_entry = bool(self.pending_entry.get("is_revenge_entry", False))
                    
                    data_spread = self._get_effective_data_spread(inst)
                    entry_fill, entry_fee, _ = self._exec.fill_entry(mid, direction, lot, vol_proxy, data_spread=data_spread)

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
                        entry_certainty=entry_certainty,
                        setup_quality=setup_quality,
                        confluence_count=confluence_count,
                        bars_since_setup=bars_since_setup,
                        deliberation_bars=deliberation_bars,
                        is_fomo_entry=is_fomo_entry,
                        is_revenge_entry=is_revenge_entry,
                        entry_context=self._capture_entry_context(inst),
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
                    if self._setup_active:
                        self._setup_taken_during_active = True
                    self._setup_rejections_since_last_trade = 0

                    # Bars between trades for patience metrics
                    if self._last_trade_entry_bar is not None:
                        bars_between = int(self.episode_bars - int(self._last_trade_entry_bar))
                        self._bars_between_trades.append(bars_between)
                        self._max_patience_bars = max(self._max_patience_bars, bars_between)
                    self._last_trade_entry_bar = int(self.episode_bars)

                    # Psychological trade counters
                    if is_fomo_entry:
                        self._fomo_trade_count += 1
                    if is_revenge_entry:
                        self._revenge_trade_count += 1
                else:
                    # Pending entry was accepted but could not fill; allow setup to be skipped
                    if self._setup_active:
                        self._setup_taken_during_active = False

                self.pending_entry = None

        # Attempt new entry
        attempted_entry = (self.position is None and self.pending_entry is None and intent in ("long", "short"))
        if intent == "long":
            entry_quality = float(q_long)
            entry_certainty = float(cert_long)
            setup_quality = float(setup_long)
            confluence_count = int(confluence_long)
        elif intent == "short":
            entry_quality = float(q_short)
            entry_certainty = float(cert_short)
            setup_quality = float(setup_short)
            confluence_count = int(confluence_short)
        else:
            entry_quality = 0.5
            entry_certainty = 0.5
            setup_quality = 0.5
            confluence_count = 0

        entry_allowed = hard_ok
        block_reason = hard_block

        if attempted_entry and not can_entry_fill:
            entry_allowed = False
            block_reason = "insufficient_bars_for_fill"

        if attempted_entry:
            # Episode trade cap (selectivity phase or explicit config)
            max_trades_ep = int(getattr(self.config, "max_trades_per_episode", 0) or 0)
            if max_trades_ep > 0 and int(self.total_trades) >= max_trades_ep:
                entry_allowed = False
                block_reason = "max_trades_per_episode"

            # Loss-layer governor: dynamic entry quality threshold
            # Stricter after consecutive losses using configurable schedule
            base_threshold = float(self.config.entry_quality_threshold)
            progressive_threshold = self._loss_layer_entry_quality_threshold(base_threshold)
            
            if self.config.entry_quality_gate_enabled and float(entry_quality) < progressive_threshold:
                entry_allowed = False
                block_reason = "entry_quality_gate"

            # Setup quality gate (optional)
            min_setup_quality = float(getattr(self.config, "min_setup_quality_for_entry", 0.0) or 0.0)
            if min_setup_quality > 0.0 and float(setup_quality) < min_setup_quality:
                entry_allowed = False
                block_reason = "setup_quality_gate"

            if entry_allowed:
                lot, initial_risk = self._calculate_lot_size(size_mult)
                deliberation_bars = 0
                if self._setup_active and self._setup_active_start_bar is not None:
                    deliberation_bars = int(self.episode_bars - int(self._setup_active_start_bar))
                is_fomo_entry = bool(setup_quality < float(getattr(self.config.reward, "setup_quality_threshold", 0.70)))
                is_revenge_entry = bool(self.consecutive_losses >= 2)
                self.pending_entry = {
                    "direction": intent,
                    "lot": lot,
                    "initial_risk": initial_risk,
                    "fill_step": self.current_step + latency,
                    "entry_quality": entry_quality,
                    "entry_certainty": entry_certainty,
                    "setup_quality": setup_quality,
                    "confluence_count": confluence_count,
                    "bars_since_setup": int(self._bars_since_last_setup),
                    "deliberation_bars": deliberation_bars,
                    "is_fomo_entry": is_fomo_entry,
                    "is_revenge_entry": is_revenge_entry,
                }
                if self._setup_active:
                    self._setup_taken_during_active = True

        # Blocked entry penalty
        if attempted_entry and not entry_allowed:
            hard_blocks = {
                "hard_close",
                "drawdown_headroom",
                "max_trades_per_day",
                "max_trades_per_session",
                "max_trades_per_episode",
                "post_loss_cooldown",
                "max_consecutive_losses",
                "weekend_block",
                "no_new_trades_window",
                "final_exit_window",
                "min_entry_spacing",
                "insufficient_bars_for_fill",
                "loss_layer_stop",  # Governor hard-stop (treated differently in shaping - zero penalty)
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

        # Flatten on truncation (reuse same-step mid/vol_proxy for determinism)
        if truncated and not terminated and self.position is not None:
            # IMPORTANT: Reuse mid and vol_proxy computed earlier in this step.
            # Recomputing can introduce floating-point non-determinism in rewards.
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

        # Episode-end activity consistency check (stage-aware)
        reward_cfg = self.config.reward
        if (terminated or truncated) and getattr(reward_cfg, "activity_consistency_enabled", False):
            episode_steps = max(self.episode_step, 1)
            actual_trades = self.total_trades

            target_per_1k = getattr(reward_cfg, "target_trades_per_1k_steps", 0.0)
            stage_activity_targets = getattr(reward_cfg, "stage_activity_targets", None)
            if stage_activity_targets is not None:
                stage_idx = getattr(self, "_curriculum_stage_idx", 0)
                target_per_1k = stage_activity_targets.get(stage_idx, target_per_1k)

            expected_trades = (episode_steps / 1000.0) * target_per_1k

            is_dd_termination = termination_reason in ("max_drawdown_breach", "daily_limit_breach", "daily_dd_breach")
            should_apply_penalty = expected_trades >= 2.0 and episode_steps >= 300 and not is_dd_termination

            if should_apply_penalty:
                trade_ratio = actual_trades / expected_trades
                activity_penalty = 0.0
                if trade_ratio < 0.2:
                    activity_penalty = getattr(reward_cfg, "min_trades_penalty", 0.0)
                elif abs(trade_ratio - 1.0) > 0.5:
                    deviation = abs(trade_ratio - 1.0) - 0.5
                    scale = getattr(reward_cfg, "activity_deviation_penalty_scale", 0.0)
                    # FIX: Cap was hardcoded at 0.2, now scales with deviation for extreme overtrading
                    # With 375 trades vs 12 target, deviation=29.75, penalty should be severe
                    max_penalty = getattr(reward_cfg, "activity_deviation_penalty_cap", 2.0)
                    activity_penalty = min(deviation * scale, max_penalty)
                
                # C7 FIX: Log activity penalty as a named component for attribution
                if activity_penalty > 0:
                    reward -= activity_penalty
                    self._episode_reward_components["activity_consistency_penalty"] = (
                        self._episode_reward_components.get("activity_consistency_penalty", 0.0) - activity_penalty
                    )
                    self._episode_reward_component_counts["activity_consistency_penalty"] = (
                        self._episode_reward_component_counts.get("activity_consistency_penalty", 0) + 1
                    )

        # Per-step shaping (if enabled)
        bars_in_pos = (self.episode_bars - self.position.entry_bar) if self.position else 0
        # entry_accepted: True ONLY when a pending_entry was created this step
        # This prevents exploration bonus farming by spamming blocked entries
        entry_accepted = (attempted_entry and entry_allowed and self.pending_entry is not None)
        if self._setup_active and self._setup_active_start_bar is not None:
            self._current_deliberation_bars = int(self.episode_bars - int(self._setup_active_start_bar))
        else:
            self._current_deliberation_bars = 0
        shaping = self._compute_per_step_shaping(
            has_position=self.position is not None,
            bars_in_position=bars_in_pos,
            entry_quality_long=q_long,
            entry_quality_short=q_short,
            entry_certainty_long=cert_long,
            entry_certainty_short=cert_short,
            setup_quality_long=setup_long,
            setup_quality_short=setup_short,
            confluence_long=confluence_long,
            confluence_short=confluence_short,
            bars_since_setup=int(self._bars_since_last_setup),
            setup_rejections_since_last_trade=int(self._setup_rejections_since_last_trade),
            entry_direction=str(intent),
            entry_accepted=entry_accepted,
        )
        reward += shaping
        
        # Session budget approaching-limit penalties (v5.5)
        # Provides gradient signal BEFORE hard blocks (loss_layer, session limits)
        reward += self._compute_governor_approaching_penalties()

        # Final clipping
        cfg = self.config.reward
        reward = float(np.clip(reward, cfg.min_reward, cfg.max_reward))

        # Accumulate episode return for curriculum
        self._episode_return += float(reward)

        # Track episode max drawdown
        self._episode_max_drawdown = max(self._episode_max_drawdown, current_dd)

        obs = self._get_observation()
        win_rate = self.winning_trades / max(self.total_trades, 1)

        info = {
            "balance": float(self.balance),
            "equity": float(self.equity),
            "drawdown": float(current_dd),
            "max_drawdown": float(self._episode_max_drawdown),
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
            "entry_certainty": float(entry_certainty),
            "entry_certainty_long": float(cert_long),
            "entry_certainty_short": float(cert_short),
            "setup_quality": float(setup_quality),
            "setup_quality_long": float(setup_long),
            "setup_quality_short": float(setup_short),
            "confluence_count": int(confluence_count),
            "confluence_long": int(confluence_long),
            "confluence_short": int(confluence_short),
            "bars_since_last_setup": int(self._bars_since_last_setup),
            "setup_quality_trend": float(self._setup_quality_trend),
            "confluence_increasing": bool(self._confluence_increasing),
            "setup_skipped_count": int(self._setup_skipped_count),
            "fomo_trade_count": int(self._fomo_trade_count),
            "revenge_trade_count": int(self._revenge_trade_count),
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
                "spread_mult": float(self._episode_spread_mult) * float(getattr(self, "_scenario_spread_mult", 1.0) or 1.0),
                "slippage_mult": float(self._episode_slip_mult) * float(getattr(self, "_scenario_slippage_mult", 1.0) or 1.0),
                "latency_bars": int(self._episode_latency_bars),
                "vol_scale": float(self._episode_vol_scale),
            },
        }
        info.update(self._curriculum_step_metadata())

        if terminated or truncated:
            info["episode_stats"] = self.get_episode_stats()
        else:
            self._track_action_mask_state_for_metrics()

        return obs, reward, terminated, truncated, info

    # ---------------------------
    # Observation
    # ---------------------------

    def _get_observation(self) -> np.ndarray:
        if self.obs_builder is None:
            return self._fallback_observation()  # raises; kept as a guard, not a path
        return self._build_observation_with_builder()

    def _build_observation_with_builder(self) -> np.ndarray:
        inst = self._episode_instrument
        market_data = self._prepare_market_data(inst)
        expert_signals = self._prepare_expert_signals(inst)
        committee_state = self._prepare_committee_state(expert_signals)
        risk_state = self._prepare_risk_state()
        memory_state = self._prepare_memory_state(inst)
        account_state = self._prepare_account_state(inst)
        trading_mode_state = self._prepare_trading_mode_state(inst)
        world_model_state = self._prepare_world_model_state(inst, expert_signals, committee_state)
        governor_state = self._get_governor_state()  # v5.5: governor/budget observation

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
            governor_state=governor_state,  # v5.5: governor/budget observation
        )
        return obs

    def _prepare_market_data(self, instrument: str) -> Dict[str, Any]:
        """
        Build multi-timeframe market_data using REAL HTF data when available.

        IMPROVED (Jan 2025): Now uses actual H1/H4/D1 data from loaded files instead
        of synthetically aggregating M15 bars. Falls back to aggregation only if
        real HTF data is not available.

        For compatibility with existing PPOObservationBuilder contracts, the keys remain:
          {"M15","H1","H4","D1"}.
        """
        # Get primary timeframe data (M15)
        o = self._get_ohlcv(instrument, lookback=120, timeframe=None)
        if not o or len(o.get("close", [])) < 2:
            return {"M15": {"close": [0.0], "high": [0.0], "low": [0.0], "open": [0.0], "volume": [1.0]}}

        base_tf = self._primary_tf()
        if base_tf != "M15":
            logger.debug(f"[OBS] primary_timeframe={base_tf}; market_data['M15'] uses primary bars for compatibility.")

        base = {
            "close": o["close"].tolist(),
            "high": o["high"].tolist(),
            "low": o["low"].tolist(),
            "open": o["open"].tolist(),
            "volume": o["volume"].tolist(),
        }

        def _get_real_htf_data(tf: str, lookback: int) -> Optional[Dict[str, Any]]:
            """Try to get real HTF data, return None if not available."""
            htf_data = self._get_ohlcv(instrument, lookback=lookback, timeframe=tf)
            if htf_data and len(htf_data.get("close", [])) >= 5:
                return {
                    "close": htf_data["close"].tolist(),
                    "high": htf_data["high"].tolist(),
                    "low": htf_data["low"].tolist(),
                    "open": htf_data["open"].tolist(),
                    "volume": htf_data["volume"].tolist(),
                }
            return None

        base_min = max(1, int(timeframe_to_minutes(base_tf))) if base_tf else 15

        def _agg_fallback(target_tf: str, max_bars: int) -> Dict[str, Any]:
            """Fallback: aggregate from primary TF if real HTF data unavailable."""
            try:
                tgt_min = max(1, int(timeframe_to_minutes(target_tf)))
            except Exception:
                return {"close": [0.0], "high": [0.0], "low": [0.0], "open": [0.0], "volume": [1.0]}

            ratio_f = tgt_min / float(base_min)
            ratio = int(max(1, round(ratio_f)))

            close = base["close"]
            high = base["high"]
            low = base["low"]
            open_ = base["open"]
            vol = base["volume"]

            out_c, out_h, out_l, out_o, out_v = [], [], [], [], []
            for i in range(0, len(close), ratio):
                cc = close[i : i + ratio]
                hh = high[i : i + ratio]
                ll = low[i : i + ratio]
                oo = open_[i : i + ratio]
                vv = vol[i : i + ratio]
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

        # Try to get REAL HTF data first, fall back to aggregation
        # Lookbacks must clear PPOObservationConfig.min_bars_htf (30) with headroom.
        # D1 previously requested 20 -> every observation build raised
        # "D1.close must have >= 30 bars"; H4 requested exactly 30, one bar from
        # the same failure. Keep these strictly above the contract minimum.
        h1_data = _get_real_htf_data("H1", 60) or _agg_fallback("H1", 60)
        h4_data = _get_real_htf_data("H4", 40) or _agg_fallback("H4", 40)
        d1_data = _get_real_htf_data("D1", 40) or _agg_fallback("D1", 40)

        return {
            "M15": base,
            "H1": h1_data,
            "H4": h4_data,
            "D1": d1_data,
        }

    def _fallback_observation(self) -> np.ndarray:
        """Removed on purpose -- see the import block at the top of this module.

        This used to return ``np.zeros(observation_size)``. Combined with the
        swallowed builder import it meant the agent trained on a constant vector
        while every dashboard metric still looked healthy. There is no safe
        fallback for a missing observation, so this now raises.
        """
        raise RuntimeError(
            "PropFirmTradingEnv has no observation builder. Training on a "
            "fallback/zero observation is not permitted -- the agent would be "
            "blind while appearing to learn. Fix the observation builder instead."
        )

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
        """Clean up resources when the environment is closed."""
        # Clear cached data to free memory
        self._ohlcv_cache.clear()
        self._ohlcv_cache_key = None
        self._quote_cache_step = None
        self._episode_trade_results.clear()
        
        # Clear position state
        self.position = None
        self.pending_entry = None
        self.pending_exit = None
        
        # Clear execution model
        self._exec = None
        self._episode_execution_cfg = None

    # ---------------------------
    # Episode Analytics
    # ---------------------------

    def get_episode_stats(self) -> Dict[str, Any]:
        decision_steps = int(getattr(self, "_mask_decision_steps", self.episode_step) or 0)
        mask_collapse_steps = int(getattr(self, "_mask_collapse_steps", 0) or 0)
        stop_mode_steps = int(getattr(self, "_stop_mode_steps", 0) or 0)
        denom = max(decision_steps, 1)

        if not self._episode_trade_results:
            return {
                "trade_count": 0,
                "max_drawdown": float(self._episode_max_drawdown),
                "win_rate": 0.0,
                "avg_pnl": 0.0,
                "avg_r_multiple": 0.0,
                "avg_mae": 0.0,
                "avg_bars_held": 0.0,
                "avg_setup_quality": 0.0,
                "avg_entry_certainty": 0.0,
                "min_setup_quality_for_entry": 0.0,
                "avg_bars_between_trades": 0.0,
                "setup_skipped_count": int(self._setup_skipped_count),
                "fomo_trade_count": int(self._fomo_trade_count),
                "revenge_trade_count": int(self._revenge_trade_count),
                "max_patience_bars": int(self._max_patience_bars),
                "exit_quality_distribution": {},
                "consecutive_losses": int(self.consecutive_losses),
                "consecutive_wins": int(self.consecutive_wins),
                "hit_max_consecutive_losses": self.consecutive_losses >= self.config.max_consecutive_losses,
                "mask_decision_steps": decision_steps,
                "mask_collapse_steps": mask_collapse_steps,
                "stop_mode_steps": stop_mode_steps,
                "mask_collapse_rate": float(mask_collapse_steps / denom),
                "stop_mode_rate": float(stop_mode_steps / denom),
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
        # Cap profit factor to prevent inf from destabilizing composite scoring
        MAX_PROFIT_FACTOR = 99.0
        if losses and sum(r.net_pnl for r in losses) != 0:
            pf = sum(r.net_pnl for r in wins) / abs(sum(r.net_pnl for r in losses))
            pf = min(pf, MAX_PROFIT_FACTOR)
        else:
            pf = MAX_PROFIT_FACTOR

        hit_max_consec_losses = self.consecutive_losses >= self.config.max_consecutive_losses

        reward_components = {}
        for key, total in self._episode_reward_components.items():
            count = self._episode_reward_component_counts.get(key, 1)
            reward_components[key] = {
                "total": float(total),
                "count": count,
                "avg": float(total / count) if count > 0 else 0.0,
            }

        # Direction breakdown (buy/sell stats)
        long_trades = [r for r in results if r.direction in ("buy", "long")]
        short_trades = [r for r in results if r.direction in ("sell", "short")]
        long_wins = [r for r in long_trades if r.net_pnl > 0]
        short_wins = [r for r in short_trades if r.net_pnl > 0]
        
        direction_stats = {
            "long_count": len(long_trades),
            "short_count": len(short_trades),
            "long_win_rate": len(long_wins) / len(long_trades) if long_trades else 0.0,
            "short_win_rate": len(short_wins) / len(short_trades) if short_trades else 0.0,
            "long_pnl": float(sum(r.net_pnl for r in long_trades)),
            "short_pnl": float(sum(r.net_pnl for r in short_trades)),
            "long_avg_pnl": float(sum(r.net_pnl for r in long_trades) / len(long_trades)) if long_trades else 0.0,
            "short_avg_pnl": float(sum(r.net_pnl for r in short_trades) / len(short_trades)) if short_trades else 0.0,
        }

        # Patience/selectivity metrics
        setup_qualities = [float(getattr(r, "setup_quality", 0.0)) for r in results]
        entry_certainties = [float(getattr(r, "entry_certainty", 0.0)) for r in results]
        min_setup_quality_for_entry = min(setup_qualities) if setup_qualities else 0.0
        avg_setup_quality = float(sum(setup_qualities) / len(setup_qualities)) if setup_qualities else 0.0
        avg_entry_certainty = float(sum(entry_certainties) / len(entry_certainties)) if entry_certainties else 0.0
        avg_bars_between = (
            float(sum(self._bars_between_trades) / len(self._bars_between_trades))
            if self._bars_between_trades
            else 0.0
        )

        return {
            "trade_count": len(results),
            "max_drawdown": float(self._episode_max_drawdown),
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
            "avg_setup_quality": avg_setup_quality,
            "avg_entry_certainty": avg_entry_certainty,
            "min_setup_quality_for_entry": float(min_setup_quality_for_entry),
            "avg_bars_between_trades": avg_bars_between,
            "setup_skipped_count": int(self._setup_skipped_count),
            "fomo_trade_count": int(self._fomo_trade_count),
            "revenge_trade_count": int(self._revenge_trade_count),
            "max_patience_bars": int(self._max_patience_bars),
            "exit_quality_distribution": exit_dist,
            "direction_stats": direction_stats,  # NEW: Buy/Sell breakdown
            "profit_factor": float(pf),
            "consecutive_losses": int(self.consecutive_losses),
            "consecutive_wins": int(self.consecutive_wins),
            "max_consecutive_losses_reached": int(self.max_consecutive_losses_reached),
            "hit_max_consecutive_losses": hit_max_consec_losses,
            "mask_decision_steps": decision_steps,
            "mask_collapse_steps": mask_collapse_steps,
            "stop_mode_steps": stop_mode_steps,
            "mask_collapse_rate": float(mask_collapse_steps / denom),
            "stop_mode_rate": float(stop_mode_steps / denom),
            "reward_components": reward_components,
            "trades_with_regime": self._build_trades_with_regime(results),
            # v5.5: Governor state for dashboard visualization
            "governor_state": self._get_governor_state(),
        }

    def _build_trades_with_regime(self, results: List[TradeResult]) -> List[Dict[str, Any]]:
        """Build regime-tagged trade list for RegimeSkillAssessment integration."""
        trades_with_regime = []
        for r in results:
            entry_context = r.entry_context or {}

            vol_regime = entry_context.get("volatility_regime", "medium")
            risk_regime = entry_context.get("risk_regime", "neutral")

            trend_strength = float(entry_context.get("structure_trend", 0.0))
            if abs(trend_strength) > 0.5:
                trend_regime = "strong_trend"
            elif abs(trend_strength) > 0.2:
                trend_regime = "weak_trend"
            else:
                trend_regime = "ranging"

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

            spread_percentile = float(entry_context.get("spread_percentile", 0.5))
            if spread_percentile < 0.3:
                spread_regime = "tight"
            elif spread_percentile > 0.7:
                spread_regime = "wide"
            else:
                spread_regime = "normal"

            trades_with_regime.append(
                {
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
                }
            )

        return trades_with_regime
