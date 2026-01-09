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
import warnings
from datetime import datetime, date
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

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

from envs.core.execution_model import ExecutionConfig, ExecutionModel

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

        self.obs_builder: Optional["PPOObservationBuilder"] = None  # type: ignore
        if OBS_BUILDER_AVAILABLE and PPOObservationBuilder is not None:
            self.obs_builder = PPOObservationBuilder()
            logger.info(f"[OBS] Using PPOObservationBuilder v{PPO_OBS_VERSION}")
        else:
            logger.warning("[OBS] PPOObservationBuilder not available -> fallback observation")

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

        # Data difficulty settings (curriculum-based filtering)
        self._data_difficulty: Optional[Any] = None
        self._valid_start_indices: Optional[np.ndarray] = None
        self._volatility_percentiles: Optional[np.ndarray] = None

        # Step-local caches (always defined to avoid attribute drift)
        self._step_entry_quality_cache: Dict[str, float] = {}
        self._step_expert_signals_cache: Optional[Dict[str, Any]] = None

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

            current_stage = getattr(self.curriculum, "current_stage", None)
            if current_stage is not None:
                self._curriculum_stage_idx = getattr(current_stage, "value", 0)
            else:
                self._curriculum_stage_idx = 0

            env_overrides = getattr(stage_cfg, "env_overrides", None)
            reward_overrides = getattr(stage_cfg, "reward_overrides", None)
            execution_overrides = getattr(stage_cfg, "execution_overrides", None)
            generic = getattr(stage_cfg, "overrides", None)

            if isinstance(generic, dict):
                self._apply_overrides_to_object(self.config, generic)
            if isinstance(env_overrides, dict):
                self._apply_overrides_to_object(self.config, env_overrides)
            if isinstance(reward_overrides, dict):
                self._apply_overrides_to_object(self.config.reward, reward_overrides)
            if isinstance(execution_overrides, dict):
                self._apply_overrides_to_object(self.config.execution, execution_overrides)
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

    def _get_ohlcv(
        self,
        instrument: str,
        lookback: int = 120,
        timeframe: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        OHLCV slice for (instrument, timeframe) ending at current_step (inclusive).

        IMPORTANT: timeframe param exists to satisfy mixin contracts and prevent future TypeErrors.
        """
        tf = str(timeframe) if timeframe is not None else self._primary_tf()

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

        end = min(self.current_step + 1, len(df))
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
        o = self._get_ohlcv(instrument, lookback=40, timeframe=None)
        if not o or len(o.get("close", [])) < 20:
            return 0.3
        close = np.asarray(o["close"], dtype=np.float64)
        returns = np.diff(close[-20:]) / np.maximum(close[-20:-1], 1e-8)
        vol = float(np.std(returns))
        vol = float(np.clip(vol * 100.0, 0.0, 1.0))
        return float(np.clip(vol * self._episode_vol_scale, 0.0, 2.0))

    def _get_step_bid_ask(self, instrument: str, mid: float, vol_proxy: float) -> Tuple[float, float]:
        """
        Quote caching: call quote() at most once per step, but validate mid/vol inputs.

        If mid/vol differ materially within the same step, re-quote to avoid stale bid/ask reuse.
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

        bid, ask, _ = self._exec.quote(mid, vol_proxy)
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

        self._ohlcv_cache_key = None
        self._ohlcv_cache = {}

        self._quote_cache_step = None
        self._quote_cache_inst = None
        self._quote_cache_mid = 0.0
        self._quote_cache_vol = 0.0
        self._quote_cache_bid = 0.0
        self._quote_cache_ask = 0.0

        self._step_entry_quality_cache = {}
        self._step_expert_signals_cache = None

        self._apply_domain_randomization()
        self._episode_execution_cfg = self._build_episode_execution_config()
        self._exec = ExecutionModel(self._episode_execution_cfg, self.np_random)

        # Apply domain randomization to execution model (critical)
        self._exec.set_episode_randomization(
            spread_mult=self._episode_spread_mult,
            slippage_mult=self._episode_slip_mult,
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
                    "fill_step": self.current_step + latency,
                    "entry_quality": entry_quality,
                }

        # Blocked entry penalty
        if attempted_entry and not entry_allowed:
            hard_blocks = {
                "hard_close",
                "drawdown_headroom",
                "max_trades_per_day",
                "max_trades_per_session",
                "post_loss_cooldown",
                "max_consecutive_losses",
                "weekend_block",
                "no_new_trades_window",
                "final_exit_window",
                "min_entry_spacing",
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
                if trade_ratio < 0.2:
                    under_trade_penalty = getattr(reward_cfg, "min_trades_penalty", 0.0)
                    reward -= under_trade_penalty
                elif abs(trade_ratio - 1.0) > 0.5:
                    deviation = abs(trade_ratio - 1.0) - 0.5
                    scale = getattr(reward_cfg, "activity_deviation_penalty_scale", 0.0)
                    deviation_penalty = min(deviation * scale, 0.2)
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
        inst = self._episode_instrument
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
        """
        Build multi-timeframe market_data from primary TF bars.

        Upgrade: aggregation ratios are computed dynamically from primary_timeframe minutes.
        For compatibility with existing PPOObservationBuilder contracts, the keys remain:
          {"M15","H1","H4","D1"}.

        Note: If primary_timeframe is not M15, "M15" here still represents the *primary* bars.
        """
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

        base_min = max(1, int(timeframe_to_minutes(base_tf))) if base_tf else 15

        def agg(target_tf: str, max_bars: int) -> Dict[str, Any]:
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

        return {
            "M15": base,
            "H1": agg("H1", 50),
            "H4": agg("H4", 30),
            "D1": agg("D1", 20),
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
                "trade_count": 0,
                "max_drawdown": float(self._episode_max_drawdown),
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
            "exit_quality_distribution": exit_dist,
            "profit_factor": float(pf),
            "consecutive_losses": int(self.consecutive_losses),
            "consecutive_wins": int(self.consecutive_wins),
            "max_consecutive_losses_reached": int(self.max_consecutive_losses_reached),
            "hit_max_consecutive_losses": hit_max_consec_losses,
            "reward_components": reward_components,
            "trades_with_regime": self._build_trades_with_regime(results),
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
