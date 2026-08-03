

from __future__ import annotations

import copy
import warnings
from collections import deque
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from pandas.api.types import is_numeric_dtype

warnings.filterwarnings("ignore", category=RuntimeWarning)


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


from modules.meta.ppo_observation_builder import (
    PPO_OBS_SIZE,
    PPO_OBS_VERSION,
    PPOObservationBuilder,
)

OBS_BUILDER_AVAILABLE = True


from modules.utils import simulation_time as simclock


def validate_observation_version(saved_version: str, saved_size: int) -> None:
    if saved_size != PPO_OBS_SIZE:
        raise ValueError(
            f"Observation SIZE MISMATCH! Model trained with {saved_size}-dim observations, "
            f"but current PPOObservationBuilder produces {PPO_OBS_SIZE}-dim. "
            f"This will cause silent policy corruption. Retrain model or rollback builder."
        )


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
    DEFAULT_PRIMARY_TIMEFRAME,
    get_envs_logger,
    timeframe_to_minutes,
)

logger = get_envs_logger("prop_firm_env")

from envs.core.env_types import (
    CloseReason,
    PropFirmConfig,
    PropPosition,
    TradeResult,
)
from envs.core.execution_model import (
    CommissionMode,
    CommissionSpec,
    ExecutionConfig,
    ExecutionModel,
)
from envs.prop_firm import (
    DataDifficultyMixin,
    EntryQualityMixin,
    ExpertSignalsMixin,
    ObservationBuildersMixin,
    RewardShapingMixin,
    SessionTimingMixin,
    TradeRewardMixin,
)


class PropFirmTradingEnv(


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


        self.curriculum: Any = None
        self._apply_curriculum_overrides = bool(apply_curriculum_overrides)
        self._reported_dropped_overrides: set = set()
        self._high_vol_idx_cache: Dict[int, Optional[np.ndarray]] = {}
        self._pending_stage_apply: bool = False
        self._last_stage_name: str = ""
        self._last_stage_epoch: int = 0
        self._curriculum_stage_idx: int = 0
        self._episode_effective_stage_name: str = ""
        self._episode_start_time: str = ""
        self._episode_start_index: int = 0
        if curriculum_manager is not None and CURRICULUM_AVAILABLE:
            self.curriculum = curriculum_manager


            self._pending_stage_apply = True
            self._last_stage_name = getattr(self.curriculum.current_stage, "name", "")
            self._last_stage_epoch = int(getattr(self.curriculum, "current_stage_epoch", 0))
            current_stage = getattr(self.curriculum, "current_stage", None)
            if current_stage is not None:
                self._curriculum_stage_idx = getattr(current_stage, "value", 0)

        self.data = data_dict
        # Mirrored frames are built on first use, not here: constructing them
        # doubles the resident dataset and most callers never enable them.
        self._base_data = data_dict
        self._mirrored_data: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None
        self._mirror_anchors: Dict[str, float] = {}
        self._mirror_active: bool = False
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


        self._episode_instrument: str = str(self.instruments[0])


        try:
            self.config.observation_size = int(PPO_OBS_SIZE)
        except Exception as e:
            logger.debug(f"Could not sync observation_size: {e}")

        self._K = len(self.config.size_buckets)
        self._ACTION_HOLD = 0
        self._ACTION_LONG_START = 1
        self._ACTION_SHORT_START = 1 + self._K
        self._ACTION_CLOSE = 1 + 2 * self._K
        self._N_ACTIONS = 2 * self._K + 2

        self.action_space = spaces.Discrete(self._N_ACTIONS)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(int(self.config.observation_size),), dtype=np.float32
        )


        self.obs_builder: "PPOObservationBuilder" = PPOObservationBuilder()
        logger.info(f"[OBS] Using PPOObservationBuilder v{PPO_OBS_VERSION} ({PPO_OBS_SIZE} dims)")


        self._primary_data_len = self._get_primary_data_length()
        self._min_data_len = self._primary_data_len


        self.balance = float(self.config.initial_balance)
        self.equity = float(self.config.initial_balance)
        self.initial_balance = float(self.config.initial_balance)
        self.day_start_balance = float(self.config.initial_balance)
        self.peak_balance = float(self.config.initial_balance)


        self.position: Optional[PropPosition] = None
        self.pending_entry: Optional[Dict[str, Any]] = None
        self.pending_exit: Optional[Dict[str, Any]] = None


        self.total_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.max_consecutive_losses_reached = 0


        self._current_day: Optional[date] = None
        self._current_session_key: Optional[Tuple[date, str]] = None
        self._session_trades = 0


        self.session_start_balance: float = float(self.config.initial_balance)
        self.session_pnl: float = 0.0
        self.session_consecutive_losses: int = 0
        self.session_start_step: int = 0


        self._last_entry_dt: Optional[datetime] = None
        self._last_loss_dt: Optional[datetime] = None


        self._last_entry_step: Optional[int] = None
        self._last_loss_step: Optional[int] = None


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


        self._exec: Optional[ExecutionModel] = None
        self._episode_execution_cfg: Optional[ExecutionConfig] = None


        # id(DataFrame) -> int64 nanosecond timestamps. See _df_time_ns: without
        # this the timestamp conversion dominated 75% of env step time.
        self._time_ns_cache: Dict[int, Optional[np.ndarray]] = {}

        self._ohlcv_cache_key: Optional[Tuple[str, str, int]] = None
        self._ohlcv_cache: Dict[int, Dict[str, Any]] = {}


        self._quote_cache_step: Optional[int] = None
        self._quote_cache_inst: Optional[str] = None
        self._quote_cache_mid: float = 0.0
        self._quote_cache_vol: float = 0.0
        self._quote_cache_bid: float = 0.0
        self._quote_cache_ask: float = 0.0


        self._avg_vol: Optional[float] = None
        self._episode_trade_results: List[TradeResult] = []
        self._last_reward_components: Dict[str, float] = {}
        self._episode_max_drawdown: float = 0.0


        self._episode_reward_components: Dict[str, float] = {}
        self._episode_reward_component_counts: Dict[str, int] = {}


        self._episode_spread_mult = 1.0
        self._episode_slip_mult = 1.0
        self._episode_latency_bars = 0
        self._episode_vol_scale = 1.0


        self._scenario_spread_mult = 1.0
        self._scenario_slippage_mult = 1.0
        self._scenario_latency_add = 0
        self._scenario_data_difficulty: Optional[Any] = None
        self._validation_start_schedule: Optional[np.ndarray] = None
        self._validation_start_cursor: int = 0


        self._data_difficulty: Optional[Any] = None
        self._valid_start_indices: Optional[np.ndarray] = None
        self._volatility_percentiles: Optional[np.ndarray] = None


        self._step_entry_quality_cache: Dict[str, float] = {}
        self._step_expert_signals_cache: Optional[Dict[str, Any]] = None

    def _track_action_mask_state_for_metrics(self) -> None:
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

    def set_scenario_data_difficulty(self, difficulty: Optional[Any]) -> None:
        """Install an explicit validation-only sampling filter.

        Raw evaluation normally clears curriculum sampling bias.  A named
        validation scenario (high-volatility, ranging, and so on) is different:
        its filter is part of the test contract and must survive the reset that
        starts the scenario.  Keeping it separate also prevents it leaking into
        the following scenario or training episode.
        """
        self._scenario_data_difficulty = copy.deepcopy(difficulty)
        self._validation_start_schedule = None
        self._validation_start_cursor = 0

    def configure_validation_episode_starts(
        self,
        *,
        min_episodes: int,
        max_episodes: int,
    ) -> List[int]:
        """Prepare chronological, non-overlapping starts for one scenario.

        DummyVecEnv automatically resets immediately after a terminal step, so
        one extra sentinel start is retained internally for that unused reset.
        The returned list contains evidence windows only.
        """
        if not bool(getattr(self.config, "raw_evaluation_mode", False)):
            raise RuntimeError("validation start schedules require raw evaluation mode")
        required = max(1, int(min_episodes))
        requested = max(required, int(max_episodes))

        # Apply the stored scenario filter and the raw-evaluation horizon before
        # deriving legal starts. This is idempotent and uses the manager's
        # episode-cached effective stage.
        self._sync_curriculum_stage_overrides()
        latency = max(0, self._exec_latency())
        low, high = self._episode_sampling_bounds(self._episode_start_buffer(), latency)
        if self._valid_start_indices is None:
            candidates = np.arange(low, high, dtype=np.int64)
        else:
            candidates = np.asarray(self._valid_start_indices, dtype=np.int64)
            candidates = candidates[(candidates >= low) & (candidates < high)]

        block = int(self.config.max_steps_per_episode)
        selected: List[int] = []
        next_legal = low
        for candidate in np.sort(np.unique(candidates)):
            value = int(candidate)
            if value < next_legal:
                continue
            selected.append(value)
            # An episode starting at s observes s and finishes after stepping
            # through s+block.  The next independent path must start after that
            # terminal bar, not on it.
            next_legal = value + block + 1
            if len(selected) >= requested:
                break

        if len(selected) < required:
            raise ValueError(
                f"only {len(selected)} independent {block}-bar validation windows "
                f"exist for the scenario; need {required}"
            )
        evidence = selected[:requested]
        self._validation_start_schedule = np.asarray(
            [*evidence, evidence[0]], dtype=np.int64
        )
        self._validation_start_cursor = 0
        return list(evidence)

    def clear_validation_episode_starts(self) -> None:
        self._validation_start_schedule = None
        self._validation_start_cursor = 0


    # Overrides that legitimately have no matching attribute on the target.
    # commission_per_lot is translated into config.execution.commission_spec by
    # _sync_curriculum_stage_overrides rather than assigned directly.
    _OVERRIDE_KEYS_HANDLED_ELSEWHERE = frozenset({"commission_per_lot"})

    def _apply_overrides_to_object(self, target: Any, overrides: Dict[str, Any]) -> None:
        if not isinstance(overrides, dict):
            return
        for k, v in overrides.items():
            if not isinstance(k, str):
                raise TypeError(f"curriculum override key must be a string, got {k!r}")

            if "." in k:
                head, rest = k.split(".", 1)
                if not hasattr(target, head):
                    raise AttributeError(
                        f"curriculum override {k!r} has no parent attribute on "
                        f"{type(target).__name__}"
                    )
                sub = getattr(target, head)
                self._apply_overrides_to_object(sub, {rest: v})
                continue
            if hasattr(target, k):
                setattr(target, k, v)
            elif k not in self._OVERRIDE_KEYS_HANDLED_ELSEWHERE:
                raise AttributeError(
                    f"curriculum override {k!r}={v!r} has no attribute on "
                    f"{type(target).__name__}"
                )

    def _sync_curriculum_stage_overrides(self) -> None:
        if not (self.curriculum and CURRICULUM_AVAILABLE and self._apply_curriculum_overrides):
            return

        try:
            if hasattr(self.curriculum, "get_effective_stage_config"):
                stage_cfg = self.curriculum.get_effective_stage_config()
            else:
                stage_cfg = getattr(self.curriculum, "stage_config", None)
            if stage_cfg is None:
                raise RuntimeError("curriculum supplied no effective stage configuration")

            effective_stage = (
                self.curriculum.get_effective_stage()
                if hasattr(self.curriculum, "get_effective_stage")
                else getattr(self.curriculum, "current_stage", None)
            )
            if effective_stage is not None:
                self._curriculum_stage_idx = getattr(effective_stage, "value", 0)
            else:
                self._curriculum_stage_idx = 0

            env_overrides = getattr(stage_cfg, "env_overrides", None)
            reward_overrides = getattr(stage_cfg, "reward_overrides", None)
            execution_overrides = getattr(stage_cfg, "execution_overrides", None)
            generic = getattr(stage_cfg, "overrides", None)


            constraints = getattr(stage_cfg, "constraints", None)
            if constraints is not None:

                constraint_mapping = {
                    "max_positions": "max_positions",
                    "max_trades_per_day": "max_trades_per_day",
                    "max_trades_per_session": "max_trades_per_session",
                    "max_trades_per_episode": "max_trades_per_episode",
                    "max_consecutive_losses": "max_consecutive_losses",
                    "loss_layer_stop": "loss_layer_stop",

                    "session_loss_limit_pct": "session_loss_limit_pct",
                    "session_consecutive_loss_limit": "session_consecutive_loss_limit",
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
                    value = getattr(constraints, constraint_key, None)
                    if value is None:
                        continue
                    if not hasattr(self.config, config_key):
                        raise AttributeError(
                            f"curriculum constraint {constraint_key!r} maps to missing "
                            f"PropFirmConfig field {config_key!r}"
                        )
                    setattr(self.config, config_key, value)
                    applied_constraints[config_key] = value


                stage_name = getattr(effective_stage, "name", "UNKNOWN")
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
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            "invalid curriculum commission_per_lot override"
                        ) from exc

            raw_evaluation = bool(getattr(self.config, "raw_evaluation_mode", False))
            if raw_evaluation:
                # Scenario filters select bars; they do not re-enable stochastic
                # price/execution augmentation.
                self.config.mirror_augmentation_prob = 0.0
                self.config.high_vol_oversample_prob = 0.0
                self.config.domain_randomization_enabled = False
                self.config.execution.deterministic_costs = True
                self.config.execution.rejection_enabled = False
                self.config.execution.spread_shock_enabled = False
                validation_steps = int(
                    getattr(self.config, "validation_max_steps_per_episode", 0) or 0
                )
                if validation_steps <= 0:
                    raise ValueError(
                        "validation_max_steps_per_episode must be positive in raw evaluation"
                    )
                self.config.max_steps_per_episode = validation_steps

            scenario_difficulty = getattr(self, "_scenario_data_difficulty", None)
            if scenario_difficulty is not None:
                desired_difficulty = copy.deepcopy(scenario_difficulty)
            elif raw_evaluation:
                # Raw chronological evaluation must not mirror prices, bias
                # episode starts, or randomize execution.  Declared stress
                # scenarios are applied separately and remain auditable.
                desired_difficulty = None
            else:
                desired_difficulty = copy.deepcopy(getattr(stage_cfg, "data_difficulty", None))
                # DataDifficulty is the single curriculum sampler; the older
                # global high-vol branch would otherwise override Explorer's
                # declared low-volatility curriculum from episode one.
                self.config.high_vol_oversample_prob = 0.0

            # Reward blending, recovery, review, and mixed-stage sampling can
            # change the effective config every episode.  Do not rebuild the
            # O(n) volatility/trend sampling cache when only reward or risk
            # fields changed and the declared data difficulty is identical.
            try:
                difficulty_changed = desired_difficulty != self._data_difficulty
            except Exception:
                difficulty_changed = True
            if difficulty_changed:
                self.set_data_difficulty(desired_difficulty)
        except Exception as exc:
            raise RuntimeError(
                "curriculum overrides could not be applied; refusing to run an "
                "episode with a partially applied stage"
            ) from exc


    def _primary_tf(self) -> str:
        tf = getattr(self.config, "primary_timeframe", None) or DEFAULT_PRIMARY_TIMEFRAME
        return str(tf)

    def _tf_minutes(self) -> int:
        try:
            return int(timeframe_to_minutes(self._primary_tf()))
        except Exception:
            return 15

    # Higher-timeframe bars requested by _prepare_market_data. The observation
    # contract requires at least min_bars_htf (30) of each, so an episode may
    # not start before this much history exists.
    # Top quintile by 5-day realised volatility counts as the high-vol band.
    _HIGH_VOL_FRACTION: float = 0.20

    _HTF_LOOKBACK_BARS: Dict[str, int] = {"H1": 60, "H4": 40, "D1": 40}

    def _episode_start_buffer(self) -> int:
        """Bars of history an episode start must have behind it.

        This previously counted only the M15 lookbacks and returned 230 bars.
        The higher timeframes were never considered, and D1 is the binding
        constraint by a wide margin: 40 D1 bars is 40 * (1440/15) = 3,840 M15
        bars. Episodes sampled near the start of the dataset therefore had as
        few as 24 D1 bars, and the observation build raised

            ObservationContractError: D1.close must have >= 30 bars. Got 24

        Intermittently, depending on the sampled start - the kind of fault that
        surfaces hours into a run rather than at launch.
        """
        obs_lb = 120
        expert_lb = int(getattr(self, "_MIN_LOOKBACK", 220) or 220)
        struct_lb = int(getattr(self, "_STRUCTURE_LOOKBACK", 0) or 0)
        margin = 10

        primary_minutes = max(1, self._tf_minutes())
        htf_bars_required = 0
        for tf, bars in self._HTF_LOOKBACK_BARS.items():
            ratio = max(1, timeframe_to_minutes(tf) // primary_minutes)
            htf_bars_required = max(htf_bars_required, bars * ratio)

        return int(max(obs_lb, expert_lb, struct_lb, htf_bars_required, 50) + margin)


    def _loss_layer(self) -> int:
        return int(max(0, getattr(self, "consecutive_losses", 0)))

    def _loss_layer_clamped(self) -> int:
        return int(np.clip(self._loss_layer(), 0, 5))

    def _loss_layer_stop(self) -> int:
        return int(getattr(self.config, "loss_layer_stop", 5))

    def _maybe_relax_loss_layer_on_session_roll(self, *, session_rolled: bool) -> None:
        if not session_rolled:
            return

        stop = int(self._loss_layer_stop())
        if stop <= 0:
            return

        if int(getattr(self, "consecutive_losses", 0)) >= stop:
            self.consecutive_losses = max(0, stop - 1)
            self.consecutive_wins = 0

    def _loss_layer_cooldown_minutes(self, base_minutes: float) -> float:
        layer = self._loss_layer_clamped()


        mins_table = getattr(self.config, "loss_layer_cooldown_minutes", None)
        if isinstance(mins_table, (list, tuple)) and len(mins_table) >= 6:
            return float(mins_table[layer])


        mults = getattr(self.config, "loss_layer_cooldown_multipliers", None)
        if isinstance(mults, (list, tuple)) and len(mults) >= 6:
            return float(base_minutes) * float(mults[layer])


        default_mult = [1.0, 1.0, 1.5, 2.5, 4.0, 8.0]
        return float(base_minutes) * default_mult[layer]

    def _loss_layer_entry_quality_threshold(self, base_threshold: float) -> float:
        layer = self._loss_layer_clamped()

        adds = getattr(self.config, "loss_layer_entry_q_add", None)
        if isinstance(adds, (list, tuple)) and len(adds) >= 6:
            add_val = float(adds[layer])
        else:

            add_val = [0.00, 0.00, 0.05, 0.10, 0.18, 0.30][layer]

        return float(np.clip(float(base_threshold) + add_val, 0.0, 0.90))

    def _loss_layer_risk_multiplier(self) -> float:
        layer = self._loss_layer_clamped()

        mults = getattr(self.config, "loss_layer_risk_mult", None)
        if isinstance(mults, (list, tuple)) and len(mults) >= 6:
            return float(mults[layer])


        return float([1.00, 0.90, 0.75, 0.60, 0.45, 0.30][layer])


    def _win_streak_risk_multiplier(self) -> float:
        """Scale risk up on a winning run, mirroring the loss-layer brake.

        loss_layer_risk_mult already cuts risk to 0.90/0.75/0.60/0.45/0.30
        through a losing streak; this is the other half. Anti-martingale: press
        while the account is proving itself, not while it is bleeding.

        Gated on drawdown headroom, because scaling up is only sound with room
        to be wrong. Inside half the max-drawdown limit the multiplier is
        withdrawn entirely - a winning streak is not evidence when the account
        is already down.
        """
        cfg = self.config
        if not bool(getattr(cfg, "win_streak_risk_enabled", True)):
            return 1.0

        wins = int(getattr(self, "consecutive_wins", 0) or 0)
        if wins < int(getattr(cfg, "win_streak_risk_min_wins", 2)):
            return 1.0

        current_dd, _daily = self._calc_dds()
        limit = max(float(getattr(cfg, "firm_max_drawdown_limit", cfg.max_drawdown_limit)), 1e-9)
        headroom = 1.0 - (current_dd / limit)
        if headroom < 0.5:
            return 1.0

        mults = getattr(cfg, "win_streak_risk_mult", None)
        if not isinstance(mults, (list, tuple)) or len(mults) < 4:
            mults = [1.0, 1.15, 1.30, 1.45]
        idx = min(wins - int(getattr(cfg, "win_streak_risk_min_wins", 2)), len(mults) - 1)
        return float(mults[max(0, idx)])

    def _decode_action(self, action_id: int) -> Tuple[str, float]:
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
        return int(getattr(self._episode_execution_cfg, "latency_bars", 0) or 0)

    def _get_close_priority(self, reason: str) -> int:
        try:
            return CloseReason(reason).close_priority
        except ValueError:
            return 40

    def _set_or_override_pending_exit(self, *, reason: str, fill_step: int) -> None:
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
        mask = np.ones(self._N_ACTIONS, dtype=np.bool_)

        next_step = int(self.current_step + 1)

        latency = max(0, self._exec_latency())
        last_idx = (self._min_data_len - 1)
        remaining_bars_next = last_idx - next_step


        can_enter_fill = remaining_bars_next >= (latency + 2)

        can_exit_fill = remaining_bars_next >= latency

        has_position_or_pending = (self.position is not None) or (self.pending_entry is not None)
        can_enter = (not has_position_or_pending) and bool(can_enter_fill)


        if self._loss_layer() >= self._loss_layer_stop():
            can_enter = False

        # Drawdown veto, outside the policy.
        #
        # Drawdown used to be represented only as a reward term, which the
        # optimizer can trade against: a large enough expected gain justifies
        # breaching. A prop-firm limit is not that kind of quantity - crossing it
        # ends the account, so it belongs in the action mask where the policy
        # cannot negotiate with it, and where live trading enforces the identical
        # rule. Exits stay legal; only new risk is refused.
        if self._entry_blocked_by_drawdown():
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
        return self.action_masks()


    def _resolve_column(self, df: pd.DataFrame, col: str) -> str:
        if col in df.columns:
            return col
        upper_col = col.capitalize()
        if upper_col in df.columns:
            return upper_col
        full_upper = col.upper()
        if full_upper in df.columns:
            return full_upper
        raise KeyError(f"Column '{col}' not found (tried: {col}, {upper_col}, {full_upper})")

    def _build_mirrored_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """A price-mirrored copy of the dataset: p' = anchor - p.

        Linear reflection, so every bar-to-bar move is exactly negated and an
        uptrend becomes an identical downtrend. Absolute point moves are
        preserved exactly, which is what P&L depends on - a log-space
        reflection would preserve percentage returns instead and shrink the
        point moves, changing trade economics.

        high and low swap: reflecting a bar turns its high into its low.

        The anchor is 2x the global maximum high, shared across every timeframe
        of an instrument so they stay aligned, and large enough that mirrored
        prices are always positive.
        """
        mirrored: Dict[str, Dict[str, pd.DataFrame]] = {}

        for instrument, frames in self.data.items():
            if not isinstance(frames, dict):
                continue

            highs = []
            for df in frames.values():
                if df is None or getattr(df, "empty", True):
                    continue
                try:
                    highs.append(float(df[self._resolve_column(df, "high")].max()))
                except (KeyError, ValueError):
                    continue
            if not highs:
                continue
            anchor = 2.0 * max(highs)

            out: Dict[str, pd.DataFrame] = {}
            for tf, df in frames.items():
                if df is None or getattr(df, "empty", True):
                    out[tf] = df
                    continue
                try:
                    o = self._resolve_column(df, "open")
                    h = self._resolve_column(df, "high")
                    low = self._resolve_column(df, "low")
                    c = self._resolve_column(df, "close")
                except KeyError:
                    out[tf] = df
                    continue

                m = df.copy(deep=True)
                m[o] = anchor - df[o].to_numpy(dtype=np.float64)
                m[c] = anchor - df[c].to_numpy(dtype=np.float64)
                # Swapped: the reflection of the high is the low.
                m[h] = anchor - df[low].to_numpy(dtype=np.float64)
                m[low] = anchor - df[h].to_numpy(dtype=np.float64)
                out[tf] = m

            mirrored[instrument] = out
            self._mirror_anchors[instrument] = anchor

        return mirrored

    def _select_episode_data(self) -> None:
        """Pick the original or mirrored dataset for this episode."""
        prob = float(getattr(self.config, "mirror_augmentation_prob", 0.0) or 0.0)
        if prob <= 0.0:
            self._mirror_active = False
            return

        if self._mirrored_data is None:
            self._mirrored_data = self._build_mirrored_data()
            if not self._mirrored_data:
                logger.warning("Mirror augmentation requested but no frame could be mirrored")
                self._mirror_active = False
                return
            self._original_data = self._base_data

        use_mirror = bool(self.np_random.random() < prob)
        if use_mirror == self._mirror_active:
            return

        self._mirror_active = use_mirror
        self.data = self._mirrored_data if use_mirror else self._base_data
        # Frame identity changed, so anything keyed on it must be recomputed.
        self._time_ns_cache = {}
        self._ohlcv_cache_key = None
        self._ohlcv_cache = {}

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
        # Cached per DataFrame. The conversion below runs tz_convert +
        # tz_localize + sort_values across the WHOLE frame (99,908 rows for
        # XAUUSD M15), and this method is called ~12x per env step. Profiling a
        # 400-step run showed tz_localize alone accounting for 156 s of 209 s
        # total - 75% of all time spent stepping the environment.
        #
        # The time column is static for the lifetime of the run, so the result
        # is computed once per frame and reused. Keyed by id() because
        # DataFrames are unhashable and these objects are held for the whole
        # run by self.data.
        if df is None or df.empty:
            return None

        cache_key = id(df)
        cached = self._time_ns_cache.get(cache_key)
        if cached is not None:
            return cached

        result = self._compute_df_time_ns(df)
        self._time_ns_cache[cache_key] = result
        return result

    def _compute_df_time_ns(self, df: pd.DataFrame) -> Optional[np.ndarray]:


        if isinstance(df.index, pd.DatetimeIndex):
            idx = df.index
            idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
            if idx.hasnans:
                raise ValueError("market-data index contains invalid timestamps")
            if not idx.is_monotonic_increasing:
                raise ValueError("market-data index timestamps are not increasing")

            # Force nanoseconds: pandas 2 can store parsed timestamps at
            # microsecond resolution, and both ``asi8`` and ``astype(int64)``
            # then expose microseconds.  Bar-duration arithmetic below is ns.
            return idx.to_numpy(dtype="datetime64[ns]").view("int64").astype(np.int64, copy=False)


        for tc in ("time", "Time", "timestamp", "datetime", "Datetime"):
            if tc not in df.columns:
                continue

            s = df[tc]
            try:
                if is_numeric_dtype(s):

                    v = float(s.iloc[-1]) if len(s) else 0.0
                    unit = "ms" if v > 1e12 else "s"
                    dt = pd.to_datetime(s.astype("int64"), unit=unit, utc=True, errors="coerce")
                else:
                    dt = pd.to_datetime(s, utc=True, errors="coerce")


                if dt.isna().any():
                    raise ValueError("market-data time column contains invalid timestamps")
                if not dt.is_monotonic_increasing:
                    raise ValueError("market-data time column is not increasing")
                return (
                    pd.DatetimeIndex(dt)
                    .to_numpy(dtype="datetime64[ns]")
                    .view("int64")
                    .astype(np.int64, copy=False)
                )

            except (TypeError, ValueError, OverflowError):
                raise

        return None


    def _current_primary_time_ns(self, instrument: str) -> Optional[int]:
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


        try:
            if isinstance(df.index, pd.DatetimeIndex) and not df.index.is_monotonic_increasing:
                df = df.sort_index()
        except Exception:
            pass


        if tf == primary_tf:
            end = min(int(self.current_step) + 1, len(df))
            start = max(0, end - lb)
        else:

            cur_t = self._current_primary_time_ns(instrument)
            htf_t = self._df_time_ns(df)

            if cur_t is not None and htf_t is not None and len(htf_t) > 0:
                # CSV timestamps are bar opens.  A decision consumes the
                # completed primary bar, so its information time is the primary
                # close.  An HTF row is legal only after *its* close; selecting
                # merely htf_open <= primary_open leaked the rest of the H1/H4/D1
                # candle into earlier M15 decisions.
                primary_minutes = max(1, int(timeframe_to_minutes(primary_tf)))
                htf_minutes = max(primary_minutes, int(timeframe_to_minutes(tf)))
                decision_t = int(cur_t) + primary_minutes * 60 * 1_000_000_000
                latest_closed_open = decision_t - htf_minutes * 60 * 1_000_000_000
                pos = int(np.searchsorted(htf_t, latest_closed_open, side="right") - 1)
                if pos < 0:
                    self._ohlcv_cache[lb] = {}
                    return {}
                end = min(pos + 1, len(df))
                start = max(0, end - lb)
            else:

                try:
                    base_min = max(1, int(timeframe_to_minutes(primary_tf)))
                    tgt_min = max(1, int(timeframe_to_minutes(tf)))
                    ratio = max(1, int(round(tgt_min / float(base_min))))
                except Exception:
                    ratio = 1

                approx_end = int((int(self.current_step) + 1) // ratio)
                if approx_end <= 0:
                    self._ohlcv_cache[lb] = {}
                    return {}
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
        inst = instrument.upper().replace("_", "").replace("/", "")
        if "XAU" in inst or "GOLD" in inst:
            return 100.0, 1.0
        if "XAG" in inst or "SILVER" in inst:
            return 50.0, 1.0
        return 10.0, 10000.0


    _VOL_SHORT_BARS = 20
    _VOL_REF_BARS = 260


    _VOL_FULL_SCALE_RATIO = 3.0

    def _atr_vol_proxy(self, instrument: str) -> float:
        o = self._get_ohlcv(instrument, lookback=self._VOL_REF_BARS, timeframe=None)
        if not o or len(o.get("close", [])) < self._VOL_SHORT_BARS + 1:
            return 0.3

        close = np.asarray(o["close"], dtype=np.float64)
        rets = np.diff(close) / np.maximum(close[:-1], 1e-8)
        if rets.size < self._VOL_SHORT_BARS:
            return 0.3

        short_vol = float(np.std(rets[-self._VOL_SHORT_BARS:]))


        ref_vol = float(np.median(np.abs(rets - np.median(rets)))) * 1.4826
        if not np.isfinite(ref_vol) or ref_vol <= 1e-12:
            ref_vol = float(np.std(rets))
        if not np.isfinite(ref_vol) or ref_vol <= 1e-12:
            return 0.3

        ratio = short_vol / ref_vol
        vol = float(np.clip(ratio / self._VOL_FULL_SCALE_RATIO, 0.0, 1.0))
        return float(np.clip(vol * self._episode_vol_scale, 0.0, 2.0))

    def _get_current_data_spread(self, instrument: str) -> Optional[float]:
        o = self._get_ohlcv(instrument, lookback=1, timeframe=None)
        if not o or "spread" not in o:
            return None
        spread_arr = o.get("spread")
        if spread_arr is None or len(spread_arr) == 0:
            return None

        raw_points = float(spread_arr[-1])
        if raw_points <= 0:
            return None


        inst_upper = instrument.upper().replace("_", "").replace("/", "")
        if "XAU" in inst_upper or "GOLD" in inst_upper:
            point_value = 0.01
        elif "XAG" in inst_upper or "SILVER" in inst_upper:
            point_value = 0.001
        else:
            point_value = 0.0001

        return raw_points * point_value

    def _get_effective_data_spread(self, instrument: str) -> Optional[float]:
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
        assert self._exec is not None

        step = int(self.current_step)
        inst = str(instrument)

        if self._quote_cache_step == step and self._quote_cache_inst == inst:

            mid_tol = max(1e-9, 1e-6 * abs(float(mid)))


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

    def _intrabar_stop_mid(self, pos: PropPosition) -> Optional[float]:
        """Return a conservative stop trigger from the current completed bar.

        Stops are executable price levels, not end-of-bar euro thresholds.  A
        gap through the level exits at the bar open; an ordinary touch exits at
        the stop.  Spread and slippage are then applied by ``fill_exit``.
        """
        stop = getattr(pos, "stop_price", None)
        if stop is None or not np.isfinite(float(stop)):
            return None
        ohlcv = self._get_ohlcv(pos.instrument, lookback=1)
        if not ohlcv:
            return None
        try:
            bar_open = float(np.asarray(ohlcv["open"], dtype=np.float64)[-1])
            bar_high = float(np.asarray(ohlcv["high"], dtype=np.float64)[-1])
            bar_low = float(np.asarray(ohlcv["low"], dtype=np.float64)[-1])
        except (KeyError, IndexError, TypeError, ValueError):
            return None
        if not all(np.isfinite(v) for v in (bar_open, bar_high, bar_low)):
            raise ValueError("non-finite OHLC at executable stop check")

        stop = float(stop)
        if pos.direction == "long" and bar_low <= stop:
            return min(bar_open, stop)
        if pos.direction == "short" and bar_high >= stop:
            return max(bar_open, stop)
        return None

    def _realize_pnl_on_exit(self, pos: PropPosition, exit_fill: float, exit_fee: float) -> float:
        pip_value, multiplier = self._get_pip_value(pos.instrument)
        diff = (exit_fill - pos.entry_price) * multiplier
        if pos.direction == "short":
            diff = -diff
        gross = diff * pip_value * pos.lot_size
        return float(gross - float(exit_fee))

    def _high_vol_starts(self, instrument: str) -> Optional[np.ndarray]:
        """Indices of bars in the top volatility band, computed once per frame.

        The post-war regime is roughly 10% of the merged dataset, so sampling
        episode starts uniformly gives the agent 10% exposure to the market it
        now has to trade. Weighting the sampler is what turns 5.5% exposure into
        enough experience to learn from.
        """
        df = self.data.get(instrument, {}).get(self._primary_tf())
        if df is None or getattr(df, "empty", True):
            return None

        key = id(df)
        cached = self._high_vol_idx_cache.get(key)
        if cached is not None:
            return cached

        try:
            close = df[self._resolve_column(df, "close")].to_numpy(dtype=np.float64)
        except KeyError:
            return None
        if close.size < 2000:
            self._high_vol_idx_cache[key] = None
            return None

        r = np.diff(np.log(np.maximum(close, 1e-9)))
        window = 96 * 5
        cs = np.cumsum(np.insert(r ** 2, 0, 0.0))
        var = (cs[window:] - cs[:-window]) / window
        vol = np.sqrt(np.maximum(var, 0.0))
        pad = close.size - vol.size
        vol = np.concatenate([np.full(pad, vol[0] if vol.size else 0.0), vol])

        cutoff = float(np.nanpercentile(vol, 100.0 * (1.0 - self._HIGH_VOL_FRACTION)))
        idx = np.flatnonzero(vol >= cutoff).astype(np.int64)
        self._high_vol_idx_cache[key] = idx if idx.size else None
        return self._high_vol_idx_cache[key]

    def _sample_episode_start(self, buffer: int, max_start: int) -> int:
        prob = float(getattr(self.config, "high_vol_oversample_prob", 0.0) or 0.0)
        if prob > 0.0 and self.np_random.random() < prob:
            idx = self._high_vol_starts(self._episode_instrument)
            if idx is not None:
                eligible = idx[(idx >= buffer) & (idx < max_start)]
                if eligible.size:
                    return int(eligible[self.np_random.integers(0, eligible.size)])
        return int(self.np_random.integers(buffer, max_start))

    def _prepare_session_state(self, instrument: str) -> Dict[str, Any]:
        """Which market session it is, and how expensive the spread is right now.

        Every other observation block is derived from XAUUSD price alone. Measured
        FTMO spread runs 37 points at 09-12 UTC against 47 at 01 UTC, so the same
        trade costs a quarter more depending on the hour, and nothing in the
        observation carried that.

        spread_ratio compares the current bar's spread to its own recent median,
        so it reads as "normal / expensive" rather than as an absolute that would
        change meaning between the historical and broker feeds.
        """
        primary = self.data.get(instrument, {}).get(self._primary_tf())
        decision_utc: Optional[pd.Timestamp] = None
        if primary is not None and not primary.empty:
            idx = int(np.clip(self.current_step, 0, len(primary) - 1))
            raw_ts: Any = None
            if isinstance(primary.index, pd.DatetimeIndex):
                raw_ts = primary.index[idx]
            else:
                for column in ("time", "timestamp", "datetime", "date"):
                    if column in primary.columns:
                        raw_ts = primary[column].iloc[idx]
                        break
            if raw_ts is not None:
                parsed = pd.Timestamp(raw_ts)
                if parsed.tzinfo is None:
                    # MT5 epoch timestamps and the canonical CSVs are UTC.  A
                    # naive value is therefore UTC-naive, not Europe/Berlin.
                    parsed = parsed.tz_localize("UTC")
                else:
                    parsed = parsed.tz_convert("UTC")
                decision_utc = parsed + pd.Timedelta(minutes=max(1, self._tf_minutes()))

        hour = (
            float(decision_utc.hour + decision_utc.minute / 60.0)
            if decision_utc is not None
            else 12.0
        )

        in_london = False
        in_new_york = False
        if decision_utc is not None:
            london = decision_utc.tz_convert(ZoneInfo("Europe/London"))
            new_york = decision_utc.tz_convert(ZoneInfo("America/New_York"))
            london_hour = london.hour + london.minute / 60.0
            ny_hour = new_york.hour + new_york.minute / 60.0
            in_london = 8.0 <= london_hour < 17.0
            in_new_york = 8.0 <= ny_hour < 17.0

        spread_ratio = 1.0
        o = self._get_ohlcv(instrument, lookback=200)
        spreads = o.get("spread") if o else None
        if spreads is not None and len(spreads) >= 20:
            arr = np.asarray(spreads, dtype=np.float64)
            prior = arr[:-1]
            prior = prior[np.isfinite(prior) & (prior > 0.0)]
            current = float(arr[-1])
            median = float(np.median(prior)) if prior.size else 0.0
            if np.isfinite(current) and current > 0.0 and median > 1e-9:
                execution_mult = float(getattr(self, "_episode_spread_mult", 1.0) or 1.0)
                execution_mult *= float(getattr(self, "_scenario_spread_mult", 1.0) or 1.0)
                spread_ratio = float(current * execution_mult / median)

        return {
            "hour_utc": hour,
            "spread_ratio": spread_ratio,
            "is_prime": bool(in_london or in_new_york),
            "is_overlap": bool(in_london and in_new_york),
        }

    def _entry_blocked_by_drawdown(self) -> bool:
        """True once drawdown has eaten the configured share of its budget.

        Blocks opening new risk while leaving closes available, so the agent can
        always work its way out of a position but cannot dig deeper.
        """
        current_dd, current_daily_dd = self._calc_dds()
        return self._entry_blocked_by_drawdown_values(current_dd, current_daily_dd)

    def _firm_drawdown_limits(self) -> Tuple[float, float]:
        daily = float(getattr(self.config, "firm_daily_drawdown_limit", 0.0) or 0.0)
        total = float(getattr(self.config, "firm_max_drawdown_limit", 0.0) or 0.0)
        if not (np.isfinite(daily) and np.isfinite(total) and 0.0 < daily < 1.0 and 0.0 < total < 1.0):
            raise ValueError(f"invalid firm drawdown limits: daily={daily!r}, total={total!r}")
        return daily, total

    def _entry_blocked_by_drawdown_values(self, current_dd: float, current_daily_dd: float) -> bool:
        reserve = float(getattr(self.config, "dd_entry_veto_fraction", 0.0) or 0.0)
        if reserve == 0.0:
            return False
        if not np.isfinite(reserve) or reserve < 0.0 or reserve >= 1.0:
            raise ValueError(f"dd_entry_veto_fraction must be in [0,1): {reserve!r}")
        if not (np.isfinite(current_dd) and np.isfinite(current_daily_dd)):
            return True

        daily_limit, max_limit = self._firm_drawdown_limits()

        if current_dd >= max_limit * reserve:
            return True

        if current_daily_dd >= daily_limit * reserve:
            return True

        return False

    def _atr_price(self, instrument: str, period: int = 14) -> float:
        """ATR in price units, for sizing the stop against volatility."""
        o = self._get_ohlcv(instrument, lookback=period + 2)
        if not o or len(o.get("close", [])) < period + 1:
            return 0.0
        high = np.asarray(o["high"], dtype=np.float64)
        low = np.asarray(o["low"], dtype=np.float64)
        close = np.asarray(o["close"], dtype=np.float64)
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
        )
        if tr.size < period:
            return 0.0
        return float(np.mean(tr[-period:]))

    def _calculate_lot_size(self, size_mult: float) -> Tuple[float, float]:
        base_risk = self.balance * self.config.risk_per_trade_pct

        base_risk *= self._loss_layer_risk_multiplier()
        base_risk *= self._win_streak_risk_multiplier()
        risk_eur = base_risk * float(np.clip(size_mult, 0.25, 1.25))
        risk_eur = min(risk_eur, self.balance * self.config.max_risk_per_trade_pct)

        # The stop was a fixed euro amount, so its distance in PRICE terms had
        # nothing to do with volatility: the same 350 EUR sat ~1.5 ATR away in a
        # calm market and ~0.75 ATR away once volatility doubled, and got hit
        # roughly twice as often for the same nominal risk. Gold's realised
        # volatility went from 0.85% to ~1.55% after February 2026, so the agent
        # was risking the same money at a very different probability of losing it.
        #
        # Sizing the lot from ATR fixes the stop at a constant statistical
        # distance instead: risk in euros stays put, and the lot shrinks when the
        # market gets wilder.
        cfg = self.config
        stop_eur = max(float(cfg.hard_stop_loss_eur), 100.0)

        if bool(getattr(cfg, "atr_stop_enabled", True)):
            inst = str(getattr(self, "_episode_instrument", "") or "")
            atr = self._atr_price(inst) if inst else 0.0
            if atr > 0.0:
                pip_value, multiplier = self._get_pip_value(inst)
                per_price_unit = max(pip_value * multiplier, 1e-9)
                stop_distance = float(getattr(cfg, "atr_stop_multiplier", 1.5)) * atr
                lot = risk_eur / max(stop_distance * per_price_unit, 1e-9)
                lot = float(np.clip(lot, 0.01, 10.0))
                # Recompute so the position's own stop matches the lot actually taken.
                initial_risk = float(lot * stop_distance * per_price_unit)
                return lot, initial_risk

        lot = risk_eur / stop_eur
        lot = float(np.clip(lot, 0.01, 10.0))
        initial_risk = float(lot * stop_eur)
        return lot, initial_risk

    def _update_peak_balance(self) -> None:
        if bool(self.config.trailing_drawdown):
            self.peak_balance = max(self.peak_balance, float(self.equity))
        else:
            self.peak_balance = max(self.peak_balance, float(self.balance))

    def _calc_dds(self) -> Tuple[float, float]:
        values = {
            "equity": float(self.equity),
            "initial_balance": float(self.config.initial_balance),
            "peak_balance": float(self.peak_balance),
            "day_start_balance": float(self.day_start_balance),
        }
        if any((not np.isfinite(v)) for v in values.values()):
            raise ValueError(f"non-finite account risk state: {values}")
        if values["initial_balance"] <= 0.0 or values["peak_balance"] <= 0.0 or values["day_start_balance"] <= 0.0:
            raise ValueError(f"non-positive account risk anchor: {values}")
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


        self.balance += realized_pnl
        self.equity = self.balance
        self.total_pnl += realized_pnl
        self.daily_pnl += realized_pnl


        self.session_pnl += realized_pnl

        self.total_trades += 1


        if net_trade_pnl > 0:
            self.winning_trades += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0

            self.session_consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

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
            entry_context=pos.entry_context if hasattr(pos, "entry_context") else None,
        )

        self._episode_trade_results.append(result)

        self.position = None
        self.pending_exit = None
        self._update_peak_balance()

        return result


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


        if self._loss_layer() >= self._loss_layer_stop():
            return False, "loss_layer_stop"


        session_loss_limit = getattr(self.config, "session_loss_limit_pct", 0.99)
        if self.session_start_balance > 0:
            session_pnl_pct = self.session_pnl / self.session_start_balance
            if session_pnl_pct < -session_loss_limit:
                return False, "session_loss_limit"


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


        if getattr(self.config, "observation_period_required", False):
            min_obs_bars = int(getattr(self.config, "min_bars_observation_before_entry", 0) or 0)
            if min_obs_bars > 0 and int(getattr(self, "episode_bars", 0)) < min_obs_bars:
                return False, "observation_period"

        if self.consecutive_losses >= self.config.max_consecutive_losses:
            return False, "max_consecutive_losses"

        # This is the authoritative non-negotiable account veto.  It is called
        # from the mask, direct action handling, and pending-fill revalidation;
        # a caller that ignores an advisory mask still cannot open risk.
        if self._entry_blocked_by_drawdown_values(current_dd, current_daily_dd):
            return False, "drawdown_entry_veto"

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


    def _get_primary_data_length(self) -> int:
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

    @staticmethod
    def _utc_timestamp_ns(value: Any) -> int:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return int(ts.value)

    def _episode_sampling_bounds(self, history_min: int, latency: int) -> Tuple[int, int]:
        """Return a half-open range of legal episode starts.

        Observation history is only a lower bound.  It must never also be
        subtracted from the tail, which previously made the newest 3,850 M15
        bars unreachable and removed the very regime this run needed to learn.
        """
        lower = max(1, int(history_min))
        upper = int(self._min_data_len) - int(self.config.max_steps_per_episode) - max(0, int(latency))

        primary = self.data.get(self._episode_instrument, {}).get(self._primary_tf())
        times = self._df_time_ns(primary) if primary is not None else None
        min_time = getattr(self.config, "episode_start_min_time", None)
        max_time = getattr(self.config, "episode_start_max_time", None)
        if times is not None and len(times) > 0:
            if min_time is not None:
                lower = max(lower, int(np.searchsorted(times, self._utc_timestamp_ns(min_time), side="left")))
            if max_time is not None:
                upper = min(upper, int(np.searchsorted(times, self._utc_timestamp_ns(max_time), side="right")))
        elif min_time is not None or max_time is not None:
            raise ValueError("episode time bounds require a valid primary timestamp column")

        if upper <= lower:
            raise ValueError(
                "no legal episode start: "
                f"history_min={history_min}, eligible=[{lower},{upper}), "
                f"data_bars={self._min_data_len}, max_steps={self.config.max_steps_per_episode}, "
                f"latency={latency}, min_time={min_time!r}, max_time={max_time!r}"
            )
        return lower, upper

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)


        if self.curriculum:
            effective_stage = getattr(self.curriculum, "current_stage", None)
            if hasattr(self.curriculum, "get_effective_stage"):
                effective_stage = self.curriculum.get_effective_stage()
            self._episode_effective_stage_name = str(getattr(effective_stage, "name", ""))

            # Effective curriculum state is episode-scoped.  In addition to a
            # stage change, reward blending, recovery, review, mixed-stage
            # rehearsal, and selectivity constraints can change between two
            # resets while current_stage stays constant.  Refresh the cheap
            # overrides every episode; the expensive difficulty cache above is
            # retained when its config did not change.
            self._sync_curriculum_stage_overrides()
            self._pending_stage_apply = False
            self._last_stage_name = getattr(self.curriculum.current_stage, "name", "")
            self._last_stage_epoch = int(getattr(self.curriculum, "current_stage_epoch", 0))
        else:
            self._episode_effective_stage_name = ""


        # Before anything reads prices this episode: swaps self.data between
        # the original and its mirror, so direction is symmetric across the run.
        self._select_episode_data()

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
        self._mask_decision_steps = 0
        self._mask_collapse_steps = 0
        self._stop_mode_steps = 0

        self._avg_vol = None
        self._episode_trade_results = []
        self._last_reward_components = {}
        self._episode_max_drawdown = 0.0

        self._episode_reward_components = {}
        self._episode_reward_component_counts = {}
        self._episode_reward_total = 0.0


        self._episode_gross_profit = 0.0
        self._episode_total_costs = 0.0


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


        effective_spread_mult = float(self._episode_spread_mult) * float(getattr(self, "_scenario_spread_mult", 1.0) or 1.0)
        effective_slip_mult = float(self._episode_slip_mult) * float(getattr(self, "_scenario_slippage_mult", 1.0) or 1.0)
        self._exec.set_episode_randomization(
            spread_mult=effective_spread_mult,
            slippage_mult=effective_slip_mult,
        )


        buffer = self._episode_start_buffer()
        latency = max(0, self._exec_latency())
        buffer, max_start = self._episode_sampling_bounds(buffer, latency)


        validation_schedule = getattr(self, "_validation_start_schedule", None)
        if validation_schedule is not None:
            cursor = int(getattr(self, "_validation_start_cursor", 0))
            if cursor >= len(validation_schedule):
                raise RuntimeError("validation episode-start schedule exhausted")
            candidate = int(validation_schedule[cursor])
            self._validation_start_cursor = cursor + 1
            if not buffer <= candidate < max_start:
                raise RuntimeError(
                    f"scheduled validation start {candidate} left legal range "
                    f"[{buffer}, {max_start})"
                )
            if self._valid_start_indices is not None and not bool(
                np.any(self._valid_start_indices == candidate)
            ):
                raise RuntimeError(
                    f"scheduled validation start {candidate} violates scenario filter"
                )
            self.current_step = candidate
        elif self._data_difficulty is not None:
            self.current_step = self._sample_episode_start_with_difficulty(buffer, max_start)
        elif max_start > buffer:
            self.current_step = self._sample_episode_start(buffer, max_start)
        else:
            raise ValueError(f"no legal episode start in [{buffer}, {max_start})")

        inst = self._episode_instrument
        dt = self._get_bar_dt(inst)
        if dt is None:
            raise RuntimeError(
                "episode start has no valid market-data timestamp: "
                f"instrument={inst!r}, step={self.current_step}"
            )
        self._episode_start_index = int(self.current_step)
        self._episode_start_time = pd.Timestamp(dt).isoformat()


        simclock.reset()
        simclock.set_bar_time(dt, step=int(self.current_step))
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
                "effective_curriculum_stage": self._episode_effective_stage_name,
                "stage_epoch": int(getattr(self.curriculum, "current_stage_epoch", 0)),
            }
        except Exception:
            return {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:

        self.current_step += 1
        self.episode_step += 1
        self.episode_bars += 1

        inst = self._episode_instrument
        dt = self._get_bar_dt(inst)


        simclock.set_bar_time(dt, step=int(self.current_step))


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


        can_entry_fill = remaining_bars >= (latency + 2)
        can_exit_fill = remaining_bars >= latency

        reward = 0.0
        trade_closed = False
        close_result: Optional[TradeResult] = None


        pnl_u = 0.0
        if self.position is not None:
            pnl_u = self._mark_unrealized_pnl_from_bid_ask(self.position, bid, ask)
            self.equity = self.balance + pnl_u
            self.position.peak_pnl = max(self.position.peak_pnl, pnl_u)
            self.position.lowest_pnl = min(self.position.lowest_pnl, pnl_u)
        else:
            self.equity = self.balance

        self._update_peak_balance()


        prev_session_key = getattr(self, "_current_session_key", None)
        self._maybe_roll_day_session(dt)
        session_rolled = prev_session_key != getattr(self, "_current_session_key", None)
        self._maybe_relax_loss_layer_on_session_roll(session_rolled=session_rolled)

        current_dd, current_daily_dd = self._calc_dds()


        if self.position is not None:
            stop_mid = self._intrabar_stop_mid(self.position)
            if stop_mid is not None:
                close_result = self._close_position_now(
                    reason=CloseReason.HARD_STOP.value,
                    dt=dt,
                    mid=stop_mid,
                    vol_proxy=vol_proxy,
                )
                trade_closed = True
                current_dd, current_daily_dd = self._calc_dds()
                reward += self._compute_trade_reward(close_result, current_dd)


        forced_close_now = False
        close_reason_str = ""

        if self.position is not None:
            pos = self.position

            # Per-position: with ATR sizing every trade has its own stop distance,
            # so a single global euro threshold would fire at the wrong place.
            stop_at = float(getattr(pos, "initial_risk_eur", 0.0) or 0.0)
            if stop_at <= 0.0:
                stop_at = float(self.config.hard_stop_loss_eur)
            if pnl_u <= -stop_at:
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


        if self.position is not None and not forced_close_now and can_exit_fill:
            if intent == "close":
                self._set_or_override_pending_exit(
                    reason=CloseReason.AGENT_CLOSE.value,
                    fill_step=self.current_step + latency,
                )


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


        q_long = self._get_step_entry_quality(inst, "long")
        q_short = self._get_step_entry_quality(inst, "short")
        cert_long = self._get_step_entry_certainty(inst, "long")
        cert_short = self._get_step_entry_certainty(inst, "short")
        setup_long, confluence_long = self._get_step_setup_quality(inst, "long")
        setup_short, confluence_short = self._get_step_setup_quality(inst, "short")


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
                    pip_value, multiplier = self._get_pip_value(inst)
                    per_price_unit = max(float(pip_value) * float(multiplier), 1e-9)
                    stop_distance = float(initial_risk) / max(float(lot) * per_price_unit, 1e-9)
                    if not (np.isfinite(stop_distance) and stop_distance > 0.0):
                        raise ValueError(f"invalid executable stop distance: {stop_distance!r}")
                    pos.stop_distance_price = stop_distance
                    pos.stop_price = (
                        float(entry_fill) - stop_distance
                        if direction == "long"
                        else float(entry_fill) + stop_distance
                    )
                    self.position = pos


                    u0 = self._mark_unrealized_pnl_from_bid_ask(pos, bid, ask)
                    self.equity = self.balance + float(u0)


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


                    if self._last_trade_entry_bar is not None:
                        bars_between = int(self.episode_bars - int(self._last_trade_entry_bar))
                        self._bars_between_trades.append(bars_between)
                        self._max_patience_bars = max(self._max_patience_bars, bars_between)
                    self._last_trade_entry_bar = int(self.episode_bars)


                    if is_fomo_entry:
                        self._fomo_trade_count += 1
                    if is_revenge_entry:
                        self._revenge_trade_count += 1
                else:

                    if self._setup_active:
                        self._setup_taken_during_active = False

                self.pending_entry = None


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

            max_trades_ep = int(getattr(self.config, "max_trades_per_episode", 0) or 0)
            if max_trades_ep > 0 and int(self.total_trades) >= max_trades_ep:
                entry_allowed = False
                block_reason = "max_trades_per_episode"


            base_threshold = float(self.config.entry_quality_threshold)
            progressive_threshold = self._loss_layer_entry_quality_threshold(base_threshold)

            if self.config.entry_quality_gate_enabled and float(entry_quality) < progressive_threshold:
                entry_allowed = False
                block_reason = "entry_quality_gate"


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
                    # Orders are decided after the current bar is complete.
                    # latency=0 therefore fills on the next bar; each extra
                    # latency bar delays it by one additional bar.
                    "fill_step": self.current_step + latency + 1,
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


        if attempted_entry and not entry_allowed:
            hard_blocks = {
                "hard_close",
                "drawdown_headroom",
                "drawdown_entry_veto",
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
                "loss_layer_stop",
            }
            is_hard = block_reason in hard_blocks
            penalty = self._compute_blocked_action_penalty(block_reason, entry_quality, is_hard)
            reward -= penalty


        current_dd, current_daily_dd = self._calc_dds()
        firm_daily_limit, firm_max_limit = self._firm_drawdown_limits()
        dd_breach = current_dd >= firm_max_limit
        daily_dd_breach = current_daily_dd >= firm_daily_limit

        did_risk_liquidate_this_step = False
        if (dd_breach or daily_dd_breach) and self.position is not None:
            close_result = self._close_position_now(
                reason=CloseReason.RISK_LIQUIDATION.value, dt=dt, mid=mid, vol_proxy=vol_proxy
            )
            did_risk_liquidate_this_step = True
            trade_closed = True
            current_dd, current_daily_dd = self._calc_dds()
            reward += self._compute_trade_reward(close_result, current_dd)

            dd_breach = current_dd >= firm_max_limit
            daily_dd_breach = current_daily_dd >= firm_daily_limit


        terminated = False
        truncated = False
        termination_reason = ""

        if self.current_step >= self._min_data_len - 1:
            truncated = True
            termination_reason = "data_exhausted"
        elif self.episode_step >= int(self.config.max_steps_per_episode):
            truncated = True
            termination_reason = "episode_length"


        if truncated and not terminated and self.position is not None:


            close_result = self._close_position_now(
                reason=CloseReason.EPISODE_TRUNCATE.value, dt=dt, mid=mid, vol_proxy=vol_proxy
            )
            trade_closed = True
            current_dd, current_daily_dd = self._calc_dds()
            reward += self._compute_trade_reward(close_result, current_dd)


        if truncated and not terminated:
            if self.pending_entry is not None:
                self.pending_entry = None
                reward -= 0.01
            if self.pending_exit is not None:
                self.pending_exit = None

        current_dd, current_daily_dd = self._calc_dds()
        dd_breach = current_dd >= firm_max_limit
        daily_dd_breach = current_daily_dd >= firm_daily_limit


        if dd_breach or daily_dd_breach:
            terminated = True
            truncated = False
            if dd_breach:
                termination_reason = "max_drawdown_breach"
            else:
                termination_reason = "daily_limit_breach"

        # Trade count is evidence, not an economic objective.  A lower-bound
        # activity reward/penalty makes the agent trade merely to satisfy the
        # curriculum and is discontinuous around the configured quota.  Keep
        # abstention neutral here; promotion/validation separately decide
        # whether enough independent trades exist to claim an edge.
        reward_cfg = self.config.reward
        if (terminated or truncated) and bool(getattr(reward_cfg, "activity_consistency_enabled", False)):
            target_per_1k = float(getattr(reward_cfg, "target_trades_per_1k_steps", 0.0) or 0.0)
            stage_targets = getattr(reward_cfg, "stage_activity_targets", None)
            if isinstance(stage_targets, dict):
                target_per_1k = float(stage_targets.get(getattr(self, "_curriculum_stage_idx", 0), target_per_1k))
            expected = (max(self.episode_step, 1) / 1000.0) * target_per_1k
            ratio = float(self.total_trades) / expected if expected >= 2.0 else 0.0
            if ratio > 1.5:
                scale = float(getattr(reward_cfg, "activity_deviation_penalty_scale", 0.0) or 0.0)
                cap = float(getattr(reward_cfg, "activity_deviation_penalty_cap", 2.0) or 2.0)
                activity_penalty = min((ratio - 1.5) * scale, cap)
                reward -= activity_penalty
                self._episode_reward_components["overactivity_penalty"] = -activity_penalty
                self._episode_reward_component_counts["overactivity_penalty"] = 1


        bars_in_pos = (self.episode_bars - self.position.entry_bar) if self.position else 0


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


        reward += self._compute_governor_approaching_penalties()


        cfg = self.config.reward
        is_dd_termination = termination_reason in ("max_drawdown_breach", "daily_limit_breach")
        if is_dd_termination:
            # Apply the account-forfeit correction after every shaping term and
            # outside the ordinary per-step clip.  This guarantees that no
            # amount of reward accumulated before a breach can leave the episode
            # profitable in reward space.
            breach_penalty = float(getattr(cfg, "dd_breach_penalty", 25.0))
            if termination_reason == "daily_limit_breach":
                breach_penalty *= 0.6
            prior_total = float(getattr(self, "_episode_reward_total", 0.0))
            reward = min(float(reward), -prior_total - breach_penalty)
        else:
            reward = float(np.clip(reward, cfg.min_reward, cfg.max_reward))


        self._episode_return += float(reward)


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

        # Running total, used to price a drawdown breach against what the
        # episode actually earned.
        self._episode_reward_total = float(getattr(self, "_episode_reward_total", 0.0)) + float(reward)

        if terminated or truncated:
            info["episode_stats"] = self.get_episode_stats()
        else:
            self._track_action_mask_state_for_metrics()

        return obs, reward, terminated, truncated, info


    def _get_observation(self) -> np.ndarray:
        if self.obs_builder is None:
            return self._fallback_observation()
        return self._build_observation_with_builder()

    def _build_observation_with_builder(self) -> np.ndarray:
        inst = self._episode_instrument
        market_data = self._prepare_market_data(inst)
        expert_signals = self._prepare_expert_signals(inst)
        risk_state = self._prepare_risk_state()
        account_state = self._prepare_account_state(inst)
        trading_mode_state = self._prepare_trading_mode_state(inst)
        governor_state = self._get_governor_state()
        session_state = self._prepare_session_state(inst)

        assert self.obs_builder is not None
        obs = self.obs_builder.build(
            market_data=market_data,
            expert_signals=expert_signals,
            risk_state=risk_state,
            account_state=account_state,
            trading_mode_state=trading_mode_state,
            governor_state=governor_state,
            session_state=session_state,
        )
        return obs

    def _prepare_market_data(self, instrument: str) -> Dict[str, Any]:

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

        self._ohlcv_cache.clear()
        self._ohlcv_cache_key = None
        self._quote_cache_step = None
        self._episode_trade_results.clear()


        self.position = None
        self.pending_entry = None
        self.pending_exit = None


        self._exec = None
        self._episode_execution_cfg = None


    def get_episode_stats(self) -> Dict[str, Any]:
        decision_steps = int(getattr(self, "_mask_decision_steps", self.episode_step) or 0)
        mask_collapse_steps = int(getattr(self, "_mask_collapse_steps", 0) or 0)
        stop_mode_steps = int(getattr(self, "_stop_mode_steps", 0) or 0)
        denom = max(decision_steps, 1)
        end_time = ""
        try:
            end_dt = self._get_bar_dt(self._episode_instrument)
            if end_dt is not None:
                end_time = pd.Timestamp(end_dt).isoformat()
        except Exception:
            pass
        episode_identity = {
            "episode_start_time": str(getattr(self, "_episode_start_time", "") or ""),
            "episode_end_time": end_time,
            "episode_start_index": int(getattr(self, "_episode_start_index", 0) or 0),
            "episode_end_index": int(self.current_step),
            "episode_length": int(self.episode_step),
            "effective_curriculum_stage": str(
                getattr(self, "_episode_effective_stage_name", "") or ""
            ),
        }

        if not self._episode_trade_results:
            return {
                **episode_identity,
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
            **episode_identity,
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
            "direction_stats": direction_stats,
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

            "governor_state": self._get_governor_state(),
            # So the augmentation is observable rather than an invisible
            # mechanism nobody can confirm is running.
            "mirrored_episode": bool(self._mirror_active),
        }

    def _build_trades_with_regime(self, results: List[TradeResult]) -> List[Dict[str, Any]]:
        trades_with_regime = []
        for r in results:
            entry_context = r.entry_context or {}

            vol_regime = entry_context.get("volatility_regime", "medium")
            risk_regime = entry_context.get("risk_regime", "neutral")

            # structure_trend is a direction, not a magnitude: the producer
            # emits exactly -1.0, 0.0 or +1.0 (HH+HL, LL+LH, neither). Banding
            # it by |v| > 0.5 / > 0.2 therefore made "weak_trend" unreachable -
            # 2,000 trades produced 1,104 strong_trend, 896 ranging and zero
            # weak_trend, and the trend_following skill score was computed over
            # a permanently empty bucket. structure_strength is the continuous
            # 0..1 magnitude the band actually wanted, and travels in the same
            # entry context.
            trend_direction = float(entry_context.get("structure_trend", 0.0))
            trend_strength = float(entry_context.get("structure_strength", 0.0))
            if trend_direction == 0.0:
                trend_regime = "ranging"
            elif trend_strength > 0.5:
                trend_regime = "strong_trend"
            else:
                trend_regime = "weak_trend"

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
                    # In R, the unit that decides whether a reward:risk target
                    # is reachable. mfe_r is how far the trade ran in our
                    # favour before we closed it; net_pnl/mfe is how much of
                    # that run we actually kept, which is the number that says
                    # whether winners are being cut short.
                    "mae_r": float(abs(r.mae) / max(r.initial_risk_eur, 1.0)),
                    "mfe_r": float(abs(r.mfe) / max(r.initial_risk_eur, 1.0)),
                    # What the trade actually risked, so size selection is
                    # visible rather than inferred from P&L.
                    "risk_eur": float(r.initial_risk_eur),
                    "lot_size": float(r.lot_size),
                    "fees_eur": float(getattr(r, "total_fees", 0.0)),
                    "entry_quality": float(r.entry_quality),
                    "exit_type": r.close_reason.value,
                    "volatility_regime": vol_regime,
                    "trend_regime": trend_regime,
                    "session_regime": session_regime,
                    "spread_regime": spread_regime,
                }
            )

        return trades_with_regime
