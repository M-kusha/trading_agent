

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import Any, ClassVar, Deque, Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np

from envs.core.shared_utils import (
    clamp as _clamp,
)
from envs.core.shared_utils import (
    get_envs_logger,
    iso_timestamp,
)
from envs.core.shared_utils import (
    safe_float as _sf,
)
from envs.core.shared_utils import (
    safe_int as _si,
)
from envs.curriculum.config.thresholds import (
    AdaptiveThresholdConfig,
    CompetenceThresholds,
    CompositeScoringConfig,
)

logger = get_envs_logger("curriculum.metrics")

DEFAULT_TZ = "Europe/Berlin"


def _now_iso(tz: str = DEFAULT_TZ) -> str:
    try:
        return datetime.now(tz=ZoneInfo(tz)).isoformat()
    except Exception as e:
        logger.debug(f"Timezone fallback for {tz}: {e}")
        return iso_timestamp()


def _linear_regression_slope(y: np.ndarray) -> float:
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=np.float64)
    try:
        coeffs = np.polyfit(x, y, 1)
        return float(coeffs[0])
    except Exception as e:
        logger.debug(f"Linear regression failed: {e}")
        return 0.0


@dataclass
class EpisodeMetrics:


    _FLOAT_FIELDS: ClassVar[set] = set()
    _INT_FIELDS: ClassVar[set] = set()
    _BOOL_FIELDS: ClassVar[set] = set()
    _STR_FIELDS: ClassVar[set] = set()

    total_pnl: float = 0.0
    win_rate: float = 0.0
    trade_count: int = 0
    winning_trades: int = 0
    losing_trades: int = 0


    max_drawdown: float = 0.0
    daily_drawdown: float = 0.0
    dd_breach: bool = False


    avg_r_multiple: float = 0.0
    profit_factor: float = 0.0
    avg_mae: float = 0.0
    avg_mfe: float = 0.0
    avg_bars_held: float = 0.0
    avg_entry_quality: float = 0.5
    avg_bars_between_trades: float = 0.0
    avg_setup_quality: float = 0.0
    avg_entry_certainty: float = 0.0
    min_setup_quality_for_entry: float = 0.0


    consecutive_losses: int = 0
    consecutive_wins: int = 0
    max_consecutive_losses_reached: int = 0
    hit_max_consecutive_losses: bool = False
    mask_collapse_steps: int = 0
    # The denominator those collapse/stop counts belong over. The mask
    # tracker runs from two call sites per env step, so dividing its
    # counts by episode length yields rates above 1.0.
    mask_decision_steps: int = 0
    stop_mode_steps: int = 0
    setup_skipped_count: int = 0
    fomo_trade_count: int = 0
    revenge_trade_count: int = 0
    max_patience_bars: int = 0


    trailing_stop_exits: int = 0
    agent_close_exits: int = 0
    hard_stop_exits: int = 0
    risk_liquidation_exits: int = 0
    other_exits: int = 0


    episode_length: int = 0
    episode_reward: float = 0.0
    termination_reason: str = ""


    stage_name: str = ""
    stage_epoch: int = 0
    global_episode_idx: int = 0


    policy_entropy: float = -1.0

    timestamp: str = field(default_factory=lambda: _now_iso(DEFAULT_TZ))

    def __post_init__(self) -> None:


        if not EpisodeMetrics._FLOAT_FIELDS:
            EpisodeMetrics._FLOAT_FIELDS = {
                "total_pnl", "win_rate", "max_drawdown", "daily_drawdown",
                "avg_r_multiple", "profit_factor", "avg_mae", "avg_mfe",
                "avg_bars_held", "avg_entry_quality", "avg_bars_between_trades",
                "avg_setup_quality", "avg_entry_certainty", "min_setup_quality_for_entry",
                "episode_reward",
                "policy_entropy",
            }
            EpisodeMetrics._INT_FIELDS = {
                "trade_count", "winning_trades", "losing_trades",
                "consecutive_losses", "consecutive_wins",
                "max_consecutive_losses_reached",
                "mask_collapse_steps", "mask_decision_steps", "stop_mode_steps",
                "setup_skipped_count", "fomo_trade_count", "revenge_trade_count",
                "max_patience_bars",
                "trailing_stop_exits", "agent_close_exits", "hard_stop_exits",
                "risk_liquidation_exits", "other_exits",
                "episode_length", "stage_epoch", "global_episode_idx",
            }
            EpisodeMetrics._BOOL_FIELDS = {"dd_breach", "hit_max_consecutive_losses"}
            EpisodeMetrics._STR_FIELDS = {"termination_reason", "stage_name", "timestamp"}


        for k in EpisodeMetrics._FLOAT_FIELDS:
            v = getattr(self, k, 0.0)
            fv = _sf(v, getattr(self, k))
            if not np.isfinite(fv):
                fv = getattr(self, k)
            setattr(self, k, float(fv))
        for k in EpisodeMetrics._INT_FIELDS:
            setattr(self, k, int(_si(getattr(self, k, 0), 0)))
        for k in EpisodeMetrics._BOOL_FIELDS:
            setattr(self, k, bool(getattr(self, k, False)))
        for k in EpisodeMetrics._STR_FIELDS:
            setattr(self, k, str(getattr(self, k, "") or ""))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EpisodeMetrics":
        allowed = {f.name for f in fields(cls)}
        payload_raw = {k: v for k, v in (d or {}).items() if k in allowed}


        payload: Dict[str, Any] = {}

        if not cls._FLOAT_FIELDS:
            _ = cls()
        for k, v in payload_raw.items():
            if k in cls._FLOAT_FIELDS:
                fv = _sf(v, 0.0)
                payload[k] = float(fv) if np.isfinite(fv) else 0.0
            elif k in cls._INT_FIELDS:
                payload[k] = int(_si(v, 0))
            elif k in cls._BOOL_FIELDS:
                payload[k] = bool(v)
            else:
                payload[k] = v
        return cls(**payload)


@dataclass
class RollingStats:
    window_size: int


    mean_pnl: float = 0.0
    std_pnl: float = 0.0
    mean_win_rate: float = 0.0
    std_win_rate: float = 0.0


    std_win_rate_trade_weighted: float = 0.0
    std_win_rate_eligible_unweighted: float = 0.0
    win_rate_stability_eligible_episodes: int = 0
    win_rate_stability_eligible_trades: int = 0
    mean_trade_count: float = 0.0
    total_trades: int = 0
    total_wins: int = 0
    total_losses: int = 0
    win_rate_trade_weighted: float = 0.0


    mean_drawdown: float = 0.0
    max_drawdown_seen: float = 0.0
    dd_breach_rate: float = 0.0


    mean_profit_factor: float = 0.0
    mean_r_multiple: float = 0.0
    mean_entry_quality: float = 0.5
    mean_bars_between_trades: float = 0.0
    std_bars_between_trades: float = 0.0
    mean_setup_skipped_per_episode: float = 0.0
    mean_entry_certainty: float = 0.0
    mean_setup_quality: float = 0.0
    mean_min_setup_quality_for_entry: float = 0.0
    fomo_trade_rate: float = 0.0
    revenge_trade_rate: float = 0.0
    mean_max_patience_bars: float = 0.0
    patience_consistency: float = 0.0


    consecutive_loss_breach_rate: float = 0.0
    avg_max_consecutive_losses: float = 0.0
    consecutive_loss_streak_rate: float = 0.0
    mask_collapse_rate: float = 0.0
    stop_mode_rate: float = 0.0


    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    win_loss_ratio: float = 0.0


    win_rate_wilson_low: float = 0.0
    win_rate_wilson_high: float = 0.0
    win_rate_wilson_width: float = 0.0
    pnl_mean_ci_low: float = 0.0
    pnl_mean_ci_high: float = 0.0


    mean_entropy: float = -1.0
    std_entropy: float = 0.0
    entropy_samples: int = 0


    trailing_stop_rate: float = 0.0
    agent_close_rate: float = 0.0
    hard_stop_rate: float = 0.0
    risk_liquidation_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:

        from dataclasses import asdict
        return asdict(self)


@dataclass
class LearningVelocity:

    metric_history: Dict[str, Deque[float]] = field(default_factory=dict)


    improvement_rates: Dict[str, float] = field(default_factory=dict)


    plateau_episodes: int = 0
    plateau_threshold: float = 0.005


    history_window: int = 100


    min_samples: int = 20


    LOWER_IS_BETTER: ClassVar[set] = {
        "max_drawdown", "daily_drawdown", "avg_mae", "consecutive_losses",
        "hard_stop_exits", "risk_liquidation_exits",
        "max_consecutive_losses_reached",
    }

    def has_sufficient_samples(self) -> bool:
        for hist in self.metric_history.values():
            if len(hist) >= self.min_samples:
                return True
        return False

    def update(self, metrics: Dict[str, float]) -> None:
        any_improving = False
        has_rate_estimates = False

        for name, value in metrics.items():

            fv = _sf(value, float('nan'))
            if not np.isfinite(fv):
                continue

            if name not in self.metric_history:
                self.metric_history[name] = deque(maxlen=self.history_window)

            self.metric_history[name].append(fv)


            if len(self.metric_history[name]) >= self.min_samples:
                has_rate_estimates = True
                y = np.array(list(self.metric_history[name]), dtype=np.float64)
                slope = _linear_regression_slope(y)


                mean_val = np.mean(y)
                if abs(mean_val) > 1e-8:
                    normalized_slope = slope / abs(mean_val)
                else:
                    normalized_slope = slope


                if name in self.LOWER_IS_BETTER:
                    normalized_slope = -normalized_slope

                self.improvement_rates[name] = float(normalized_slope)


                if normalized_slope > self.plateau_threshold:
                    any_improving = True


        if not has_rate_estimates:

            pass
        elif any_improving:
            self.plateau_episodes = 0
        else:
            self.plateau_episodes += 1

    def is_plateaued(self, threshold_episodes: int = 100) -> bool:
        return self.plateau_episodes >= threshold_episodes

    def get_average_improvement(self) -> float:
        if not self.improvement_rates:
            return 0.0
        return float(np.mean(list(self.improvement_rates.values())))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "improvement_rates": dict(self.improvement_rates),
            "plateau_episodes": self.plateau_episodes,
            "plateau_threshold": self.plateau_threshold,
            "average_improvement": self.get_average_improvement(),
            "has_sufficient_samples": self.has_sufficient_samples(),

            "is_plateaued_default": self.is_plateaued(),
        }


@dataclass
class CompositeScore:
    total_score: float = 0.0
    component_scores: Dict[str, float] = field(default_factory=dict)
    meets_hard_floors: bool = True
    hard_floor_failures: List[str] = field(default_factory=list)
    promotion_ready: bool = False
    demotion_risk: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_score": self.total_score,
            "component_scores": self.component_scores,
            "meets_hard_floors": self.meets_hard_floors,
            "hard_floor_failures": self.hard_floor_failures,
            "promotion_ready": self.promotion_ready,
            "demotion_risk": self.demotion_risk,
        }


def compute_composite_score(
    stats: RollingStats,
    thresholds: CompetenceThresholds,
    config: CompositeScoringConfig,
) -> CompositeScore:
    if not config.enabled:
        return CompositeScore()

    components: Dict[str, float] = {}


    # A stage that sets no requirement has not been passed - it has not been
    # assessed. Dividing by an epsilon floor (1e-6, 0.01) turned "no bar to
    # clear" into a perfect 1.000: at stage 0, where min_win_rate and
    # min_profit_factor are both 0.0, a 49.7% coin flip and a profit factor of
    # 1.02 each scored 1.000 and carried the composite to 0.94 with
    # promotion_ready True. 0.5 is the honest encoding of no evidence either
    # way, and matches how the drawdown and consistency branches below already
    # treat an absent threshold.
    UNASSESSED = 0.5

    min_win_rate = float(getattr(thresholds, "min_win_rate", 0.0) or 0.0)
    if min_win_rate > 0.0:
        components["win_rate"] = min(stats.mean_win_rate / min_win_rate, 1.5) / 1.5
    else:
        components["win_rate"] = UNASSESSED

    min_profit_factor = float(getattr(thresholds, "min_profit_factor", 0.0) or 0.0)
    if min_profit_factor > 0.0:
        components["profit_factor"] = min(stats.mean_profit_factor / min_profit_factor, 2.0) / 2.0
    else:
        components["profit_factor"] = UNASSESSED


    if thresholds.max_avg_drawdown > 0:
        dd_ratio = stats.mean_drawdown / thresholds.max_avg_drawdown
        dd_score = 1.0 - min(dd_ratio, 1.5) / 1.5
    else:
        dd_score = 1.0 if stats.mean_drawdown <= 0.01 else 0.5
    components["drawdown"] = max(0, dd_score)


    cons_std = float(getattr(stats, "std_win_rate_trade_weighted", stats.std_win_rate))
    if thresholds.max_win_rate_std > 0:
        cons_ratio = cons_std / thresholds.max_win_rate_std
        cons_score = 1.0 - min(cons_ratio, 1.5) / 1.5
    else:
        cons_score = 1.0 if cons_std <= 0.05 else 0.5
    components["consistency"] = max(0, cons_score)


    r_score = _clamp((stats.mean_r_multiple + 0.5) / 1.0, 0.0, 1.0)
    components["r_multiple"] = r_score


    if thresholds.max_dd_breach_rate > 0:
        breach_ratio = stats.dd_breach_rate / thresholds.max_dd_breach_rate
        breach_score = 1.0 - min(breach_ratio, 1.5) / 1.5
    else:
        breach_score = 1.0 if stats.dd_breach_rate <= 0.05 else 0.5
    components["dd_breach_rate"] = max(0, breach_score)


    # Two-sided. The old score was min(count/min_count, 2)/2 - monotonically
    # increasing in trade count with no ceiling - so a policy trading 366 times
    # against a stage target of 12 scored a perfect 1.0, exactly like one
    # trading 24 times. The promotion gate was paying for the over-trading the
    # activity_consistency_penalty charges for.
    max_trades = float(getattr(thresholds, "max_trade_count_avg", 0.0) or 0.0)
    if thresholds.min_trade_count_avg > 0:
        trade_score = min(stats.mean_trade_count / thresholds.min_trade_count_avg, 2.0) / 2.0
    else:
        trade_score = 0.5
    if max_trades > 0 and stats.mean_trade_count > max_trades:
        # Decay to zero by 3x the ceiling, so the score keeps a gradient across
        # the whole range a runaway policy actually reaches instead of pinning.
        excess = (stats.mean_trade_count - max_trades) / (2.0 * max_trades)
        trade_score *= max(0.0, 1.0 - min(excess, 1.0))
    components["trade_activity"] = trade_score


    if thresholds.max_consecutive_loss_rate > 0:
        cl_ratio = stats.consecutive_loss_breach_rate / thresholds.max_consecutive_loss_rate
        cl_score = 1.0 - min(cl_ratio, 1.5) / 1.5
    else:
        cl_score = 1.0 if stats.consecutive_loss_breach_rate <= 0.05 else 0.5
    components["consecutive_loss_rate"] = max(0, cl_score)


    weight_sum = sum(config.weights.values())
    if weight_sum <= 0:
        weight_sum = 1.0

    total_score = sum(
        components.get(name, 0.5) * weight
        for name, weight in config.weights.items()
    ) / weight_sum


    hard_floor_failures: List[str] = []
    meets_floors = True

    if "win_rate" in config.hard_floors:
        if stats.mean_win_rate < config.hard_floors["win_rate"]:
            meets_floors = False
            hard_floor_failures.append("win_rate")

    if "max_drawdown" in config.hard_floors:
        if stats.mean_drawdown > config.hard_floors["max_drawdown"]:
            meets_floors = False
            hard_floor_failures.append("max_drawdown")

    if "dd_breach_rate" in config.hard_floors:
        if stats.dd_breach_rate > config.hard_floors["dd_breach_rate"]:
            meets_floors = False
            hard_floor_failures.append("dd_breach_rate")


    if "profit_factor" in config.hard_floors:
        if stats.mean_profit_factor < config.hard_floors["profit_factor"]:
            meets_floors = False
            hard_floor_failures.append("profit_factor")


    if "r_multiple" in config.hard_floors:
        if stats.mean_r_multiple < config.hard_floors["r_multiple"]:
            meets_floors = False
            hard_floor_failures.append("r_multiple")


    promotion_ready = meets_floors and (total_score >= config.promotion_threshold)
    demotion_risk = total_score < config.demotion_threshold

    return CompositeScore(
        total_score=total_score,
        component_scores=components,
        meets_hard_floors=meets_floors,
        hard_floor_failures=hard_floor_failures,
        promotion_ready=promotion_ready,
        demotion_risk=demotion_risk,
    )


def compute_adjusted_thresholds(
    base: CompetenceThresholds,
    velocity: LearningVelocity,
    config: AdaptiveThresholdConfig,
    composite_score: Optional["CompositeScore"] = None,
    promotion_threshold: float = 0.7,
) -> CompetenceThresholds:
    if not config.enabled:
        return base

    if velocity.plateau_episodes < config.plateau_episodes_threshold:
        return base


    if composite_score is not None:
        proximity_margin = 0.15
        if composite_score.total_score < (promotion_threshold - proximity_margin):
            return base


    plateau_beyond_threshold = velocity.plateau_episodes - config.plateau_episodes_threshold
    buildup = getattr(config, "relaxation_buildup_episodes", 0)
    if buildup <= 0:
        relax_progress = 1.0
    else:
        relax_progress = min(1.0, plateau_beyond_threshold / buildup)
    relax_factor = config.max_relaxation * relax_progress


    adjusted = copy.deepcopy(base)


    if "min_win_rate" in config.relaxable_metrics:
        adjusted.min_win_rate = base.min_win_rate * (1 - relax_factor * 0.5)



    if "min_profit_factor" in config.relaxable_metrics:
        adjusted.min_profit_factor = base.min_profit_factor * (1 - relax_factor * 0.3)

    if "min_avg_pnl" in config.relaxable_metrics:
        if base.min_avg_pnl < 0:


            adjusted.min_avg_pnl = base.min_avg_pnl * (1 - relax_factor)
        else:
            adjusted.min_avg_pnl = base.min_avg_pnl * (1 - relax_factor * 0.3)

    if "min_trade_count_avg" in config.relaxable_metrics:
        adjusted.min_trade_count_avg = base.min_trade_count_avg * (1 - relax_factor * 0.3)

    if "max_win_rate_std" in config.relaxable_metrics:
        adjusted.max_win_rate_std = base.max_win_rate_std * (1 + relax_factor * 0.3)

    if "max_pnl_std" in config.relaxable_metrics:
        adjusted.max_pnl_std = base.max_pnl_std * (1 + relax_factor * 0.3)


    for metric in config.never_relax:
        if hasattr(base, metric) and hasattr(adjusted, metric):
            setattr(adjusted, metric, getattr(base, metric))


    adjusted.min_win_rate = _clamp(adjusted.min_win_rate, 0.0, 1.0)
    adjusted.min_profit_factor = max(0.0, adjusted.min_profit_factor)
    adjusted.min_trade_count_avg = max(0.0, adjusted.min_trade_count_avg)
    adjusted.max_win_rate_std = max(0.0, adjusted.max_win_rate_std)
    adjusted.max_pnl_std = max(0.0, adjusted.max_pnl_std)
    adjusted.max_avg_drawdown = _clamp(adjusted.max_avg_drawdown, 0.0, 1.0)
    adjusted.max_dd_breach_rate = _clamp(adjusted.max_dd_breach_rate, 0.0, 1.0)

    return adjusted


__all__ = [
    "CompositeScore",
    "EpisodeMetrics",
    "LearningVelocity",
    "RollingStats",
    "compute_adjusted_thresholds",
    "compute_composite_score",
]
