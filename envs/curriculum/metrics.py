# envs/curriculum/metrics.py
"""
Curriculum metrics and scoring components.

Contains:
- EpisodeMetrics: Metrics collected from a single episode
- RollingStats: Rolling statistics computed over a window of episodes
- LearningVelocity: Tracks rate of improvement for plateau detection
- CompositeScore: Result of composite competence scoring
- compute_composite_score(): Compute weighted composite competence score
- compute_adjusted_thresholds(): Compute adjusted thresholds based on learning velocity
"""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import Any, ClassVar, Deque, Dict, List, Optional, TYPE_CHECKING

import numpy as np
from zoneinfo import ZoneInfo

from envs.curriculum.config.thresholds import (
    CompetenceThresholds,
    CompositeScoringConfig,
    AdaptiveThresholdConfig,
)

from envs.core.shared_utils import (
    safe_float as _sf,
    safe_int as _si,
    clamp as _clamp,
    get_envs_logger,
    iso_timestamp,
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
    """Compute slope of linear regression for trend detection."""
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=np.float64)
    try:
        coeffs = np.polyfit(x, y, 1)
        return float(coeffs[0])
    except Exception as e:
        logger.debug(f"Linear regression failed: {e}")
        return 0.0


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EpisodeMetrics:
    """Metrics collected from a single episode."""
    # Explicit field groups for safe (de)serialization. We avoid relying on
    # __annotations__ types because `from __future__ import annotations` can
    # defer/alter runtime typing behavior.
    _FLOAT_FIELDS: ClassVar[set] = set()
    _INT_FIELDS: ClassVar[set] = set()
    _BOOL_FIELDS: ClassVar[set] = set()
    _STR_FIELDS: ClassVar[set] = set()
    # Core performance
    total_pnl: float = 0.0
    win_rate: float = 0.0
    trade_count: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    # Risk metrics
    max_drawdown: float = 0.0
    daily_drawdown: float = 0.0
    dd_breach: bool = False

    # Quality metrics
    avg_r_multiple: float = 0.0
    profit_factor: float = 0.0
    avg_mae: float = 0.0
    avg_mfe: float = 0.0
    avg_bars_held: float = 0.0
    avg_entry_quality: float = 0.5

    # Behavior metrics
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    max_consecutive_losses_reached: int = 0  # Peak consecutive losses during episode
    hit_max_consecutive_losses: bool = False

    # Exit quality tracking
    trailing_stop_exits: int = 0
    agent_close_exits: int = 0
    hard_stop_exits: int = 0
    risk_liquidation_exits: int = 0
    other_exits: int = 0

    # Episode metadata
    episode_length: int = 0
    episode_reward: float = 0.0
    termination_reason: str = ""

    # Curriculum metadata
    stage_name: str = ""
    stage_epoch: int = 0
    global_episode_idx: int = 0

    # Entropy tracking (if available)
    policy_entropy: float = -1.0  # -1 indicates not available

    timestamp: str = field(default_factory=lambda: _now_iso(DEFAULT_TZ))

    def __post_init__(self) -> None:
        # Populate field groups once (class-level) for safe parsing.
        # This is done lazily to keep diffs minimal and avoid import-time complexity.
        if not EpisodeMetrics._FLOAT_FIELDS:
            EpisodeMetrics._FLOAT_FIELDS = {
                "total_pnl", "win_rate", "max_drawdown", "daily_drawdown",
                "avg_r_multiple", "profit_factor", "avg_mae", "avg_mfe",
                "avg_bars_held", "avg_entry_quality", "episode_reward",
                "policy_entropy",
            }
            EpisodeMetrics._INT_FIELDS = {
                "trade_count", "winning_trades", "losing_trades",
                "consecutive_losses", "consecutive_wins",
                "max_consecutive_losses_reached",
                "trailing_stop_exits", "agent_close_exits", "hard_stop_exits",
                "risk_liquidation_exits", "other_exits",
                "episode_length", "stage_epoch", "global_episode_idx",
            }
            EpisodeMetrics._BOOL_FIELDS = {"dd_breach", "hit_max_consecutive_losses"}
            EpisodeMetrics._STR_FIELDS = {"termination_reason", "stage_name", "timestamp"}

        # Defensive normalization (avoids downstream NaN / type surprises)
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
        """Backward/forward compatible constructor."""
        allowed = {f.name for f in fields(cls)}
        payload_raw = {k: v for k, v in (d or {}).items() if k in allowed}

        # Coerce types to keep older checkpoints/loggers from breaking training.
        payload: Dict[str, Any] = {}
        # Ensure field groups exist even if __post_init__ hasn't run yet
        if not cls._FLOAT_FIELDS:
            _ = cls()  # triggers __post_init__ and initializes groups
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
    """Rolling statistics computed over a window of episodes."""
    window_size: int

    # Performance stats
    mean_pnl: float = 0.0
    std_pnl: float = 0.0
    mean_win_rate: float = 0.0      # Episode-averaged win rate
    std_win_rate: float = 0.0
    mean_trade_count: float = 0.0
    total_trades: int = 0
    total_wins: int = 0
    total_losses: int = 0
    win_rate_trade_weighted: float = 0.0  # Trade-pooled: total_wins / total_trades

    # Risk stats
    mean_drawdown: float = 0.0
    max_drawdown_seen: float = 0.0
    dd_breach_rate: float = 0.0

    # Quality stats
    mean_profit_factor: float = 0.0
    mean_r_multiple: float = 0.0
    mean_entry_quality: float = 0.5

    # Behavior stats
    consecutive_loss_breach_rate: float = 0.0
    avg_max_consecutive_losses: float = 0.0  # Average of peak consecutive losses per episode
    consecutive_loss_streak_rate: float = 0.0  # Rate of episodes with 3+ consecutive losses

    # Computed metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    win_loss_ratio: float = 0.0

    # Statistical confidence
    win_rate_wilson_low: float = 0.0
    win_rate_wilson_high: float = 0.0
    pnl_mean_ci_low: float = 0.0
    pnl_mean_ci_high: float = 0.0

    # Entropy stats
    mean_entropy: float = -1.0
    std_entropy: float = 0.0
    entropy_samples: int = 0  # Number of valid entropy samples (for gating)

    # Exit quality stats
    trailing_stop_rate: float = 0.0
    agent_close_rate: float = 0.0
    hard_stop_rate: float = 0.0
    risk_liquidation_rate: float = 0.0


@dataclass
class LearningVelocity:
    """
    Tracks rate of improvement for plateau detection.
    
    Monitors multiple metrics over time to determine if agent is still learning
    or has plateaued at current performance level.
    """
    # Rolling history per metric (metric_name -> deque of values)
    metric_history: Dict[str, Deque[float]] = field(default_factory=dict)
    
    # Computed improvement rates (metric_name -> slope)
    improvement_rates: Dict[str, float] = field(default_factory=dict)
    
    # Plateau tracking
    plateau_episodes: int = 0
    plateau_threshold: float = 0.005  # Minimum improvement to not count as plateau
    
    # History window size
    history_window: int = 100
    
    # Minimum samples for rate calculation
    min_samples: int = 20
    
    # Metrics where LOWER is better (inverted direction)
    LOWER_IS_BETTER: ClassVar[set] = {
        "max_drawdown", "daily_drawdown", "avg_mae", "consecutive_losses",
        "hard_stop_exits", "risk_liquidation_exits",
        "max_consecutive_losses_reached",
    }
    
    def has_sufficient_samples(self) -> bool:
        """True if at least one metric has enough samples to compute an improvement rate."""
        for hist in self.metric_history.values():
            if len(hist) >= self.min_samples:
                return True
        return False
    
    def update(self, metrics: Dict[str, float]) -> None:
        """Update velocity tracking with new metrics."""
        any_improving = False
        has_rate_estimates = False  # Track if ANY metric has enough samples
        
        for name, value in metrics.items():
            # Defensive: ignore NaN/inf inputs (can permanently poison plateau logic)
            fv = _sf(value, float('nan'))
            if not np.isfinite(fv):
                continue

            if name not in self.metric_history:
                self.metric_history[name] = deque(maxlen=self.history_window)
            
            self.metric_history[name].append(fv)
            
            # Compute improvement rate via linear regression
            if len(self.metric_history[name]) >= self.min_samples:
                has_rate_estimates = True  # We can compute at least one slope
                y = np.array(list(self.metric_history[name]), dtype=np.float64)
                slope = _linear_regression_slope(y)
                
                # Normalize by mean to get relative improvement
                mean_val = np.mean(y)
                if abs(mean_val) > 1e-8:
                    normalized_slope = slope / abs(mean_val)
                else:
                    normalized_slope = slope
                
                # For "lower is better" metrics, negate slope for storage
                # (so positive means improvement regardless of direction)
                if name in self.LOWER_IS_BETTER:
                    normalized_slope = -normalized_slope
                
                self.improvement_rates[name] = float(normalized_slope)
                
                # Check if this metric is improving
                if normalized_slope > self.plateau_threshold:
                    any_improving = True
        
        # Update plateau counter ONLY if we have enough data to measure improvement
        # Don't count early episodes as "plateaued" when we can't compute slopes yet
        if not has_rate_estimates:
            # Not enough data yet - don't increment plateau (stay at 0 or current)
            pass
        elif any_improving:
            self.plateau_episodes = 0
        else:
            self.plateau_episodes += 1
    
    def is_plateaued(self, threshold_episodes: int = 100) -> bool:
        """Check if learning has plateaued."""
        return self.plateau_episodes >= threshold_episodes
    
    def get_average_improvement(self) -> float:
        """Get average improvement rate across all metrics."""
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
            # Note: is_plateaued should be checked with stage-specific threshold externally
            "is_plateaued_default": self.is_plateaued(),
        }


@dataclass
class CompositeScore:
    """Result of composite competence scoring."""
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


# =============================================================================
# Composite Scoring
# =============================================================================

def compute_composite_score(
    stats: RollingStats,
    thresholds: CompetenceThresholds,
    config: CompositeScoringConfig,
) -> CompositeScore:
    """
    Compute weighted composite competence score.
    
    Allows nuanced evaluation rather than all-or-nothing gating.
    """
    if not config.enabled:
        return CompositeScore()
    
    components: Dict[str, float] = {}
    
    # Win rate: score 0-1 based on distance to threshold
    # Use a tiny epsilon, not 0.01, so extremely low thresholds don't distort scoring.
    wr_den = max(getattr(thresholds, "min_win_rate", 0.0), 1e-6)
    wr_score = min(stats.mean_win_rate / wr_den, 1.5) / 1.5
    components["win_rate"] = wr_score
    
    # Profit factor
    pf_score = min(stats.mean_profit_factor / max(thresholds.min_profit_factor, 0.01), 2.0) / 2.0
    components["profit_factor"] = pf_score
    
    # Drawdown (inverted - lower is better)
    if thresholds.max_avg_drawdown > 0:
        dd_ratio = stats.mean_drawdown / thresholds.max_avg_drawdown
        dd_score = 1.0 - min(dd_ratio, 1.5) / 1.5
    else:
        dd_score = 1.0 if stats.mean_drawdown <= 0.01 else 0.5
    components["drawdown"] = max(0, dd_score)
    
    # Consistency
    if thresholds.max_win_rate_std > 0:
        cons_ratio = stats.std_win_rate / thresholds.max_win_rate_std
        cons_score = 1.0 - min(cons_ratio, 1.5) / 1.5
    else:
        cons_score = 1.0 if stats.std_win_rate <= 0.05 else 0.5
    components["consistency"] = max(0, cons_score)
    
    # R-multiple
    r_score = _clamp((stats.mean_r_multiple + 0.5) / 1.0, 0.0, 1.0)
    components["r_multiple"] = r_score
    
    # DD breach rate (inverted)
    if thresholds.max_dd_breach_rate > 0:
        breach_ratio = stats.dd_breach_rate / thresholds.max_dd_breach_rate
        breach_score = 1.0 - min(breach_ratio, 1.5) / 1.5
    else:
        breach_score = 1.0 if stats.dd_breach_rate <= 0.05 else 0.5
    components["dd_breach_rate"] = max(0, breach_score)
    
    # Trade activity
    if thresholds.min_trade_count_avg > 0:
        trade_score = min(stats.mean_trade_count / thresholds.min_trade_count_avg, 2.0) / 2.0
    else:
        trade_score = 0.5
    components["trade_activity"] = trade_score
    
    # Consecutive loss rate (inverted)
    if thresholds.max_consecutive_loss_rate > 0:
        cl_ratio = stats.consecutive_loss_breach_rate / thresholds.max_consecutive_loss_rate
        cl_score = 1.0 - min(cl_ratio, 1.5) / 1.5
    else:
        cl_score = 1.0 if stats.consecutive_loss_breach_rate <= 0.05 else 0.5
    components["consecutive_loss_rate"] = max(0, cl_score)
    
    # Compute weighted composite (normalized by weight sum for stable scale)
    weight_sum = sum(config.weights.values())
    if weight_sum <= 0:
        weight_sum = 1.0  # Avoid division by zero
    
    total_score = sum(
        components.get(name, 0.5) * weight
        for name, weight in config.weights.items()
    ) / weight_sum  # Normalize to [0, 1] range
    
    # Check hard floors
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
    
    # NEW: Check profit_factor hard floor
    if "profit_factor" in config.hard_floors:
        if stats.mean_profit_factor < config.hard_floors["profit_factor"]:
            meets_floors = False
            hard_floor_failures.append("profit_factor")
    
    # NEW: Check r_multiple hard floor
    if "r_multiple" in config.hard_floors:
        if stats.mean_r_multiple < config.hard_floors["r_multiple"]:
            meets_floors = False
            hard_floor_failures.append("r_multiple")
    
    # Determine promotion/demotion status
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


# =============================================================================
# Adaptive Thresholds
# =============================================================================

def compute_adjusted_thresholds(
    base: CompetenceThresholds,
    velocity: LearningVelocity,
    config: AdaptiveThresholdConfig,
    composite_score: Optional["CompositeScore"] = None,
    promotion_threshold: float = 0.7,
) -> CompetenceThresholds:
    """
    Compute adjusted thresholds based on learning velocity.
    
    Slightly relaxes thresholds if agent is plateaued AND close to promotion.
    The proximity check prevents wasting relaxation on agents that are nowhere
    near promotion anyway.
    """
    if not config.enabled:
        return base
    
    if velocity.plateau_episodes < config.plateau_episodes_threshold:
        return base  # No adjustment needed
    
    # PROXIMITY CHECK: Only relax if agent is reasonably close to promotion
    # This prevents wasting relaxation on agents that are far from ready.
    if composite_score is not None:
        proximity_margin = 0.15  # Must be within 15% of promotion threshold
        if composite_score.total_score < (promotion_threshold - proximity_margin):
            return base  # Too far from promotion, don't relax
    
    # Calculate relaxation factor
    # Linear buildup from 0 to max_relaxation over relaxation_buildup_episodes
    plateau_beyond_threshold = velocity.plateau_episodes - config.plateau_episodes_threshold
    buildup = getattr(config, "relaxation_buildup_episodes", 0)
    if buildup <= 0:
        relax_progress = 1.0
    else:
        relax_progress = min(1.0, plateau_beyond_threshold / buildup)
    relax_factor = config.max_relaxation * relax_progress
    
    # Create adjusted thresholds
    adjusted = copy.deepcopy(base)
    
    # Apply relaxation to allowed metrics
    if "min_win_rate" in config.relaxable_metrics:
        adjusted.min_win_rate = base.min_win_rate * (1 - relax_factor * 0.5)

        # Keep Wilson lower-bound gate consistent if present in thresholds.
        # Otherwise you can relax mean win-rate but still be blocked by an unrelaxed Wilson gate.
        if hasattr(adjusted, "min_win_rate_wilson_low") and hasattr(base, "min_win_rate_wilson_low"):
            try:
                base_w = float(getattr(base, "min_win_rate_wilson_low"))
                setattr(adjusted, "min_win_rate_wilson_low", base_w * (1 - relax_factor * 0.5))
            except Exception:
                pass
    
    if "min_profit_factor" in config.relaxable_metrics:
        adjusted.min_profit_factor = base.min_profit_factor * (1 - relax_factor * 0.3)
    
    if "min_avg_pnl" in config.relaxable_metrics:
        if base.min_avg_pnl < 0:
            # FIX: To make a negative number "less negative", multiply by (1 - factor)
            # e.g., -100 * (1 - 0.2) = -80 (closer to 0, easier threshold)
            adjusted.min_avg_pnl = base.min_avg_pnl * (1 - relax_factor)
        else:
            adjusted.min_avg_pnl = base.min_avg_pnl * (1 - relax_factor * 0.3)
    
    if "min_trade_count_avg" in config.relaxable_metrics:
        adjusted.min_trade_count_avg = base.min_trade_count_avg * (1 - relax_factor * 0.3)
    
    if "max_win_rate_std" in config.relaxable_metrics:
        adjusted.max_win_rate_std = base.max_win_rate_std * (1 + relax_factor * 0.3)
    
    if "max_pnl_std" in config.relaxable_metrics:
        adjusted.max_pnl_std = base.max_pnl_std * (1 + relax_factor * 0.3)
    
    # NEVER relax safety metrics
    for metric in config.never_relax:
        if hasattr(base, metric) and hasattr(adjusted, metric):
            setattr(adjusted, metric, getattr(base, metric))
    
    # CLAMP all thresholds to valid domains to prevent pathological configs
    adjusted.min_win_rate = _clamp(adjusted.min_win_rate, 0.0, 1.0)
    if hasattr(adjusted, "min_win_rate_wilson_low"):
        try:
            setattr(adjusted, "min_win_rate_wilson_low", _clamp(float(getattr(adjusted, "min_win_rate_wilson_low")), 0.0, 1.0))
        except Exception:
            pass
    adjusted.min_profit_factor = max(0.0, adjusted.min_profit_factor)
    adjusted.min_trade_count_avg = max(0.0, adjusted.min_trade_count_avg)
    adjusted.max_win_rate_std = max(0.0, adjusted.max_win_rate_std)
    adjusted.max_pnl_std = max(0.0, adjusted.max_pnl_std)
    adjusted.max_avg_drawdown = _clamp(adjusted.max_avg_drawdown, 0.0, 1.0)
    adjusted.max_dd_breach_rate = _clamp(adjusted.max_dd_breach_rate, 0.0, 1.0)
    
    return adjusted


# Re-export for backward compatibility
__all__ = [
    "EpisodeMetrics",
    "RollingStats",
    "LearningVelocity",
    "CompositeScore",
    "compute_composite_score",
    "compute_adjusted_thresholds",
]
