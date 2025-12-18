# envs/curriculum_manager.py
"""
Curriculum Manager for Trading RL Agent
========================================

Tracks agent competence across multiple metrics and determines when to promote
(or demote) the agent to the next curriculum stage.

Key Principles:
1. Progress by STATISTICAL CONFIDENCE, not time or luck
2. Multiple metrics must ALL pass thresholds (no single-metric gaming)
3. Stability matters as much as performance (low variance required)
4. Demotion is possible if performance degrades significantly
5. Rolling window evaluation prevents overfitting to recent episodes

Enhancements in this version:
- Stage-epoch isolation: prevents "old" stage history from contaminating re-entries
- Trade-level win-rate confidence bounds (Wilson lower bound)
- Robust parsing of info/episode_stats with safe coercion + clamping
- Stronger save/load compatibility (unknown/missing keys tolerated)
"""

from __future__ import annotations

import json
import logging
import math
from collections import deque
from dataclasses import dataclass, field, asdict, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np

from envs.curriculum_config import (
    CurriculumStage,
    CurriculumStageConfig,
    CompetenceThresholds,
    DataDifficulty,
    TransitionSettings,
    MIN_EVALUATION_EPISODES,
    get_stage_config,
    get_next_stage,
    get_previous_stage,
    get_stage_progression,
)

logger = logging.getLogger("curriculum_manager")

STATE_VERSION = "1.2"  # Updated for transition state
DEFAULT_TZ = "Europe/Berlin"


def _now_iso(tz: str = DEFAULT_TZ) -> str:
    try:
        return datetime.now(tz=ZoneInfo(tz)).isoformat()
    except Exception:
        return datetime.now().isoformat()


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Wilson score confidence interval for a Bernoulli proportion.
    Returns (low, high). For promotion we usually use the LOWER bound.
    """
    if n <= 0:
        return 0.0, 0.0
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    radius = (z / denom) * math.sqrt((phat * (1.0 - phat) / n) + (z * z) / (4.0 * n * n))
    return _clamp(center - radius, 0.0, 1.0), _clamp(center + radius, 0.0, 1.0)


def _mean_ci_normal(mean: float, std: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 1:
        return mean, mean
    se = std / math.sqrt(max(n, 1))
    return mean - z * se, mean + z * se


@dataclass
class EpisodeMetrics:
    """Metrics collected from a single episode."""
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
    hit_max_consecutive_losses: bool = False

    # Episode metadata
    episode_length: int = 0
    episode_reward: float = 0.0
    termination_reason: str = ""

    # Curriculum metadata (added in v1.1)
    stage_name: str = ""
    stage_epoch: int = 0
    global_episode_idx: int = 0

    timestamp: str = field(default_factory=lambda: _now_iso(DEFAULT_TZ))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EpisodeMetrics":
        """
        Backward/forward compatible constructor:
        - ignores unknown keys
        - fills missing with defaults
        """
        allowed = {f.name for f in fields(cls)}
        payload = {k: v for k, v in (d or {}).items() if k in allowed}
        return cls(**payload)


@dataclass
class RollingStats:
    """Rolling statistics computed over a window of episodes."""
    window_size: int

    # Performance stats
    mean_pnl: float = 0.0
    std_pnl: float = 0.0
    mean_win_rate: float = 0.0
    std_win_rate: float = 0.0
    mean_trade_count: float = 0.0
    total_trades: int = 0
    total_wins: int = 0
    total_losses: int = 0

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

    # Computed metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    win_loss_ratio: float = 0.0

    # Statistical confidence (added)
    win_rate_wilson_low: float = 0.0
    win_rate_wilson_high: float = 0.0
    pnl_mean_ci_low: float = 0.0
    pnl_mean_ci_high: float = 0.0


class CurriculumManager:
    """
    Manages curriculum progression for the trading RL agent.

    Responsibilities:
    - Track episode metrics history
    - Compute rolling statistics
    - Evaluate promotion/demotion criteria
    - Persist and restore state
    - Manage transition effects (LR warmup, reward blending)

    IMPORTANT:
    - Promotion/demotion evaluation uses ONLY episodes from the CURRENT stage-epoch
      to prevent stale history contamination when revisiting a stage.
    """

    def __init__(
        self,
        initial_stage: CurriculumStage = CurriculumStage.FOUNDATION,
        max_history_size: int = 1000,
        auto_promote: bool = True,
        auto_demote: bool = True,
        verbose: bool = True,
        tz: str = DEFAULT_TZ,
        on_transition_callback: Optional[Any] = None,  # Callable[[str, CurriculumStage, CurriculumStage, Dict], None]
    ) -> None:
        self.current_stage = initial_stage
        self.max_history_size = max_history_size
        self.auto_promote = auto_promote
        self.auto_demote = auto_demote
        self.verbose = verbose
        self.tz = tz
        self.on_transition_callback = on_transition_callback

        # Metrics history per stage (stores all epochs; evaluation filters by epoch)
        self._history: Dict[CurriculumStage, Deque[EpisodeMetrics]] = {
            stage: deque(maxlen=max_history_size) for stage in CurriculumStage
        }

        # Totals per stage (lifetime)
        self._stage_timesteps_total: Dict[CurriculumStage, int] = {stage: 0 for stage in CurriculumStage}
        self._stage_episodes_total: Dict[CurriculumStage, int] = {stage: 0 for stage in CurriculumStage}

        # Current stage-epoch counters (reset on stage entry)
        self.stage_timesteps: int = 0
        self.stage_episodes: int = 0

        # Stage epoch counters (increment each time we ENTER a stage)
        self._stage_epoch_counter: Dict[CurriculumStage, int] = {stage: 0 for stage in CurriculumStage}
        self._current_stage_epoch: int = 0
        self._enter_stage(self.current_stage, reason="init")

        # Promotion/demotion history
        self._transitions: List[Dict[str, Any]] = []

        # Cached rolling stats
        self._rolling_stats: Optional[RollingStats] = None
        self._rolling_stats_dirty: bool = True

        # Track totals
        self.total_timesteps: int = 0
        self.total_episodes: int = 0
        
        # Transition state tracking
        self._transition_cooldown_remaining: int = 0  # Episodes until next transition allowed
        self._reward_blend_remaining: int = 0  # Episodes of reward blending left
        self._previous_stage_config: Optional[CurriculumStageConfig] = None
        self._lr_warmup_active: bool = False
        self._lr_warmup_steps_remaining: int = 0
        self._lr_warmup_factor: float = 1.0

        logger.info(f"CurriculumManager initialized at stage: {initial_stage.name}")

    def _enter_stage(self, stage: CurriculumStage, reason: str) -> None:
        # Save previous config for reward blending
        if hasattr(self, 'current_stage'):
            self._previous_stage_config = get_stage_config(self.current_stage)
        
        self._stage_epoch_counter[stage] += 1
        self._current_stage_epoch = self._stage_epoch_counter[stage]
        self.stage_timesteps = 0
        self.stage_episodes = 0
        self._rolling_stats_dirty = True
        
        # Setup transition effects from new stage config
        new_config = get_stage_config(stage)
        transition = new_config.transition
        
        # Setup transition cooldown
        self._transition_cooldown_remaining = transition.transition_cooldown_episodes
        
        # Setup reward blending if enabled and we have previous config
        if transition.reward_blend_enabled and self._previous_stage_config is not None:
            self._reward_blend_remaining = transition.reward_blend_episodes
        else:
            self._reward_blend_remaining = 0
        
        # Setup LR warmup
        if transition.lr_warmup_enabled and reason != "init":
            self._lr_warmup_active = True
            self._lr_warmup_steps_remaining = transition.lr_warmup_steps
            self._lr_warmup_factor = transition.lr_warmup_factor
        else:
            self._lr_warmup_active = False
            self._lr_warmup_steps_remaining = 0
            self._lr_warmup_factor = 1.0
        
        if self.verbose:
            logger.info(f"Entered stage {stage.name} (epoch={self._current_stage_epoch}, reason={reason})")
            if self._lr_warmup_active:
                logger.info(f"  LR warmup: {transition.lr_warmup_factor:.0%} → 100% over {transition.lr_warmup_steps:,} steps")
            if self._reward_blend_remaining > 0:
                logger.info(f"  Reward blend: {self._reward_blend_remaining} episodes")

    @property
    def stage_config(self) -> CurriculumStageConfig:
        return get_stage_config(self.current_stage)

    @property
    def competence_thresholds(self) -> CompetenceThresholds:
        return self.stage_config.competence

    @property
    def current_stage_epoch(self) -> int:
        return self._current_stage_epoch
    
    @property
    def is_in_transition(self) -> bool:
        """Check if we're currently in a transition period."""
        return self._lr_warmup_active or self._reward_blend_remaining > 0
    
    @property
    def reward_blend_factor(self) -> float:
        """
        Get the blend factor for reward configs.
        Returns 1.0 when fully on new config, 0.0 when fully on old config.
        """
        if self._reward_blend_remaining <= 0 or self._previous_stage_config is None:
            return 1.0
        
        transition = self.stage_config.transition
        total_blend = transition.reward_blend_episodes
        if total_blend <= 0:
            return 1.0
        
        progress = 1.0 - (self._reward_blend_remaining / total_blend)
        return max(0.0, min(1.0, progress))
    
    def get_lr_multiplier(self, base_steps_since_transition: int = 0) -> float:
        """
        Get the learning rate multiplier for warmup.
        
        Args:
            base_steps_since_transition: Steps since last transition (optional, for fine control)
            
        Returns:
            Multiplier in [lr_warmup_factor, 1.0]
        """
        if not self._lr_warmup_active:
            return 1.0
        
        transition = self.stage_config.transition
        if transition.lr_warmup_steps <= 0:
            return 1.0
        
        # Linear warmup from lr_warmup_factor to 1.0
        warmup_progress = 1.0 - (self._lr_warmup_steps_remaining / transition.lr_warmup_steps)
        warmup_progress = max(0.0, min(1.0, warmup_progress))
        
        return self._lr_warmup_factor + (1.0 - self._lr_warmup_factor) * warmup_progress
    
    def step_transition_state(self, timesteps: int = 1) -> None:
        """
        Update transition state counters.
        Call this each training step to progress LR warmup.
        """
        if self._lr_warmup_active and self._lr_warmup_steps_remaining > 0:
            self._lr_warmup_steps_remaining -= timesteps
            if self._lr_warmup_steps_remaining <= 0:
                self._lr_warmup_active = False
                self._lr_warmup_factor = 1.0
                if self.verbose:
                    logger.info(f"LR warmup complete for stage {self.current_stage.name}")
    
    def episode_transition_tick(self) -> None:
        """
        Update per-episode transition counters.
        Call this at end of each episode.
        """
        if self._transition_cooldown_remaining > 0:
            self._transition_cooldown_remaining -= 1
        
        if self._reward_blend_remaining > 0:
            self._reward_blend_remaining -= 1
            if self._reward_blend_remaining <= 0 and self.verbose:
                logger.info(f"Reward blending complete for stage {self.current_stage.name}")

    def record_episode(self, metrics: EpisodeMetrics, timesteps: int = 0) -> None:
        # Fill curriculum metadata defensively
        metrics.stage_name = metrics.stage_name or self.current_stage.name
        metrics.stage_epoch = metrics.stage_epoch or self._current_stage_epoch
        metrics.global_episode_idx = metrics.global_episode_idx or (self.total_episodes + 1)
        if not metrics.timestamp:
            metrics.timestamp = _now_iso(self.tz)

        self._history[self.current_stage].append(metrics)

        self._stage_timesteps_total[self.current_stage] += timesteps
        self._stage_episodes_total[self.current_stage] += 1

        self.stage_timesteps += timesteps
        self.stage_episodes += 1

        self.total_timesteps += timesteps
        self.total_episodes += 1
        self._rolling_stats_dirty = True

    def record_episode_from_info(self, info: Dict[str, Any], episode_reward: float, episode_length: int) -> None:
        ep_stats = info.get("episode_stats", {}) or {}

        # Prefer explicit values when present
        total_pnl = _safe_float(info.get("total_pnl", ep_stats.get("total_pnl", 0.0)), 0.0)
        win_rate = _clamp(_safe_float(info.get("win_rate", ep_stats.get("win_rate", 0.0)), 0.0), 0.0, 1.0)

        # Trade counts: support multiple schemas
        trade_count = _safe_int(info.get("trade_count", ep_stats.get("total_trades", ep_stats.get("trade_count", 0))), 0)
        winning_trades = _safe_int(ep_stats.get("winning_trades", info.get("winning_trades", 0)), 0)
        losing_trades = _safe_int(ep_stats.get("losing_trades", info.get("losing_trades", 0)), 0)

        # Fallback if env only reports win_rate and trade_count
        if (winning_trades == 0 and losing_trades == 0) and trade_count > 0:
            approx_wins = int(round(win_rate * trade_count))
            winning_trades = max(0, min(trade_count, approx_wins))
            losing_trades = max(0, trade_count - winning_trades)

        max_dd = _clamp(_safe_float(info.get("drawdown", ep_stats.get("max_drawdown", 0.0)), 0.0), 0.0, 1.0)
        daily_dd = _clamp(_safe_float(info.get("daily_drawdown", ep_stats.get("daily_drawdown", 0.0)), 0.0), 0.0, 1.0)

        termination_reason = str(info.get("termination_reason", ep_stats.get("termination_reason", "")) or "")

        # DD breach: prefer explicit bools; fallback to reason scan
        dd_breach = bool(info.get("dd_breach", ep_stats.get("dd_breach", False)))
        if not dd_breach and termination_reason:
            low = termination_reason.lower()
            dd_breach = ("drawdown" in low) or ("dd breach" in low) or ("daily_dd" in low) or ("max_dd" in low)

        # Profit factor: cap very large values to avoid exploding means
        pf_raw = ep_stats.get("profit_factor", info.get("profit_factor", 0.0))
        pf = _safe_float(pf_raw, 0.0)
        if math.isinf(pf) or pf > 1e6:
            pf = 10.0
        pf = _clamp(pf, 0.0, 10.0)

        hit_max_consec = bool(info.get("hit_max_consecutive_losses", ep_stats.get("hit_max_consecutive_losses", False)))
        if not hit_max_consec and termination_reason:
            hit_max_consec = "consecutive" in termination_reason.lower()

        metrics = EpisodeMetrics(
            total_pnl=total_pnl,
            win_rate=win_rate,
            trade_count=trade_count,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            max_drawdown=max_dd,
            daily_drawdown=daily_dd,
            dd_breach=dd_breach,
            avg_r_multiple=_safe_float(ep_stats.get("avg_r_multiple", info.get("avg_r_multiple", 0.0)), 0.0),
            profit_factor=pf,
            avg_mae=_safe_float(ep_stats.get("avg_mae", info.get("avg_mae", 0.0)), 0.0),
            avg_mfe=_safe_float(ep_stats.get("avg_mfe", info.get("avg_mfe", 0.0)), 0.0),
            avg_bars_held=_safe_float(ep_stats.get("avg_bars_held", info.get("avg_bars_held", 0.0)), 0.0),
            avg_entry_quality=_clamp(_safe_float(ep_stats.get("avg_entry_quality", info.get("avg_entry_quality", 0.5)), 0.5), 0.0, 1.0),
            consecutive_losses=_safe_int(info.get("consecutive_losses", ep_stats.get("consecutive_losses", 0)), 0),
            consecutive_wins=_safe_int(info.get("consecutive_wins", ep_stats.get("consecutive_wins", 0)), 0),
            hit_max_consecutive_losses=hit_max_consec,
            episode_length=_safe_int(episode_length, 0),
            episode_reward=_safe_float(episode_reward, 0.0),
            termination_reason=termination_reason,
            stage_name=self.current_stage.name,
            stage_epoch=self._current_stage_epoch,
            global_episode_idx=self.total_episodes + 1,
            timestamp=_now_iso(self.tz),
        )

        self.record_episode(metrics, timesteps=episode_length)

    def _current_epoch_window(self, window_size: int) -> List[EpisodeMetrics]:
        history = list(self._history[self.current_stage])
        # Filter to current stage epoch to avoid stale contamination
        epoch = self._current_stage_epoch
        filtered = [m for m in history if m.stage_epoch == epoch]
        return filtered[-window_size:] if len(filtered) > window_size else filtered

    def get_rolling_stats(self, force_refresh: bool = False) -> RollingStats:
        if not force_refresh and not self._rolling_stats_dirty and self._rolling_stats is not None:
            return self._rolling_stats

        thresholds = self.competence_thresholds
        window_size = thresholds.evaluation_window

        window = self._current_epoch_window(window_size)

        if not window:
            self._rolling_stats = RollingStats(window_size=0)
            self._rolling_stats_dirty = False
            return self._rolling_stats

        pnls = np.array([_safe_float(m.total_pnl, 0.0) for m in window], dtype=np.float64)
        win_rates = np.array([_clamp(_safe_float(m.win_rate, 0.0), 0.0, 1.0) for m in window], dtype=np.float64)
        trade_counts = np.array([max(0.0, float(m.trade_count)) for m in window], dtype=np.float64)
        drawdowns = np.array([_clamp(_safe_float(m.max_drawdown, 0.0), 0.0, 1.0) for m in window], dtype=np.float64)
        dd_breaches = np.array([bool(m.dd_breach) for m in window], dtype=np.bool_)
        profit_factors = np.array([_clamp(_safe_float(m.profit_factor, 0.0), 0.0, 10.0) for m in window], dtype=np.float64)
        r_multiples = np.array([_safe_float(m.avg_r_multiple, 0.0) for m in window], dtype=np.float64)
        entry_qualities = np.array([_clamp(_safe_float(m.avg_entry_quality, 0.5), 0.0, 1.0) for m in window], dtype=np.float64)
        consec_loss_breaches = np.array([bool(m.hit_max_consecutive_losses) for m in window], dtype=np.bool_)

        mean_pnl = float(np.mean(pnls))
        std_pnl = float(np.std(pnls))
        mean_win_rate = float(np.mean(win_rates))
        std_win_rate = float(np.std(win_rates))
        mean_trade_count = float(np.mean(trade_counts))

        total_trades = int(np.sum(trade_counts))
        total_wins = int(sum(max(0, m.winning_trades) for m in window))
        total_losses = int(sum(max(0, m.losing_trades) for m in window))

        mean_drawdown = float(np.mean(drawdowns))
        max_drawdown_seen = float(np.max(drawdowns)) if len(drawdowns) else 0.0
        dd_breach_rate = float(np.mean(dd_breaches)) if len(dd_breaches) else 0.0

        pf_pos = profit_factors[profit_factors > 0]
        mean_profit_factor = float(np.mean(pf_pos)) if pf_pos.size else 0.0

        mean_r_multiple = float(np.mean(r_multiples)) if len(r_multiples) else 0.0
        mean_entry_quality = float(np.mean(entry_qualities)) if len(entry_qualities) else 0.5
        consecutive_loss_breach_rate = float(np.mean(consec_loss_breaches)) if len(consec_loss_breaches) else 0.0

        # Sharpe/Sortino on episode PnL (proxy)
        if std_pnl > 1e-9:
            sharpe = mean_pnl / std_pnl
        else:
            sharpe = mean_pnl if mean_pnl > 0 else 0.0

        negative_pnls = pnls[pnls < 0]
        if negative_pnls.size:
            downside_std = float(np.std(negative_pnls))
            sortino = mean_pnl / downside_std if downside_std > 1e-9 else sharpe
        else:
            sortino = sharpe * 1.5 if mean_pnl > 0 else 0.0

        win_loss_ratio = total_wins / max(total_losses, 1)

        # Trade-level win-rate Wilson bounds (more statistically meaningful than episode mean)
        wl_n = max(total_wins + total_losses, 0)
        win_low, win_high = _wilson_interval(total_wins, wl_n, z=1.96) if wl_n > 0 else (0.0, 0.0)

        # Mean PnL CI (normal approximation)
        ci_low, ci_high = _mean_ci_normal(mean_pnl, std_pnl, n=len(window), z=1.96)

        stats = RollingStats(
            window_size=len(window),
            mean_pnl=mean_pnl,
            std_pnl=std_pnl,
            mean_win_rate=mean_win_rate,
            std_win_rate=std_win_rate,
            mean_trade_count=mean_trade_count,
            total_trades=total_trades,
            total_wins=total_wins,
            total_losses=total_losses,
            mean_drawdown=mean_drawdown,
            max_drawdown_seen=max_drawdown_seen,
            dd_breach_rate=dd_breach_rate,
            mean_profit_factor=mean_profit_factor,
            mean_r_multiple=mean_r_multiple,
            mean_entry_quality=mean_entry_quality,
            consecutive_loss_breach_rate=consecutive_loss_breach_rate,
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino),
            win_loss_ratio=float(win_loss_ratio),
            win_rate_wilson_low=float(win_low),
            win_rate_wilson_high=float(win_high),
            pnl_mean_ci_low=float(ci_low),
            pnl_mean_ci_high=float(ci_high),
        )

        self._rolling_stats = stats
        self._rolling_stats_dirty = False
        return stats

    def check_promotion_criteria(self) -> Tuple[bool, Dict[str, Any]]:
        thresholds = self.competence_thresholds
        stats = self.get_rolling_stats()

        results: Dict[str, Any] = {
            "stage": self.current_stage.name,
            "stage_epoch": self._current_stage_epoch,
            "evaluation_window": stats.window_size,
            "checks": {},
        }

        all_passed = True

        # Minimum episodes/timesteps in CURRENT stage entry (not lifetime)
        passed = self.stage_episodes >= thresholds.min_episodes
        results["checks"]["min_episodes"] = {"required": thresholds.min_episodes, "actual": self.stage_episodes, "passed": passed}
        all_passed = all_passed and passed

        passed = self.stage_timesteps >= thresholds.min_timesteps
        results["checks"]["min_timesteps"] = {"required": thresholds.min_timesteps, "actual": self.stage_timesteps, "passed": passed}
        all_passed = all_passed and passed

        # Data sufficiency guard (window episodes)
        min_required = max(MIN_EVALUATION_EPISODES, min(thresholds.evaluation_window, max(1, thresholds.min_episodes // 2)))
        if stats.window_size < min_required:
            results["checks"]["insufficient_data"] = {"required": min_required, "actual": stats.window_size, "passed": False}
            results["promotion_ready"] = False
            results["stats"] = asdict(stats)
            return False, results

        # Performance checks (episode-level)
        passed = stats.mean_profit_factor >= thresholds.min_profit_factor
        results["checks"]["profit_factor"] = {"required": thresholds.min_profit_factor, "actual": stats.mean_profit_factor, "passed": passed}
        all_passed = all_passed and passed

        passed = stats.mean_drawdown <= thresholds.max_avg_drawdown
        results["checks"]["avg_drawdown"] = {"required": thresholds.max_avg_drawdown, "actual": stats.mean_drawdown, "passed": passed}
        all_passed = all_passed and passed

        passed = stats.mean_pnl >= thresholds.min_avg_pnl
        results["checks"]["avg_pnl"] = {"required": thresholds.min_avg_pnl, "actual": stats.mean_pnl, "passed": passed}
        all_passed = all_passed and passed
        
        # R-Multiple check (if threshold > 0)
        if thresholds.min_avg_r_multiple > 0:
            passed = stats.mean_r_multiple >= thresholds.min_avg_r_multiple
            results["checks"]["r_multiple"] = {"required": thresholds.min_avg_r_multiple, "actual": stats.mean_r_multiple, "passed": passed}
            all_passed = all_passed and passed

        # Win-rate checks (both episode mean and trade-level lower bound)
        passed_mean = stats.mean_win_rate >= thresholds.min_win_rate
        results["checks"]["win_rate_mean"] = {"required": thresholds.min_win_rate, "actual": stats.mean_win_rate, "passed": passed_mean}

        passed_wilson = stats.win_rate_wilson_low >= thresholds.min_win_rate
        results["checks"]["win_rate_wilson_low"] = {
            "required": thresholds.min_win_rate,
            "actual": stats.win_rate_wilson_low,
            "passed": passed_wilson,
            "note": "Trade-level 95% Wilson lower bound",
        }

        # Require BOTH: prevents promotion on luck with few trades
        all_passed = all_passed and passed_mean and passed_wilson

        # Consistency checks
        passed = stats.std_win_rate <= thresholds.max_win_rate_std
        results["checks"]["win_rate_stability"] = {"required": thresholds.max_win_rate_std, "actual": stats.std_win_rate, "passed": passed}
        all_passed = all_passed and passed

        passed = stats.std_pnl <= thresholds.max_pnl_std
        results["checks"]["pnl_stability"] = {"required": thresholds.max_pnl_std, "actual": stats.std_pnl, "passed": passed}
        all_passed = all_passed and passed

        passed = stats.mean_trade_count >= thresholds.min_trade_count_avg
        results["checks"]["trade_activity"] = {"required": thresholds.min_trade_count_avg, "actual": stats.mean_trade_count, "passed": passed}
        all_passed = all_passed and passed

        # Behavior checks
        passed = stats.dd_breach_rate <= thresholds.max_dd_breach_rate
        results["checks"]["dd_breach_rate"] = {"required": thresholds.max_dd_breach_rate, "actual": stats.dd_breach_rate, "passed": passed}
        all_passed = all_passed and passed

        passed = stats.consecutive_loss_breach_rate <= thresholds.max_consecutive_loss_rate
        results["checks"]["consecutive_loss_rate"] = {"required": thresholds.max_consecutive_loss_rate, "actual": stats.consecutive_loss_breach_rate, "passed": passed}
        all_passed = all_passed and passed

        results["promotion_ready"] = all_passed
        results["stats"] = asdict(stats)
        return all_passed, results

    def check_demotion_criteria(self) -> Tuple[bool, Dict[str, Any]]:
        config = self.stage_config
        if not config.allow_demotion:
            return False, {"reason": "demotion_disabled"}

        if self.current_stage == CurriculumStage.FOUNDATION:
            return False, {"reason": "at_foundation"}

        thresholds = self.competence_thresholds
        stats = self.get_rolling_stats()

        results: Dict[str, Any] = {
            "stage": self.current_stage.name,
            "stage_epoch": self._current_stage_epoch,
            "evaluation_window": stats.window_size,
            "checks": {},
        }

        # Need enough data in current epoch
        min_episodes_for_demotion = max(50, thresholds.evaluation_window // 2)
        if stats.window_size < min_episodes_for_demotion:
            results["should_demote"] = False
            results["reason"] = "insufficient_data"
            results["stats"] = asdict(stats)
            return False, results

        demotion_factor = 0.70
        critical_failures = 0

        if stats.mean_win_rate < thresholds.min_win_rate * demotion_factor:
            critical_failures += 1
            results["checks"]["win_rate_critical"] = True

        if stats.mean_profit_factor < thresholds.min_profit_factor * demotion_factor:
            critical_failures += 1
            results["checks"]["profit_factor_critical"] = True

        if stats.mean_drawdown > thresholds.max_avg_drawdown * (1.0 / demotion_factor):
            critical_failures += 1
            results["checks"]["drawdown_critical"] = True

        if stats.dd_breach_rate > thresholds.max_dd_breach_rate * (1.0 / demotion_factor):
            critical_failures += 1
            results["checks"]["dd_breach_critical"] = True

        should_demote = critical_failures >= 2

        results["critical_failures"] = critical_failures
        results["should_demote"] = should_demote
        results["stats"] = asdict(stats)

        return should_demote, results

    def try_promote(self) -> Tuple[bool, Optional[CurriculumStage]]:
        if not self.auto_promote:
            return False, None

        config = self.stage_config
        if config.is_terminal:
            return False, None

        next_stage = get_next_stage(self.current_stage)
        if next_stage is None:
            return False, None
        
        # Respect transition cooldown
        if self._transition_cooldown_remaining > 0:
            return False, None

        meets_criteria, results = self.check_promotion_criteria()
        if not meets_criteria:
            return False, None

        old_stage = self.current_stage
        self.current_stage = next_stage
        self._enter_stage(next_stage, reason="promotion")

        transition_info = {
            "type": "promotion",
            "from_stage": old_stage.name,
            "to_stage": next_stage.name,
            "timestamp": _now_iso(self.tz),
            "total_timesteps": self.total_timesteps,
            "total_episodes": self.total_episodes,
            "criteria_results": results,
        }
        self._transitions.append(transition_info)

        if self.verbose:
            logger.info(f"🎓 PROMOTION: {old_stage.name} → {next_stage.name} (epoch={self._current_stage_epoch})")
        
        # Trigger callback for checkpoint/LR adjustment
        if self.on_transition_callback is not None:
            try:
                self.on_transition_callback("promotion", old_stage, next_stage, transition_info)
            except Exception as e:
                logger.warning(f"Transition callback error: {e}")

        return True, next_stage

    def try_demote(self) -> Tuple[bool, Optional[CurriculumStage]]:
        if not self.auto_demote:
            return False, None

        previous_stage = get_previous_stage(self.current_stage)
        if previous_stage is None:
            return False, None
        
        # Respect transition cooldown
        if self._transition_cooldown_remaining > 0:
            return False, None

        should_demote, results = self.check_demotion_criteria()
        if not should_demote:
            return False, None

        old_stage = self.current_stage
        self.current_stage = previous_stage
        self._enter_stage(previous_stage, reason="demotion")

        transition_info = {
            "type": "demotion",
            "from_stage": old_stage.name,
            "to_stage": previous_stage.name,
            "timestamp": _now_iso(self.tz),
            "total_timesteps": self.total_timesteps,
            "total_episodes": self.total_episodes,
            "criteria_results": results,
        }
        self._transitions.append(transition_info)

        if self.verbose:
            logger.warning(f"📉 DEMOTION: {old_stage.name} → {previous_stage.name} (epoch={self._current_stage_epoch})")
        
        # Trigger callback for checkpoint/LR adjustment
        if self.on_transition_callback is not None:
            try:
                self.on_transition_callback("demotion", old_stage, previous_stage, transition_info)
            except Exception as e:
                logger.warning(f"Transition callback error: {e}")

        return True, previous_stage

    def update(self) -> Tuple[bool, Optional[CurriculumStage]]:
        promoted, new_stage = self.try_promote()
        if promoted:
            return True, new_stage

        demoted, new_stage = self.try_demote()
        if demoted:
            return True, new_stage

        return False, None

    def force_stage(self, stage: CurriculumStage, reason: str = "manual") -> None:
        old_stage = self.current_stage
        self.current_stage = stage
        self._enter_stage(stage, reason=f"force:{reason}")

        self._transitions.append({
            "type": "force",
            "from_stage": old_stage.name,
            "to_stage": stage.name,
            "reason": reason,
            "timestamp": _now_iso(self.tz),
            "total_timesteps": self.total_timesteps,
            "total_episodes": self.total_episodes,
        })

        if self.verbose:
            logger.info(f"⚡ FORCE STAGE: {old_stage.name} → {stage.name} (reason={reason})")

    def get_progress_report(self) -> Dict[str, Any]:
        stats = self.get_rolling_stats()
        meets_promotion, promotion_results = self.check_promotion_criteria()
        meets_demotion, demotion_results = self.check_demotion_criteria()

        progression = get_stage_progression()
        stage_idx = progression.index(self.current_stage)

        return {
            "current_stage": self.current_stage.name,
            "current_stage_epoch": self._current_stage_epoch,
            "stage_index": stage_idx,
            "total_stages": len(progression),
            "progress_pct": (stage_idx / (len(progression) - 1)) * 100 if len(progression) > 1 else 100,
            "total_timesteps": self.total_timesteps,
            "total_episodes": self.total_episodes,
            "stage_timesteps": self.stage_timesteps,
            "stage_episodes": self.stage_episodes,
            "rolling_stats": asdict(stats),
            "promotion_ready": meets_promotion,
            "promotion_checks": promotion_results.get("checks", {}),
            "demotion_risk": meets_demotion,
            "demotion_checks": demotion_results.get("checks", {}),
            "transitions_count": len(self._transitions),
            "transitions": self._transitions[-5:],
            # Transition state info
            "is_in_transition": self.is_in_transition,
            "transition_cooldown_remaining": self._transition_cooldown_remaining,
            "reward_blend_remaining": self._reward_blend_remaining,
            "reward_blend_factor": self.reward_blend_factor,
            "lr_warmup_active": self._lr_warmup_active,
            "lr_warmup_steps_remaining": self._lr_warmup_steps_remaining,
            "lr_multiplier": self.get_lr_multiplier(),
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        history_serialized: Dict[str, List[Dict[str, Any]]] = {}
        for stage, episodes in self._history.items():
            history_serialized[stage.name] = [asdict(ep) for ep in episodes]

        state = {
            "version": STATE_VERSION,
            "tz": self.tz,
            "current_stage": self.current_stage.name,
            "current_stage_epoch": self._current_stage_epoch,
            "total_timesteps": self.total_timesteps,
            "total_episodes": self.total_episodes,
            "stage_timesteps_total": {s.name: t for s, t in self._stage_timesteps_total.items()},
            "stage_episodes_total": {s.name: e for s, e in self._stage_episodes_total.items()},
            "stage_epoch_counter": {s.name: e for s, e in self._stage_epoch_counter.items()},
            "stage_timesteps_current": self.stage_timesteps,
            "stage_episodes_current": self.stage_episodes,
            "history": history_serialized,
            "transitions": self._transitions,
            "saved_at": _now_iso(self.tz),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, sort_keys=False)

        logger.info(f"Curriculum state saved to {path}")

    @classmethod
    def load(cls, path: Path, **kwargs: Any) -> "CurriculumManager":
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)

        tz = state.get("tz", DEFAULT_TZ)
        current_stage = CurriculumStage[state["current_stage"]]

        manager = cls(initial_stage=current_stage, tz=tz, **kwargs)
        manager.total_timesteps = _safe_int(state.get("total_timesteps", 0), 0)
        manager.total_episodes = _safe_int(state.get("total_episodes", 0), 0)

        # Restore totals
        for stage_name, timesteps in (state.get("stage_timesteps_total", {}) or {}).items():
            try:
                manager._stage_timesteps_total[CurriculumStage[stage_name]] = _safe_int(timesteps, 0)
            except KeyError:
                pass

        for stage_name, episodes in (state.get("stage_episodes_total", {}) or {}).items():
            try:
                manager._stage_episodes_total[CurriculumStage[stage_name]] = _safe_int(episodes, 0)
            except KeyError:
                pass

        # Restore epoch counters + current epoch
        for stage_name, epoch in (state.get("stage_epoch_counter", {}) or {}).items():
            try:
                manager._stage_epoch_counter[CurriculumStage[stage_name]] = _safe_int(epoch, 0)
            except KeyError:
                pass

        manager._current_stage_epoch = _safe_int(state.get("current_stage_epoch", manager._current_stage_epoch), manager._current_stage_epoch)
        manager.stage_timesteps = _safe_int(state.get("stage_timesteps_current", 0), 0)
        manager.stage_episodes = _safe_int(state.get("stage_episodes_current", 0), 0)

        # Restore history
        manager._history = {stage: deque(maxlen=manager.max_history_size) for stage in CurriculumStage}
        for stage_name, episodes_data in (state.get("history", {}) or {}).items():
            try:
                stage = CurriculumStage[stage_name]
            except KeyError:
                continue
            for ep_data in (episodes_data or []):
                try:
                    manager._history[stage].append(EpisodeMetrics.from_dict(ep_data))
                except Exception:
                    continue

        manager._transitions = state.get("transitions", []) or []
        manager._rolling_stats_dirty = True

        logger.info(f"Curriculum state loaded from {path} (stage={current_stage.name}, epoch={manager._current_stage_epoch})")
        return manager

    def __repr__(self) -> str:
        return f"CurriculumManager(stage={self.current_stage.name}, epoch={self._current_stage_epoch}, episodes={self.total_episodes}, timesteps={self.total_timesteps:,})"
