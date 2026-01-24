# envs/curriculum/config/thresholds.py
"""
Competence thresholds and skill requirements.

Contains:
- CompetenceThresholds: Performance thresholds for stage promotion
- SkillRequirements: Per-stage skill requirements
- EntropyTargets: Entropy targets for exploration management
- CompositeScoringConfig: Weighted composite competence scoring config
- AdaptiveThresholdConfig: Adaptive threshold relaxation config

Upgrades (Jan 2026):
- Fixed composite hard_floors key mismatch: "drawdown" is the composite component key.
- Added __post_init__ canonicalization for composite keys (supports legacy "max_drawdown").
- Optional weight normalization (prevents accidental sum != 1 causing inconsistent scaling).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from envs.curriculum.config.stages import TradingSkill


MIN_EVALUATION_EPISODES = 25


@dataclass
class CompetenceThresholds:
    """Performance thresholds for stage promotion."""
    min_episodes: int = 100
    min_timesteps: int = 50_000

    min_win_rate: float = 0.40
    min_profit_factor: float = 0.8
    max_avg_drawdown: float = 0.20
    min_avg_pnl: float = -1000.0

    min_avg_r_multiple: float = 0.0
    min_entropy: float = 0.0

    max_win_rate_std: float = 0.30
    # Trade-aware stability: ignore very-low-trade episodes when computing win-rate variance
    # (e.g., stop-mode / constraint-induced low activity).
    min_trades_per_episode_for_win_rate_stability: int = 3
    # Optional uncertainty gate: pooled-trade Wilson interval width must be below this.
    # Set to 0 to disable.
    max_win_rate_wilson_width: float = 0.0
    max_pnl_std: float = 10000.0
    min_trade_count_avg: float = 1.0

    # Patience / discipline metrics (v2.1)
    min_avg_bars_between_trades: float = 0.0
    min_setup_skipped_per_episode: float = 0.0
    min_entry_certainty_avg: float = 0.0
    min_avg_setup_quality: float = 0.0
    max_fomo_trade_rate: float = 1.0
    max_revenge_trade_rate: float = 1.0

    max_dd_breach_rate: float = 0.50
    max_consecutive_loss_rate: float = 0.30
    # Sample-efficiency / behavioral collapse gates (0..1 fraction of steps)
    max_mask_collapse_rate: float = 1.0
    max_stop_mode_rate: float = 1.0

    # Consistency streak gating (Stage 8+)
    consistency_streak_required: int = 0
    consistency_streak_criteria: Dict[str, float] = field(default_factory=dict)

    evaluation_window: int = 50


@dataclass
class SkillRequirements:
    """
    Per-stage skill requirements for promotion.

    Maps skills to minimum scores [0, 1] required to pass.
    """
    required_skills: Dict["TradingSkill", float] = field(default_factory=dict)

    min_confidence: float = 0.5

    require_all_skills: bool = False
    weighted_threshold: float = 0.6

    skill_weights: Dict["TradingSkill", float] = field(default_factory=dict)

    def get_weight(self, skill: "TradingSkill") -> float:
        return float(self.skill_weights.get(skill, 1.0))


@dataclass
class EntropyTargets:
    """
    Entropy targets for exploration management.
    """
    min_entropy: float = 0.1
    max_entropy: float = 0.8

    low_entropy_penalty_scale: float = 0.1
    high_entropy_penalty_scale: float = 0.05

    use_in_promotion: bool = True


@dataclass
class CompositeScoringConfig:
    """
    Configuration for weighted composite competence scoring.

    Component keys produced by compute_composite_score() must match these:
        - "win_rate", "profit_factor", "drawdown", "consistency",
        - "r_multiple", "dd_breach_rate", "trade_activity", "consecutive_loss_rate"
    """
    enabled: bool = True

    weights: Dict[str, float] = field(default_factory=lambda: {
        "win_rate": 0.20,
        "profit_factor": 0.20,
        "drawdown": 0.15,
        "consistency": 0.15,
        "r_multiple": 0.10,
        "dd_breach_rate": 0.10,
        "trade_activity": 0.05,
        "consecutive_loss_rate": 0.05,
    })

    # Hard floors: must meet regardless of composite score
    hard_floors: Dict[str, float] = field(default_factory=lambda: {
        "win_rate": 0.30,
        "drawdown": 0.25,
        "dd_breach_rate": 0.40,
    })

    promotion_threshold: float = 0.70
    demotion_threshold: float = 0.35

    normalize_weights: bool = True

    def __post_init__(self) -> None:
        # Canonicalize legacy keys (e.g., max_drawdown -> drawdown)
        from envs.curriculum.config.registry import canonicalize_composite_key

        def canon_map(d: Dict[str, float]) -> Dict[str, float]:
            out: Dict[str, float] = {}
            for k, v in (d or {}).items():
                ck = canonicalize_composite_key(str(k))
                out[ck] = float(out.get(ck, 0.0) + float(v))
            return out

        self.weights = canon_map(self.weights)
        self.hard_floors = canon_map(self.hard_floors)

        # Optional normalization for stability (prevents accidental sum drift)
        if self.normalize_weights:
            s = float(sum(max(0.0, float(v)) for v in self.weights.values()))
            if s > 0:
                self.weights = {k: float(max(0.0, v)) / s for k, v in self.weights.items()}


@dataclass
class AdaptiveThresholdConfig:
    """
    Configuration for adaptive threshold relaxation.
    """
    enabled: bool = True

    plateau_episodes_threshold: int = 100
    plateau_improvement_threshold: float = 0.01

    max_relaxation: float = 0.10
    relaxation_buildup_episodes: int = 500

    relaxable_metrics: Set[str] = field(default_factory=lambda: {
        "min_win_rate",
        "min_profit_factor",
        "min_avg_pnl",
        "min_trade_count_avg",
        "max_win_rate_std",
        "max_pnl_std",
    })

    never_relax: Set[str] = field(default_factory=lambda: {
        "max_avg_drawdown",
        "max_dd_breach_rate",
        "max_consecutive_loss_rate",
    })


__all__ = [
    "CompetenceThresholds",
    "SkillRequirements",
    "EntropyTargets",
    "CompositeScoringConfig",
    "AdaptiveThresholdConfig",
    "MIN_EVALUATION_EPISODES",
]
