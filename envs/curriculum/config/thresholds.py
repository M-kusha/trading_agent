

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Set

if TYPE_CHECKING:
    from envs.curriculum.config.stages import TradingSkill


MIN_EVALUATION_EPISODES = 25


@dataclass
class CompetenceThresholds:
    min_episodes: int = 100
    min_timesteps: int = 50_000

    min_win_rate: float = 0.40
    min_profit_factor: float = 0.8
    max_avg_drawdown: float = 0.20
    min_avg_pnl: float = -1000.0

    min_avg_r_multiple: float = 0.0
    min_entropy: float = 0.0

    max_win_rate_std: float = 0.30


    min_trades_per_episode_for_win_rate_stability: int = 3


    max_win_rate_wilson_width: float = 0.0
    max_pnl_std: float = 10000.0
    min_trade_count_avg: float = 1.0


    min_avg_bars_between_trades: float = 0.0
    min_setup_skipped_per_episode: float = 0.0
    min_entry_certainty_avg: float = 0.0
    min_avg_setup_quality: float = 0.0
    max_fomo_trade_rate: float = 1.0
    max_revenge_trade_rate: float = 1.0

    max_dd_breach_rate: float = 0.50
    max_consecutive_loss_rate: float = 0.30

    max_mask_collapse_rate: float = 1.0
    max_stop_mode_rate: float = 1.0


    consistency_streak_required: int = 0
    consistency_streak_criteria: Dict[str, float] = field(default_factory=dict)

    evaluation_window: int = 50


@dataclass
class SkillRequirements:
    required_skills: Dict["TradingSkill", float] = field(default_factory=dict)

    min_confidence: float = 0.5

    require_all_skills: bool = False
    weighted_threshold: float = 0.6

    skill_weights: Dict["TradingSkill", float] = field(default_factory=dict)

    def get_weight(self, skill: "TradingSkill") -> float:
        return float(self.skill_weights.get(skill, 1.0))


@dataclass
class EntropyTargets:
    min_entropy: float = 0.1
    max_entropy: float = 0.8

    low_entropy_penalty_scale: float = 0.1
    high_entropy_penalty_scale: float = 0.05

    use_in_promotion: bool = True


@dataclass
class CompositeScoringConfig:
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


    hard_floors: Dict[str, float] = field(default_factory=lambda: {
        "win_rate": 0.30,
        "drawdown": 0.25,
        "dd_breach_rate": 0.40,
    })

    promotion_threshold: float = 0.70
    demotion_threshold: float = 0.35

    normalize_weights: bool = True

    def __post_init__(self) -> None:

        from envs.curriculum.config.registry import canonicalize_composite_key

        def canon_map(d: Dict[str, float]) -> Dict[str, float]:
            out: Dict[str, float] = {}
            for k, v in (d or {}).items():
                ck = canonicalize_composite_key(str(k))
                out[ck] = float(out.get(ck, 0.0) + float(v))
            return out

        self.weights = canon_map(self.weights)
        self.hard_floors = canon_map(self.hard_floors)


        if self.normalize_weights:
            s = float(sum(max(0.0, float(v)) for v in self.weights.values()))
            if s > 0:
                self.weights = {k: float(max(0.0, v)) / s for k, v in self.weights.items()}


@dataclass
class AdaptiveThresholdConfig:
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
    "MIN_EVALUATION_EPISODES",
    "AdaptiveThresholdConfig",
    "CompetenceThresholds",
    "CompositeScoringConfig",
    "EntropyTargets",
    "SkillRequirements",
]
