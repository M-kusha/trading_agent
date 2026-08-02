

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from envs.curriculum.config.stages import CurriculumStage, MarketRegime, TradingSkill


@dataclass
class RecoveryProtocolConfig:
    enabled: bool = True


    trigger_after_demotions: int = 2


    recovery_duration_episodes: int = 200


    focus_skill: Optional["TradingSkill"] = None


    reward_modifications: Dict[str, float] = field(default_factory=dict)


    constraint_modifications: Dict[str, float] = field(default_factory=dict)


@dataclass
class MixedStageSamplingConfig:
    enabled: bool = True


    current_stage_weight: float = 0.70


    recent_stages_weight: float = 0.20


    foundation_weight: float = 0.10


    recent_stage_depth: int = 2


@dataclass
class ReviewSessionConfig:
    enabled: bool = True


    review_frequency: int = 500


    review_duration: int = 50


    review_depth: int = 2


    min_stage_for_review: Optional["CurriculumStage"] = None

    def __post_init__(self):
        if self.min_stage_for_review is None:
            from envs.curriculum.config.stages import CurriculumStage
            self.min_stage_for_review = CurriculumStage.INTEGRATOR


@dataclass
class ValidationConfig:
    enabled: bool = False


    validation_episodes: int = 100


    min_performance_ratio: float = 0.85


    max_performance_drop: float = 0.15


    required_regimes: List["MarketRegime"] = field(default_factory=lambda: _default_required_regimes())


    min_episodes_per_regime: int = 20


def _default_required_regimes():
    from envs.curriculum.config.stages import MarketRegime
    return [
        MarketRegime.TRENDING_UP,
        MarketRegime.TRENDING_DOWN,
        MarketRegime.RANGING,
    ]


__all__ = [
    "MixedStageSamplingConfig",
    "RecoveryProtocolConfig",
    "ReviewSessionConfig",
    "ValidationConfig",
]
