# envs/curriculum/config/protocols.py
"""
Recovery, sampling, review, and validation protocol configurations.

Contains:
- RecoveryProtocolConfig: Configuration for recovery after repeated failures
- MixedStageSamplingConfig: Training on mixture of stages
- ReviewSessionConfig: Periodic review sessions on earlier stages
- ValidationConfig: Hold-out validation before promotion
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from envs.curriculum.config.stages import TradingSkill, MarketRegime, CurriculumStage


@dataclass
class RecoveryProtocolConfig:
    """
    Configuration for recovery protocol after repeated failures.
    """
    enabled: bool = True
    
    # Trigger after this many demotions from the same stage
    trigger_after_demotions: int = 2
    
    # Duration of recovery protocol in episodes
    recovery_duration_episodes: int = 200
    
    # Focus skill override (if None, determined from diagnosis)
    focus_skill: Optional["TradingSkill"] = None
    
    # Reward modifications during recovery
    reward_modifications: Dict[str, float] = field(default_factory=dict)
    
    # Constraint modifications during recovery (stricter limits)
    constraint_modifications: Dict[str, float] = field(default_factory=dict)


@dataclass
class MixedStageSamplingConfig:
    """
    Configuration for training on mixture of stages.
    
    Prevents catastrophic forgetting by occasionally training on earlier stages.
    """
    enabled: bool = True
    
    # Weight for current stage
    current_stage_weight: float = 0.70
    
    # Weight for recent stages (1-2 stages back)
    recent_stages_weight: float = 0.20
    
    # Weight for foundation stage (always some basics)
    foundation_weight: float = 0.10
    
    # How many stages back to sample from
    recent_stage_depth: int = 2


@dataclass
class ReviewSessionConfig:
    """
    Configuration for periodic review sessions on earlier stages.
    
    Ensures agent hasn't forgotten earlier skills.
    """
    enabled: bool = True
    
    # Episodes between review sessions
    review_frequency: int = 500
    
    # Episodes per review session
    review_duration: int = 50
    
    # How many stages back to review
    review_depth: int = 2
    
    # Minimum stage to trigger reviews (no reviews in early discovery)
    # Note: This will be set to CurriculumStage.INTEGRATOR at import time
    min_stage_for_review: Optional["CurriculumStage"] = None
    
    def __post_init__(self):
        if self.min_stage_for_review is None:
            from envs.curriculum.config.stages import CurriculumStage
            self.min_stage_for_review = CurriculumStage.INTEGRATOR


@dataclass
class ValidationConfig:
    """
    Configuration for hold-out validation before promotion.
    """
    enabled: bool = False  # Disabled by default (requires separate validation data)
    
    # Number of validation episodes
    validation_episodes: int = 100
    
    # Minimum performance ratio (validation / training)
    min_performance_ratio: float = 0.85
    
    # Maximum acceptable performance drop
    max_performance_drop: float = 0.15
    
    # Required regimes to validate on
    required_regimes: List["MarketRegime"] = field(default_factory=lambda: _default_required_regimes())
    
    # Minimum episodes per regime
    min_episodes_per_regime: int = 20


def _default_required_regimes():
    """Default required regimes for validation."""
    from envs.curriculum.config.stages import MarketRegime
    return [
        MarketRegime.TRENDING_UP,
        MarketRegime.TRENDING_DOWN,
        MarketRegime.RANGING,
    ]


__all__ = [
    "RecoveryProtocolConfig",
    "MixedStageSamplingConfig",
    "ReviewSessionConfig",
    "ValidationConfig",
]
