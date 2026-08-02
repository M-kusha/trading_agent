# envs/curriculum/config/__init__.py
"""
Curriculum configuration classes.

Re-exports all configuration classes for backward compatibility.
"""

from envs.curriculum.config.constraints import (
    RewardShaping,
    TradingConstraints,
)
from envs.curriculum.config.execution import (
    DataDifficulty,
    ExecutionDifficulty,
    TransitionSettings,
)
from envs.curriculum.config.protocols import (
    MixedStageSamplingConfig,
    RecoveryProtocolConfig,
    ReviewSessionConfig,
    ValidationConfig,
)
from envs.curriculum.config.registry import (
    COMPOSITE_HARD_FLOOR_KEYS,
    COMPOSITE_WEIGHT_KEYS,
    METRIC_CANONICAL_NAMES,
    METRIC_TO_THRESHOLD,
    OBSERVED_ALIASES,
    # Metric registries
    OBSERVED_METRICS,
    THRESHOLD_ALIASES,
    THRESHOLD_FIELDS,
    canonicalize_metric,
    # Helper functions
    canonicalize_observed_metric,
    canonicalize_threshold_field,
    get_threshold_for_metric,
    is_valid_observed_metric,
    is_valid_threshold_field,
)
from envs.curriculum.config.stages import (
    CurriculumStage,
    CurriculumStageConfig,
    MarketRegime,
    TradingSkill,
)
from envs.curriculum.config.thresholds import (
    MIN_EVALUATION_EPISODES,
    AdaptiveThresholdConfig,
    CompetenceThresholds,
    CompositeScoringConfig,
    EntropyTargets,
    SkillRequirements,
)

__all__ = [
    # Stages
    "CurriculumStage",
    "TradingSkill",
    "MarketRegime",
    "CurriculumStageConfig",
    # Execution
    "ExecutionDifficulty",
    "DataDifficulty",
    "TransitionSettings",
    # Thresholds
    "CompetenceThresholds",
    "SkillRequirements",
    "EntropyTargets",
    "CompositeScoringConfig",
    "AdaptiveThresholdConfig",
    "MIN_EVALUATION_EPISODES",
    # Constraints
    "TradingConstraints",
    "RewardShaping",
    # Protocols
    "RecoveryProtocolConfig",
    "MixedStageSamplingConfig",
    "ReviewSessionConfig",
    "ValidationConfig",
    # Registry
    "OBSERVED_METRICS",
    "OBSERVED_ALIASES",
    "THRESHOLD_FIELDS",
    "THRESHOLD_ALIASES",
    "METRIC_TO_THRESHOLD",
    "COMPOSITE_WEIGHT_KEYS",
    "COMPOSITE_HARD_FLOOR_KEYS",
    "METRIC_CANONICAL_NAMES",
    "canonicalize_observed_metric",
    "canonicalize_threshold_field",
    "canonicalize_metric",
    "is_valid_observed_metric",
    "is_valid_threshold_field",
    "get_threshold_for_metric",
]
