# envs/curriculum/__init__.py
"""
Curriculum learning components for trading RL agent.

This package contains:
- curriculum_manager: CurriculumManager, CurriculumStage
- curriculum_config: Stage configs, DataDifficulty, ExecutionDifficulty, etc.
- curriculum_invariants: Curriculum invariant checks
- validation_gates: Stage promotion validation
- regime_skill_assessment: Regime-specific skill tracking
- metrics, skills, protocols: Supporting classes
"""

# Core curriculum components (moved from envs/)
from envs.curriculum.curriculum_manager import (
    CurriculumManager,
    CurriculumStage,
)

from envs.curriculum.curriculum_config import (
    DataDifficulty,
    ExecutionDifficulty,
    TradingConstraints,
    RewardShaping,
    CurriculumStageConfig,
    get_stage_config,
    get_stage_progression,
)

from envs.curriculum.curriculum_invariants import (
    CurriculumInvariantChecker,
    InvariantViolation,
    AntiGamingChecker,
    reconcile_trade_accounting,
)

from envs.curriculum.validation_gates import (
    ValidationGateChecker,
    ValidationGateConfig,
    ValidationGateResult,
    StressTestRunner,
    StressTestConfig,
)

from envs.curriculum.regime_skill_assessment import (
    RegimeSkillAssessment,
    RegimePerformance,
)

# Supporting classes (already in curriculum/)
from envs.curriculum.metrics import (
    EpisodeMetrics,
    RollingStats,
    LearningVelocity,
    CompositeScore,
    compute_composite_score,
    compute_adjusted_thresholds,
)

from envs.curriculum.skills import (
    SkillAssessment,
    DemotionRecord,
    DemotionAnalyzer,
)

from envs.curriculum.protocols import (
    RecoveryProtocolState,
    ReviewSessionState,
)

__all__ = [
    # Core
    "CurriculumManager",
    "CurriculumStage",
    # Config
    "DataDifficulty",
    "ExecutionDifficulty",
    "TradingConstraints",
    "RewardShaping",
    "CurriculumStageConfig",
    "get_stage_config",
    "get_stage_progression",
    # Invariants
    "CurriculumInvariantChecker",
    "InvariantViolation",
    "AntiGamingChecker",
    "reconcile_trade_accounting",
    # Validation
    "ValidationGateChecker",
    "ValidationGateConfig",
    "ValidationGateResult",
    "StressTestRunner",
    "StressTestConfig",
    # Regime skills
    "RegimeSkillAssessment",
    "RegimePerformance",
    # Metrics
    "EpisodeMetrics",
    "RollingStats",
    "LearningVelocity",
    "CompositeScore",
    "compute_composite_score",
    "compute_adjusted_thresholds",
    # Skills
    "SkillAssessment",
    "DemotionRecord",
    "DemotionAnalyzer",
    # Protocols
    "RecoveryProtocolState",
    "ReviewSessionState",
]

