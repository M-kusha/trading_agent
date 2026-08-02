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
from envs.curriculum.curriculum_config import (
    CurriculumStageConfig,
    DataDifficulty,
    ExecutionDifficulty,
    RewardShaping,
    TradingConstraints,
    get_stage_config,
    get_stage_progression,
)
from envs.curriculum.curriculum_invariants import (
    AntiGamingChecker,
    CurriculumInvariantChecker,
    InvariantViolation,
    reconcile_trade_accounting,
)
from envs.curriculum.curriculum_manager import (
    CurriculumManager,
    CurriculumStage,
)

# Supporting classes (already in curriculum/)
from envs.curriculum.metrics import (
    CompositeScore,
    EpisodeMetrics,
    LearningVelocity,
    RollingStats,
    compute_adjusted_thresholds,
    compute_composite_score,
)
from envs.curriculum.protocols import (
    RecoveryProtocolState,
    ReviewSessionState,
)
from envs.curriculum.regime_skill_assessment import (
    RegimePerformance,
    RegimeSkillAssessment,
)
from envs.curriculum.skills import (
    DemotionAnalyzer,
    DemotionRecord,
    SkillAssessment,
)
from envs.curriculum.validation_gates import (
    StressTestConfig,
    StressTestRunner,
    ValidationGateChecker,
    ValidationGateConfig,
    ValidationGateResult,
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

