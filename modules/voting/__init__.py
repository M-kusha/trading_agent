# modules/voting/__init__.py
"""
Unified Voting System v5.0

This package provides a self-contained, modular voting infrastructure for the trading system.
All modules are independent and do NOT depend on legacy voting_wrappers.py or other old modules.

Architecture Overview:
├── core/       - Base types, constants, and abstract classes
│   ├── constants.py  - VotingAction enum, thresholds, bus keys
│   ├── types.py      - VotingProposal, VoteBundle, ConsensusResult, CollusionResult
│   └── base.py       - VotingModuleBase (shared infrastructure)
│
├── experts/    - Voting members that generate proposals
│   ├── base.py       - VotingExpertBase abstract class
│   ├── theme.py      - ThemeExpert (risk-on/off, volatility, trending)
│   └── seasonality.py- SeasonalityRiskExpert (session-based adjustments)
│
├── stages/     - Pipeline stages for processing votes
│   ├── committee.py  - CommitteeCoordinator (vote collection)
│   ├── consensus.py  - ConsensusAnalyzer (agreement scoring)
│   ├── collusion.py  - CollusionDetector (suspicious pattern detection)
│   ├── horizon.py    - HorizonAligner (time-based weight adjustments)
│   ├── uncertainty.py- UncertaintySampler (robustness sampling)
│   └── arbiter.py    - FinalArbiter (final gate decision)
│
├── pipeline/   - Orchestration
│   └── kernel.py     - SlimVotingKernel (end-to-end pipeline)
│
└── utils/      - Shared utilities
    ├── validators.py - Input validation utilities
    └── metrics.py    - Consensus and agreement metrics

Contracts are defined in modules/contracts.py (v5.0.0 entries).
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════════
# CORE - Always available (base infrastructure)
# ═══════════════════════════════════════════════════════════════════════════════

from .core.types import (
    VotingProposal,
    VoteBundle,
    ConsensusResult,
    CollusionResult,
    create_empty_bundle,
    create_abstain_bundle,
    make_vote_bundle,
    make_consensus_result,
)

from .core.constants import (
    VotingAction,
    PipelineStage,
    VotingQuality,
    VotingBusKeys,
    VOTING_DEFAULTS,
    KNOWN_VOTING_MEMBERS,
    CONSENSUS_THRESHOLDS,
    COLLUSION_THRESHOLDS,
    HORIZON_WEIGHTS,
)

from .core.base import VotingModuleBase

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERTS - Voting members that generate proposals
# ═══════════════════════════════════════════════════════════════════════════════

from .experts.base import VotingExpertBase
from .experts.theme import ThemeExpert
from .experts.seasonality import SeasonalityRiskExpert

# ═══════════════════════════════════════════════════════════════════════════════
# STAGES - Pipeline processing stages
# ═══════════════════════════════════════════════════════════════════════════════

from .stages.committee import CommitteeCoordinator
from .stages.consensus import ConsensusAnalyzer
from .stages.collusion import CollusionDetector
from .stages.horizon import HorizonAligner
from .stages.uncertainty import UncertaintySampler
from .stages.arbiter import FinalArbiter

# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE - Orchestration
# ═══════════════════════════════════════════════════════════════════════════════

from .pipeline.kernel import SlimVotingKernel

# ═══════════════════════════════════════════════════════════════════════════════
# UTILS - Shared utilities
# ═══════════════════════════════════════════════════════════════════════════════

from .utils.validators import (
    validate_proposal,
    validate_confidence,
    validate_voting_action,
    sanitize_bus_key,
)

from .utils.metrics import (
    calculate_agreement_score,
    calculate_weighted_consensus,
    calculate_diversity_index,
    calculate_collusion_score,
    calculate_fragility_score,
)


__all__ = [
    # Core types
    "VotingProposal",
    "VoteBundle",
    "ConsensusResult",
    "CollusionResult",
    "create_empty_bundle",
    "create_abstain_bundle",
    "make_vote_bundle",
    "make_consensus_result",
    # Enums/constants
    "VotingAction",
    "PipelineStage",
    "VotingQuality",
    "VotingBusKeys",
    "VOTING_DEFAULTS",
    "KNOWN_VOTING_MEMBERS",
    "CONSENSUS_THRESHOLDS",
    "COLLUSION_THRESHOLDS",
    "HORIZON_WEIGHTS",
    # Base classes
    "VotingModuleBase",
    "VotingExpertBase",
    # Experts
    "ThemeExpert",
    "SeasonalityRiskExpert",
    # Stages
    "CommitteeCoordinator",
    "ConsensusAnalyzer",
    "CollusionDetector",
    "HorizonAligner",
    "UncertaintySampler",
    "FinalArbiter",
    # Pipeline
    "SlimVotingKernel",
    # Utils
    "validate_proposal",
    "validate_confidence",
    "validate_voting_action",
    "sanitize_bus_key",
    "calculate_agreement_score",
    "calculate_weighted_consensus",
    "calculate_diversity_index",
    "calculate_collusion_score",
    "calculate_fragility_score",
]
