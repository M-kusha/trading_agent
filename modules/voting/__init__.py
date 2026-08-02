# modules/voting/__init__.py
"""Shared voting types, constants and helpers.

The voting IMPLEMENTATION was removed on 2026-08-02:

  experts/   TrendExpert, MomentumExpert, ThemeExpert, SeasonalityRiskExpert
  stages/    CommitteeCoordinator, ConsensusAnalyzer, CollusionDetector,
             HorizonAligner, UncertaintySampler, FinalArbiter
  pipeline/  SlimVotingKernel

Why - measured over 1,600 samples of XAUUSD M15 against forward returns:

    strategy                   h=4 bp    h=16 bp    h=96 bp
    ALWAYS LONG (control)       -0.00       1.15      10.18
    theme                        1.04       2.00       6.32
    committee                    0.78       2.14       6.11
    trend                        0.49       2.01       4.50
    momentum                    -0.32      -1.06       0.64

Every expert underperformed a trivial always-long control at the daily horizon.
Hit rates sat at 49-52% and the best information coefficient was +0.043 at
p=0.089 - not significant. They were also largely one signal wearing four hats:
trend and theme correlate 0.817, and the committee correlates 0.944 with the
trend expert alone.

Cost side: running them in training measured 419 ms/step against a ~12 ms
baseline - a 36x slowdown that turns a 3.3-hour run into 120 hours - while
occupying observation dimensions the dataset cannot support (roughly 3,223
effectively independent samples for the entire feature vector).

What remains is the shared vocabulary: types, constants, thresholds, metrics
and validators. Live code still imports these, notably
modules.voting.core.constants.is_training_mode.

To recover an expert for an ablation:
    git log --diff-filter=D -- modules/voting/experts/seasonality.py

SeasonalityRiskExpert is the one worth re-testing. Session and seasonality is
the only genuinely orthogonal idea in the set, and the training-side version
was always a hardcoded stub that has never been evaluated.
"""

from __future__ import annotations

from .core.base import VotingModuleBase
from .core.constants import (
    COLLUSION_THRESHOLDS,
    CONSENSUS_THRESHOLDS,
    HORIZON_WEIGHTS,
    KNOWN_VOTING_MEMBERS,
    VOTING_DEFAULTS,
    PipelineStage,
    VotingAction,
    VotingBusKeys,
    VotingQuality,
)
from .core.types import (
    CollusionResult,
    ConsensusResult,
    VoteBundle,
    VotingProposal,
    create_abstain_bundle,
    create_empty_bundle,
    make_consensus_result,
    make_vote_bundle,
)
from .utils.metrics import (
    calculate_agreement_score,
    calculate_collusion_score,
    calculate_diversity_index,
    calculate_fragility_score,
    calculate_weighted_consensus,
)
from .utils.validators import (
    sanitize_bus_key,
    validate_confidence,
    validate_proposal,
    validate_voting_action,
)

__all__ = [
    "COLLUSION_THRESHOLDS",
    "CONSENSUS_THRESHOLDS",
    "HORIZON_WEIGHTS",
    "KNOWN_VOTING_MEMBERS",
    "VOTING_DEFAULTS",
    "CollusionResult",
    "ConsensusResult",
    "PipelineStage",
    "VoteBundle",
    "VotingAction",
    "VotingBusKeys",
    "VotingModuleBase",
    "VotingProposal",
    "VotingQuality",
    "calculate_agreement_score",
    "calculate_collusion_score",
    "calculate_diversity_index",
    "calculate_fragility_score",
    "calculate_weighted_consensus",
    "create_abstain_bundle",
    "create_empty_bundle",
    "make_consensus_result",
    "make_vote_bundle",
    "sanitize_bus_key",
    "validate_confidence",
    "validate_proposal",
    "validate_voting_action",
]
