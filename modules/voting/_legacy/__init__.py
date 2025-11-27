"""
Legacy Voting Modules
=====================
DEPRECATED: These modules have been replaced by the unified voting system in v5.0.

The new architecture is located in:
- modules/voting/core/       - Base types and constants
- modules/voting/experts/    - ThemeExpert, SeasonalityRiskExpert
- modules/voting/stages/     - Pipeline stages
- modules/voting/pipeline/   - SlimVotingKernel orchestrator
- modules/voting/utils/      - Validators and metrics

These legacy files are preserved for reference only and should NOT be imported.
All code is commented out to prevent accidental usage.

Migration Guide:
----------------
Old Module                          -> New Module
voting_kernel.VotingKernel          -> pipeline.kernel.SlimVotingKernel
voting_wrappers.EnhancedThemeExpert -> experts.theme.ThemeExpert
voting_wrappers.EnhancedSeasonality -> experts.seasonality.SeasonalityRiskExpert
voting_wrappers.EnhancedCommittee   -> stages.committee.CommitteeCoordinator
strategy_arbiter.StrategyArbiter    -> stages.arbiter.FinalArbiter
consensus_detector.ConsensusDetector-> stages.consensus.ConsensusAnalyzer
collusion_auditor.CollusionAuditor  -> stages.collusion.CollusionDetector
time_horizon_aligner.TimeHorizon... -> stages.horizon.HorizonAligner
alternative_reality_sampler.Alter...-> stages.uncertainty.UncertaintySampler

Deprecated: November 2025
"""

# No imports - all code is commented out in the legacy files

__all__ = []
