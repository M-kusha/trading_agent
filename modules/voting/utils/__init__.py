"""
Voting Utilities Package
========================
Shared utilities for the voting system.
"""

from .validators import (
    validate_proposal,
    validate_confidence,
    validate_action,
    validate_voting_action,
    sanitize_bus_key,
)

from .metrics import (
    calculate_agreement_score,
    calculate_weighted_consensus,
    calculate_diversity_index,
    calculate_correlation,
    calculate_collusion_score,
    calculate_fragility_score,
    safe_mean,
    safe_std,
    clip_value,
)

__all__ = [
    # Validators
    "validate_proposal",
    "validate_confidence",
    "validate_action",
    "validate_voting_action",
    "sanitize_bus_key",
    # Metrics
    "calculate_agreement_score",
    "calculate_weighted_consensus",
    "calculate_diversity_index",
    "calculate_correlation",
    "calculate_collusion_score",
    "calculate_fragility_score",
    "safe_mean",
    "safe_std",
    "clip_value",
]
