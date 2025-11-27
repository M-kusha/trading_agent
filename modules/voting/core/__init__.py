# modules/voting/core/__init__.py
"""
Core voting system types, base classes, and constants.
"""

from .types import VotingProposal, VoteBundle, ConsensusResult, CollusionResult
from .constants import VotingAction, PipelineStage, VOTING_DEFAULTS
from .base import VotingModuleBase

__all__ = [
    # Types
    "VotingProposal",
    "VoteBundle", 
    "ConsensusResult",
    "CollusionResult",
    # Enums
    "VotingAction",
    "PipelineStage",
    # Constants
    "VOTING_DEFAULTS",
    # Base
    "VotingModuleBase",
]
