# modules/voting/core/__init__.py
"""
Core voting system types, base classes, and constants.
"""

from .base import VotingModuleBase
from .constants import VOTING_DEFAULTS, PipelineStage, VotingAction
from .types import CollusionResult, ConsensusResult, VoteBundle, VotingProposal

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
