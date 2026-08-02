"""
Voting Experts Package
======================
Contains voting expert implementations and their base class.

Experts are voting members that analyze specific aspects of market data
and generate voting proposals with confidence scores.
"""

from modules.utils.session_utils import normalize_session_name
from modules.voting.experts.base import VotingExpertBase
from modules.voting.experts.momentum import MomentumExpert
from modules.voting.experts.seasonality import SeasonalityRiskExpert
from modules.voting.experts.theme import ThemeExpert
from modules.voting.experts.trend import TrendExpert

__all__ = [
    # Base
    'VotingExpertBase',
    
    # Experts
    'ThemeExpert',
    'SeasonalityRiskExpert',
    'MomentumExpert',
    'TrendExpert',
    
    # Utilities
    'normalize_session_name',
]
