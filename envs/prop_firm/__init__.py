# envs/prop_firm/__init__.py
"""
Prop Firm Trading Environment Components.

This package contains modular components extracted from the main PropFirmTradingEnv class.
All mixins are re-exported here for backward compatibility.

Mixins:
- ExpertSignalsMixin: Expert signal computation (trend, momentum, theme)
- MarketStructureMixin: Institutional-grade S/R, order blocks, liquidity
- EntryQualityMixin: Entry quality computation
- TradeRewardMixin: Trade reward calculation
- RewardShapingMixin: Per-step shaping and blocked action penalties
- ObservationBuildersMixin: State builders for observation construction
- SessionTimingMixin: Session windows, weekend detection, timing
- DataDifficultyMixin: Curriculum-based data filtering
"""

from envs.prop_firm.signals.expert_signals import ExpertSignalsMixin
from envs.prop_firm.signals.market_structure import MarketStructureMixin
from envs.prop_firm.signals.entry_quality import EntryQualityMixin
from envs.prop_firm.rewards.trade_reward import TradeRewardMixin
from envs.prop_firm.rewards.shaping import RewardShapingMixin
from envs.prop_firm.observation.state_builders import ObservationBuildersMixin
from envs.prop_firm.session.timing import SessionTimingMixin
from envs.prop_firm.data.difficulty import DataDifficultyMixin

__all__ = [
    "ExpertSignalsMixin",
    "MarketStructureMixin",
    "EntryQualityMixin",
    "TradeRewardMixin",
    "RewardShapingMixin",
    "ObservationBuildersMixin",
    "SessionTimingMixin",
    "DataDifficultyMixin",
]
