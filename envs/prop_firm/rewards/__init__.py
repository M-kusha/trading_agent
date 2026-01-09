# envs/prop_firm/rewards/__init__.py
"""
Reward computation components for PropFirmTradingEnv.

Contains:
- TradeRewardMixin: Main trade-close reward calculation
- RewardShapingMixin: Per-step shaping and blocked action penalties
"""

from envs.prop_firm.rewards.trade_reward import TradeRewardMixin
from envs.prop_firm.rewards.shaping import RewardShapingMixin

__all__ = [
    "TradeRewardMixin",
    "RewardShapingMixin",
]
