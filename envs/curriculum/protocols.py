# envs/curriculum/protocols.py
"""
Recovery protocol and review session state components.

Contains:
- RecoveryProtocolState: State for active recovery protocol
- ReviewSessionState: State for review session management
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from envs.curriculum.config.stages import (
    CurriculumStage,
    TradingSkill,
)

from envs.core.shared_utils import get_envs_logger

logger = get_envs_logger("curriculum.protocols")


def _foundation_stage() -> CurriculumStage:
    """Get the foundation stage (first stage in progression)."""
    # Use sorted list of CurriculumStage enum values to get first stage
    # This avoids circular import with curriculum_config.get_stage_progression()
    return sorted(CurriculumStage, key=lambda s: s.value)[0]


@dataclass
class RecoveryProtocolState:
    """State for active recovery protocol."""
    triggered: bool = False
    focus_skill: Optional[TradingSkill] = None
    reward_modifications: Dict[str, float] = field(default_factory=dict)
    constraint_modifications: Dict[str, float] = field(default_factory=dict)
    episodes_remaining: int = 0
    trigger_reason: str = ""
    
    def is_active(self) -> bool:
        return self.triggered and self.episodes_remaining > 0
    
    def tick(self) -> None:
        """Decrease episode counter."""
        if self.episodes_remaining > 0:
            self.episodes_remaining -= 1
            if self.episodes_remaining <= 0:
                self.triggered = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "triggered": self.triggered,
            "focus_skill": self.focus_skill.value if self.focus_skill else None,
            "reward_modifications": self.reward_modifications,
            "constraint_modifications": self.constraint_modifications,
            "episodes_remaining": self.episodes_remaining,
            "trigger_reason": self.trigger_reason,
            "is_active": self.is_active(),
        }

    @staticmethod
    def get_patience_focused_recovery() -> Dict[str, Any]:
        """Return a patience/discipline-focused recovery template."""
        return {
            "focus_skill": TradingSkill.DISCIPLINE.value,
            "reward_modifications": {
                # Increase selectivity and discourage impulsive entries
                "churn_penalty_per_trade": 1.5,
                "loss_streak_caution_base": 1.5,
                "fear_of_missing_out_penalty": 1.5,
                "revenge_trading_penalty": 1.5,
                "daily_trade_soft_limit": 0.6,
            },
            "constraint_modifications": {
                # Reduce frequency + enforce cooling off
                "max_trades_per_day": 0.5,
                "min_bars_between_entries": 1.5,
                "min_bars_after_loss": 1.5,
                "entry_quality_gate_enabled": 1.0,
                "entry_quality_threshold": 1.1,
                "min_setup_quality_for_entry": 1.1,
            },
            "description": "focus_patience_discipline",
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecoveryProtocolState":
        """Restore state from dictionary."""
        focus_skill = None
        if data.get("focus_skill"):
            try:
                focus_skill = TradingSkill(data["focus_skill"])
            except (ValueError, KeyError):
                logger.debug(f"Unknown focus_skill in recovery state: {data.get('focus_skill')}")
        
        return cls(
            triggered=data.get("triggered", False),
            focus_skill=focus_skill,
            reward_modifications=data.get("reward_modifications", {}),
            constraint_modifications=data.get("constraint_modifications", {}),
            episodes_remaining=data.get("episodes_remaining", 0),
            trigger_reason=data.get("trigger_reason", ""),
        )


@dataclass
class ReviewSessionState:
    """State for review session management."""
    episodes_since_review: int = 0
    in_review: bool = False
    review_stage: Optional[CurriculumStage] = None
    review_episodes_remaining: int = 0
    home_stage: Optional[CurriculumStage] = None
    return_to_home_latch: bool = False  # One-episode latch after review ends
    
    def is_active(self) -> bool:
        return self.in_review and self.review_episodes_remaining > 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "episodes_since_review": self.episodes_since_review,
            "in_review": self.in_review,
            "review_stage": self.review_stage.name if self.review_stage else None,
            "review_episodes_remaining": self.review_episodes_remaining,
            "home_stage": self.home_stage.name if self.home_stage else None,
            "return_to_home_latch": self.return_to_home_latch,
            "is_active": self.is_active(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReviewSessionState":
        """Restore state from dictionary."""
        review_stage = None
        if data.get("review_stage"):
            try:
                review_stage = CurriculumStage[data["review_stage"]]
            except KeyError:
                logger.debug(f"Unknown review_stage: {data.get('review_stage')}")
        
        home_stage = None
        if data.get("home_stage"):
            try:
                home_stage = CurriculumStage[data["home_stage"]]
            except KeyError:
                logger.debug(f"Unknown home_stage: {data.get('home_stage')}")
        
        return cls(
            episodes_since_review=data.get("episodes_since_review", 0),
            in_review=data.get("in_review", False),
            review_stage=review_stage,
            review_episodes_remaining=data.get("review_episodes_remaining", 0),
            home_stage=home_stage,
            return_to_home_latch=bool(data.get("return_to_home_latch", False)),
        )


# Re-export for backward compatibility
__all__ = [
    "RecoveryProtocolState",
    "ReviewSessionState",
]
