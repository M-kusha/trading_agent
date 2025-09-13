# modules/reward/shared/reward_config.py
"""
Shared Configuration for Reward System
Centralized configuration management with validation and normalization
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class RewardMode(Enum):
    """Reward system operating modes"""
    TRAINING = "training"
    VALIDATION = "validation"
    LIVE_TRADING = "live_trading"
    EMERGENCY = "emergency"
    OPTIMIZATION = "optimization"


@dataclass
class RewardConfig:
    """Configuration for Risk-Adjusted Reward System"""

    # Balance configuration
    initial_balance: Optional[float] = None

    # History and memory
    history_size: int = 50
    min_trade_bonus: float = 0.5

    # Regime weights
    regime_weights: List[float] = field(default_factory=lambda: [0.3, 0.4, 0.3])

    # Penalty weights
    dd_pen_weight: float = 2.0
    risk_pen_weight: float = 0.1
    tail_pen_weight: float = 0.5
    mistake_pen_weight: float = 0.3
    no_trade_penalty_weight: float = 0.05

    # Bonus weights
    win_bonus_weight: float = 1.0
    consistency_bonus_weight: float = 0.5
    sharpe_bonus_weight: float = 0.3
    trade_frequency_bonus: float = 0.2

    # Advanced parameters
    volatility_adjustment: float = 1.0
    regime_bonus_weight: float = 0.2
    momentum_bonus_weight: float = 0.1

    # Performance thresholds
    max_processing_time_ms: float = 100.0
    circuit_breaker_threshold: int = 5
    min_reward_quality: float = 0.3

    # Adaptation parameters
    confidence_decay: float = 0.98
    performance_smoothing: float = 0.95
    adaptive_learning_rate: float = 0.01

    # ─────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        self._validate_and_normalize()

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────
    def get_weights(self) -> Dict[str, Any]:
        """Get all weight parameters"""
        return {
            'regime_weights': self.regime_weights,
            'dd_pen_weight': self.dd_pen_weight,
            'risk_pen_weight': self.risk_pen_weight,
            'tail_pen_weight': self.tail_pen_weight,
            'mistake_pen_weight': self.mistake_pen_weight,
            'no_trade_penalty_weight': self.no_trade_penalty_weight,
            'win_bonus_weight': self.win_bonus_weight,
            'consistency_bonus_weight': self.consistency_bonus_weight,
            'sharpe_bonus_weight': self.sharpe_bonus_weight,
            'trade_frequency_bonus': self.trade_frequency_bonus,
            'volatility_adjustment': self.volatility_adjustment,
            'regime_bonus_weight': self.regime_bonus_weight,
            'momentum_bonus_weight': self.momentum_bonus_weight,
        }

    def apply_genome(self, genome: Dict[str, Any]) -> None:
        """Apply genome parameters to configuration with coercion & revalidation"""
        for key, value in genome.items():
            if not hasattr(self, key):
                continue
            try:
                # Coerce types for known numeric fields
                if key in {
                    'initial_balance', 'min_trade_bonus', 'dd_pen_weight', 'risk_pen_weight',
                    'tail_pen_weight', 'mistake_pen_weight', 'no_trade_penalty_weight',
                    'win_bonus_weight', 'consistency_bonus_weight', 'sharpe_bonus_weight',
                    'trade_frequency_bonus', 'volatility_adjustment', 'regime_bonus_weight',
                    'momentum_bonus_weight', 'max_processing_time_ms', 'min_reward_quality',
                    'confidence_decay', 'performance_smoothing', 'adaptive_learning_rate'
                }:
                    value = float(value)
                elif key in {'history_size', 'circuit_breaker_threshold'}:
                    value = int(value)
                elif key == 'regime_weights':
                    value = [float(v) for v in value] if isinstance(value, (list, tuple)) else self.regime_weights
            except Exception:
                # If coercion fails, skip the assignment
                continue
            setattr(self, key, value)

        # Re-run validation/normalization after applying genome
        self._validate_and_normalize()

    # ─────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────
    def _validate_and_normalize(self) -> None:
        # Coerce numeric types and clamp ranges
        self.history_size = max(1, int(self.history_size))
        self.min_trade_bonus = max(0.0, float(self.min_trade_bonus))

        # Normalize regime_weights to non-negative and sum to 1.0
        rw = self.regime_weights or []
        rw = [float(max(0.0, v)) for v in rw]
        if not rw or sum(rw) <= 0.0:
            rw = [0.3, 0.4, 0.3]
        total = sum(rw)
        self.regime_weights = [v / total for v in rw]

        # Clamp all non-negative weights
        for name in [
            'dd_pen_weight', 'risk_pen_weight', 'tail_pen_weight', 'mistake_pen_weight',
            'no_trade_penalty_weight', 'win_bonus_weight', 'consistency_bonus_weight',
            'sharpe_bonus_weight', 'trade_frequency_bonus', 'volatility_adjustment',
            'regime_bonus_weight', 'momentum_bonus_weight'
        ]:
            setattr(self, name, max(0.0, float(getattr(self, name))))

        # Processing thresholds & bounds
        self.max_processing_time_ms = max(1.0, float(self.max_processing_time_ms))
        self.circuit_breaker_threshold = max(1, int(self.circuit_breaker_threshold))
        self.min_reward_quality = self._clamp01(self.min_reward_quality)
        self.confidence_decay = self._clamp01(self.confidence_decay)
        self.performance_smoothing = self._clamp01(self.performance_smoothing)
        self.adaptive_learning_rate = self._clamp01(self.adaptive_learning_rate)

        # initial_balance: accept None or non-negative float
        if self.initial_balance is not None:
            try:
                ib = float(self.initial_balance)
            except Exception:
                ib = None
            self.initial_balance = None if ib is None or ib < 0.0 else ib

    @staticmethod
    def _clamp01(v: float) -> float:
        try:
            vf = float(v)
        except Exception:
            return 0.0
        return 0.0 if vf < 0.0 else (1.0 if vf > 1.0 else vf)
