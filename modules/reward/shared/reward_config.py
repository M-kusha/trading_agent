# modules/reward/shared/reward_config.py
"""
Shared Configuration for Reward System
Centralized configuration management with validation and normalization

MODE-AWARE: Penalty weights automatically switch between LIVE (conservative)
and TRAINING (exploratory) modes. Call set_reward_mode() at startup.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class RewardMode(Enum):
    """Reward system operating modes"""
    TRAINING = "training"
    VALIDATION = "validation"
    LIVE_TRADING = "live_trading"
    EMERGENCY = "emergency"
    OPTIMIZATION = "optimization"


def _load_reward_config_from_yaml() -> Dict[str, Any]:
    """Load reward config values from risk_policy.yaml."""
    import os

    import yaml
    defaults = {}
    try:
        config_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "config", "risk_policy.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                policy = yaml.safe_load(f) or {}
            
            prop_firm = policy.get("prop_firm", {})
            limits = policy.get("limits", {})
            
            # Prop firm limits
            defaults["daily_dd_limit"] = float(prop_firm.get("daily_drawdown_limit", 0.05))
            defaults["max_dd_limit"] = float(prop_firm.get("max_drawdown_limit", 0.10))
            defaults["profit_target"] = float(prop_firm.get("profit_target", 0.10))
            defaults["prop_firm_enabled"] = bool(prop_firm.get("enabled", True))
    except Exception:
        pass  # Fall back to dataclass defaults
    return defaults


# ═══════════════════════════════════════════════════════════════════
# MODE-AWARE REWARD PARAMETERS
# ═══════════════════════════════════════════════════════════════════

# Global mode flag
_REWARD_MODE: str = "TRAINING"  # "LIVE" or "TRAINING"

# ───────────────────────────────────────────────────────────────────
# LIVE MODE PARAMETERS (Conservative - QUALITY TRADES ONLY)
# ───────────────────────────────────────────────────────────────────
_LIVE_REWARD_PARAMS = {
    # Penalty weights (strong penalties for risky behavior)
    "dd_pen_weight": 3.0,           # Strong drawdown penalty
    "risk_pen_weight": 0.25,        # Strong risk penalty
    "tail_pen_weight": 0.8,         # Strong tail risk penalty
    "mistake_pen_weight": 0.5,      # Strong mistake penalty
    "no_trade_penalty_weight": 0.0,   # NO penalty for patience
    
    # TRADE COST: Higher in live (real spread/commission)
    "trade_open_cost": 0.20,        # Cost per new trade
    
    # Prop firm penalties (critical for funded accounts)
    "prop_firm_dd_penalty_weight": 5.0,    # Heavy penalty near DD limits
    "prop_firm_violation_penalty": 10.0,   # Severe penalty for rule breach
    "profit_target_bonus_weight": 2.0,     # Bonus for reaching targets
    
    # Bonus weights (conservative - reward quality)
    "win_bonus_weight": 1.2,        # Reward winners
    "trade_frequency_bonus": 0.0,   # NO frequency bonus
    
    # Quality bonuses
    "win_rate_bonus_weight": 0.8,   # Higher win rate bonus in live
    "patience_bonus_weight": 0.05,  # Small patience bonus
}

# ───────────────────────────────────────────────────────────────────
# TRAINING MODE PARAMETERS (QUALITY OVER QUANTITY)
# Teach the agent to be SELECTIVE - only trade on strong signals
# ───────────────────────────────────────────────────────────────────
_TRAINING_REWARD_PARAMS = {
    # Penalty weights (moderate for exploration)
    "dd_pen_weight": 2.0,           # Moderate drawdown penalty
    "risk_pen_weight": 0.1,         # Moderate risk penalty
    "tail_pen_weight": 0.5,         # Moderate tail risk penalty
    "mistake_pen_weight": 0.3,      # Moderate mistake penalty
    "no_trade_penalty_weight": 0.0,   # NO penalty for not trading (patience is OK)
    
    # TRADE COST: Penalize opening new positions (teaches selectivity)
    # Agent must overcome this cost with profits to net positive reward
    "trade_open_cost": 0.15,        # Cost per new trade opened (like spread/commission)
    
    # Prop firm penalties (learn to respect limits)
    "prop_firm_dd_penalty_weight": 3.0,    # Progressive penalty near DD limits
    "prop_firm_violation_penalty": 5.0,    # Learn to avoid rule breaches
    "profit_target_bonus_weight": 1.5,     # Incentive for profit targets
    
    # Bonus weights (reward quality not quantity)
    "win_bonus_weight": 1.5,        # HIGHER win bonus (reward winners more)
    "trade_frequency_bonus": 0.0,   # NO frequency bonus (don't reward churning)
    
    # NEW: Quality bonuses
    "win_rate_bonus_weight": 0.5,   # Bonus for maintaining high win rate
    "patience_bonus_weight": 0.1,   # Small bonus for holding during low-signal periods
}


def set_reward_mode(mode: str) -> None:
    """
    Set the global reward mode. Call this at startup based on config.
    
    Args:
        mode: "LIVE" for conservative real-money trading,
              "TRAINING" for exploratory learning mode
    """
    global _REWARD_MODE
    mode = mode.upper().strip()
    if mode not in ("LIVE", "TRAINING"):
        mode = "TRAINING"  # Default to exploratory mode
    _REWARD_MODE = mode


def get_reward_mode() -> str:
    """Get the current reward mode."""
    return _REWARD_MODE


def get_reward_params() -> Dict[str, float]:
    """Get the current reward parameters based on mode."""
    if _REWARD_MODE == "LIVE":
        return _LIVE_REWARD_PARAMS.copy()
    return _TRAINING_REWARD_PARAMS.copy()


@dataclass
class RewardConfig:
    """
    Configuration for Risk-Adjusted Reward System.
    
    MODE-AWARE: Default values are overridden based on global mode.
    Call set_reward_mode("LIVE") or set_reward_mode("TRAINING") at startup.
    """

    # Balance configuration
    initial_balance: Optional[float] = None

    # History and memory
    history_size: int = 50
    min_trade_bonus: float = 0.5

    # Regime weights
    regime_weights: List[float] = field(default_factory=lambda: [0.3, 0.4, 0.3])

    # Penalty weights (defaults - will be overridden by mode in __post_init__)
    dd_pen_weight: float = 2.0
    risk_pen_weight: float = 0.1
    tail_pen_weight: float = 0.5
    mistake_pen_weight: float = 0.3
    no_trade_penalty_weight: float = 0.0  # No penalty for patience

    # Trade cost (penalizes opening new positions - teaches selectivity)
    trade_open_cost: float = 0.15

    # Bonus weights
    win_bonus_weight: float = 1.5
    consistency_bonus_weight: float = 0.5
    sharpe_bonus_weight: float = 0.3
    trade_frequency_bonus: float = 0.0  # No bonus for frequent trading
    
    # Quality bonuses (reward selective trading)
    win_rate_bonus_weight: float = 0.5
    patience_bonus_weight: float = 0.1

    # Advanced parameters
    volatility_adjustment: float = 1.0
    regime_bonus_weight: float = 0.2
    momentum_bonus_weight: float = 0.1

    # Prop firm parameters (read from risk_policy.yaml via bus)
    prop_firm_enabled: bool = True
    prop_firm_dd_penalty_weight: float = 3.0     # Progressive penalty as DD approaches limits
    prop_firm_violation_penalty: float = 5.0     # Penalty for violating prop firm rules
    profit_target_bonus_weight: float = 1.5      # Bonus for progress toward profit target
    daily_dd_limit: float = 0.05                 # 5% daily drawdown limit
    max_dd_limit: float = 0.10                   # 10% max drawdown limit
    profit_target: float = 0.10                  # 10% profit target for challenge

    # Performance thresholds
    max_processing_time_ms: float = 100.0
    circuit_breaker_threshold: int = 5
    min_reward_quality: float = 0.3

    # Adaptation parameters
    confidence_decay: float = 0.98
    performance_smoothing: float = 0.95
    adaptive_learning_rate: float = 0.01
    
    # Mode override flag (set to False to use static values)
    use_mode_aware_defaults: bool = True

    # ─────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        # Load prop firm values from risk_policy.yaml
        self._load_from_risk_policy()
        # Apply mode-aware defaults if enabled
        if self.use_mode_aware_defaults:
            self._apply_mode_defaults()
        self._validate_and_normalize()
    
    def _load_from_risk_policy(self) -> None:
        """Load prop firm parameters from risk_policy.yaml."""
        yaml_config = _load_reward_config_from_yaml()
        for key, value in yaml_config.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def _apply_mode_defaults(self) -> None:
        """Apply mode-aware default values for penalty/bonus weights."""
        params = get_reward_params()
        
        # Only override if using default values (not explicitly set)
        # This allows explicit overrides to take precedence
        self.dd_pen_weight = params.get("dd_pen_weight", self.dd_pen_weight)
        self.risk_pen_weight = params.get("risk_pen_weight", self.risk_pen_weight)
        self.tail_pen_weight = params.get("tail_pen_weight", self.tail_pen_weight)
        self.mistake_pen_weight = params.get("mistake_pen_weight", self.mistake_pen_weight)
        self.no_trade_penalty_weight = params.get("no_trade_penalty_weight", self.no_trade_penalty_weight)
        self.win_bonus_weight = params.get("win_bonus_weight", self.win_bonus_weight)
        self.trade_frequency_bonus = params.get("trade_frequency_bonus", self.trade_frequency_bonus)
        
        # Trade selectivity parameters (quality over quantity)
        self.trade_open_cost = params.get("trade_open_cost", self.trade_open_cost)
        self.win_rate_bonus_weight = params.get("win_rate_bonus_weight", getattr(self, 'win_rate_bonus_weight', 0.5))
        self.patience_bonus_weight = params.get("patience_bonus_weight", getattr(self, 'patience_bonus_weight', 0.1))
        
        # Prop firm parameters
        self.prop_firm_dd_penalty_weight = params.get("prop_firm_dd_penalty_weight", self.prop_firm_dd_penalty_weight)
        self.prop_firm_violation_penalty = params.get("prop_firm_violation_penalty", self.prop_firm_violation_penalty)
        self.profit_target_bonus_weight = params.get("profit_target_bonus_weight", self.profit_target_bonus_weight)

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
            # Prop firm weights
            'prop_firm_dd_penalty_weight': self.prop_firm_dd_penalty_weight,
            'prop_firm_violation_penalty': self.prop_firm_violation_penalty,
            'profit_target_bonus_weight': self.profit_target_bonus_weight,
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
