

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class RewardMode(Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    LIVE_TRADING = "live_trading"
    EMERGENCY = "emergency"
    OPTIMIZATION = "optimization"


def _load_reward_config_from_yaml() -> Dict[str, Any]:
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


            defaults["daily_dd_limit"] = float(prop_firm.get("daily_drawdown_limit", 0.05))
            defaults["max_dd_limit"] = float(prop_firm.get("max_drawdown_limit", 0.10))
            defaults["profit_target"] = float(prop_firm.get("profit_target", 0.10))
            defaults["prop_firm_enabled"] = bool(prop_firm.get("enabled", True))
    except Exception:
        pass
    return defaults


_REWARD_MODE: str = "TRAINING"


_LIVE_REWARD_PARAMS = {

    "dd_pen_weight": 3.0,
    "risk_pen_weight": 0.25,
    "tail_pen_weight": 0.8,
    "mistake_pen_weight": 0.5,
    "no_trade_penalty_weight": 0.0,


    "trade_open_cost": 0.20,


    "prop_firm_dd_penalty_weight": 5.0,
    "prop_firm_violation_penalty": 10.0,
    "profit_target_bonus_weight": 2.0,


    "win_bonus_weight": 1.2,
    "trade_frequency_bonus": 0.0,


    "win_rate_bonus_weight": 0.8,
    "patience_bonus_weight": 0.05,
}


_TRAINING_REWARD_PARAMS = {

    "dd_pen_weight": 2.0,
    "risk_pen_weight": 0.1,
    "tail_pen_weight": 0.5,
    "mistake_pen_weight": 0.3,
    "no_trade_penalty_weight": 0.0,


    "trade_open_cost": 0.15,


    "prop_firm_dd_penalty_weight": 3.0,
    "prop_firm_violation_penalty": 5.0,
    "profit_target_bonus_weight": 1.5,


    "win_bonus_weight": 1.5,
    "trade_frequency_bonus": 0.0,


    "win_rate_bonus_weight": 0.5,
    "patience_bonus_weight": 0.1,
}


def set_reward_mode(mode: str) -> None:
    global _REWARD_MODE
    mode = mode.upper().strip()
    if mode not in ("LIVE", "TRAINING"):
        mode = "TRAINING"
    _REWARD_MODE = mode


def get_reward_mode() -> str:
    return _REWARD_MODE


def get_reward_params() -> Dict[str, float]:
    if _REWARD_MODE == "LIVE":
        return _LIVE_REWARD_PARAMS.copy()
    return _TRAINING_REWARD_PARAMS.copy()


@dataclass
class RewardConfig:


    initial_balance: Optional[float] = None


    history_size: int = 50
    min_trade_bonus: float = 0.5


    regime_weights: List[float] = field(default_factory=lambda: [0.3, 0.4, 0.3])


    dd_pen_weight: float = 2.0
    risk_pen_weight: float = 0.1
    tail_pen_weight: float = 0.5
    mistake_pen_weight: float = 0.3
    no_trade_penalty_weight: float = 0.0


    trade_open_cost: float = 0.15


    win_bonus_weight: float = 1.5
    consistency_bonus_weight: float = 0.5
    sharpe_bonus_weight: float = 0.3
    trade_frequency_bonus: float = 0.0


    win_rate_bonus_weight: float = 0.5
    patience_bonus_weight: float = 0.1


    volatility_adjustment: float = 1.0
    regime_bonus_weight: float = 0.2
    momentum_bonus_weight: float = 0.1


    prop_firm_enabled: bool = True
    prop_firm_dd_penalty_weight: float = 3.0
    prop_firm_violation_penalty: float = 5.0
    profit_target_bonus_weight: float = 1.5
    daily_dd_limit: float = 0.05
    max_dd_limit: float = 0.10
    profit_target: float = 0.10


    max_processing_time_ms: float = 100.0
    circuit_breaker_threshold: int = 5
    min_reward_quality: float = 0.3


    confidence_decay: float = 0.98
    performance_smoothing: float = 0.95
    adaptive_learning_rate: float = 0.01


    use_mode_aware_defaults: bool = True


    def __post_init__(self) -> None:

        self._load_from_risk_policy()

        if self.use_mode_aware_defaults:
            self._apply_mode_defaults()
        self._validate_and_normalize()

    def _load_from_risk_policy(self) -> None:
        yaml_config = _load_reward_config_from_yaml()
        for key, value in yaml_config.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def _apply_mode_defaults(self) -> None:
        params = get_reward_params()


        self.dd_pen_weight = params.get("dd_pen_weight", self.dd_pen_weight)
        self.risk_pen_weight = params.get("risk_pen_weight", self.risk_pen_weight)
        self.tail_pen_weight = params.get("tail_pen_weight", self.tail_pen_weight)
        self.mistake_pen_weight = params.get("mistake_pen_weight", self.mistake_pen_weight)
        self.no_trade_penalty_weight = params.get("no_trade_penalty_weight", self.no_trade_penalty_weight)
        self.win_bonus_weight = params.get("win_bonus_weight", self.win_bonus_weight)
        self.trade_frequency_bonus = params.get("trade_frequency_bonus", self.trade_frequency_bonus)


        self.trade_open_cost = params.get("trade_open_cost", self.trade_open_cost)
        self.win_rate_bonus_weight = params.get("win_rate_bonus_weight", getattr(self, 'win_rate_bonus_weight', 0.5))
        self.patience_bonus_weight = params.get("patience_bonus_weight", getattr(self, 'patience_bonus_weight', 0.1))


        self.prop_firm_dd_penalty_weight = params.get("prop_firm_dd_penalty_weight", self.prop_firm_dd_penalty_weight)
        self.prop_firm_violation_penalty = params.get("prop_firm_violation_penalty", self.prop_firm_violation_penalty)
        self.profit_target_bonus_weight = params.get("profit_target_bonus_weight", self.profit_target_bonus_weight)


    def get_weights(self) -> Dict[str, Any]:
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

            'prop_firm_dd_penalty_weight': self.prop_firm_dd_penalty_weight,
            'prop_firm_violation_penalty': self.prop_firm_violation_penalty,
            'profit_target_bonus_weight': self.profit_target_bonus_weight,
        }

    def apply_genome(self, genome: Dict[str, Any]) -> None:
        for key, value in genome.items():
            if not hasattr(self, key):
                continue
            try:

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

                continue
            setattr(self, key, value)


        self._validate_and_normalize()


    def _validate_and_normalize(self) -> None:

        self.history_size = max(1, int(self.history_size))
        self.min_trade_bonus = max(0.0, float(self.min_trade_bonus))


        rw = self.regime_weights or []
        rw = [float(max(0.0, v)) for v in rw]
        if not rw or sum(rw) <= 0.0:
            rw = [0.3, 0.4, 0.3]
        total = sum(rw)
        self.regime_weights = [v / total for v in rw]


        for name in [
            'dd_pen_weight', 'risk_pen_weight', 'tail_pen_weight', 'mistake_pen_weight',
            'no_trade_penalty_weight', 'win_bonus_weight', 'consistency_bonus_weight',
            'sharpe_bonus_weight', 'trade_frequency_bonus', 'volatility_adjustment',
            'regime_bonus_weight', 'momentum_bonus_weight'
        ]:
            setattr(self, name, max(0.0, float(getattr(self, name))))


        self.max_processing_time_ms = max(1.0, float(self.max_processing_time_ms))
        self.circuit_breaker_threshold = max(1, int(self.circuit_breaker_threshold))
        self.min_reward_quality = self._clamp01(self.min_reward_quality)
        self.confidence_decay = self._clamp01(self.confidence_decay)
        self.performance_smoothing = self._clamp01(self.performance_smoothing)
        self.adaptive_learning_rate = self._clamp01(self.adaptive_learning_rate)


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
