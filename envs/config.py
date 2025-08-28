# envs/config.py
"""
Enhanced Configuration System for InfoBus-Integrated Trading Environment
Includes presets, factory methods, and full InfoBus compatibility
(Hardened & type-safe; drop-in compatible)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, replace
from typing import Any, Dict, List, Optional, Iterable, Tuple
from pathlib import Path


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _ensure_dir(p: str) -> None:
    if p:
        Path(p).mkdir(parents=True, exist_ok=True)


def _to_builtin(obj: Any) -> Any:
    """
    Recursively convert values to JSON-serializable builtins.
    Keeps lists/dicts shape, converts Paths, sets, tuples, numpy scalars, etc.
    """
    try:
        # Fast path for simple builtins
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (list, tuple)):
            return [_to_builtin(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): _to_builtin(v) for k, v in obj.items()}
    except Exception:
        pass
    # Fallback to string
    return str(obj)


@dataclass
class TradingConfig:
    """Centralized configuration for InfoBus-integrated trading environment"""

    # ===================================================================
    # Core Environment Parameters
    # ===================================================================
    initial_balance: float = 3000.0
    max_steps: int = 200
    debug: bool = True
    init_seed: int = 42
    max_steps_per_episode: int = field(init=False)

    # InfoBus Configuration
    info_bus_enabled: bool = True
    info_bus_audit_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    info_bus_validation: bool = True
    # Initialization behavior
    info_bus_init_timeout: float = 2.0
    orchestrator_init_timeout: float = 10.0
    orchestrator_async_init: bool = True

    # ===================================================================
    # Data and Instruments
    # ===================================================================
    data_dir: str = "data/processed"
    instruments: List[str] = field(default_factory=lambda: ["EUR/USD", "XAU/USD"])
    timeframes: List[str] = field(default_factory=lambda: ["H1", "H4", "D1"])

    # ===================================================================
    # Trading Parameters
    # ===================================================================
    no_trade_penalty: float = 0.3
    consensus_min: float = 0.30
    consensus_max: float = 0.70
    max_episodes: int = 10000

    # ===================================================================
    # Risk Management
    # ===================================================================
    min_intensity: float = 0.25
    min_inst_confidence: float = 0.60
    rotation_gap: int = 5
    max_position_pct: float = 0.10
    max_total_exposure: float = 0.30
    max_drawdown: float = 0.20
    max_correlation: float = 0.8

    # Position Management Specific
    max_consecutive_losses: int = 5
    loss_reduction: float = 0.2
    max_instrument_concentration: float = 0.25
    min_volatility: float = 0.015
    hard_loss_eur: float = 30.0
    trail_pct: float = 0.10
    trail_abs_eur: float = 10.0
    pips_tolerance: int = 20
    min_size_pct: float = 0.01
    min_signal_threshold: float = 0.15
    position_scale_threshold: float = 0.30
    emergency_close_threshold: float = 0.85
    confidence_decay: float = 0.95

    # Performance thresholds for Position Manager
    position_max_processing_time_ms: float = 100
    position_circuit_breaker_threshold: int = 3

    # Enhanced Risk Parameters for InfoBus
    risk_check_frequency: int = 1  # Steps between risk checks
    risk_alert_cooldown: int = 5   # Steps between similar alerts
    max_concurrent_alerts: int = 10

    # ===================================================================
    # Training Environment
    # ===================================================================
    num_envs: int = 1
    test_mode: bool = False
    live_mode: bool = False
    enable_shadow_sim: bool = True
    enable_news_sentiment: bool = False

    # Module Enablement Flags
    enable_meta_rl: bool = True
    enable_memory_systems: bool = True
    enable_strategy_evolution: bool = True
    enable_risk_monitoring: bool = True
    enable_visualization: bool = True

    # ===================================================================
    # PPO Hyperparameters
    # ===================================================================
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.01

    # ===================================================================
    # Network Architecture
    # ===================================================================
    policy_hidden_size: int = 256
    value_hidden_size: int = 256

    # ===================================================================
    # Training Schedule
    # ===================================================================
    final_training_steps: int = 100000
    log_interval: int = 10
    checkpoint_freq: int = 10000
    eval_freq: int = 5000
    n_eval_episodes: int = 5

    # ===================================================================
    # Enhanced Directory Structure with Rotation
    # ===================================================================
    log_dir: str = "logs"
    log_level: str = "INFO"
    log_rotation_lines: int = 2000  # Mandatory 2000-line rotation
    checkpoint_dir: str = "checkpoints"
    model_dir: str = "models"
    tensorboard_dir: str = "logs/tensorboard"

    # InfoBus-specific directories
    info_bus_log_dir: str = "logs/info_bus"
    audit_log_dir: str = "logs/audit"
    operator_log_dir: str = "logs/operator"

    def __post_init__(self) -> None:
        """Post-initialization setup with InfoBus support + guard rails."""
        # Alias for backward compatibility
        object.__setattr__(self, "max_steps_per_episode", int(self.max_steps))

        # Sanitize lists
        self.instruments = [str(x) for x in (self.instruments or []) if str(x).strip()]
        self.timeframes = [str(x) for x in (self.timeframes or []) if str(x).strip()]

        # Clamp key probabilities/ratios into valid ranges
        self.max_drawdown = _clamp(float(self.max_drawdown), 0.0, 0.95)
        self.max_total_exposure = _clamp(float(self.max_total_exposure), 0.0, 5.0)
        self.max_position_pct = _clamp(float(self.max_position_pct), 0.0, 1.0)
        self.min_inst_confidence = _clamp(float(self.min_inst_confidence), 0.0, 1.0)
        self.consensus_min = _clamp(float(self.consensus_min), 0.0, 1.0)
        self.consensus_max = _clamp(float(self.consensus_max), 0.0, 1.0)
        if self.consensus_min > self.consensus_max:
            # keep invariant
            self.consensus_min, self.consensus_max = self.consensus_max, self.consensus_min

        self.emergency_close_threshold = _clamp(float(self.emergency_close_threshold), 0.0, 1.0)
        self.trail_pct = _clamp(float(self.trail_pct), 0.0, 1.0)
        self.confidence_decay = _clamp(float(self.confidence_decay), 0.0, 1.0)

        self.info_bus_init_timeout = max(0.0, float(self.info_bus_init_timeout))
        self.orchestrator_init_timeout = max(0.0, float(self.orchestrator_init_timeout))
        self.risk_check_frequency = max(1, int(self.risk_check_frequency))
        self.risk_alert_cooldown = max(0, int(self.risk_alert_cooldown))
        self.max_concurrent_alerts = max(1, int(self.max_concurrent_alerts))
        self.max_steps = max(1, int(self.max_steps))
        self.max_steps_per_episode = self.max_steps  # keep alias in sync

        # Ensure directories exist
        all_dirs = [
            self.log_dir, self.checkpoint_dir, self.model_dir,
            self.tensorboard_dir, self.data_dir, self.info_bus_log_dir,
            self.audit_log_dir, self.operator_log_dir
        ]
        for directory in all_dirs:
            _ensure_dir(directory)

        # Create module-specific log directories
        module_log_dirs = [
            "logs/trading", "logs/risk", "logs/strategy", "logs/memory",
            "logs/voting", "logs/market", "logs/position", "logs/features"
        ]
        for directory in module_log_dirs:
            _ensure_dir(directory)

    # ------------------------------------------------------------------
    # Structured views
    # ------------------------------------------------------------------
    def get_model_config(self) -> Dict[str, Any]:
        """Get PPO model configuration dictionary"""
        return {
            "learning_rate": float(self.learning_rate),
            "n_steps": int(self.n_steps),
            "batch_size": int(self.batch_size),
            "n_epochs": int(self.n_epochs),
            "gamma": float(self.gamma),
            "gae_lambda": float(self.gae_lambda),
            "clip_range": float(self.clip_range),
            "clip_range_vf": (None if self.clip_range_vf is None else float(self.clip_range_vf)),
            "ent_coef": float(self.ent_coef),
            "vf_coef": float(self.vf_coef),
            "max_grad_norm": float(self.max_grad_norm),
            "target_kl": (None if self.target_kl is None else float(self.target_kl)),
            "policy_hidden_size": int(self.policy_hidden_size),
            "value_hidden_size": int(self.value_hidden_size),
        }

    def get_info_bus_config(self) -> Dict[str, Any]:
        """Get InfoBus configuration dictionary"""
        return {
            "enabled": bool(self.info_bus_enabled),
            "audit_level": str(self.info_bus_audit_level),
            "validation": bool(self.info_bus_validation),
            "log_dir": str(self.info_bus_log_dir),
            "rotation_lines": int(self.log_rotation_lines),
        }

    def get_module_config(self) -> Dict[str, Any]:
        """Get module configuration dictionary"""
        return {
            "debug": bool(self.debug),
            "max_history": 100,
            "audit_enabled": True,
            "log_rotation_lines": int(self.log_rotation_lines),
            "health_check_interval": 100,
            "info_bus_enabled": bool(self.info_bus_enabled),
        }

    # ------------------------------------------------------------------
    # (De)serialization helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict of this config (converts complex types to builtins)."""
        return _to_builtin(asdict(self))

    def save_config(self, path: str) -> None:
        """Save configuration to JSON file"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_config(cls, path: str) -> "TradingConfig":
        """Load configuration from JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        # Filter unexpected keys gracefully
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        clean = {k: v for k, v in config_dict.items() if k in allowed}
        return cls(**clean)

    # ------------------------------------------------------------------
    # Ergonomics
    # ------------------------------------------------------------------
    def clone_with(self, **overrides: Any) -> "TradingConfig":
        """Return a copy with selected fields overridden."""
        # Only apply known fields
        known = {k: v for k, v in overrides.items() if hasattr(self, k)}
        return replace(self, **known)

    def apply_overrides(self, **overrides: Any) -> None:
        """In-place override for known fields (keeps compatibility)."""
        for k, v in overrides.items():
            if hasattr(self, k):
                setattr(self, k, v)
        # Re-run clamps/dirs if critical fields changed
        self.__post_init__()

    def __str__(self) -> str:
        """String representation for logging"""
        return (
            f"TradingConfig(\n"
            f"  Mode: {'LIVE' if self.live_mode else ('TEST' if self.test_mode else 'BACKTEST')}\n"
            f"  InfoBus: {'ENABLED' if self.info_bus_enabled else 'DISABLED'}\n"
            f"  Balance: ${self.initial_balance:,.2f}\n"
            f"  Max Steps: {self.max_steps}\n"
            f"  Instruments: {self.instruments}\n"
            f"  Training Steps: {self.final_training_steps:,}\n"
            f"  Learning Rate: {self.learning_rate}\n"
            f"  Risk Limits: DD={self.max_drawdown:.1%}, Exposure={self.max_total_exposure:.1%}\n"
            f"  Log Rotation: {self.log_rotation_lines} lines\n"
            f")"
        )


@dataclass
class MarketState:
    """Enhanced market state with InfoBus integration"""
    balance: float
    peak_balance: float
    current_step: int
    current_drawdown: float
    last_trade_step: Dict[str, int] = field(default_factory=dict)

    # Enhanced state tracking
    session_start_balance: float = field(init=False)
    session_trades: int = 0
    session_pnl: float = 0.0
    last_info_bus_update: int = 0

    def __post_init__(self):
        object.__setattr__(self, "session_start_balance", float(self.balance))


@dataclass
class EpisodeMetrics:
    """Enhanced episode metrics with InfoBus tracking"""
    pnls: List[float] = field(default_factory=list)
    durations: List[int] = field(default_factory=list)
    drawdowns: List[float] = field(default_factory=list)
    trades: List[Dict[str, Any]] = field(default_factory=list)
    votes_log: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_trace: List[str] = field(default_factory=list)

    # InfoBus-specific metrics
    info_bus_events: List[Dict[str, Any]] = field(default_factory=list)
    module_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    consensus_history: List[float] = field(default_factory=list)
    risk_alerts: List[Dict[str, Any]] = field(default_factory=list)


class ConfigPresets:
    """Enhanced preset configurations for InfoBus-integrated environment"""

    @staticmethod
    def conservative_live() -> TradingConfig:
        """Conservative configuration for live trading with InfoBus"""
        return TradingConfig(
            # Conservative risk settings
            initial_balance=1000.0,
            max_position_pct=0.05,
            max_total_exposure=0.15,
            max_drawdown=0.10,
            min_inst_confidence=0.75,
            consensus_min=0.50,

            # Live trading settings
            live_mode=True,
            debug=False,
            enable_shadow_sim=False,

            # InfoBus settings
            info_bus_enabled=True,
            info_bus_audit_level="WARNING",  # Only important events
            info_bus_validation=True,

            # Enhanced monitoring
            risk_check_frequency=1,
            risk_alert_cooldown=3,
            max_concurrent_alerts=5,

            # Conservative learning
            learning_rate=1e-4,
            ent_coef=0.001,
            n_steps=1024,

            # Shorter episodes for safety
            max_steps=100,
            final_training_steps=50000,

            # More frequent monitoring
            log_interval=5,
            checkpoint_freq=2500,
            eval_freq=1000,

            # Single instrument to start
            instruments=["EUR/USD"],
            timeframes=["H1", "H4", "D1"],
        )

    @staticmethod
    def research_mode() -> TradingConfig:
        """Research configuration with full InfoBus debugging"""
        return TradingConfig(
            # Research settings
            initial_balance=5000.0,
            max_position_pct=0.10,
            max_total_exposure=0.30,
            max_drawdown=0.25,

            # Full debugging mode
            live_mode=False,
            test_mode=True,
            debug=True,
            enable_shadow_sim=True,
            enable_news_sentiment=True,

            # InfoBus full debugging
            info_bus_enabled=True,
            info_bus_audit_level="DEBUG",  # All events
            info_bus_validation=True,

            # Detailed monitoring
            risk_check_frequency=1,
            risk_alert_cooldown=1,
            max_concurrent_alerts=20,

            # Fast iteration
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=32,

            # Short episodes for experiments
            max_steps=100,
            final_training_steps=25000,

            # Frequent logging
            log_interval=1,
            checkpoint_freq=5000,
            eval_freq=2500,

            # Full instrument set
            instruments=["EUR/USD", "XAU/USD"],
            timeframes=["H1", "H4", "D1"],
        )

    @staticmethod
    def production_backtest() -> TradingConfig:
        """Production backtesting with balanced InfoBus monitoring"""
        return TradingConfig(
            # Production settings
            initial_balance=10000.0,
            max_position_pct=0.15,
            max_total_exposure=0.40,
            max_drawdown=0.25,
            min_inst_confidence=0.50,
            consensus_min=0.30,

            # Backtest mode
            live_mode=False,
            test_mode=False,
            debug=False,
            enable_shadow_sim=True,

            # Balanced InfoBus monitoring
            info_bus_enabled=True,
            info_bus_audit_level="INFO",
            info_bus_validation=True,

            # Standard monitoring
            risk_check_frequency=1,
            risk_alert_cooldown=5,
            max_concurrent_alerts=10,

            # Production learning
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,

            # Full episodes
            max_steps=200,
            final_training_steps=100000,

            # Standard monitoring
            log_interval=10,
            checkpoint_freq=10000,
            eval_freq=5000,

            # Multiple instruments
            instruments=["EUR/USD", "XAU/USD"],
            timeframes=["H1", "H4", "D1"],
        )


class ConfigFactory:
    """Enhanced factory for creating InfoBus-compatible configurations"""

    @staticmethod
    def create_config(
        mode: str = "backtest",
        risk_level: str = "moderate",
        info_bus_level: str = "auto",
        **overrides: Any
    ) -> TradingConfig:
        """Create configuration with InfoBus integration"""

        # Base configurations by mode
        if mode == "live":
            config = ConfigPresets.conservative_live()
        elif mode == "research":
            config = ConfigPresets.research_mode()
        elif mode == "production":
            config = ConfigPresets.production_backtest()
        else:  # backtest
            config = TradingConfig()

        # Adjust InfoBus level
        if info_bus_level == "auto":
            if config.debug:
                config.info_bus_audit_level = "DEBUG"
            elif config.live_mode:
                config.info_bus_audit_level = "WARNING"
            else:
                config.info_bus_audit_level = "INFO"
        else:
            level = str(info_bus_level).upper()
            config.info_bus_audit_level = level

        # Adjust risk level
        if risk_level == "conservative":
            config.max_position_pct *= 0.5
            config.max_total_exposure *= 0.7
            config.max_drawdown *= 0.8
            config.min_inst_confidence = min(config.min_inst_confidence + 0.1, 0.9)
            config.risk_check_frequency = 1  # More frequent checks
            config.risk_alert_cooldown = 3   # Faster alerts
        elif risk_level == "aggressive":
            config.max_position_pct *= 1.5
            config.max_total_exposure *= 1.3
            config.max_drawdown *= 1.2
            config.min_inst_confidence = max(config.min_inst_confidence - 0.1, 0.3)
            config.risk_check_frequency = 2  # Less frequent checks
            config.risk_alert_cooldown = 10  # Slower alerts

        # Apply overrides (known fields only)
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                print(f"Warning: Unknown config parameter '{key}'")

        # Re-run clamps/invariants after overrides
        config.__post_init__()
        return config


# ===================================================================
# Configuration Validation with InfoBus Checks
# ===================================================================
def validate_config(config: TradingConfig) -> List[str]:
    """Enhanced validation with InfoBus compatibility checks"""
    warnings: List[str] = []

    # Standard risk validation
    if config.max_total_exposure > 1.0:
        warnings.append("⚠️ Total exposure > 100% is very risky")

    if config.max_drawdown > 0.5:
        warnings.append("⚠️ Max drawdown > 50% is extremely risky")

    if config.max_position_pct > 0.3:
        warnings.append("⚠️ Position size > 30% per trade is very risky")

    # InfoBus validation
    if config.info_bus_enabled:
        if config.info_bus_audit_level not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            warnings.append("⚠️ Invalid InfoBus audit level")

        if config.risk_check_frequency < 1:
            warnings.append("⚠️ Risk check frequency too low for InfoBus")

        if config.log_rotation_lines > 5000:
            warnings.append("⚠️ Log rotation lines > 5000 may impact performance")

    # Live mode validation
    if config.live_mode:
        if not config.info_bus_enabled:
            warnings.append("⚠️ InfoBus recommended for live trading")

        if config.debug and config.max_position_pct > 0.1:
            warnings.append("⚠️ Large positions in live debug mode")

    return warnings


# ===================================================================
# Example Usage
# ===================================================================
if __name__ == "__main__":
    print("🔧 Enhanced Trading Configuration System with InfoBus")
    print("=" * 70)

    # Test configurations
    configs = {
        "Conservative Live": ConfigPresets.conservative_live(),
        "Research Mode": ConfigPresets.research_mode(),
        "Production Backtest": ConfigPresets.production_backtest(),
    }

    for name, config in configs.items():
        print(f"\n📋 {name}:")
        print(f"  InfoBus: {config.info_bus_enabled} ({config.info_bus_audit_level})")
        print(f"  Risk Level: {config.max_drawdown:.1%} DD, {config.max_total_exposure:.1%} Exposure")
        print(f"  Log Rotation: {config.log_rotation_lines} lines")

        warns = validate_config(config)
        if warns:
            print(f"  ⚠️ Warnings: {len(warns)}")
            for warning in warns[:2]:
                print(f"    - {warning}")

    print("\n✅ Enhanced configuration system test completed!")
