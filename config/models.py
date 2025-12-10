from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModeConfig:
    """Mode toggles shared by training and live pipelines."""

    name: str = "training"
    live: bool = False
    training: bool = True
    test_mode: bool = False
    enable_shadow_sim: bool = True


@dataclass
class LoggingConfig:
    """Central logging configuration with a global debug switch."""

    debug: bool = False
    level: str = "INFO"
    log_dir: str = "logs"
    filename: str = "trading_agent.log"
    max_bytes: int = 10_485_760  # 10 MB
    backup_count: int = 5

    @property
    def effective_level(self) -> str:
        return "DEBUG" if self.debug else str(self.level).upper()


@dataclass
class PathsConfig:
    """Filesystem layout."""

    logs: str = "logs"
    checkpoints: str = "checkpoints"
    models: str = "models"
    tensorboard: str = "logs/tensorboard"
    data: str = "data/processed"


@dataclass
class EnvironmentConfig:
    """Environment defaults shared between training and live runs."""

    instruments: List[str] = field(default_factory=lambda: ["EUR_USD", "XAU_USD"])
    timeframes: List[str] = field(default_factory=lambda: ["M15", "H1", "H4", "D1"])
    data_source: str = "local"
    initial_balance: float = 100_000.0
    max_steps: int = 200
    num_envs: int = 8
    bus_first: bool = True
    primary_timeframe: str = "M15"
    update_interval: int = 5
    min_trade_interval: int = 60
    use_trailing_stop: bool = True
    enable_shadow_sim: bool = True


@dataclass
class RLConfig:
    """RL hyperparameters and training loop controls."""

    learning_rate: float = 1e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip_range: float = 0.15
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.015

    final_training_steps: int = 100_000
    checkpoint_freq: int = 10_000
    eval_freq: int = 5_000
    log_interval: int = 10
    n_eval_episodes: int = 5


@dataclass
class MT5Config:
    """MT5 connection parameters."""

    account: Optional[int] = None
    password: Optional[str] = None
    server: str = "MetaQuotes-Demo"
    path: Optional[str] = None


@dataclass
class RiskConfig:
    """Risk policy wrapper to keep YAML typed while allowing passthrough."""

    policy: Dict[str, Any] = field(default_factory=dict)
    overrides: Dict[str, Any] = field(default_factory=dict)

    def to_trading_overrides(self) -> Dict[str, Any]:
        """Map known risk overrides into TradingConfig fields."""
        out: Dict[str, Any] = {}
        for key in ["max_total_exposure", "max_position_pct", "max_drawdown", "emergency_drawdown_trigger"]:
            if key in self.overrides and self.overrides[key] is not None:
                out[key] = self.overrides[key]
        return out


@dataclass
class TradingAgentConfig:
    """Complete config object consumed by both training and live flows."""

    mode: ModeConfig = field(default_factory=ModeConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    mt5: MT5Config = field(default_factory=MT5Config)
    risk: RiskConfig = field(default_factory=RiskConfig)

    def to_trading_config(self):
        """Convert to the existing TradingConfig used by envs.*."""
        from envs.config import TradingConfig  # Lazy import to avoid cycles

        overrides = {
            # Modes and flags
            "debug": self.logging.debug,
            "log_level": self.logging.effective_level,
            "live_mode": self.mode.live,
            "training_mode": self.mode.training,
            "test_mode": self.mode.test_mode,
            "enable_shadow_sim": self.mode.enable_shadow_sim,
            # Paths
            "log_dir": self.paths.logs,
            "checkpoint_dir": self.paths.checkpoints,
            "model_dir": self.paths.models,
            "tensorboard_dir": self.paths.tensorboard,
            "data_dir": self.paths.data,
            # Env
            "initial_balance": self.environment.initial_balance,
            "max_steps": self.environment.max_steps,
            "num_envs": self.environment.num_envs,
            "instruments": self.environment.instruments,
            "timeframes": self.environment.timeframes,
            "bus_first": self.environment.bus_first,
            "primary_timeframe": self.environment.primary_timeframe,
            # RL hyperparameters
            "learning_rate": self.rl.learning_rate,
            "n_steps": self.rl.n_steps,
            "batch_size": self.rl.batch_size,
            "n_epochs": self.rl.n_epochs,
            "gamma": self.rl.gamma,
            "gae_lambda": self.rl.gae_lambda,
            "clip_range": self.rl.clip_range,
            "ent_coef": self.rl.ent_coef,
            "vf_coef": self.rl.vf_coef,
            "max_grad_norm": self.rl.max_grad_norm,
            "target_kl": self.rl.target_kl,
            # Training loop
            "final_training_steps": self.rl.final_training_steps,
            "checkpoint_freq": self.rl.checkpoint_freq,
            "eval_freq": self.rl.eval_freq,
            "log_interval": self.rl.log_interval,
            "n_eval_episodes": self.rl.n_eval_episodes,
        }
        overrides.update(self.risk.to_trading_overrides())
        overrides["risk_policy_payload"] = self.risk.policy
        overrides["risk_overrides_payload"] = self.risk.overrides

        return TradingConfig(**overrides)
