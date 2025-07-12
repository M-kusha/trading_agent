# envs/config.py
"""
Enhanced Configuration System for InfoBus-Integrated Trading Environment
Includes presets, factory methods, and full InfoBus compatibility
"""
import os
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path


@dataclass
class TradingConfig:
    """Centralized configuration for InfoBus-integrated trading environment"""
    
    # ═══════════════════════════════════════════════════════════════════
    # Core Environment Parameters
    # ═══════════════════════════════════════════════════════════════════
    initial_balance: float = 3000.0
    max_steps: int = 200
    debug: bool = True
    init_seed: int = 42
    max_steps_per_episode: int = field(init=False)
    
    # InfoBus Configuration
    info_bus_enabled: bool = True
    info_bus_audit_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    info_bus_validation: bool = True
    
    # ═══════════════════════════════════════════════════════════════════
    # Data and Instruments
    # ═══════════════════════════════════════════════════════════════════
    data_dir: str = "data/processed"
    instruments: List[str] = field(default_factory=lambda: ["EUR/USD", "XAU/USD"])
    timeframes: List[str] = field(default_factory=lambda: ["H1", "H4", "D1"])
    
    # ═══════════════════════════════════════════════════════════════════
    # Trading Parameters  
    # ═══════════════════════════════════════════════════════════════════
    no_trade_penalty: float = 0.3
    consensus_min: float = 0.30
    consensus_max: float = 0.70
    max_episodes: int = 10000
    
    # ═══════════════════════════════════════════════════════════════════
    # Risk Management
    # ═══════════════════════════════════════════════════════════════════
    min_intensity: float = 0.25
    min_inst_confidence: float = 0.60
    rotation_gap: int = 5
    max_position_pct: float = 0.10
    max_total_exposure: float = 0.30
    max_drawdown: float = 0.20
    max_correlation: float = 0.8
    
    # Enhanced Risk Parameters for InfoBus
    risk_check_frequency: int = 1  # Steps between risk checks
    risk_alert_cooldown: int = 5   # Steps between similar alerts
    max_concurrent_alerts: int = 10
    
    # ═══════════════════════════════════════════════════════════════════
    # Training Environment
    # ═══════════════════════════════════════════════════════════════════
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
    
    # ═══════════════════════════════════════════════════════════════════
    # PPO Hyperparameters
    # ═══════════════════════════════════════════════════════════════════
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
    
    # ═══════════════════════════════════════════════════════════════════
    # Network Architecture
    # ═══════════════════════════════════════════════════════════════════
    policy_hidden_size: int = 256
    value_hidden_size: int = 256
    
    # ═══════════════════════════════════════════════════════════════════
    # Training Schedule
    # ═══════════════════════════════════════════════════════════════════
    final_training_steps: int = 100000
    log_interval: int = 10
    checkpoint_freq: int = 10000
    eval_freq: int = 5000
    n_eval_episodes: int = 5
    
    # ═══════════════════════════════════════════════════════════════════
    # Enhanced Directory Structure with Rotation
    # ═══════════════════════════════════════════════════════════════════
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
        """Post-initialization setup with InfoBus support"""
        # Create the alias after dataclass finishes init
        object.__setattr__(self, "max_steps_per_episode", self.max_steps)
        
        # Ensure all directories exist
        all_dirs = [
            self.log_dir, self.checkpoint_dir, self.model_dir, 
            self.tensorboard_dir, self.data_dir, self.info_bus_log_dir,
            self.audit_log_dir, self.operator_log_dir
        ]
        
        for directory in all_dirs:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
        # Create module-specific log directories
        module_log_dirs = [
            "logs/trading", "logs/risk", "logs/strategy", "logs/memory",
            "logs/voting", "logs/market", "logs/position", "logs/features"
        ]
        
        for directory in module_log_dirs:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get PPO model configuration dictionary"""
        return {
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "clip_range_vf": self.clip_range_vf,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "target_kl": self.target_kl,
            "policy_hidden_size": self.policy_hidden_size,
            "value_hidden_size": self.value_hidden_size,
        }
    
    def get_info_bus_config(self) -> Dict[str, Any]:
        """Get InfoBus configuration dictionary"""
        return {
            "enabled": self.info_bus_enabled,
            "audit_level": self.info_bus_audit_level,
            "validation": self.info_bus_validation,
            "log_dir": self.info_bus_log_dir,
            "rotation_lines": self.log_rotation_lines,
        }
    
    def get_module_config(self) -> Dict[str, Any]:
        """Get module configuration dictionary"""
        return {
            "debug": self.debug,
            "max_history": 100,
            "audit_enabled": True,
            "log_rotation_lines": self.log_rotation_lines,
            "health_check_interval": 100,
            "info_bus_enabled": self.info_bus_enabled,
        }
    
    def save_config(self, path: str):
        """Save configuration to JSON file"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, (str, int, float, bool, list)):
                config_dict[key] = value
            elif value is None:
                config_dict[key] = None
            else:
                config_dict[key] = str(value)
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_config(cls, path: str) -> 'TradingConfig':
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
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
        object.__setattr__(self, "session_start_balance", self.balance)


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
        **overrides
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
            config.info_bus_audit_level = info_bus_level.upper()
        
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
        
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                print(f"Warning: Unknown config parameter '{key}'")
        
        return config


# ═══════════════════════════════════════════════════════════════════
# Configuration Validation with InfoBus Checks
# ═══════════════════════════════════════════════════════════════════

def validate_config(config: TradingConfig) -> List[str]:
    """Enhanced validation with InfoBus compatibility checks"""
    warnings = []
    
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


# ═══════════════════════════════════════════════════════════════════
# Example Usage
# ═══════════════════════════════════════════════════════════════════

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
        
        warnings = validate_config(config)
        if warnings:
            print(f"  ⚠️ Warnings: {len(warnings)}")
            for warning in warnings[:2]:
                print(f"    - {warning}")
    
    print("\n✅ Enhanced configuration system test completed!")