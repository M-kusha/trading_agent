# envs/config.py
"""
Complete Configuration System for the Trading Environment
Includes presets, factory methods, and all required parameters
"""
import os
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path


@dataclass
class TradingConfig:
    """Centralized configuration for the trading environment"""
    
    # ═══════════════════════════════════════════════════════════════════
    # Core Environment Parameters
    # ═══════════════════════════════════════════════════════════════════
    initial_balance: float = 3000.0
    max_steps: int = 200
    debug: bool = True
    init_seed: int = 42
    max_steps_per_episode: int = field(init=False)
    
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
    
    # ═══════════════════════════════════════════════════════════════════
    # Training Environment
    # ═══════════════════════════════════════════════════════════════════
    num_envs: int = 1
    test_mode: bool = False
    live_mode: bool = False
    enable_shadow_sim: bool = True
    enable_news_sentiment: bool = False
    
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
    # Directory Structure
    # ═══════════════════════════════════════════════════════════════════
    log_dir: str = "logs"
    log_level: str = "INFO"
    checkpoint_dir: str = "checkpoints"
    model_dir: str = "models"
    tensorboard_dir: str = "logs/tensorboard"
    
    def __post_init__(self) -> None:
        """Post-initialization setup"""
        # Create the alias after dataclass finishes init
        object.__setattr__(self, "max_steps_per_episode", self.max_steps)
        
        # Ensure directories exist
        for directory in [self.log_dir, self.checkpoint_dir, self.model_dir, 
                         self.tensorboard_dir, self.data_dir]:
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
            f"  Balance: ${self.initial_balance:,.2f}\n"
            f"  Max Steps: {self.max_steps}\n"
            f"  Instruments: {self.instruments}\n"
            f"  Training Steps: {self.final_training_steps:,}\n"
            f"  Learning Rate: {self.learning_rate}\n"
            f"  Risk Limits: DD={self.max_drawdown:.1%}, Exposure={self.max_total_exposure:.1%}\n"
            f")"
        )


@dataclass
class MarketState:
    """Encapsulates current market state"""
    balance: float
    peak_balance: float
    current_step: int
    current_drawdown: float
    last_trade_step: Dict[str, int] = field(default_factory=dict)


@dataclass
class EpisodeMetrics:
    """Tracks episode-level metrics"""
    pnls: List[float] = field(default_factory=list)
    durations: List[int] = field(default_factory=list)
    drawdowns: List[float] = field(default_factory=list)
    trades: List[Dict[str, Any]] = field(default_factory=list)
    votes_log: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_trace: List[str] = field(default_factory=list)


class ConfigPresets:
    """Preset configurations for different trading scenarios"""
    
    @staticmethod
    def conservative_live() -> TradingConfig:
        """Conservative configuration for live trading"""
        return TradingConfig(
            # Conservative risk settings
            initial_balance=1000.0,
            max_position_pct=0.05,  # Very small positions
            max_total_exposure=0.15,  # Low total exposure
            max_drawdown=0.10,  # Strict drawdown limit
            min_inst_confidence=0.75,  # High confidence required
            consensus_min=0.50,  # Higher consensus required
            
            # Live trading settings
            live_mode=True,
            debug=False,
            enable_shadow_sim=False,  # No simulation in live mode
            
            # Conservative learning
            learning_rate=1e-4,  # Slower learning
            ent_coef=0.001,  # Less exploration
            n_steps=1024,  # Smaller batches
            
            # Shorter episodes for safety
            max_steps=100,
            final_training_steps=50000,
            
            # More frequent monitoring
            log_interval=5,
            checkpoint_freq=2500,
            eval_freq=1000,
            
            # Single instrument to start
            instruments=["EUR/USD"],
            timeframes=["H1", "H4", "D1"],  # Longer timeframes for stability
        )
    
    @staticmethod
    def aggressive_backtest() -> TradingConfig:
        """Aggressive configuration for backtesting"""
        return TradingConfig(
            # Aggressive risk settings
            initial_balance=10000.0,
            max_position_pct=0.20,  # Larger positions
            max_total_exposure=0.50,  # Higher exposure
            max_drawdown=0.30,  # Allow bigger drawdowns
            min_inst_confidence=0.40,  # Lower confidence threshold
            consensus_min=0.20,  # Lower consensus required
            
            # Backtest settings
            live_mode=False,
            test_mode=False,
            debug=False,
            enable_shadow_sim=True,
            
            # Aggressive learning
            learning_rate=5e-4,  # Faster learning
            ent_coef=0.02,  # More exploration
            n_steps=4096,  # Larger batches
            batch_size=128,
            
            # Longer episodes
            max_steps=500,
            final_training_steps=200000,
            
            # Standard monitoring
            log_interval=20,
            checkpoint_freq=20000,
            eval_freq=10000,
            
            # Multiple instruments
            instruments=["EUR/USD", "XAU/USD", "GBP/USD"],
            timeframes=["H1", "H4", "D1"],
        )
    
    @staticmethod
    def research_mode() -> TradingConfig:
        """Research configuration for experimentation"""
        return TradingConfig(
            # Research settings
            initial_balance=5000.0,
            max_position_pct=0.10,
            max_total_exposure=0.30,
            max_drawdown=0.25,
            
            # Research mode flags
            live_mode=False,
            test_mode=True,
            debug=True,  # Full debugging
            enable_shadow_sim=True,
            enable_news_sentiment=True,  # All features enabled
            
            # Fast iteration
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=32,  # Small batches for fast updates
            
            # Short episodes for quick experiments
            max_steps=100,
            final_training_steps=25000,
            
            # Frequent logging for analysis
            log_interval=5,
            checkpoint_freq=5000,
            eval_freq=2500,
            n_eval_episodes=3,
            
            # Full instrument set
            instruments=["EUR/USD", "XAU/USD"],
            timeframes=["H1", "H4", "D1"],
        )
    
    @staticmethod
    def demo_online() -> TradingConfig:
        """Configuration for online demo trading"""
        return TradingConfig(
            # Demo trading settings
            initial_balance=2000.0,
            max_position_pct=0.08,
            max_total_exposure=0.25,
            max_drawdown=0.15,
            min_inst_confidence=0.65,
            consensus_min=0.40,
            
            # Online demo mode
            live_mode=True,  # Use live execution
            test_mode=False,
            debug=True,  # Keep debugging for monitoring
            enable_shadow_sim=False,  # No simulation needed
            
            # Online learning parameters
            learning_rate=2e-4,  # Moderate learning rate
            ent_coef=0.015,  # Balanced exploration
            n_steps=512,  # Smaller steps for online updates
            batch_size=32,
            n_epochs=5,  # Fewer epochs for faster updates
            
            # Real-time constraints
            max_steps=50,  # Shorter episodes
            final_training_steps=10000,  # Continuous learning
            
            # Real-time monitoring
            log_interval=1,  # Log every step
            checkpoint_freq=1000,
            eval_freq=500,
            
            # Focus on major pairs
            instruments=["EUR/USD", "XAU/USD"],
            timeframes=["H1", "H4", "D1"],
        )
    
    @staticmethod
    def get_available_presets() -> List[str]:
        """Get list of available preset names"""
        return ["conservative", "aggressive", "research", "demo_online"]


class ConfigFactory:
    """Factory for creating and modifying configurations"""
    
    @staticmethod
    def create_config(
        mode: str = "backtest",
        risk_level: str = "moderate",
        **overrides
    ) -> TradingConfig:
        """Create configuration with specified parameters"""
        
        # Base configurations by mode
        if mode == "live":
            config = ConfigPresets.conservative_live()
        elif mode == "demo":
            config = ConfigPresets.demo_online()
        elif mode == "research":
            config = ConfigPresets.research_mode()
        elif mode == "aggressive":
            config = ConfigPresets.aggressive_backtest()
        else:  # backtest
            config = TradingConfig()
        
        # Adjust risk level
        if risk_level == "conservative":
            config.max_position_pct *= 0.5
            config.max_total_exposure *= 0.7
            config.max_drawdown *= 0.8
            config.min_inst_confidence = min(config.min_inst_confidence + 0.1, 0.9)
        elif risk_level == "aggressive":
            config.max_position_pct *= 1.5
            config.max_total_exposure *= 1.3
            config.max_drawdown *= 1.2
            config.min_inst_confidence = max(config.min_inst_confidence - 0.1, 0.3)
        
        # Apply any overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                print(f"Warning: Unknown config parameter '{key}'")
        
        return config
    
    @staticmethod
    def quick_config(
        balance: float = 3000.0,
        instruments: List[str] = None,
        live: bool = False,
        debug: bool = True
    ) -> TradingConfig:
        """Quick configuration for common use cases"""
        if instruments is None:
            instruments = ["EUR/USD"]
        
        return TradingConfig(
            initial_balance=balance,
            instruments=instruments,
            live_mode=live,
            debug=debug,
            max_steps=200 if not live else 50,
            final_training_steps=50000 if not live else 10000,
        )


# ═══════════════════════════════════════════════════════════════════
# Configuration Validation
# ═══════════════════════════════════════════════════════════════════

def validate_config(config: TradingConfig) -> List[str]:
    """Validate configuration and return list of warnings/errors"""
    warnings = []
    
    # Risk validation
    if config.max_total_exposure > 1.0:
        warnings.append("Total exposure > 100% is very risky")
    
    if config.max_drawdown > 0.5:
        warnings.append("Max drawdown > 50% is extremely risky")
    
    if config.max_position_pct > 0.3:
        warnings.append("Position size > 30% per trade is very risky")
    
    # Training validation
    if config.learning_rate > 1e-3:
        warnings.append("Learning rate might be too high")
    
    if config.batch_size > config.n_steps:
        warnings.append("Batch size should not exceed n_steps")
    
    # Live mode validation
    if config.live_mode:
        if config.debug and config.max_position_pct > 0.1:
            warnings.append("Large positions in live mode - consider reducing")
        
        if config.final_training_steps > 50000:
            warnings.append("Very long training in live mode")
    
    return warnings


# ═══════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════

def print_config_comparison(config1: TradingConfig, config2: TradingConfig):
    """Print comparison between two configurations"""
    print("Configuration Comparison:")
    print("=" * 50)
    
    for field in config1.__dataclass_fields__:
        val1 = getattr(config1, field)
        val2 = getattr(config2, field)
        
        if val1 != val2:
            print(f"{field:25} | {val1:15} | {val2:15}")


def get_config_summary(config: TradingConfig) -> Dict[str, Any]:
    """Get a summary of key configuration parameters"""
    return {
        "mode": "LIVE" if config.live_mode else ("TEST" if config.test_mode else "BACKTEST"),
        "balance": config.initial_balance,
        "risk_level": "HIGH" if config.max_drawdown > 0.25 else ("LOW" if config.max_drawdown < 0.15 else "MEDIUM"),
        "instruments": len(config.instruments),
        "training_steps": config.final_training_steps,
        "learning_rate": config.learning_rate,
    }


# ═══════════════════════════════════════════════════════════════════
# Example Usage and Testing
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🔧 Trading Configuration System")
    print("=" * 60)
    
    # Test different presets
    configs = {
        "Conservative Live": ConfigPresets.conservative_live(),
        "Aggressive Backtest": ConfigPresets.aggressive_backtest(),
        "Research Mode": ConfigPresets.research_mode(),
        "Demo Online": ConfigPresets.demo_online(),
    }
    
    for name, config in configs.items():
        print(f"\n📋 {name}:")
        summary = get_config_summary(config)
        for key, value in summary.items():
            print(f"  {key:15}: {value}")
        
        # Validate
        warnings = validate_config(config)
        if warnings:
            print(f"  ⚠️  Warnings: {len(warnings)}")
            for warning in warnings[:2]:  # Show first 2
                print(f"    - {warning}")
    
    # Test factory
    print(f"\n🏭 Factory Examples:")
    custom_config = ConfigFactory.create_config(
        mode="live",
        risk_level="conservative",
        initial_balance=5000.0,
        instruments=["EUR/USD", "GBP/USD"]
    )
    print(f"Custom config balance: ${custom_config.initial_balance}")
    print(f"Custom config instruments: {custom_config.instruments}")
    
    print("\n✅ Configuration system test completed!")