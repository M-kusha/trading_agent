# config/trading_config.py
"""
CENTRALIZED TRADING CONFIGURATION
Single source of truth for ALL system parameters
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import os
from pathlib import Path

@dataclass
class TradingConfig:
    """
    MASTER CONFIGURATION CLASS
    All parameters flow from here - NO MORE HARDCODING!
    """
    
    # ═══════════════════════════════════════════════════════════════
    # CORE SYSTEM PARAMETERS
    # ═══════════════════════════════════════════════════════════════
    
    # Environment basics
    initial_balance: float = 3000.0
    max_steps: int = 200
    max_steps_per_episode: int = field(init=False)  # Alias for compatibility
    debug: bool = True
    init_seed: int = 42
    
    # Mode settings
    live_mode: bool = False
    test_mode: bool = True
    
    # Data and instruments
    instruments: List[str] = field(default_factory=lambda: ["EURUSD", "XAUUSD"])
    timeframes: List[str] = field(default_factory=lambda: ["H1", "H4", "D1"])
    primary_timeframe: str = "D1"
    
    # Point values for PnL calculation
    point_values: Dict[str, float] = field(default_factory=lambda: {
        "EURUSD": 100000, "EUR/USD": 100000,
        "XAUUSD": 100, "XAU/USD": 100,
        "GBPUSD": 100000, "GBP/USD": 100000,
        "USDJPY": 100000, "USD/JPY": 100000,
    })
    
    # ═══════════════════════════════════════════════════════════════
    # TRADING PARAMETERS
    # ═══════════════════════════════════════════════════════════════
    
    # Action thresholds
    min_intensity: float = 0.15  # Reduced from 0.25 for more trading
    min_inst_confidence: float = 0.55  # Reduced from 0.60
    rotation_gap: int = 3  # Reduced from 5
    
    # Consensus settings
    consensus_min: float = 0.25  # Reduced from 0.30
    consensus_max: float = 0.75  # Increased from 0.70
    consensus_window: int = 10
    
    # Position sizing
    base_risk_pct: float = 0.02  # 2% risk per trade
    max_position_pct: float = 0.08  # 8% max single position
    max_total_exposure: float = 0.25  # 25% max total exposure
    min_lot_size: float = 0.01
    max_lot_size: float = 5.0
    
    # Trade execution
    slippage_buffer: float = 0.0002  # 2 pips buffer
    max_spread_pct: float = 0.001  # 0.1% max spread
    
    # ═══════════════════════════════════════════════════════════════
    # RISK MANAGEMENT PARAMETERS
    # ═══════════════════════════════════════════════════════════════
    
    # Drawdown limits
    max_drawdown_daily: float = 0.05  # 5% daily
    max_drawdown_total: float = 0.20  # 20% total
    dd_emergency_stop: float = 0.30  # 30% emergency stop
    
    # Risk scaling
    volatility_threshold: float = 1.5
    correlation_limit: float = 0.75
    var_confidence: float = 0.95
    var_window: int = 50
    
    # Dynamic risk parameters
    risk_freeze_duration: int = 5
    risk_vol_history_len: int = 100
    risk_dd_threshold: float = 0.15
    risk_vol_ratio_threshold: float = 1.8
    
    # Portfolio risk
    portfolio_var_window: int = 50
    portfolio_dd_limit: float = 0.20
    max_positions: int = 8
    
    # ═══════════════════════════════════════════════════════════════
    # REWARD AND PENALTY PARAMETERS
    # ═══════════════════════════════════════════════════════════════
    
    # Reward shaping
    no_trade_penalty: float = 0.1  # Reduced from 0.3
    consensus_bonus_weight: float = 0.2
    drawdown_penalty_weight: float = 2.0
    profit_reward_weight: float = 1.0
    sharpe_bonus_weight: float = 0.5
    
    # Risk-adjusted reward parameters
    reward_risk_free_rate: float = 0.02  # 2% annual
    reward_volatility_penalty: float = 0.1
    reward_max_dd_penalty: float = 1.0
    
    # ═══════════════════════════════════════════════════════════════
    # MODEL AND TRAINING PARAMETERS
    # ═══════════════════════════════════════════════════════════════
    
    # PPO-Lagrangian hyperparameters
    learning_rate: float = 3e-4
    learning_rate_schedule: str = "constant"  # constant, linear, cosine
    
    # Network architecture
    policy_hidden_size: int = 512
    value_hidden_size: int = 256
    features_dim: int = 256
    
    # PPO specific
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
    
    # Lagrangian constraint parameters
    cost_limit: float = 20.0  # Reduced from 25.0
    lagrangian_pid_ki: float = 0.01
    lagrangian_pid_kp: float = 0.1
    lagrangian_pid_kd: float = 0.01
    
    # ═══════════════════════════════════════════════════════════════
    # TRAINING CONFIGURATION
    # ═══════════════════════════════════════════════════════════════
    
    # Environment setup
    num_envs: int = 4
    
    # Optimization settings
    n_trials: int = 0  # Set in __post_init__
    timesteps_per_trial: int = 0
    final_training_steps: int = 0
    
    # Pruning settings
    pruner_startup_trials: int = 0
    pruner_warmup_steps: int = 0
    pruner_interval_steps: int = 0
    
    # Evaluation
    eval_freq: int = 0
    n_eval_episodes: int = 0
    eval_deterministic: bool = True
    
    # ═══════════════════════════════════════════════════════════════
    # LOGGING AND MONITORING
    # ═══════════════════════════════════════════════════════════════
    
    # Directories
    log_dir: str = "logs/trading"
    model_dir: str = "models"
    checkpoint_dir: str = "checkpoints"
    tensorboard_dir: str = "logs/tensorboard"
    data_dir: str = "data/processed"
    
    # Logging frequencies
    log_interval: int = 0  # Set in __post_init__
    tb_log_freq: int = 0
    checkpoint_freq: int = 0
    log_level: str = "INFO"
    
    # Console settings
    console_width: int = 80
    use_emojis: bool = False  # Disabled for Windows compatibility
    
    # ═══════════════════════════════════════════════════════════════
    # MODULE CONFIGURATION
    # ═══════════════════════════════════════════════════════════════
    
    # Feature engineering
    feature_engine_window: int = 32
    multiscale_levels: int = 4
    feature_compression_ratio: float = 0.8
    
    # Memory systems
    mistake_memory_interval: int = 10
    mistake_memory_clusters: int = 3
    memory_compressor_capacity: int = 50
    memory_compressor_levels: int = 5
    playbook_memory_capacity: int = 100
    memory_budget_total: int = 1000
    memory_budget_critical: int = 500
    memory_budget_reserve: int = 300
    
    # Neural memory
    neural_memory_embedding_dim: int = 32
    neural_memory_num_heads: int = 4
    neural_memory_capacity: int = 500
    
    # Strategy systems
    strategy_pool_size: int = 20
    strategy_mutation_rate: float = 0.1
    curriculum_difficulty_levels: int = 5
    
    # Market analysis
    theme_detector_lookback: int = 100
    theme_detector_n_themes: int = 4
    fractal_confirmation_window: int = 100
    regime_matrix_history: int = 200
    
    # Voting and consensus
    voting_committee_size: int = 8
    voting_weight_decay: float = 0.95
    voting_adaptation_rate: float = 0.1
    horizon_alignment_levels: List[int] = field(default_factory=lambda: [1, 4, 24, 96])
    
    # Monitoring
    active_monitor_max_duration: int = 200  # matches max_steps
    execution_quality_window: int = 50
    anomaly_detection_sensitivity: float = 2.0
    
    # ═══════════════════════════════════════════════════════════════
    # LIVE TRADING PARAMETERS
    # ═══════════════════════════════════════════════════════════════
    
    # MT5 connection
    mt5_timeout: int = 10000  # milliseconds
    mt5_deviation: int = 20  # points
    mt5_magic_number: int = 202406
    mt5_comment_prefix: str = "AI_Trade"
    
    # Data feed
    live_data_update_interval: int = 5  # seconds
    live_data_buffer_size: int = 1000
    live_data_reconnect_attempts: int = 3
    
    # Order management
    order_retry_attempts: int = 3
    order_retry_delay: float = 1.0  # seconds
    position_monitoring_interval: int = 30  # seconds
    
    # ═══════════════════════════════════════════════════════════════
    # TARGET PERFORMANCE METRICS
    # ═══════════════════════════════════════════════════════════════
    
    target_daily_profit: float = 150.0  # EUR/day
    target_monthly_return: float = 0.15  # 15% monthly
    target_annual_sharpe: float = 2.0
    target_win_rate: float = 0.60  # 60%
    target_profit_factor: float = 1.5
    max_consecutive_losses: int = 5
    
    # ═══════════════════════════════════════════════════════════════
    # MODULE ENABLE/DISABLE FLAGS
    # ═══════════════════════════════════════════════════════════════
    
    enable_shadow_simulation: bool = True
    enable_news_sentiment: bool = False
    enable_regime_detection: bool = True
    enable_theme_analysis: bool = True
    enable_meta_rl: bool = True
    enable_memory_systems: bool = True
    enable_opponent_simulation: bool = False  # CPU intensive
    enable_visualization: bool = False  # For live mode
    
    # Advanced features
    enable_bias_auditor: bool = True
    enable_thesis_evolution: bool = True
    enable_explanation_generation: bool = True
    enable_playbook_clustering: bool = True
    
    def __post_init__(self):
        """Set mode-specific parameters and validate configuration"""
        
        # Set max_steps_per_episode alias
        object.__setattr__(self, "max_steps_per_episode", self.max_steps)
        
        # Configure based on mode
        if self.test_mode:
            self._configure_test_mode()
        elif self.live_mode:
            self._configure_live_mode()
        else:
            self._configure_production_mode()
            
        # Create directories
        self._create_directories()
        
        # Validate configuration
        self._validate_config()
        
    def _configure_test_mode(self):
        """Configure for quick testing"""
        # Quick training settings
        self.n_trials = 3
        self.timesteps_per_trial = 5_000
        self.final_training_steps = 20_000
        self.pruner_startup_trials = 1
        self.pruner_warmup_steps = 1_000
        self.pruner_interval_steps = 1_000
        
        # Frequent logging
        self.log_interval = 50
        self.tb_log_freq = 100
        self.checkpoint_freq = 2_000
        self.eval_freq = 1_000
        self.n_eval_episodes = 2
        
        # Reduced complexity
        self.num_envs = 2
        self.n_steps = 256
        self.batch_size = 32
        self.strategy_pool_size = 5
        
        # Disable CPU-intensive features
        self.enable_opponent_simulation = False
        self.enable_shadow_simulation = False
        
        print("TEST MODE: Quick training with extensive logging")
        
    def _configure_live_mode(self):
        """Configure for live trading"""
        # Live-specific settings
        self.max_steps = 10_000_000  # Essentially unlimited
        self.debug = False
        self.test_mode = False
        
        # Conservative trading
        self.min_intensity = 0.20
        self.min_inst_confidence = 0.65
        self.consensus_min = 0.35
        
        # Tighter risk management
        self.max_drawdown_daily = 0.03  # 3% daily
        self.base_risk_pct = 0.015  # 1.5% per trade
        self.cost_limit = 15.0
        
        # Enable monitoring
        self.enable_visualization = True
        
        # Less frequent logging
        self.log_interval = 300  # 5 minutes
        self.tb_log_freq = 1000
        
        print("LIVE MODE: Conservative settings with enhanced monitoring")
        
    def _configure_production_mode(self):
        """Configure for full production training"""
        # Full training settings
        self.n_trials = 20
        self.timesteps_per_trial = 500_000
        self.final_training_steps = 5_000_000
        self.pruner_startup_trials = 5
        self.pruner_warmup_steps = 100_000
        self.pruner_interval_steps = 50_000
        
        # Balanced logging
        self.log_interval = 1_000
        self.tb_log_freq = 1_000
        self.checkpoint_freq = 50_000
        self.eval_freq = 10_000
        self.n_eval_episodes = 10
        
        # Full capability
        self.num_envs = 4
        self.enable_opponent_simulation = True
        self.enable_shadow_simulation = True
        
        print("PRODUCTION MODE: Full training pipeline")
        
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.log_dir,
            self.model_dir, 
            self.checkpoint_dir,
            self.tensorboard_dir,
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
    def _validate_config(self):
        """Validate configuration parameters"""
        # Core validations
        assert self.initial_balance > 0, "Initial balance must be positive"
        assert self.max_steps > 0, "Max steps must be positive"
        assert len(self.instruments) > 0, "Must have at least one instrument"
        
        # Risk validations
        assert 0 < self.max_drawdown_total < 1, "Max drawdown must be between 0 and 1"
        assert 0 < self.correlation_limit < 1, "Correlation limit must be between 0 and 1"
        assert self.base_risk_pct > 0, "Base risk percentage must be positive"
        
        # Trading validations
        assert 0 < self.consensus_min < self.consensus_max < 1, "Invalid consensus range"
        assert 0 < self.min_intensity < 1, "Min intensity must be between 0 and 1"
        assert self.rotation_gap >= 0, "Rotation gap cannot be negative"
        
        # Training validations
        assert self.n_steps > 0, "N steps must be positive"
        assert self.batch_size > 0, "Batch size must be positive"
        assert 0 < self.gamma < 1, "Gamma must be between 0 and 1"
        
        # Path validations
        for instrument in self.instruments:
            assert len(instrument) > 0, f"Invalid instrument name: {instrument}"
            
        print("Configuration validated successfully")
        
    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration"""
        return {
            'learning_rate': self.learning_rate,
            'n_steps': self.n_steps,
            'batch_size': self.batch_size,
            'n_epochs': self.n_epochs,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'clip_range': self.clip_range,
            'clip_range_vf': self.clip_range_vf,
            'ent_coef': self.ent_coef,
            'vf_coef': self.vf_coef,
            'max_grad_norm': self.max_grad_norm,
            'policy_hidden_size': self.policy_hidden_size,
            'value_hidden_size': self.value_hidden_size,
            'features_dim': self.features_dim,
        }
        
    def get_risk_config(self) -> Dict[str, Any]:
        """Get risk management configuration"""
        return {
            'max_drawdown_daily': self.max_drawdown_daily,
            'max_drawdown_total': self.max_drawdown_total,
            'dd_emergency_stop': self.dd_emergency_stop,
            'volatility_threshold': self.volatility_threshold,
            'correlation_limit': self.correlation_limit,
            'var_confidence': self.var_confidence,
            'var_window': self.var_window,
            'base_risk_pct': self.base_risk_pct,
            'max_position_pct': self.max_position_pct,
            'max_total_exposure': self.max_total_exposure,
            'max_positions': self.max_positions,
        }
        
    def get_trading_config(self) -> Dict[str, Any]:
        """Get trading-specific configuration"""
        return {
            'min_intensity': self.min_intensity,
            'min_inst_confidence': self.min_inst_confidence,
            'rotation_gap': self.rotation_gap,
            'consensus_min': self.consensus_min,
            'consensus_max': self.consensus_max,
            'no_trade_penalty': self.no_trade_penalty,
            'slippage_buffer': self.slippage_buffer,
            'max_spread_pct': self.max_spread_pct,
        }
        
    def get_module_config(self, module_name: str) -> Dict[str, Any]:
        """Get configuration for specific module"""
        configs = {
            'feature_engine': {
                'window': self.feature_engine_window,
                'levels': self.multiscale_levels,
                'debug': self.debug,
            },
            'position_manager': {
                'initial_balance': self.initial_balance,
                'instruments': self.instruments,
                'debug': self.debug,
                'min_lot_size': self.min_lot_size,
                'max_lot_size': self.max_lot_size,
            },
            'risk_controller': {
                'freeze_duration': self.risk_freeze_duration,
                'vol_history_len': self.risk_vol_history_len,
                'dd_threshold': self.risk_dd_threshold,
                'vol_ratio_threshold': self.risk_vol_ratio_threshold,
                'debug': self.debug,
            },
            'strategy_pool': {
                'pool_size': self.strategy_pool_size,
                'mutation_rate': self.strategy_mutation_rate,
                'debug': self.debug,
            },
            'memory_systems': {
                'mistake_interval': self.mistake_memory_interval,
                'mistake_clusters': self.mistake_memory_clusters,
                'compressor_capacity': self.memory_compressor_capacity,
                'compressor_levels': self.memory_compressor_levels,
                'neural_embedding_dim': self.neural_memory_embedding_dim,
                'neural_num_heads': self.neural_memory_num_heads,
                'neural_capacity': self.neural_memory_capacity,
                'debug': self.debug,
            }
        }
        
        return configs.get(module_name, {'debug': self.debug})
        
    def save_config(self, filepath: str):
        """Save configuration to file"""
        import json
        
        # Convert to serializable dict
        config_dict = {}
        for field_name, field_type in self.__dataclass_fields__.items():
            value = getattr(self, field_name)
            # Handle non-serializable types
            if isinstance(value, Path):
                config_dict[field_name] = str(value)
            else:
                config_dict[field_name] = value
                
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
            
        print(f"Configuration saved to {filepath}")
        
    @classmethod
    def load_config(cls, filepath: str) -> 'TradingConfig':
        """Load configuration from file"""
        import json
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
            
        return cls(**config_dict)
        
    def update_from_dict(self, updates: Dict[str, Any]):
        """Update configuration from dictionary"""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown config parameter: {key}")
                
    def __str__(self) -> str:
        """String representation for logging"""
        mode = "LIVE" if self.live_mode else ("TEST" if self.test_mode else "PRODUCTION")
        return (
            f"TradingConfig({mode}):\n"
            f"  Balance: €{self.initial_balance:,.2f}\n"
            f"  Instruments: {self.instruments}\n"
            f"  Max Steps: {self.max_steps:,}\n"
            f"  Risk Limit: {self.max_drawdown_total:.1%}\n"
            f"  Min Intensity: {self.min_intensity:.2f}\n"
            f"  Consensus Range: {self.consensus_min:.2f}-{self.consensus_max:.2f}"
        )


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION PRESETS
# ═══════════════════════════════════════════════════════════════════

class ConfigPresets:
    """Predefined configuration presets for common scenarios"""
    
    @staticmethod
    def conservative_live() -> TradingConfig:
        """Conservative settings for live trading"""
        return TradingConfig(
            live_mode=True,
            test_mode=False,
            initial_balance=5000.0,
            min_intensity=0.25,
            min_inst_confidence=0.70,
            consensus_min=0.40,
            base_risk_pct=0.01,  # 1% per trade
            max_drawdown_daily=0.02,  # 2% daily
            cost_limit=10.0,
        )
        
    @staticmethod
    def aggressive_backtest() -> TradingConfig:
        """Aggressive settings for backtesting"""
        return TradingConfig(
            live_mode=False,
            test_mode=False,
            min_intensity=0.10,
            min_inst_confidence=0.50,
            consensus_min=0.20,
            base_risk_pct=0.03,  # 3% per trade
            max_position_pct=0.12,
            cost_limit=30.0,
        )
        
    @staticmethod
    def quick_test() -> TradingConfig:
        """Quick test settings"""
        return TradingConfig(
            test_mode=True,
            live_mode=False,
            max_steps=50,
            n_trials=2,
            timesteps_per_trial=1000,
            num_envs=1,
        )
        
    @staticmethod
    def research_mode() -> TradingConfig:
        """Settings optimized for research and experimentation"""
        return TradingConfig(
            live_mode=False,
            test_mode=False,
            debug=True,
            enable_opponent_simulation=True,
            enable_shadow_simulation=True,
            enable_bias_auditor=True,
            enable_thesis_evolution=True,
            log_interval=100,  # Detailed logging
        )


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION FACTORY
# ═══════════════════════════════════════════════════════════════════

class ConfigFactory:
    """Factory for creating configurations based on environment variables or arguments"""
    
    @staticmethod
    def from_env() -> TradingConfig:
        """Create configuration from environment variables"""
        config = TradingConfig()
        
        # Override with environment variables
        env_mappings = {
            'TRADING_MODE': 'live_mode',
            'INITIAL_BALANCE': 'initial_balance',
            'DEBUG': 'debug',
            'INSTRUMENTS': 'instruments',
            'MIN_INTENSITY': 'min_intensity',
            'RISK_PCT': 'base_risk_pct',
        }
        
        for env_var, config_attr in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Type conversion
                if config_attr == 'live_mode':
                    setattr(config, config_attr, env_value.lower() == 'true')
                elif config_attr == 'debug':
                    setattr(config, config_attr, env_value.lower() == 'true')
                elif config_attr == 'instruments':
                    setattr(config, config_attr, env_value.split(','))
                elif config_attr in ['initial_balance', 'min_intensity', 'base_risk_pct']:
                    setattr(config, config_attr, float(env_value))
                    
        return config
        
    @staticmethod
    def from_args(args) -> TradingConfig:
        """Create configuration from command line arguments"""
        config = TradingConfig()
        
        if hasattr(args, 'live') and args.live:
            config.live_mode = True
            config.test_mode = False
        elif hasattr(args, 'test') and args.test:
            config.test_mode = True
            config.live_mode = False
            
        if hasattr(args, 'debug'):
            config.debug = args.debug
            
        if hasattr(args, 'balance'):
            config.initial_balance = float(args.balance)
            
        return config


# ═══════════════════════════════════════════════════════════════════
# USAGE EXAMPLES
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Example usage
    
    # Default configuration
    config = TradingConfig()
    print("Default config:")
    print(config)
    print()
    
    # Test mode
    test_config = TradingConfig(test_mode=True)
    print("Test config:")
    print(test_config)
    print()
    
    # Preset configurations
    conservative = ConfigPresets.conservative_live()
    print("Conservative live config:")
    print(conservative)
    print()
    
    # Save and load
    config.save_config("config_example.json")
    loaded_config = TradingConfig.load_config("config_example.json")
    print("Loaded config matches:", str(config) == str(loaded_config))