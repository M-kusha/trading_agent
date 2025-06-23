# train_ppo_lag.py
"""
Advanced PPO-Lagrangian Training Script for Forex/Gold Trading
Integrates all modules with detailed logging and Optuna optimization
"""

import os
import warnings
import random
import logging
import logging.handlers
from sys import platform
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import json
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Optuna for hyperparameter optimization
import optuna
from optuna.pruners import MedianPruner
from optuna.exceptions import TrialPruned
from optuna.trial import TrialState

# Progress tracking
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# PPO-LAG specific imports
from stable_baselines3.common.vec_env import  DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import get_schedule_fn

# Custom imports
from envs.env import EnhancedTradingEnv, TradingConfig
from utils.data_utils import load_data

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TrainingConfig:
    """Comprehensive training configuration"""
    # Mode settings
    test_mode: bool = True
    debug: bool = True
    
    # Environment settings
    num_envs: int = 4
    global_seed: int = 42
    initial_balance: float = 3000.0
    max_steps_per_episode: int = 200
    
    # Optimization settings
    n_trials: int = 0
    timesteps_per_trial: int = 0
    final_training_steps: int = 0
    
    # Pruning settings
    pruner_startup_trials: int = 0
    pruner_warmup_steps: int = 0
    pruner_interval_steps: int = 0
    
    # Logging frequencies
    tb_log_freq: int = 0
    checkpoint_freq: int = 0
    eval_freq: int = 0
    n_eval_episodes: int = 0
    log_interval: int = 0
    
    # PPO-LAG specific
    cost_limit: float = 25.0  # Maximum allowed cost (risk)
    lagrangian_pid_ki: float = 0.01
    lagrangian_pid_kp: float = 0.1
    lagrangian_pid_kd: float = 0.01
    
    # Paths
    log_dir: str = "logs/training"
    model_dir: str = "models"
    checkpoint_dir: str = "checkpoints"
    tensorboard_dir: str = "logs/tensorboard"
    
    # Target metrics
    target_daily_profit: float = 150.0  # €150/day
    max_drawdown_limit: float = 0.20   # 20% max drawdown
    min_win_rate: float = 0.55         # 55% minimum win rate
    
    def __post_init__(self):
        """Set mode-specific parameters"""
        if self.test_mode:
            # Quick test settings
            self.n_trials = 2
            self.timesteps_per_trial = 10_000
            self.final_training_steps = 50_000
            self.pruner_startup_trials = 1
            self.pruner_warmup_steps = 2_000
            self.pruner_interval_steps = 2_000
            self.tb_log_freq = 1_000
            self.checkpoint_freq = 5_000
            self.eval_freq = 2_000
            self.n_eval_episodes = 3
            self.log_interval = 500
        else:
            # Full training settings
            self.n_trials = 20
            self.timesteps_per_trial = 500_000
            self.final_training_steps = 5_000_000
            self.pruner_startup_trials = 5
            self.pruner_warmup_steps = 100_000
            self.pruner_interval_steps = 50_000
            self.tb_log_freq = 1_000
            self.checkpoint_freq = 50_000
            self.eval_freq = 10_000
            self.n_eval_episodes = 10
            self.log_interval = 1_000
            
        # Create directories
        for path in [self.log_dir, self.model_dir, self.checkpoint_dir, self.tensorboard_dir]:
            os.makedirs(path, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# PPO-LAG Implementation
# ═══════════════════════════════════════════════════════════════════

class PPOLagrangian(nn.Module):
    """PPO with Lagrangian constraints for safe trading"""
    
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        cost_limit: float = 25.0,
        lagrangian_pid_ki: float = 0.01,
        lagrangian_pid_kp: float = 0.1,
        lagrangian_pid_kd: float = 0.01,
        **kwargs
    ):
        super().__init__()
        self.num_timesteps = 0
        self.logger = logging.getLogger("PPOLagrangian")
        
        self.observation_space = observation_space
        self.action_space = action_space
        self.cost_limit = cost_limit
        
        # Fixed: Store observation dimension
        self.obs_dim = observation_space.shape[0]
        
        # Networks
        self.features_dim = 256
        
        # Fixed: Use fixed input dimension instead of LazyLinear
        self.shared_net = nn.Sequential(
            nn.Linear(self.obs_dim, 512),  # Fixed input size
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.features_dim),
            nn.ReLU()
        )
        
        # Actor (policy) head
        self.policy_net = nn.Sequential(
            nn.Linear(self.features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_space.shape[0] * 2)  # Mean and log_std
        )
        
        # Value head
        self.value_net = nn.Sequential(
            nn.Linear(self.features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Cost value head (for constraint)
        self.cost_value_net = nn.Sequential(
            nn.Linear(self.features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Lagrangian multiplier
        self.log_lagrangian = nn.Parameter(torch.zeros(1))
        
        # PID controller for Lagrangian
        self.pid_ki = lagrangian_pid_ki
        self.pid_kp = lagrangian_pid_kp
        self.pid_kd = lagrangian_pid_kd
        self.pid_i = 0
        self.prev_error = 0
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_schedule(1))
        
    def forward(self, obs):
        """Forward pass through networks"""
        # Fixed: Ensure correct input shape
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            
        # Fixed: Handle variable observation sizes
        if obs.shape[-1] != self.obs_dim:
            if obs.shape[-1] < self.obs_dim:
                # Pad with zeros
                padding = torch.zeros(
                    (*obs.shape[:-1], self.obs_dim - obs.shape[-1]),
                    device=obs.device,
                    dtype=obs.dtype
                )
                obs = torch.cat([obs, padding], dim=-1)
            else:
                # Truncate
                obs = obs[..., :self.obs_dim]
        
        features = self.shared_net(obs)
        
        # Policy
        policy_out = self.policy_net(features)
        mean, log_std = torch.chunk(policy_out, 2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        
        # Values
        value = self.value_net(features)
        cost_value = self.cost_value_net(features)
        
        return mean, log_std, value, cost_value
    
    def get_lagrangian(self):
        """Get current Lagrangian multiplier"""
        return torch.exp(self.log_lagrangian).item()
    
    def update_lagrangian(self, cost_error):
        """Update Lagrangian using PID controller"""
        # PID update
        self.pid_i += cost_error
        pid_d = cost_error - self.prev_error
        self.prev_error = cost_error
        
        # Update log_lagrangian
        delta = self.pid_kp * cost_error + self.pid_ki * self.pid_i + self.pid_kd * pid_d
        self.log_lagrangian.data += delta
# ═══════════════════════════════════════════════════════════════════
# Custom Callbacks
# ═══════════════════════════════════════════════════════════════════

class DetailedLoggingCallback(BaseCallback):
    """Comprehensive logging callback with InfoBus integration"""
    
    def __init__(self, config: TrainingConfig, trial_id: int = -1):
        super().__init__(verbose=1)
        self.config = config
        self.trial_id = trial_id
        self.episode_count = 0
        self.step_count = 0


        
        # Setup loggers
        self.setup_loggers()
        
        # Metrics tracking
        self.episode_metrics = []
        self.trade_history = []
        self.best_metrics = {
            'reward': -float('inf'),
            'profit': -float('inf'),
            'sharpe': -float('inf')
        }

    def init_callback(self, model, env=None):
        super().init_callback(model)
        self._env = env
        
    def setup_loggers(self):
        """Setup multiple specialized loggers"""
        # Main training logger
        self.train_logger = self._create_logger(
            'training', 
            os.path.join(self.config.log_dir, 'training.log')
        )
        
        # Trade logger
        self.trade_logger = self._create_logger(
            'trades',
            os.path.join(self.config.log_dir, 'trades.log')
        )
        
        # Performance logger
        self.perf_logger = self._create_logger(
            'performance',
            os.path.join(self.config.log_dir, 'performance.log')
        )
        
        # Risk logger
        self.risk_logger = self._create_logger(
            'risk',
            os.path.join(self.config.log_dir, 'risk.log')
        )
        
    def _create_logger(self, name: str, filename: str) -> logging.Logger:
        """Create a specialized logger"""
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        
        # File handler
        fh = logging.handlers.RotatingFileHandler(
            filename, maxBytes=10*1024*1024, backupCount=5
        )
        fh.setFormatter(
            logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        )
        
        # Console handler for important messages
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(
            logging.Formatter('[%(name)s] %(message)s')
        )
        
        logger.handlers.clear()
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def _on_step(self) -> bool:
        """Called at every step"""
        self.step_count += 1
        
        # Log at intervals
        if self.step_count % self.config.log_interval == 0:
            self._log_step_metrics()
            
        return True
    
    def _on_rollout_end(self):
        """Called at the end of a rollout"""
        self.episode_count += 1
        self._log_episode_metrics()
        
    def _log_step_metrics(self):
        """Log step-level metrics"""
        try:
            # Get environment metrics
            env = self._env.envs[0]
            info_bus = getattr(env, 'info_bus', None)
            
            if info_bus:
                # Extract key metrics from InfoBus
                risk = info_bus.get('risk', {})
                positions = info_bus.get('positions', [])
                
                self.train_logger.info(
                    f"Step {self.step_count} | "
                    f"Balance: ${risk.get('balance', 0):,.2f} | "
                    f"Drawdown: {risk.get('current_drawdown', 0):.2%} | "
                    f"Positions: {len(positions)} | "
                    f"P&L Today: ${info_bus.get('pnl_today', 0):,.2f}"
                )
                
                # Log any alerts
                for alert in info_bus.get('alerts', []):
                    self.risk_logger.warning(f"ALERT: {alert}")
                    
        except Exception as e:
            self.train_logger.error(f"Error logging step metrics: {e}")
            
    def _log_episode_metrics(self):
        """Log episode-level metrics"""
        try:
            env = self._env.envs[0]
            
            # Get comprehensive metrics
            metrics = env.get_metrics()
            
            # Performance summary
            self.perf_logger.info(
                f"\n{'='*60}\n"
                f"Episode {self.episode_count} Summary (Trial {self.trial_id})\n"
                f"{'='*60}\n"
                f"Final Balance: ${metrics['balance']:,.2f}\n"
                f"Total P&L: ${metrics['balance'] - self.config.initial_balance:,.2f}\n"
                f"Win Rate: {metrics.get('win_rate', 0):.2%}\n"
                f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}\n"
                f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}\n"
                f"Total Trades: {metrics.get('total_trades', 0)}\n"
                f"Avg Trade P&L: ${metrics.get('avg_pnl', 0):.2f}\n"
                f"{'='*60}"
            )
            
            # Get trade explanations if available
            if hasattr(env, 'trade_auditor'):
                analysis = env.trade_auditor.get_trade_analysis()
                
                self.trade_logger.info(
                    f"\nTrade Analysis:\n"
                    f"Top Strategies: {analysis.get('top_strategies', [])}\n"
                    f"Regime Performance: {analysis.get('regime_breakdown', {})}"
                )
                
            # Check for new bests
            self._check_best_performance(metrics)
            
        except Exception as e:
            self.train_logger.error(f"Error logging episode metrics: {e}")
            
    def _check_best_performance(self, metrics: Dict[str, Any]):
        """Check and log if we have new best performance"""
        updated = False
        
        profit = metrics['balance'] - self.config.initial_balance
        if profit > self.best_metrics['profit']:
            self.best_metrics['profit'] = profit
            updated = True
            
        sharpe = metrics.get('sharpe_ratio', 0)
        if sharpe > self.best_metrics['sharpe']:
            self.best_metrics['sharpe'] = sharpe
            updated = True
            
        if updated:
            self.perf_logger.info(
                f"\n NEW BEST PERFORMANCE!\n"
                f"Best Profit: ${self.best_metrics['profit']:,.2f}\n"
                f"Best Sharpe: {self.best_metrics['sharpe']:.3f}\n"
            )

        
    def update_locals(self, locals_dict):
        """Update local variables"""
        self.locals.update(locals_dict)
        
    def on_training_start(self, locals_=None, globals_=None):
        """Called at the beginning of training"""
        self._on_training_start()
        
    def on_training_end(self):
        """Called at the end of training"""
        pass
            

class TensorboardCallback(BaseCallback):
    """Enhanced TensorBoard logging with module metrics"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__(verbose=0)
        self.config = config
        self.writer = None
    
    def init_callback(self, model, env=None):
        super().init_callback(model)
        self._env = env
        
    def _on_training_start(self):
        """Initialize TensorBoard writer"""
        log_dir = os.path.join(
            self.config.tensorboard_dir,
            f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.writer = SummaryWriter(log_dir=log_dir)
        
    def _on_step(self) -> bool:
        """Log to TensorBoard"""
        if self.n_calls % self.config.tb_log_freq == 0 and self.writer:
            self._log_metrics()
        return True
    
    def _log_metrics(self):
        """Log comprehensive metrics to TensorBoard"""
        try:
            env = self._env.envs[0]
            metrics = env.get_metrics()
            
            # Basic metrics
            self.writer.add_scalar('env/balance', metrics['balance'], self.num_timesteps)
            self.writer.add_scalar('env/pnl', metrics['balance'] - self.config.initial_balance, self.num_timesteps)
            self.writer.add_scalar('env/drawdown', metrics['drawdown'], self.num_timesteps)
            self.writer.add_scalar('env/win_rate', metrics.get('win_rate', 0), self.num_timesteps)
            self.writer.add_scalar('env/sharpe_ratio', metrics.get('sharpe_ratio', 0), self.num_timesteps)
            
            # Trading metrics
            self.writer.add_scalar('trading/total_trades', metrics.get('total_trades', 0), self.num_timesteps)
            self.writer.add_scalar('trading/avg_pnl', metrics.get('avg_pnl', 0), self.num_timesteps)
            self.writer.add_scalar('trading/profit_factor', metrics.get('profit_factor', 0), self.num_timesteps)
            
            # Module-specific metrics
            if hasattr(env, 'info_bus') and env.info_bus:
                info_bus = env.info_bus
                
                # Risk metrics
                risk = info_bus.get('risk', {})
                self.writer.add_scalar('risk/var_95', risk.get('var_95', 0), self.num_timesteps)
                self.writer.add_scalar('risk/margin_used', risk.get('margin_used', 0), self.num_timesteps)
                
                # Market context
                context = info_bus.get('market_context', {})
                for inst, vol in context.get('volatility', {}).items():
                    self.writer.add_scalar(f'volatility/{inst}', vol, self.num_timesteps)
                    
                # Voting consensus
                self.writer.add_scalar('voting/consensus', info_bus.get('consensus', 0), self.num_timesteps)
                
                # Module health
                for module_name, module in env.pipeline._module_map.items():
                    health = module.get_health_status()
                    self.writer.add_scalar(
                        f'health/{module_name}',
                        1.0 if health['status'] == 'OK' else 0.0,
                        self.num_timesteps
                    )
                    
        except Exception as e:
            if self.config.debug:
                print(f"TensorBoard logging error: {e}")
                
    def _on_training_end(self):
        """Close TensorBoard writer"""
        if self.writer:
            self.writer.close()

class CheckpointCallback(BaseCallback):
    """Enhanced checkpoint callback with full state preservation"""
    
    def __init__(self, config: TrainingConfig, trial_id: int = -1):
        super().__init__(verbose=1)
        self.config = config
        self.trial_id = trial_id
        self.save_path = os.path.join(
            config.checkpoint_dir,
            f"trial_{trial_id}" if trial_id >= 0 else "final"
        )
        os.makedirs(self.save_path, exist_ok=True)

    def init_callback(self, model, env=None):
        super().init_callback(model)
        self._env = env
        
    def _on_step(self) -> bool:
        """Save checkpoint at intervals"""
        if self.n_calls % self.config.checkpoint_freq == 0:
            self._save_checkpoint()
        return True
    
    def _save_checkpoint(self):
        """Save comprehensive checkpoint"""
        try:
            checkpoint_path = os.path.join(
                self.save_path,
                f"checkpoint_{self.num_timesteps}.pkl"
            )
            
            # Prepare checkpoint data
            checkpoint = {
                'timesteps': self.num_timesteps,
                'trial_id': self.trial_id,
                'model_state': self.model.policy.state_dict(),
                'optimizer_state': self.model.policy.optimizer.state_dict(),
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save environment state
            env = self._env.envs[0]
            checkpoint['env_state'] = env.get_state()
            
            # Save module states
            module_states = {}
            for name, module in env.pipeline._module_map.items():
                module_states[name] = module.get_state()
            checkpoint['module_states'] = module_states
            
            # Save checkpoint
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)
                
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Keep only recent checkpoints
            self._cleanup_old_checkpoints()
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            
    def _cleanup_old_checkpoints(self, keep_last: int = 5):
        """Remove old checkpoints to save space"""
        checkpoints = sorted([
            f for f in os.listdir(self.save_path)
            if f.startswith('checkpoint_') and f.endswith('.pkl')
        ])
        
        if len(checkpoints) > keep_last:
            for old_ckpt in checkpoints[:-keep_last]:
                os.remove(os.path.join(self.save_path, old_ckpt))

class OptunaPruningCallback(BaseCallback):
    """
    Callback for Optuna trial pruning based on performance.
    Now accepts the vectorised training environment passed in by the
    outer loop so that BaseCallback.init_callback is not over-called.
    """

    def __init__(self, trial: optuna.Trial, config: TrainingConfig):
        super().__init__(verbose=0)
        self.trial = trial
        self.config = config
        self.eval_env = None
        self._last_eval_timestep = 0

    def init_callback(self, model, env=None):
        """
        Override SB3 initialisation.
        We *do not* assign to self.training_env because it is a read-only
        property in recent SB3 versions.  Instead, we simply build the
        separate evaluation environment if a training env is supplied.
        """
        super().init_callback(model)     # SB3 base initialisation
        if env is not None:
            self._setup_eval_env(env)    # create self.eval_env once
    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _setup_eval_env(self, train_env):
        """
        Create a lightweight evaluation environment that shares the same
        market data but has its own state trajectory.
        """
        # Get data from first training environment
        first_env = train_env.envs[0]
        data = first_env.orig_data
        config = first_env.config
        
        # Create evaluation environment with same configuration
        self.eval_env = DummyVecEnv([
            lambda: EnhancedTradingEnv(
                data,
                TradingConfig(
                    initial_balance=config.initial_balance,
                    max_steps=config.max_steps_per_episode,
                    debug=False
                )
            )
        ])
        
    def _on_step(self) -> bool:
        """Check for pruning at intervals"""
        if self.num_timesteps - self._last_eval_timestep >= self.config.pruner_interval_steps:
            self._last_eval_timestep = self.num_timesteps
            
            # Evaluate current performance
            mean_reward, std_reward = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.config.n_eval_episodes,
                deterministic=True
            )
            
            # Get additional metrics
            env = self.eval_env.envs[0]
            metrics = env.get_metrics()
            
            # Compute composite score
            score = self._compute_trial_score(mean_reward, metrics)
            
            # Report to Optuna
            self.trial.report(score, self.num_timesteps)
            
            # Store metrics in trial
            self.trial.set_user_attr('metrics', metrics)
            self.trial.set_user_attr('mean_reward', mean_reward)
            
            # Check if should prune
            if self.trial.should_prune():
                raise TrialPruned()
                
        return True
    
    def _compute_trial_score(self, mean_reward: float, metrics: Dict[str, Any]) -> float:
        """Compute composite score for trial evaluation"""
        # Normalize components
        profit = (metrics['balance'] - self.config.initial_balance) / self.config.initial_balance
        win_rate = metrics.get('win_rate', 0)
        sharpe = metrics.get('sharpe_ratio', 0) / 3.0  # Normalize assuming max 3.0
        drawdown = 1.0 - metrics.get('max_drawdown', 0)
        
        # Composite score with emphasis on profitability and risk
        score = (
            0.30 * profit +
            0.25 * win_rate +
            0.25 * sharpe +
            0.20 * drawdown
        )
        
        return float(np.clip(score, -1, 1))

# ═══════════════════════════════════════════════════════════════════
# Training Functions
# ═══════════════════════════════════════════════════════════════════

def create_ppo_lag_model(
    env: DummyVecEnv,
    config: TrainingConfig,
    hyperparams: Dict[str, Any]
) -> PPOLagrangian:
    """Create PPO-LAG model with hyperparameters"""
    
    # Create learning rate schedule
    lr_schedule = get_schedule_fn(hyperparams['learning_rate'])
    
    # Initialize model
    model = PPOLagrangian(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lr_schedule,
        cost_limit=config.cost_limit,
        lagrangian_pid_ki=config.lagrangian_pid_ki,
        lagrangian_pid_kp=config.lagrangian_pid_kp,
        lagrangian_pid_kd=config.lagrangian_pid_kd,
        **hyperparams
    )
    
    # Move to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # FIXED: Store device as an attribute on the model
    model.device = device
    
    return model

def create_trading_env(
    data: Dict,
    config: TrainingConfig,
    seed: int = 0
) -> EnhancedTradingEnv:
    """Create a configured trading environment"""
    
    env_config = TradingConfig(
        initial_balance=config.initial_balance,
        max_steps=config.max_steps_per_episode,
        debug=config.debug,
        live_mode=False,
        checkpoint_dir=config.checkpoint_dir
    )
    
    env = EnhancedTradingEnv(data, env_config)
    env._set_seeds(seed)
    
    return env

def optimize_hyperparameters(trial: optuna.Trial, config: TrainingConfig) -> float:
    """Optimize hyperparameters using Optuna"""
    
    trial_logger = logging.getLogger(f"trial_{trial.number}")
    trial_logger.info(f"Starting trial {trial.number}")
    
    # Sample hyperparameters
    hyperparams = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'n_steps': trial.suggest_categorical('n_steps', [1024, 2048, 4096]),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'n_epochs': trial.suggest_int('n_epochs', 5, 15),
        'gamma': trial.suggest_float('gamma', 0.95, 0.999),
        'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 0.99),
        'clip_range': trial.suggest_float('clip_range', 0.1, 0.3),
        'ent_coef': trial.suggest_float('ent_coef', 0.0, 0.01),
        'vf_coef': trial.suggest_float('vf_coef', 0.1, 1.0),
        'max_grad_norm': trial.suggest_float('max_grad_norm', 0.3, 1.0),
    }
    
    # Load data
    data = load_data("data/processed")
    
    # Create environments
    train_envs = []
    for i in range(config.num_envs):
        env = create_trading_env(data, config, seed=config.global_seed + i)
        train_envs.append(env)
        
    # Vectorize environments
    vec_env = DummyVecEnv([lambda env=env: env for env in train_envs])
    
    # Create model
    model = create_ppo_lag_model(vec_env, config, hyperparams)
    device = model.device
    
    # Setup callbacks
    callbacks = [
        DetailedLoggingCallback(config, trial_id=trial.number),
        TensorboardCallback(config),
        CheckpointCallback(config, trial_id=trial.number),
        OptunaPruningCallback(trial, config),
    ]
    
    CUSTOM_CALLBACKS = (
        DetailedLoggingCallback,
        TensorboardCallback,
        CheckpointCallback,
        OptunaPruningCallback,
    )

    for callback in callbacks:
        if isinstance(callback, CUSTOM_CALLBACKS):
            callback.init_callback(model, vec_env)
        else:
            callback.init_callback(model)
        callback.on_training_start(locals(), globals())

    # Train model
    try:
        total_timesteps = config.timesteps_per_trial
        n_envs = vec_env.num_envs
        n_steps = hyperparams['n_steps']
        batch_size = hyperparams['batch_size']
        n_epochs = hyperparams['n_epochs']
        
        # Buffer for storing rollouts
        obs_buffer = np.zeros((n_steps, n_envs) + vec_env.observation_space.shape, dtype=np.float32)
        actions_buffer = np.zeros((n_steps, n_envs) + vec_env.action_space.shape, dtype=np.float32)
        rewards_buffer = np.zeros((n_steps, n_envs), dtype=np.float32)
        costs_buffer = np.zeros((n_steps, n_envs), dtype=np.float32)
        dones_buffer = np.zeros((n_steps, n_envs), dtype=np.float32)
        values_buffer = np.zeros((n_steps, n_envs), dtype=np.float32)
        cost_values_buffer = np.zeros((n_steps, n_envs), dtype=np.float32)
        log_probs_buffer = np.zeros((n_steps, n_envs), dtype=np.float32)
        
        # Initialize observation
        obs = vec_env.reset()
        
        # Training loop
        num_updates = total_timesteps // (n_steps * n_envs)
        
        for update in range(num_updates):
            # Collect rollout
            for step in range(n_steps):
                obs_buffer[step] = obs
                
                # Get actions from policy
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).to(device)
                    mean, log_std, value, cost_value = model(obs_tensor)
                    
                    # Sample actions
                    std = torch.exp(log_std)
                    dist = torch.distributions.Normal(mean, std)
                    actions = dist.sample()
                    log_probs = dist.log_prob(actions).sum(dim=-1)
                    
                    # Store values
                    actions_buffer[step] = actions.cpu().numpy()
                    values_buffer[step] = value.squeeze(-1).cpu().numpy()
                    cost_values_buffer[step] = cost_value.squeeze(-1).cpu().numpy()
                    log_probs_buffer[step] = log_probs.cpu().numpy()
                
                # Step environment
                obs, rewards, dones, infos = vec_env.step(actions_buffer[step])
                model.num_timesteps += n_envs
                
                # Extract costs (risks) from infos
                costs = np.array([info.get('drawdown', 0) * 100 for info in infos])  # Use drawdown as cost
                
                rewards_buffer[step] = rewards
                costs_buffer[step] = costs
                dones_buffer[step] = dones
                
                # Call step callbacks
                for callback in callbacks:
                    callback.update_locals(locals())
                    if not callback.on_step():
                        return -float('inf')
            
            # Compute returns and advantages
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).to(device)
                _, _, next_value, next_cost_value = model(obs_tensor)
                next_value = next_value.squeeze(-1).cpu().numpy()
                next_cost_value = next_cost_value.squeeze(-1).cpu().numpy()
            
            # GAE for rewards
            returns = np.zeros_like(rewards_buffer)
            advantages = np.zeros_like(rewards_buffer)
            lastgaelam = 0
            
            for t in reversed(range(n_steps)):
                if t == n_steps - 1:
                    nextnonterminal = 1.0 - dones_buffer[t]
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buffer[t]
                    nextvalues = values_buffer[t + 1]
                
                delta = rewards_buffer[t] + hyperparams['gamma'] * nextvalues * nextnonterminal - values_buffer[t]
                advantages[t] = lastgaelam = delta + hyperparams['gamma'] * hyperparams['gae_lambda'] * nextnonterminal * lastgaelam
                returns[t] = advantages[t] + values_buffer[t]
            
            # GAE for costs
            cost_returns = np.zeros_like(costs_buffer)
            cost_advantages = np.zeros_like(costs_buffer)
            lastgaelam = 0
            
            for t in reversed(range(n_steps)):
                if t == n_steps - 1:
                    nextnonterminal = 1.0 - dones_buffer[t]
                    nextvalues = next_cost_value
                else:
                    nextnonterminal = 1.0 - dones_buffer[t]
                    nextvalues = cost_values_buffer[t + 1]
                
                delta = costs_buffer[t] + hyperparams['gamma'] * nextvalues * nextnonterminal - cost_values_buffer[t]
                cost_advantages[t] = lastgaelam = delta + hyperparams['gamma'] * hyperparams['gae_lambda'] * nextnonterminal * lastgaelam
                cost_returns[t] = cost_advantages[t] + cost_values_buffer[t]
            
            # Flatten the buffers
            obs_flat = obs_buffer.reshape(-1, *vec_env.observation_space.shape)
            actions_flat = actions_buffer.reshape(-1, *vec_env.action_space.shape)
            log_probs_flat = log_probs_buffer.reshape(-1)
            advantages_flat = advantages.reshape(-1)
            cost_advantages_flat = cost_advantages.reshape(-1)
            returns_flat = returns.reshape(-1)
            cost_returns_flat = cost_returns.reshape(-1)
            
            # Normalize advantages
            advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)
            cost_advantages_flat = (cost_advantages_flat - cost_advantages_flat.mean()) / (cost_advantages_flat.std() + 1e-8)
            
            # PPO update
            for epoch in range(n_epochs):
                # Create random mini-batches
                indices = np.random.permutation(len(obs_flat))
                
                for start_idx in range(0, len(obs_flat), batch_size):
                    end_idx = start_idx + batch_size
                    batch_indices = indices[start_idx:end_idx]
                    
                    # Get batch data
                    obs_batch = torch.FloatTensor(obs_flat[batch_indices]).to(device)
                    actions_batch = torch.FloatTensor(actions_flat[batch_indices]).to(device)
                    old_log_probs_batch = torch.FloatTensor(log_probs_flat[batch_indices]).to(device)
                    advantages_batch = torch.FloatTensor(advantages_flat[batch_indices]).to(device)
                    cost_advantages_batch = torch.FloatTensor(cost_advantages_flat[batch_indices]).to(device)
                    returns_batch = torch.FloatTensor(returns_flat[batch_indices]).to(device)
                    cost_returns_batch = torch.FloatTensor(cost_returns_flat[batch_indices]).to(device)
                    
                    # Forward pass
                    mean, log_std, values, cost_values = model(obs_batch)
                    
                    # Calculate new log probs
                    std = torch.exp(log_std)
                    dist = torch.distributions.Normal(mean, std)
                    new_log_probs = dist.log_prob(actions_batch).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1).mean()
                    
                    # Ratio for PPO
                    ratio = torch.exp(new_log_probs - old_log_probs_batch)
                    
                    # Clipped surrogate loss
                    surr1 = ratio * advantages_batch
                    surr2 = torch.clamp(ratio, 1.0 - hyperparams['clip_range'], 1.0 + hyperparams['clip_range']) * advantages_batch
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value loss
                    value_loss = ((values.squeeze(-1) - returns_batch) ** 2).mean()
                    
                    # Cost value loss
                    cost_value_loss = ((cost_values.squeeze(-1) - cost_returns_batch) ** 2).mean()
                    
                    # Lagrangian loss for cost constraint
                    lagrangian = model.get_lagrangian()
                    cost_surr1 = ratio * cost_advantages_batch
                    cost_surr2 = torch.clamp(ratio, 1.0 - hyperparams['clip_range'], 1.0 + hyperparams['clip_range']) * cost_advantages_batch
                    cost_loss = torch.max(cost_surr1, cost_surr2).mean()
                    
                    # Total loss
                    loss = (
                        policy_loss + 
                        hyperparams['vf_coef'] * value_loss + 
                        hyperparams['vf_coef'] * cost_value_loss - 
                        hyperparams['ent_coef'] * entropy +
                        lagrangian * cost_loss
                    )
                    
                    # Optimize
                    model.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), hyperparams['max_grad_norm'])
                    model.optimizer.step()
            
            # Update Lagrangian multiplier
            avg_cost = costs_buffer.mean()
            cost_error = avg_cost - config.cost_limit
            model.update_lagrangian(cost_error)
            
            # Log progress
            if update % 10 == 0:
                avg_reward = rewards_buffer.mean()
                trial_logger.info(
                    f"Update {update}/{num_updates}: "
                    f"Avg Reward: {avg_reward:.3f}, "
                    f"Avg Cost: {avg_cost:.3f}, "
                    f"Lagrangian: {model.get_lagrangian():.3f}"
                )
                
            # Call rollout end callbacks
            for callback in callbacks:
                callback.on_rollout_end()
                
    except TrialPruned:
        trial_logger.info(f"Trial {trial.number} pruned")
        raise
        
    except Exception as e:
        trial_logger.error(f"Trial {trial.number} failed: {e}")
        raise
    finally:
        # Call training end callbacks
        for callback in callbacks:
            callback.on_training_end()
        
    # Evaluate final performance
    eval_env = DummyVecEnv([lambda: create_trading_env(data, config)])
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=config.n_eval_episodes)
    
    # Get final metrics
    final_metrics = eval_env.envs[0].get_metrics()
    trial.set_user_attr('final_metrics', final_metrics)
    
    # Compute final score
    score = compute_trial_score(mean_reward, final_metrics, config)
    
    trial_logger.info(f"Trial {trial.number} completed with score: {score:.3f}")
    
    return score

def compute_trial_score(
    mean_reward: float,
    metrics: Dict[str, Any],
    config: TrainingConfig
) -> float:
    """Compute comprehensive trial score"""
    
    # Extract key metrics
    profit = (metrics['balance'] - config.initial_balance) / config.initial_balance
    win_rate = metrics.get('win_rate', 0)
    sharpe = metrics.get('sharpe_ratio', 0)
    max_dd = metrics.get('max_drawdown', 0)
    
    # Check if targets are met
    daily_profit = profit * 365 / config.max_steps_per_episode  # Annualized
    target_met = daily_profit >= config.target_daily_profit / config.initial_balance
    
    # Compute weighted score
    score = (
        0.35 * (profit * 10) +  # Emphasize profitability
        0.25 * win_rate +
        0.20 * (sharpe / 3.0) +
        0.15 * (1 - max_dd) +
        0.05 * float(target_met)
    )
    
    # Penalties
    if max_dd > config.max_drawdown_limit:
        score *= 0.5  # Heavy penalty for excessive drawdown
        
    if win_rate < config.min_win_rate:
        score *= 0.8  # Penalty for low win rate
        
    return float(np.clip(score, 0, 1))

def train_final_model(
    best_hyperparams: Dict[str, Any],
    config: TrainingConfig,
    checkpoint_path: Optional[str] = None
) -> PPOLagrangian:
    """Train final model with best hyperparameters"""
    
    logger = logging.getLogger("final_training")
    logger.info("Starting final model training")
    
    # Load data
    data = load_data("data/processed")
    
    # Create environment
    envs = []
    for i in range(config.num_envs):
        env = create_trading_env(data, config, seed=config.global_seed + i)
        envs.append(env)
        
    vec_env = DummyVecEnv([lambda env=env: env for env in envs])
    
    # Create or load model
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
            
        model = create_ppo_lag_model(vec_env, config, best_hyperparams)
        model.load_state_dict(checkpoint['model_state'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        start_timestep = checkpoint['timesteps']
        
        # Restore environment states
        for i, env in enumerate(envs):
            if 'env_state' in checkpoint:
                env.set_state(checkpoint['env_state'])
                
    else:
        model = create_ppo_lag_model(vec_env, config, best_hyperparams)
        start_timestep = 0
        
    # Setup callbacks
    callbacks = [
        DetailedLoggingCallback(config),
        TensorboardCallback(config),
        CheckpointCallback(config),
    ]
    
    # Initialize callbacks
    CUSTOM_CALLBACKS = (
        DetailedLoggingCallback,
        TensorboardCallback,
        CheckpointCallback,
        OptunaPruningCallback,
    )

    for callback in callbacks:
        if isinstance(callback, CUSTOM_CALLBACKS):
            callback.init_callback(model, vec_env)
        else:
            callback.init_callback(model)
        callback.on_training_start(locals(), globals())


    
    # Train
    remaining_timesteps = config.final_training_steps - start_timestep
    if remaining_timesteps > 0:
        logger.info(f"Training for {remaining_timesteps} timesteps")
        
        # Training parameters
        n_envs = vec_env.num_envs
        n_steps = best_hyperparams['n_steps']
        batch_size = best_hyperparams['batch_size']
        n_epochs = best_hyperparams['n_epochs']
        device = model.device
        
        # Buffer for storing rollouts
        obs_buffer = np.zeros((n_steps, n_envs) + vec_env.observation_space.shape, dtype=np.float32)
        actions_buffer = np.zeros((n_steps, n_envs) + vec_env.action_space.shape, dtype=np.float32)
        rewards_buffer = np.zeros((n_steps, n_envs), dtype=np.float32)
        costs_buffer = np.zeros((n_steps, n_envs), dtype=np.float32)
        dones_buffer = np.zeros((n_steps, n_envs), dtype=np.float32)
        values_buffer = np.zeros((n_steps, n_envs), dtype=np.float32)
        cost_values_buffer = np.zeros((n_steps, n_envs), dtype=np.float32)
        log_probs_buffer = np.zeros((n_steps, n_envs), dtype=np.float32)
        
        # Initialize observation
        obs = vec_env.reset()
        
        # Calculate number of updates
        num_updates = remaining_timesteps // (n_steps * n_envs)
        
        # Progress bar
        pbar = tqdm(total=remaining_timesteps, desc="Training", ncols=100)
        
        for update in range(num_updates):
            # Collect rollout
            for step in range(n_steps):
                obs_buffer[step] = obs
                
                # Get actions from policy
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).to(device)
                    mean, log_std, value, cost_value = model(obs_tensor)
                    
                    # Sample actions
                    std = torch.exp(log_std)
                    dist = torch.distributions.Normal(mean, std)
                    actions = dist.sample()
                    log_probs = dist.log_prob(actions).sum(dim=-1)
                    
                    # Store values
                    actions_buffer[step] = actions.cpu().numpy()
                    values_buffer[step] = value.squeeze(-1).cpu().numpy()
                    cost_values_buffer[step] = cost_value.squeeze(-1).cpu().numpy()
                    log_probs_buffer[step] = log_probs.cpu().numpy()
                
                # Step environment
                obs, rewards, dones, infos = vec_env.step(actions_buffer[step])
                model.num_timesteps += n_envs
                
                # Extract costs
                costs = np.array([info.get('drawdown', 0) * 100 for info in infos])
                
                rewards_buffer[step] = rewards
                costs_buffer[step] = costs
                dones_buffer[step] = dones
                
                # Update progress
                pbar.update(n_envs)
                
                # Call step callbacks
                for callback in callbacks:
                    callback.num_timesteps = start_timestep + update * n_steps * n_envs + step * n_envs
                    callback.update_locals(locals())
                    if not callback.on_step():
                        pbar.close()
                        return model
            
            # Compute returns and advantages (same as in optimize_hyperparameters)
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).to(device)
                _, _, next_value, next_cost_value = model(obs_tensor)
                next_value = next_value.squeeze(-1).cpu().numpy()
                next_cost_value = next_cost_value.squeeze(-1).cpu().numpy()
            
            # GAE calculation (same as before)
            returns = np.zeros_like(rewards_buffer)
            advantages = np.zeros_like(rewards_buffer)
            lastgaelam = 0
            
            for t in reversed(range(n_steps)):
                if t == n_steps - 1:
                    nextnonterminal = 1.0 - dones_buffer[t]
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buffer[t]
                    nextvalues = values_buffer[t + 1]
                
                delta = rewards_buffer[t] + best_hyperparams['gamma'] * nextvalues * nextnonterminal - values_buffer[t]
                advantages[t] = lastgaelam = delta + best_hyperparams['gamma'] * best_hyperparams['gae_lambda'] * nextnonterminal * lastgaelam
                returns[t] = advantages[t] + values_buffer[t]
            
            # Cost GAE
            cost_returns = np.zeros_like(costs_buffer)
            cost_advantages = np.zeros_like(costs_buffer)
            lastgaelam = 0
            
            for t in reversed(range(n_steps)):
                if t == n_steps - 1:
                    nextnonterminal = 1.0 - dones_buffer[t]
                    nextvalues = next_cost_value
                else:
                    nextnonterminal = 1.0 - dones_buffer[t]
                    nextvalues = cost_values_buffer[t + 1]
                
                delta = costs_buffer[t] + best_hyperparams['gamma'] * nextvalues * nextnonterminal - cost_values_buffer[t]
                cost_advantages[t] = lastgaelam = delta + best_hyperparams['gamma'] * best_hyperparams['gae_lambda'] * nextnonterminal * lastgaelam
                cost_returns[t] = cost_advantages[t] + cost_values_buffer[t]
            
            # Flatten buffers
            obs_flat = obs_buffer.reshape(-1, *vec_env.observation_space.shape)
            actions_flat = actions_buffer.reshape(-1, *vec_env.action_space.shape)
            log_probs_flat = log_probs_buffer.reshape(-1)
            advantages_flat = advantages.reshape(-1)
            cost_advantages_flat = cost_advantages.reshape(-1)
            returns_flat = returns.reshape(-1)
            cost_returns_flat = cost_returns.reshape(-1)
            
            # Normalize advantages
            advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)
            cost_advantages_flat = (cost_advantages_flat - cost_advantages_flat.mean()) / (cost_advantages_flat.std() + 1e-8)
            
            # PPO update (same as before)
            for epoch in range(n_epochs):
                indices = np.random.permutation(len(obs_flat))
                
                for start_idx in range(0, len(obs_flat), batch_size):
                    end_idx = start_idx + batch_size
                    batch_indices = indices[start_idx:end_idx]
                    
                    # Get batch data
                    obs_batch = torch.FloatTensor(obs_flat[batch_indices]).to(device)
                    actions_batch = torch.FloatTensor(actions_flat[batch_indices]).to(device)
                    old_log_probs_batch = torch.FloatTensor(log_probs_flat[batch_indices]).to(device)
                    advantages_batch = torch.FloatTensor(advantages_flat[batch_indices]).to(device)
                    cost_advantages_batch = torch.FloatTensor(cost_advantages_flat[batch_indices]).to(device)
                    returns_batch = torch.FloatTensor(returns_flat[batch_indices]).to(device)
                    cost_returns_batch = torch.FloatTensor(cost_returns_flat[batch_indices]).to(device)
                    
                    # Forward pass
                    mean, log_std, values, cost_values = model(obs_batch)
                    
                    # Calculate new log probs
                    std = torch.exp(log_std)
                    dist = torch.distributions.Normal(mean, std)
                    new_log_probs = dist.log_prob(actions_batch).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1).mean()
                    
                    # PPO loss calculation
                    ratio = torch.exp(new_log_probs - old_log_probs_batch)
                    surr1 = ratio * advantages_batch
                    surr2 = torch.clamp(ratio, 1.0 - best_hyperparams['clip_range'], 1.0 + best_hyperparams['clip_range']) * advantages_batch
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    value_loss = ((values.squeeze(-1) - returns_batch) ** 2).mean()
                    cost_value_loss = ((cost_values.squeeze(-1) - cost_returns_batch) ** 2).mean()
                    
                    # Lagrangian loss
                    lagrangian = model.get_lagrangian()
                    cost_surr1 = ratio * cost_advantages_batch
                    cost_surr2 = torch.clamp(ratio, 1.0 - best_hyperparams['clip_range'], 1.0 + best_hyperparams['clip_range']) * cost_advantages_batch
                    cost_loss = torch.max(cost_surr1, cost_surr2).mean()
                    
                    # Total loss
                    loss = (
                        policy_loss + 
                        best_hyperparams['vf_coef'] * value_loss + 
                        best_hyperparams['vf_coef'] * cost_value_loss - 
                        best_hyperparams['ent_coef'] * entropy +
                        lagrangian * cost_loss
                    )
                    
                    # Optimize
                    model.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), best_hyperparams['max_grad_norm'])
                    model.optimizer.step()
            
            # Update Lagrangian
            avg_cost = costs_buffer.mean()
            cost_error = avg_cost - config.cost_limit
            model.update_lagrangian(cost_error)
            
            # Log progress
            if update % 100 == 0:
                avg_reward = rewards_buffer.mean()
                logger.info(
                    f"Update {update}/{num_updates}: "
                    f"Avg Reward: {avg_reward:.3f}, "
                    f"Avg Cost: {avg_cost:.3f}, "
                    f"Lagrangian: {model.get_lagrangian():.3f}"
                )
            
            # Call rollout end callbacks
            for callback in callbacks:
                callback.on_rollout_end()
                
        pbar.close()
        
    else:
        logger.info("Training already completed")
    
    # Call training end callbacks
    for callback in callbacks:
        callback.on_training_end()
        
    return model

# ═══════════════════════════════════════════════════════════════════
# Main Training Script
# ═══════════════════════════════════════════════════════════════════

def main():
    """Main training entry point"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="PPO-LAG Trading Agent Training")
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--skip-optimization', action='store_true', 
                       help='Skip hyperparameter optimization')
    args = parser.parse_args()
    
    # Create configuration
    config = TrainingConfig(test_mode=args.test)
    
    # Setup main logger
    logger = logging.getLogger("main")
    logger.info(
        f"\n{'='*60}\n"
        f"PPO-LAG Trading Agent Training\n"
        f"{'='*60}\n"
        f"Mode: {'TEST' if config.test_mode else 'PRODUCTION'}\n"
        f"Target: €{config.target_daily_profit}/day\n"
        f"Max Drawdown: {config.max_drawdown_limit:.1%}\n"
        f"{'='*60}\n"
    )
    
    # Set global seed
    set_global_seed(config.global_seed)
    
    # Hyperparameter optimization
    if not args.skip_optimization:
        logger.info("Starting hyperparameter optimization...")
        
        # Create Optuna study
        study = optuna.create_study(
            study_name="ppo_lag_trading",
            storage=f"sqlite:///{config.log_dir}/optuna.db",
            direction="maximize",
            load_if_exists=True,
            pruner=MedianPruner(
                n_startup_trials=config.pruner_startup_trials,
                n_warmup_steps=config.pruner_warmup_steps,
                interval_steps=config.pruner_interval_steps
            )
        )
        
        # Optimize
        study.optimize(
            lambda trial: optimize_hyperparameters(trial, config),
            n_trials=config.n_trials,
            n_jobs=1,  # PPO doesn't parallelize well
            show_progress_bar=True
        )
        
        # Get best hyperparameters
        best_trial = study.best_trial
        best_hyperparams = best_trial.params
        
        logger.info(
            f"\nBest trial: {best_trial.number}\n"
            f"Score: {best_trial.value:.3f}\n"
            f"Hyperparameters: {best_hyperparams}\n"
        )
        
    else:
        # Use default hyperparameters
        best_hyperparams = {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
        }
        
    # Train final model
    logger.info("Training final model with best hyperparameters...")
    
    final_model = train_final_model(
        best_hyperparams,
        config,
        checkpoint_path=args.resume
    )
    
    # Save final model
    model_path = os.path.join(config.model_dir, "ppo_lag_final.pkl")
    torch.save(final_model.state_dict(), model_path)
    logger.info(f"Final model saved to: {model_path}")
    
    # Final evaluation
    logger.info("Running final evaluation...")
    
    data = load_data("data/processed")
    eval_env = create_trading_env(data, config)
    
    # Run comprehensive evaluation
    total_episodes = 50 if not config.test_mode else 10
    episode_metrics = []
    
    for ep in tqdm(range(total_episodes), desc="Final Evaluation"):
        obs = eval_env.reset()
        done = False
        
        while not done:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(final_model.device)
                mean, _, _, _ = final_model(obs_tensor)
                action = mean.squeeze(0).cpu().numpy()
                
            obs, reward, done, info = eval_env.step(action)
            
        metrics = eval_env.get_metrics()
        episode_metrics.append(metrics)
        
    # Aggregate results
    avg_balance = np.mean([m['balance'] for m in episode_metrics])
    avg_profit = avg_balance - config.initial_balance
    avg_win_rate = np.mean([m.get('win_rate', 0) for m in episode_metrics])
    avg_sharpe = np.mean([m.get('sharpe_ratio', 0) for m in episode_metrics])
    avg_max_dd = np.mean([m.get('max_drawdown', 0) for m in episode_metrics])
    
    logger.info(
        f"\n{'='*60}\n"
        f"FINAL EVALUATION RESULTS\n"
        f"{'='*60}\n"
        f"Average Final Balance: €{avg_balance:,.2f}\n"
        f"Average Profit: €{avg_profit:,.2f}\n"
        f"Average Win Rate: {avg_win_rate:.2%}\n"
        f"Average Sharpe Ratio: {avg_sharpe:.3f}\n"
        f"Average Max Drawdown: {avg_max_dd:.2%}\n"
        f"Daily Profit Target: €{config.target_daily_profit}\n"
        f"Achieved Daily: €{avg_profit * 365 / config.max_steps_per_episode:.2f}\n"
        f"{'='*60}\n"
    )
    
    # Generate final report
    generate_training_report(config, episode_metrics)
    
    logger.info("Training completed successfully!")

def generate_training_report(config: TrainingConfig, episode_metrics: List[Dict[str, Any]]):
    """Generate comprehensive training report"""
    
    report_path = os.path.join(config.log_dir, "training_report.txt")
    
    with open(report_path, 'w') as f:
        f.write(
            f"PPO-LAG Trading Agent Training Report\n"
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'='*80}\n\n"
            f"Configuration:\n"
            f"- Mode: {'TEST' if config.test_mode else 'PRODUCTION'}\n"
            f"- Initial Balance: €{config.initial_balance:,.2f}\n"
            f"- Target Daily Profit: €{config.target_daily_profit}\n"
            f"- Max Drawdown Limit: {config.max_drawdown_limit:.1%}\n"
            f"- Cost Limit: {config.cost_limit}\n\n"
            f"Training Details:\n"
            f"- Total Timesteps: {config.final_training_steps:,}\n"
            f"- Number of Trials: {config.n_trials}\n"
            f"- Episodes per Trial: {config.timesteps_per_trial // config.max_steps_per_episode}\n\n"
            f"Final Performance:\n"
        )
        
        # Add detailed metrics
        for i, metrics in enumerate(episode_metrics[:10]):  # First 10 episodes
            f.write(
                f"\nEpisode {i+1}:\n"
                f"  Balance: €{metrics['balance']:,.2f}\n"
                f"  Win Rate: {metrics.get('win_rate', 0):.2%}\n"
                f"  Sharpe: {metrics.get('sharpe_ratio', 0):.3f}\n"
                f"  Max DD: {metrics.get('max_drawdown', 0):.2%}\n"
            )
            
    # Also generate a JSON report for programmatic access
    json_report = {
        'config': config.__dict__,
        'timestamp': datetime.now().isoformat(),
        'episode_metrics': episode_metrics,
        'summary': {
            'avg_profit': np.mean([m['balance'] - config.initial_balance for m in episode_metrics]),
            'avg_win_rate': np.mean([m.get('win_rate', 0) for m in episode_metrics]),
            'avg_sharpe': np.mean([m.get('sharpe_ratio', 0) for m in episode_metrics]),
            'success_rate': sum(1 for m in episode_metrics if m['balance'] > config.initial_balance) / len(episode_metrics)
        }
    }
    
    with open(os.path.join(config.log_dir, 'training_report.json'), 'w') as f:
        json.dump(json_report, f, indent=2, default=str)

def set_global_seed(seed: int):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    main()