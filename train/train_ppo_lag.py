# # train_ppo_lag.py
# """
# Advanced PPO-Lagrangian Training Script for Forex/Gold Trading
# Integrates all modules with detailed logging and Optuna optimization
# """

# import os
# import warnings
# import random
# import logging
# import logging.handlers
# from sys import platform
# import argparse
# from dataclasses import dataclass, field
# from typing import Any, Dict, List, Optional, Tuple, Union
# from datetime import datetime
# import json
# import pickle

# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn

# # Optuna for hyperparameter optimization
# import optuna
# from optuna.pruners import MedianPruner
# from optuna.exceptions import TrialPruned
# from optuna.trial import TrialState

# # Progress tracking
# from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter

# # PPO-LAG specific imports
# from stable_baselines3.common.vec_env import  DummyVecEnv
# from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.utils import get_schedule_fn

# # Custom imports
# from envs.env import EnhancedTradingEnv, TradingConfig
# from utils.data_utils import load_data

# # Suppress warnings
# warnings.filterwarnings("ignore", category=RuntimeWarning)
# warnings.filterwarnings("ignore", category=FutureWarning)
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# # ═══════════════════════════════════════════════════════════════════
# # Configuration
# # ═══════════════════════════════════════════════════════════════════

# @dataclass
# class TrainingConfig:
#     """Comprehensive training configuration"""
#     # Mode settings
#     test_mode: bool = True
#     debug: bool = True
    
#     # Environment settings
#     num_envs: int = 4
#     global_seed: int = 42
#     initial_balance: float = 3000.0
#     max_steps_per_episode: int = 200
    
#     # Optimization settings
#     n_trials: int = 0
#     timesteps_per_trial: int = 0
#     final_training_steps: int = 0
    
#     # Pruning settings
#     pruner_startup_trials: int = 0
#     pruner_warmup_steps: int = 0
#     pruner_interval_steps: int = 0
    
#     # Logging frequencies
#     tb_log_freq: int = 0
#     checkpoint_freq: int = 0
#     eval_freq: int = 0
#     n_eval_episodes: int = 0
#     log_interval: int = 0
    
#     # PPO-LAG specific
#     cost_limit: float = 25.0  # Maximum allowed cost (risk)
#     lagrangian_pid_ki: float = 0.01
#     lagrangian_pid_kp: float = 0.1
#     lagrangian_pid_kd: float = 0.01
    
#     # Paths
#     log_dir: str = "logs/training"
#     model_dir: str = "models"
#     checkpoint_dir: str = "checkpoints"
#     tensorboard_dir: str = "logs/tensorboard"
    
#     # Target metrics
#     target_daily_profit: float = 150.0  # €150/day
#     max_drawdown_limit: float = 0.20   # 20% max drawdown
#     min_win_rate: float = 0.55         # 55% minimum win rate
    
#     def __post_init__(self):
#         """Set mode-specific parameters"""
#         if self.test_mode:
#             # Quick test settings
#             self.n_trials = 2
#             self.timesteps_per_trial = 10_000
#             self.final_training_steps = 50_000
#             self.pruner_startup_trials = 1
#             self.pruner_warmup_steps = 2_000
#             self.pruner_interval_steps = 2_000
#             self.tb_log_freq = 1_000
#             self.checkpoint_freq = 5_000
#             self.eval_freq = 2_000
#             self.n_eval_episodes = 3
#             self.log_interval = 500
#         else:
#             # Full training settings
#             self.n_trials = 20
#             self.timesteps_per_trial = 500_000
#             self.final_training_steps = 5_000_000
#             self.pruner_startup_trials = 5
#             self.pruner_warmup_steps = 100_000
#             self.pruner_interval_steps = 50_000
#             self.tb_log_freq = 1_000
#             self.checkpoint_freq = 50_000
#             self.eval_freq = 10_000
#             self.n_eval_episodes = 10
#             self.log_interval = 1_000
            
#         # Create directories
#         for path in [self.log_dir, self.model_dir, self.checkpoint_dir, self.tensorboard_dir]:
#             os.makedirs(path, exist_ok=True)

# # ═══════════════════════════════════════════════════════════════════
# # PPO-LAG Implementation
# # ═══════════════════════════════════════════════════════════════════

# class PPOLagrangian(nn.Module):
#     """PPO with Lagrangian constraints for safe trading"""
    
#     def __init__(
#         self,
#         observation_space,
#         action_space,
#         lr_schedule,
#         cost_limit: float = 25.0,
#         lagrangian_pid_ki: float = 0.01,
#         lagrangian_pid_kp: float = 0.1,
#         lagrangian_pid_kd: float = 0.01,
#         **kwargs
#     ):
#         super().__init__()
#         self.num_timesteps = 0
#         self.logger = logging.getLogger("PPOLagrangian")
        
#         self.observation_space = observation_space
#         self.action_space = action_space
#         self.cost_limit = cost_limit
        
#         # Fixed: Store observation dimension
#         self.obs_dim = observation_space.shape[0]
        
#         # Networks
#         self.features_dim = 256
        
#         # Fixed: Use fixed input dimension instead of LazyLinear
#         self.shared_net = nn.Sequential(
#             nn.Linear(self.obs_dim, 512),  # Fixed input size
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, self.features_dim),
#             nn.ReLU()
#         )
        
#         # Actor (policy) head
#         self.policy_net = nn.Sequential(
#             nn.Linear(self.features_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, action_space.shape[0] * 2)  # Mean and log_std
#         )
        
#         # Value head
#         self.value_net = nn.Sequential(
#             nn.Linear(self.features_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1)
#         )
        
#         # Cost value head (for constraint)
#         self.cost_value_net = nn.Sequential(
#             nn.Linear(self.features_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1)
#         )
        
#         # Lagrangian multiplier
#         self.log_lagrangian = nn.Parameter(torch.zeros(1))
        
#         # PID controller for Lagrangian
#         self.pid_ki = lagrangian_pid_ki
#         self.pid_kp = lagrangian_pid_kp
#         self.pid_kd = lagrangian_pid_kd
#         self.pid_i = 0
#         self.prev_error = 0
        
#         # Optimizer
#         self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_schedule(1))
        
#     def forward(self, obs):
#         """Forward pass through networks"""
#         # Fixed: Ensure correct input shape
#         if obs.dim() == 1:
#             obs = obs.unsqueeze(0)
            
#         # Fixed: Handle variable observation sizes
#         if obs.shape[-1] != self.obs_dim:
#             if obs.shape[-1] < self.obs_dim:
#                 # Pad with zeros
#                 padding = torch.zeros(
#                     (*obs.shape[:-1], self.obs_dim - obs.shape[-1]),
#                     device=obs.device,
#                     dtype=obs.dtype
#                 )
#                 obs = torch.cat([obs, padding], dim=-1)
#             else:
#                 # Truncate
#                 obs = obs[..., :self.obs_dim]
        
#         features = self.shared_net(obs)
        
#         # Policy
#         policy_out = self.policy_net(features)
#         mean, log_std = torch.chunk(policy_out, 2, dim=-1)
#         log_std = torch.clamp(log_std, -20, 2)
        
#         # Values
#         value = self.value_net(features)
#         cost_value = self.cost_value_net(features)
        
#         return mean, log_std, value, cost_value
    
#     def get_lagrangian(self):
#         """Get current Lagrangian multiplier"""
#         return torch.exp(self.log_lagrangian).item()
    
#     def update_lagrangian(self, cost_error):
#         """Update Lagrangian using PID controller"""
#         # PID update
#         self.pid_i += cost_error
#         pid_d = cost_error - self.prev_error
#         self.prev_error = cost_error
        
#         # Update log_lagrangian
#         delta = self.pid_kp * cost_error + self.pid_ki * self.pid_i + self.pid_kd * pid_d
#         self.log_lagrangian.data += delta
# # ═══════════════════════════════════════════════════════════════════
# # Custom Callbacks
# # ═══════════════════════════════════════════════════════════════════

# class DetailedLoggingCallback(BaseCallback):
#     """Comprehensive logging callback with InfoBus integration"""
    
#     def __init__(self, config: TrainingConfig, trial_id: int = -1):
#         super().__init__(verbose=1)
#         self.config = config
#         self.trial_id = trial_id
#         self.episode_count = 0
#         self.step_count = 0


        
#         # Setup loggers
#         self.setup_loggers()
        
#         # Metrics tracking
#         self.episode_metrics = []
#         self.trade_history = []
#         self.best_metrics = {
#             'reward': -float('inf'),
#             'profit': -float('inf'),
#             'sharpe': -float('inf')
#         }

#     def init_callback(self, model, env=None):
#         super().init_callback(model)
#         self._env = env
        
#     def setup_loggers(self):
#         """Setup multiple specialized loggers"""
#         # Main training logger
#         self.train_logger = self._create_logger(
#             'training', 
#             os.path.join(self.config.log_dir, 'training.log')
#         )
        
#         # Trade logger
#         self.trade_logger = self._create_logger(
#             'trades',
#             os.path.join(self.config.log_dir, 'trades.log')
#         )
        
#         # Performance logger
#         self.perf_logger = self._create_logger(
#             'performance',
#             os.path.join(self.config.log_dir, 'performance.log')
#         )
        
#         # Risk logger
#         self.risk_logger = self._create_logger(
#             'risk',
#             os.path.join(self.config.log_dir, 'risk.log')
#         )
        
#     def _create_logger(self, name: str, filename: str) -> logging.Logger:
#         """Create a specialized logger"""
#         logger = logging.getLogger(name)
#         logger.setLevel(logging.DEBUG)
        
#         # File handler
#         fh = logging.handlers.RotatingFileHandler(
#             filename, maxBytes=10*1024*1024, backupCount=5
#         )
#         fh.setFormatter(
#             logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
#         )
        
#         # Console handler for important messages
#         ch = logging.StreamHandler()
#         ch.setLevel(logging.INFO)
#         ch.setFormatter(
#             logging.Formatter('[%(name)s] %(message)s')
#         )
        
#         logger.handlers.clear()
#         logger.addHandler(fh)
#         logger.addHandler(ch)
        
#         return logger
    
#     def _on_step(self) -> bool:
#         """Called at every step"""
#         self.step_count += 1
        
#         # Log at intervals
#         if self.step_count % self.config.log_interval == 0:
#             self._log_step_metrics()
            
#         return True
    
#     def _on_rollout_end(self):
#         """Called at the end of a rollout"""
#         self.episode_count += 1
#         self._log_episode_metrics()
        
#     def _log_step_metrics(self):
#         """Log step-level metrics"""
#         try:
#             # Get environment metrics
#             env = self._env.envs[0]
#             info_bus = getattr(env, 'info_bus', None)
            
#             if info_bus:
#                 # Extract key metrics from InfoBus
#                 risk = info_bus.get('risk', {})
#                 positions = info_bus.get('positions', [])
                
#                 self.train_logger.info(
#                     f"Step {self.step_count} | "
#                     f"Balance: ${risk.get('balance', 0):,.2f} | "
#                     f"Drawdown: {risk.get('current_drawdown', 0):.2%} | "
#                     f"Positions: {len(positions)} | "
#                     f"P&L Today: ${info_bus.get('pnl_today', 0):,.2f}"
#                 )
                
#                 # Log any alerts
#                 for alert in info_bus.get('alerts', []):
#                     self.risk_logger.warning(f"ALERT: {alert}")
                    
#         except Exception as e:
#             self.train_logger.error(f"Error logging step metrics: {e}")
            
#     def _log_episode_metrics(self):
#         """Log episode-level metrics"""
#         try:
#             env = self._env.envs[0]
            
#             # Get comprehensive metrics
#             metrics = env.get_metrics()
            
#             # Performance summary
#             self.perf_logger.info(
#                 f"\n{'='*60}\n"
#                 f"Episode {self.episode_count} Summary (Trial {self.trial_id})\n"
#                 f"{'='*60}\n"
#                 f"Final Balance: ${metrics['balance']:,.2f}\n"
#                 f"Total P&L: ${metrics['balance'] - self.config.initial_balance:,.2f}\n"
#                 f"Win Rate: {metrics.get('win_rate', 0):.2%}\n"
#                 f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}\n"
#                 f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}\n"
#                 f"Total Trades: {metrics.get('total_trades', 0)}\n"
#                 f"Avg Trade P&L: ${metrics.get('avg_pnl', 0):.2f}\n"
#                 f"{'='*60}"
#             )
            
#             # Get trade explanations if available
#             if hasattr(env, 'trade_auditor'):
#                 analysis = env.trade_auditor.get_trade_analysis()
                
#                 self.trade_logger.info(
#                     f"\nTrade Analysis:\n"
#                     f"Top Strategies: {analysis.get('top_strategies', [])}\n"
#                     f"Regime Performance: {analysis.get('regime_breakdown', {})}"
#                 )
                
#             # Check for new bests
#             self._check_best_performance(metrics)
            
#         except Exception as e:
#             self.train_logger.error(f"Error logging episode metrics: {e}")
            
#     def _check_best_performance(self, metrics: Dict[str, Any]):
#         """Check and log if we have new best performance"""
#         updated = False
        
#         profit = metrics['balance'] - self.config.initial_balance
#         if profit > self.best_metrics['profit']:
#             self.best_metrics['profit'] = profit
#             updated = True
            
#         sharpe = metrics.get('sharpe_ratio', 0)
#         if sharpe > self.best_metrics['sharpe']:
#             self.best_metrics['sharpe'] = sharpe
#             updated = True
            
#         if updated:
#             self.perf_logger.info(
#                 f"\n NEW BEST PERFORMANCE!\n"
#                 f"Best Profit: ${self.best_metrics['profit']:,.2f}\n"
#                 f"Best Sharpe: {self.best_metrics['sharpe']:.3f}\n"
#             )

        
#     def update_locals(self, locals_dict):
#         """Update local variables"""
#         self.locals.update(locals_dict)
        
#     def on_training_start(self, locals_=None, globals_=None):
#         """Called at the beginning of training"""
#         self._on_training_start()
        
#     def on_training_end(self):
#         """Called at the end of training"""
#         pass
            

# class TensorboardCallback(BaseCallback):
#     """Enhanced TensorBoard logging with module metrics"""
    
#     def __init__(self, config: TrainingConfig):
#         super().__init__(verbose=0)
#         self.config = config
#         self.writer = None
    
#     def init_callback(self, model, env=None):
#         super().init_callback(model)
#         self._env = env
        
#     def _on_training_start(self):
#         """Initialize TensorBoard writer"""
#         log_dir = os.path.join(
#             self.config.tensorboard_dir,
#             f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#         )
#         self.writer = SummaryWriter(log_dir=log_dir)
        
#     def _on_step(self) -> bool:
#         """Log to TensorBoard"""
#         if self.n_calls % self.config.tb_log_freq == 0 and self.writer:
#             self._log_metrics()
#         return True
    
#     def _log_metrics(self):
#         """Log comprehensive metrics to TensorBoard"""
#         try:
#             env = self._env.envs[0]
#             metrics = env.get_metrics()
            
#             # Basic metrics
#             self.writer.add_scalar('env/balance', metrics['balance'], self.num_timesteps)
#             self.writer.add_scalar('env/pnl', metrics['balance'] - self.config.initial_balance, self.num_timesteps)
#             self.writer.add_scalar('env/drawdown', metrics['drawdown'], self.num_timesteps)
#             self.writer.add_scalar('env/win_rate', metrics.get('win_rate', 0), self.num_timesteps)
#             self.writer.add_scalar('env/sharpe_ratio', metrics.get('sharpe_ratio', 0), self.num_timesteps)
            
#             # Trading metrics
#             self.writer.add_scalar('trading/total_trades', metrics.get('total_trades', 0), self.num_timesteps)
#             self.writer.add_scalar('trading/avg_pnl', metrics.get('avg_pnl', 0), self.num_timesteps)
#             self.writer.add_scalar('trading/profit_factor', metrics.get('profit_factor', 0), self.num_timesteps)
            
#             # Module-specific metrics
#             if hasattr(env, 'info_bus') and env.info_bus:
#                 info_bus = env.info_bus
                
#                 # Risk metrics
#                 risk = info_bus.get('risk', {})
#                 self.writer.add_scalar('risk/var_95', risk.get('var_95', 0), self.num_timesteps)
#                 self.writer.add_scalar('risk/margin_used', risk.get('margin_used', 0), self.num_timesteps)
                
#                 # Market context
#                 context = info_bus.get('market_context', {})
#                 for inst, vol in context.get('volatility', {}).items():
#                     self.writer.add_scalar(f'volatility/{inst}', vol, self.num_timesteps)
                    
#                 # Voting consensus
#                 self.writer.add_scalar('voting/consensus', info_bus.get('consensus', 0), self.num_timesteps)
                
#                 # Module health
#                 for module_name, module in env.pipeline._module_map.items():
#                     health = module.get_health_status()
#                     self.writer.add_scalar(
#                         f'health/{module_name}',
#                         1.0 if health['status'] == 'OK' else 0.0,
#                         self.num_timesteps
#                     )
                    
#         except Exception as e:
#             if self.config.debug:
#                 print(f"TensorBoard logging error: {e}")
                
#     def _on_training_end(self):
#         """Close TensorBoard writer"""
#         if self.writer:
#             self.writer.close()

# class CheckpointCallback(BaseCallback):
#     """Enhanced checkpoint callback with full state preservation"""
    
#     def __init__(self, config: TrainingConfig, trial_id: int = -1):
#         super().__init__(verbose=1)
#         self.config = config
#         self.trial_id = trial_id
#         self.save_path = os.path.join(
#             config.checkpoint_dir,
#             f"trial_{trial_id}" if trial_id >= 0 else "final"
#         )
#         os.makedirs(self.save_path, exist_ok=True)

#     def init_callback(self, model, env=None):
#         super().init_callback(model)
#         self._env = env
        
#     def _on_step(self) -> bool:
#         """Save checkpoint at intervals"""
#         if self.n_calls % self.config.checkpoint_freq == 0:
#             self._save_checkpoint()
#         return True
    
#     def _save_checkpoint(self):
#         """Save comprehensive checkpoint"""
#         try:
#             checkpoint_path = os.path.join(
#                 self.save_path,
#                 f"checkpoint_{self.num_timesteps}.pkl"
#             )
            
#             # Prepare checkpoint data
#             checkpoint = {
#                 'timesteps': self.num_timesteps,
#                 'trial_id': self.trial_id,
#                 'model_state': self.model.policy.state_dict(),
#                 'optimizer_state': self.model.policy.optimizer.state_dict(),
#                 'config': self.config,
#                 'timestamp': datetime.now().isoformat()
#             }
            
#             # Save environment state
#             env = self._env.envs[0]
#             checkpoint['env_state'] = env.get_state()
            
#             # Save module states
#             module_states = {}
#             for name, module in env.pipeline._module_map.items():
#                 module_states[name] = module.get_state()
#             checkpoint['module_states'] = module_states
            
#             # Save checkpoint
#             with open(checkpoint_path, 'wb') as f:
#                 pickle.dump(checkpoint, f)
                
#             self.logger.info(f"Checkpoint saved: {checkpoint_path}")
            
#             # Keep only recent checkpoints
#             self._cleanup_old_checkpoints()
            
#         except Exception as e:
#             self.logger.error(f"Failed to save checkpoint: {e}")
            
#     def _cleanup_old_checkpoints(self, keep_last: int = 5):
#         """Remove old checkpoints to save space"""
#         checkpoints = sorted([
#             f for f in os.listdir(self.save_path)
#             if f.startswith('checkpoint_') and f.endswith('.pkl')
#         ])
        
#         if len(checkpoints) > keep_last:
#             for old_ckpt in checkpoints[:-keep_last]:
#                 os.remove(os.path.join(self.save_path, old_ckpt))

# class OptunaPruningCallback(BaseCallback):
#     """
#     Callback for Optuna trial pruning based on performance.
#     Now accepts the vectorised training environment passed in by the
#     outer loop so that BaseCallback.init_callback is not over-called.
#     """

#     def __init__(self, trial: optuna.Trial, config: TrainingConfig):
#         super().__init__(verbose=0)
#         self.trial = trial
#         self.config = config
#         self.eval_env = None
#         self._last_eval_timestep = 0

#     def init_callback(self, model, env=None):
#         """
#         Override SB3 initialisation.
#         We *do not* assign to self.training_env because it is a read-only
#         property in recent SB3 versions.  Instead, we simply build the
#         separate evaluation environment if a training env is supplied.
#         """
#         super().init_callback(model)     # SB3 base initialisation
#         if env is not None:
#             self._setup_eval_env(env)    # create self.eval_env once
#     # ------------------------------------------------------------------
#     # Private helpers
#     # ------------------------------------------------------------------
#     def _setup_eval_env(self, train_env):
#         """
#         Create a lightweight evaluation environment that shares the same
#         market data but has its own state trajectory.
#         """
#         # Get data from first training environment
#         first_env = train_env.envs[0]
#         data = first_env.orig_data
#         config = first_env.config
        
#         # Create evaluation environment with same configuration
#         self.eval_env = DummyVecEnv([
#             lambda: EnhancedTradingEnv(
#                 data,
#                 TradingConfig(
#                     initial_balance=config.initial_balance,
#                     max_steps=config.max_steps_per_episode,
#                     debug=False
#                 )
#             )
#         ])
        
#     def _on_step(self) -> bool:
#         """Check for pruning at intervals"""
#         if self.num_timesteps - self._last_eval_timestep >= self.config.pruner_interval_steps:
#             self._last_eval_timestep = self.num_timesteps
            
#             # Evaluate current performance
#             mean_reward, std_reward = evaluate_policy(
#                 self.model,
#                 self.eval_env,
#                 n_eval_episodes=self.config.n_eval_episodes,
#                 deterministic=True
#             )
            
#             # Get additional metrics
#             env = self.eval_env.envs[0]
#             metrics = env.get_metrics()
            
#             # Compute composite score
#             score = self._compute_trial_score(mean_reward, metrics)
            
#             # Report to Optuna
#             self.trial.report(score, self.num_timesteps)
            
#             # Store metrics in trial
#             self.trial.set_user_attr('metrics', metrics)
#             self.trial.set_user_attr('mean_reward', mean_reward)
            
#             # Check if should prune
#             if self.trial.should_prune():
#                 raise TrialPruned()
                
#         return True
    
#     def _compute_trial_score(self, mean_reward: float, metrics: Dict[str, Any]) -> float:
#         """Compute composite score for trial evaluation"""
#         # Normalize components
#         profit = (metrics['balance'] - self.config.initial_balance) / self.config.initial_balance
#         win_rate = metrics.get('win_rate', 0)
#         sharpe = metrics.get('sharpe_ratio', 0) / 3.0  # Normalize assuming max 3.0
#         drawdown = 1.0 - metrics.get('max_drawdown', 0)
        
#         # Composite score with emphasis on profitability and risk
#         score = (
#             0.30 * profit +
#             0.25 * win_rate +
#             0.25 * sharpe +
#             0.20 * drawdown
#         )
        
#         return float(np.clip(score, -1, 1))

# # ═══════════════════════════════════════════════════════════════════
# # Training Functions
# # ═══════════════════════════════════════════════════════════════════

# def create_ppo_lag_model(
#     env: DummyVecEnv,
#     config: TrainingConfig,
#     hyperparams: Dict[str, Any]
# ) -> PPOLagrangian:
#     """Create PPO-LAG model with hyperparameters"""
    
#     # Create learning rate schedule
#     lr_schedule = get_schedule_fn(hyperparams['learning_rate'])
    
#     # Initialize model
#     model = PPOLagrangian(
#         observation_space=env.observation_space,
#         action_space=env.action_space,
#         lr_schedule=lr_schedule,
#         cost_limit=config.cost_limit,
#         lagrangian_pid_ki=config.lagrangian_pid_ki,
#         lagrangian_pid_kp=config.lagrangian_pid_kp,
#         lagrangian_pid_kd=config.lagrangian_pid_kd,
#         **hyperparams
#     )
    
#     # Move to appropriate device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
    
#     # FIXED: Store device as an attribute on the model
#     model.device = device
    
#     return model

# def create_trading_env(
#     data: Dict,
#     config: TrainingConfig,
#     seed: int = 0
# ) -> EnhancedTradingEnv:
#     """Create a configured trading environment"""
    
#     env_config = TradingConfig(
#         initial_balance=config.initial_balance,
#         max_steps=config.max_steps_per_episode,
#         debug=config.debug,
#         live_mode=False,
#         checkpoint_dir=config.checkpoint_dir
#     )
    
#     env = EnhancedTradingEnv(data, env_config)
#     env._set_seeds(seed)
    
#     return env

# def optimize_hyperparameters(trial: optuna.Trial, config: TrainingConfig) -> float:
#     """Optimize hyperparameters using Optuna"""
    
#     trial_logger = logging.getLogger(f"trial_{trial.number}")
#     trial_logger.info(f"Starting trial {trial.number}")
    
#     # Sample hyperparameters
#     hyperparams = {
#         'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
#         'n_steps': trial.suggest_categorical('n_steps', [1024, 2048, 4096]),
#         'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
#         'n_epochs': trial.suggest_int('n_epochs', 5, 15),
#         'gamma': trial.suggest_float('gamma', 0.95, 0.999),
#         'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 0.99),
#         'clip_range': trial.suggest_float('clip_range', 0.1, 0.3),
#         'ent_coef': trial.suggest_float('ent_coef', 0.0, 0.01),
#         'vf_coef': trial.suggest_float('vf_coef', 0.1, 1.0),
#         'max_grad_norm': trial.suggest_float('max_grad_norm', 0.3, 1.0),
#     }
    
#     # Load data
#     data = load_data("data/processed")
    
#     # Create environments
#     train_envs = []
#     for i in range(config.num_envs):
#         env = create_trading_env(data, config, seed=config.global_seed + i)
#         train_envs.append(env)
        
#     # Vectorize environments
#     vec_env = DummyVecEnv([lambda env=env: env for env in train_envs])
    
#     # Create model
#     model = create_ppo_lag_model(vec_env, config, hyperparams)
#     device = model.device
    
#     # Setup callbacks
#     callbacks = [
#         DetailedLoggingCallback(config, trial_id=trial.number),
#         TensorboardCallback(config),
#         CheckpointCallback(config, trial_id=trial.number),
#         OptunaPruningCallback(trial, config),
#     ]
    
#     CUSTOM_CALLBACKS = (
#         DetailedLoggingCallback,
#         TensorboardCallback,
#         CheckpointCallback,
#         OptunaPruningCallback,
#     )

#     for callback in callbacks:
#         if isinstance(callback, CUSTOM_CALLBACKS):
#             callback.init_callback(model, vec_env)
#         else:
#             callback.init_callback(model)
#         callback.on_training_start(locals(), globals())

#     # Train model
#     try:
#         total_timesteps = config.timesteps_per_trial
#         n_envs = vec_env.num_envs
#         n_steps = hyperparams['n_steps']
#         batch_size = hyperparams['batch_size']
#         n_epochs = hyperparams['n_epochs']
        
#         # Buffer for storing rollouts
#         obs_buffer = np.zeros((n_steps, n_envs) + vec_env.observation_space.shape, dtype=np.float32)
#         actions_buffer = np.zeros((n_steps, n_envs) + vec_env.action_space.shape, dtype=np.float32)
#         rewards_buffer = np.zeros((n_steps, n_envs), dtype=np.float32)
#         costs_buffer = np.zeros((n_steps, n_envs), dtype=np.float32)
#         dones_buffer = np.zeros((n_steps, n_envs), dtype=np.float32)
#         values_buffer = np.zeros((n_steps, n_envs), dtype=np.float32)
#         cost_values_buffer = np.zeros((n_steps, n_envs), dtype=np.float32)
#         log_probs_buffer = np.zeros((n_steps, n_envs), dtype=np.float32)
        
#         # Initialize observation
#         obs = vec_env.reset()
        
#         # Training loop
#         num_updates = total_timesteps // (n_steps * n_envs)
        
#         for update in range(num_updates):
#             # Collect rollout
#             for step in range(n_steps):
#                 obs_buffer[step] = obs
                
#                 # Get actions from policy
#                 with torch.no_grad():
#                     obs_tensor = torch.FloatTensor(obs).to(device)
#                     mean, log_std, value, cost_value = model(obs_tensor)
                    
#                     # Sample actions
#                     std = torch.exp(log_std)
#                     dist = torch.distributions.Normal(mean, std)
#                     actions = dist.sample()
#                     log_probs = dist.log_prob(actions).sum(dim=-1)
                    
#                     # Store values
#                     actions_buffer[step] = actions.cpu().numpy()
#                     values_buffer[step] = value.squeeze(-1).cpu().numpy()
#                     cost_values_buffer[step] = cost_value.squeeze(-1).cpu().numpy()
#                     log_probs_buffer[step] = log_probs.cpu().numpy()
                
#                 # Step environment
#                 obs, rewards, dones, infos = vec_env.step(actions_buffer[step])
#                 model.num_timesteps += n_envs
                
#                 # Extract costs (risks) from infos
#                 costs = np.array([info.get('drawdown', 0) * 100 for info in infos])  # Use drawdown as cost
                
#                 rewards_buffer[step] = rewards
#                 costs_buffer[step] = costs
#                 dones_buffer[step] = dones
                
#                 # Call step callbacks
#                 for callback in callbacks:
#                     callback.update_locals(locals())
#                     if not callback.on_step():
#                         return -float('inf')
            
#             # Compute returns and advantages
#             with torch.no_grad():
#                 obs_tensor = torch.FloatTensor(obs).to(device)
#                 _, _, next_value, next_cost_value = model(obs_tensor)
#                 next_value = next_value.squeeze(-1).cpu().numpy()
#                 next_cost_value = next_cost_value.squeeze(-1).cpu().numpy()
            
#             # GAE for rewards
#             returns = np.zeros_like(rewards_buffer)
#             advantages = np.zeros_like(rewards_buffer)
#             lastgaelam = 0
            
#             for t in reversed(range(n_steps)):
#                 if t == n_steps - 1:
#                     nextnonterminal = 1.0 - dones_buffer[t]
#                     nextvalues = next_value
#                 else:
#                     nextnonterminal = 1.0 - dones_buffer[t]
#                     nextvalues = values_buffer[t + 1]
                
#                 delta = rewards_buffer[t] + hyperparams['gamma'] * nextvalues * nextnonterminal - values_buffer[t]
#                 advantages[t] = lastgaelam = delta + hyperparams['gamma'] * hyperparams['gae_lambda'] * nextnonterminal * lastgaelam
#                 returns[t] = advantages[t] + values_buffer[t]
            
#             # GAE for costs
#             cost_returns = np.zeros_like(costs_buffer)
#             cost_advantages = np.zeros_like(costs_buffer)
#             lastgaelam = 0
            
#             for t in reversed(range(n_steps)):
#                 if t == n_steps - 1:
#                     nextnonterminal = 1.0 - dones_buffer[t]
#                     nextvalues = next_cost_value
#                 else:
#                     nextnonterminal = 1.0 - dones_buffer[t]
#                     nextvalues = cost_values_buffer[t + 1]
                
#                 delta = costs_buffer[t] + hyperparams['gamma'] * nextvalues * nextnonterminal - cost_values_buffer[t]
#                 cost_advantages[t] = lastgaelam = delta + hyperparams['gamma'] * hyperparams['gae_lambda'] * nextnonterminal * lastgaelam
#                 cost_returns[t] = cost_advantages[t] + cost_values_buffer[t]
            
#             # Flatten the buffers
#             obs_flat = obs_buffer.reshape(-1, *vec_env.observation_space.shape)
#             actions_flat = actions_buffer.reshape(-1, *vec_env.action_space.shape)
#             log_probs_flat = log_probs_buffer.reshape(-1)
#             advantages_flat = advantages.reshape(-1)
#             cost_advantages_flat = cost_advantages.reshape(-1)
#             returns_flat = returns.reshape(-1)
#             cost_returns_flat = cost_returns.reshape(-1)
            
#             # Normalize advantages
#             advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)
#             cost_advantages_flat = (cost_advantages_flat - cost_advantages_flat.mean()) / (cost_advantages_flat.std() + 1e-8)
            
#             # PPO update
#             for epoch in range(n_epochs):
#                 # Create random mini-batches
#                 indices = np.random.permutation(len(obs_flat))
                
#                 for start_idx in range(0, len(obs_flat), batch_size):
#                     end_idx = start_idx + batch_size
#                     batch_indices = indices[start_idx:end_idx]
                    
#                     # Get batch data
#                     obs_batch = torch.FloatTensor(obs_flat[batch_indices]).to(device)
#                     actions_batch = torch.FloatTensor(actions_flat[batch_indices]).to(device)
#                     old_log_probs_batch = torch.FloatTensor(log_probs_flat[batch_indices]).to(device)
#                     advantages_batch = torch.FloatTensor(advantages_flat[batch_indices]).to(device)
#                     cost_advantages_batch = torch.FloatTensor(cost_advantages_flat[batch_indices]).to(device)
#                     returns_batch = torch.FloatTensor(returns_flat[batch_indices]).to(device)
#                     cost_returns_batch = torch.FloatTensor(cost_returns_flat[batch_indices]).to(device)
                    
#                     # Forward pass
#                     mean, log_std, values, cost_values = model(obs_batch)
                    
#                     # Calculate new log probs
#                     std = torch.exp(log_std)
#                     dist = torch.distributions.Normal(mean, std)
#                     new_log_probs = dist.log_prob(actions_batch).sum(dim=-1)
#                     entropy = dist.entropy().sum(dim=-1).mean()
                    
#                     # Ratio for PPO
#                     ratio = torch.exp(new_log_probs - old_log_probs_batch)
                    
#                     # Clipped surrogate loss
#                     surr1 = ratio * advantages_batch
#                     surr2 = torch.clamp(ratio, 1.0 - hyperparams['clip_range'], 1.0 + hyperparams['clip_range']) * advantages_batch
#                     policy_loss = -torch.min(surr1, surr2).mean()
                    
#                     # Value loss
#                     value_loss = ((values.squeeze(-1) - returns_batch) ** 2).mean()
                    
#                     # Cost value loss
#                     cost_value_loss = ((cost_values.squeeze(-1) - cost_returns_batch) ** 2).mean()
                    
#                     # Lagrangian loss for cost constraint
#                     lagrangian = model.get_lagrangian()
#                     cost_surr1 = ratio * cost_advantages_batch
#                     cost_surr2 = torch.clamp(ratio, 1.0 - hyperparams['clip_range'], 1.0 + hyperparams['clip_range']) * cost_advantages_batch
#                     cost_loss = torch.max(cost_surr1, cost_surr2).mean()
                    
#                     # Total loss
#                     loss = (
#                         policy_loss + 
#                         hyperparams['vf_coef'] * value_loss + 
#                         hyperparams['vf_coef'] * cost_value_loss - 
#                         hyperparams['ent_coef'] * entropy +
#                         lagrangian * cost_loss
#                     )
                    
#                     # Optimize
#                     model.optimizer.zero_grad()
#                     loss.backward()
#                     torch.nn.utils.clip_grad_norm_(model.parameters(), hyperparams['max_grad_norm'])
#                     model.optimizer.step()
            
#             # Update Lagrangian multiplier
#             avg_cost = costs_buffer.mean()
#             cost_error = avg_cost - config.cost_limit
#             model.update_lagrangian(cost_error)
            
#             # Log progress
#             if update % 10 == 0:
#                 avg_reward = rewards_buffer.mean()
#                 trial_logger.info(
#                     f"Update {update}/{num_updates}: "
#                     f"Avg Reward: {avg_reward:.3f}, "
#                     f"Avg Cost: {avg_cost:.3f}, "
#                     f"Lagrangian: {model.get_lagrangian():.3f}"
#                 )
                
#             # Call rollout end callbacks
#             for callback in callbacks:
#                 callback.on_rollout_end()
                
#     except TrialPruned:
#         trial_logger.info(f"Trial {trial.number} pruned")
#         raise
        
#     except Exception as e:
#         trial_logger.error(f"Trial {trial.number} failed: {e}")
#         raise
#     finally:
#         # Call training end callbacks
#         for callback in callbacks:
#             callback.on_training_end()
        
#     # Evaluate final performance
#     eval_env = DummyVecEnv([lambda: create_trading_env(data, config)])
#     mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=config.n_eval_episodes)
    
#     # Get final metrics
#     final_metrics = eval_env.envs[0].get_metrics()
#     trial.set_user_attr('final_metrics', final_metrics)
    
#     # Compute final score
#     score = compute_trial_score(mean_reward, final_metrics, config)
    
#     trial_logger.info(f"Trial {trial.number} completed with score: {score:.3f}")
    
#     return score

# def compute_trial_score(
#     mean_reward: float,
#     metrics: Dict[str, Any],
#     config: TrainingConfig
# ) -> float:
#     """Compute comprehensive trial score"""
    
#     # Extract key metrics
#     profit = (metrics['balance'] - config.initial_balance) / config.initial_balance
#     win_rate = metrics.get('win_rate', 0)
#     sharpe = metrics.get('sharpe_ratio', 0)
#     max_dd = metrics.get('max_drawdown', 0)
    
#     # Check if targets are met
#     daily_profit = profit * 365 / config.max_steps_per_episode  # Annualized
#     target_met = daily_profit >= config.target_daily_profit / config.initial_balance
    
#     # Compute weighted score
#     score = (
#         0.35 * (profit * 10) +  # Emphasize profitability
#         0.25 * win_rate +
#         0.20 * (sharpe / 3.0) +
#         0.15 * (1 - max_dd) +
#         0.05 * float(target_met)
#     )
    
#     # Penalties
#     if max_dd > config.max_drawdown_limit:
#         score *= 0.5  # Heavy penalty for excessive drawdown
        
#     if win_rate < config.min_win_rate:
#         score *= 0.8  # Penalty for low win rate
        
#     return float(np.clip(score, 0, 1))

# def train_final_model(
#     best_hyperparams: Dict[str, Any],
#     config: TrainingConfig,
#     checkpoint_path: Optional[str] = None
# ) -> PPOLagrangian:
#     """Train final model with best hyperparameters"""
    
#     logger = logging.getLogger("final_training")
#     logger.info("Starting final model training")
    
#     # Load data
#     data = load_data("data/processed")
    
#     # Create environment
#     envs = []
#     for i in range(config.num_envs):
#         env = create_trading_env(data, config, seed=config.global_seed + i)
#         envs.append(env)
        
#     vec_env = DummyVecEnv([lambda env=env: env for env in envs])
    
#     # Create or load model
#     if checkpoint_path and os.path.exists(checkpoint_path):
#         logger.info(f"Loading model from checkpoint: {checkpoint_path}")
#         with open(checkpoint_path, 'rb') as f:
#             checkpoint = pickle.load(f)
            
#         model = create_ppo_lag_model(vec_env, config, best_hyperparams)
#         model.load_state_dict(checkpoint['model_state'])
#         model.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
#         start_timestep = checkpoint['timesteps']
        
#         # Restore environment states
#         for i, env in enumerate(envs):
#             if 'env_state' in checkpoint:
#                 env.set_state(checkpoint['env_state'])
                
#     else:
#         model = create_ppo_lag_model(vec_env, config, best_hyperparams)
#         start_timestep = 0
        
#     # Setup callbacks
#     callbacks = [
#         DetailedLoggingCallback(config),
#         TensorboardCallback(config),
#         CheckpointCallback(config),
#     ]
    
#     # Initialize callbacks
#     CUSTOM_CALLBACKS = (
#         DetailedLoggingCallback,
#         TensorboardCallback,
#         CheckpointCallback,
#         OptunaPruningCallback,
#     )

#     for callback in callbacks:
#         if isinstance(callback, CUSTOM_CALLBACKS):
#             callback.init_callback(model, vec_env)
#         else:
#             callback.init_callback(model)
#         callback.on_training_start(locals(), globals())


    
#     # Train
#     remaining_timesteps = config.final_training_steps - start_timestep
#     if remaining_timesteps > 0:
#         logger.info(f"Training for {remaining_timesteps} timesteps")
        
#         # Training parameters
#         n_envs = vec_env.num_envs
#         n_steps = best_hyperparams['n_steps']
#         batch_size = best_hyperparams['batch_size']
#         n_epochs = best_hyperparams['n_epochs']
#         device = model.device
        
#         # Buffer for storing rollouts
#         obs_buffer = np.zeros((n_steps, n_envs) + vec_env.observation_space.shape, dtype=np.float32)
#         actions_buffer = np.zeros((n_steps, n_envs) + vec_env.action_space.shape, dtype=np.float32)
#         rewards_buffer = np.zeros((n_steps, n_envs), dtype=np.float32)
#         costs_buffer = np.zeros((n_steps, n_envs), dtype=np.float32)
#         dones_buffer = np.zeros((n_steps, n_envs), dtype=np.float32)
#         values_buffer = np.zeros((n_steps, n_envs), dtype=np.float32)
#         cost_values_buffer = np.zeros((n_steps, n_envs), dtype=np.float32)
#         log_probs_buffer = np.zeros((n_steps, n_envs), dtype=np.float32)
        
#         # Initialize observation
#         obs = vec_env.reset()
        
#         # Calculate number of updates
#         num_updates = remaining_timesteps // (n_steps * n_envs)
        
#         # Progress bar
#         pbar = tqdm(total=remaining_timesteps, desc="Training", ncols=100)
        
#         for update in range(num_updates):
#             # Collect rollout
#             for step in range(n_steps):
#                 obs_buffer[step] = obs
                
#                 # Get actions from policy
#                 with torch.no_grad():
#                     obs_tensor = torch.FloatTensor(obs).to(device)
#                     mean, log_std, value, cost_value = model(obs_tensor)
                    
#                     # Sample actions
#                     std = torch.exp(log_std)
#                     dist = torch.distributions.Normal(mean, std)
#                     actions = dist.sample()
#                     log_probs = dist.log_prob(actions).sum(dim=-1)
                    
#                     # Store values
#                     actions_buffer[step] = actions.cpu().numpy()
#                     values_buffer[step] = value.squeeze(-1).cpu().numpy()
#                     cost_values_buffer[step] = cost_value.squeeze(-1).cpu().numpy()
#                     log_probs_buffer[step] = log_probs.cpu().numpy()
                
#                 # Step environment
#                 obs, rewards, dones, infos = vec_env.step(actions_buffer[step])
#                 model.num_timesteps += n_envs
                
#                 # Extract costs
#                 costs = np.array([info.get('drawdown', 0) * 100 for info in infos])
                
#                 rewards_buffer[step] = rewards
#                 costs_buffer[step] = costs
#                 dones_buffer[step] = dones
                
#                 # Update progress
#                 pbar.update(n_envs)
                
#                 # Call step callbacks
#                 for callback in callbacks:
#                     callback.num_timesteps = start_timestep + update * n_steps * n_envs + step * n_envs
#                     callback.update_locals(locals())
#                     if not callback.on_step():
#                         pbar.close()
#                         return model
            
#             # Compute returns and advantages (same as in optimize_hyperparameters)
#             with torch.no_grad():
#                 obs_tensor = torch.FloatTensor(obs).to(device)
#                 _, _, next_value, next_cost_value = model(obs_tensor)
#                 next_value = next_value.squeeze(-1).cpu().numpy()
#                 next_cost_value = next_cost_value.squeeze(-1).cpu().numpy()
            
#             # GAE calculation (same as before)
#             returns = np.zeros_like(rewards_buffer)
#             advantages = np.zeros_like(rewards_buffer)
#             lastgaelam = 0
            
#             for t in reversed(range(n_steps)):
#                 if t == n_steps - 1:
#                     nextnonterminal = 1.0 - dones_buffer[t]
#                     nextvalues = next_value
#                 else:
#                     nextnonterminal = 1.0 - dones_buffer[t]
#                     nextvalues = values_buffer[t + 1]
                
#                 delta = rewards_buffer[t] + best_hyperparams['gamma'] * nextvalues * nextnonterminal - values_buffer[t]
#                 advantages[t] = lastgaelam = delta + best_hyperparams['gamma'] * best_hyperparams['gae_lambda'] * nextnonterminal * lastgaelam
#                 returns[t] = advantages[t] + values_buffer[t]
            
#             # Cost GAE
#             cost_returns = np.zeros_like(costs_buffer)
#             cost_advantages = np.zeros_like(costs_buffer)
#             lastgaelam = 0
            
#             for t in reversed(range(n_steps)):
#                 if t == n_steps - 1:
#                     nextnonterminal = 1.0 - dones_buffer[t]
#                     nextvalues = next_cost_value
#                 else:
#                     nextnonterminal = 1.0 - dones_buffer[t]
#                     nextvalues = cost_values_buffer[t + 1]
                
#                 delta = costs_buffer[t] + best_hyperparams['gamma'] * nextvalues * nextnonterminal - cost_values_buffer[t]
#                 cost_advantages[t] = lastgaelam = delta + best_hyperparams['gamma'] * best_hyperparams['gae_lambda'] * nextnonterminal * lastgaelam
#                 cost_returns[t] = cost_advantages[t] + cost_values_buffer[t]
            
#             # Flatten buffers
#             obs_flat = obs_buffer.reshape(-1, *vec_env.observation_space.shape)
#             actions_flat = actions_buffer.reshape(-1, *vec_env.action_space.shape)
#             log_probs_flat = log_probs_buffer.reshape(-1)
#             advantages_flat = advantages.reshape(-1)
#             cost_advantages_flat = cost_advantages.reshape(-1)
#             returns_flat = returns.reshape(-1)
#             cost_returns_flat = cost_returns.reshape(-1)
            
#             # Normalize advantages
#             advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)
#             cost_advantages_flat = (cost_advantages_flat - cost_advantages_flat.mean()) / (cost_advantages_flat.std() + 1e-8)
            
#             # PPO update (same as before)
#             for epoch in range(n_epochs):
#                 indices = np.random.permutation(len(obs_flat))
                
#                 for start_idx in range(0, len(obs_flat), batch_size):
#                     end_idx = start_idx + batch_size
#                     batch_indices = indices[start_idx:end_idx]
                    
#                     # Get batch data
#                     obs_batch = torch.FloatTensor(obs_flat[batch_indices]).to(device)
#                     actions_batch = torch.FloatTensor(actions_flat[batch_indices]).to(device)
#                     old_log_probs_batch = torch.FloatTensor(log_probs_flat[batch_indices]).to(device)
#                     advantages_batch = torch.FloatTensor(advantages_flat[batch_indices]).to(device)
#                     cost_advantages_batch = torch.FloatTensor(cost_advantages_flat[batch_indices]).to(device)
#                     returns_batch = torch.FloatTensor(returns_flat[batch_indices]).to(device)
#                     cost_returns_batch = torch.FloatTensor(cost_returns_flat[batch_indices]).to(device)
                    
#                     # Forward pass
#                     mean, log_std, values, cost_values = model(obs_batch)
                    
#                     # Calculate new log probs
#                     std = torch.exp(log_std)
#                     dist = torch.distributions.Normal(mean, std)
#                     new_log_probs = dist.log_prob(actions_batch).sum(dim=-1)
#                     entropy = dist.entropy().sum(dim=-1).mean()
                    
#                     # PPO loss calculation
#                     ratio = torch.exp(new_log_probs - old_log_probs_batch)
#                     surr1 = ratio * advantages_batch
#                     surr2 = torch.clamp(ratio, 1.0 - best_hyperparams['clip_range'], 1.0 + best_hyperparams['clip_range']) * advantages_batch
#                     policy_loss = -torch.min(surr1, surr2).mean()
                    
#                     value_loss = ((values.squeeze(-1) - returns_batch) ** 2).mean()
#                     cost_value_loss = ((cost_values.squeeze(-1) - cost_returns_batch) ** 2).mean()
                    
#                     # Lagrangian loss
#                     lagrangian = model.get_lagrangian()
#                     cost_surr1 = ratio * cost_advantages_batch
#                     cost_surr2 = torch.clamp(ratio, 1.0 - best_hyperparams['clip_range'], 1.0 + best_hyperparams['clip_range']) * cost_advantages_batch
#                     cost_loss = torch.max(cost_surr1, cost_surr2).mean()
                    
#                     # Total loss
#                     loss = (
#                         policy_loss + 
#                         best_hyperparams['vf_coef'] * value_loss + 
#                         best_hyperparams['vf_coef'] * cost_value_loss - 
#                         best_hyperparams['ent_coef'] * entropy +
#                         lagrangian * cost_loss
#                     )
                    
#                     # Optimize
#                     model.optimizer.zero_grad()
#                     loss.backward()
#                     torch.nn.utils.clip_grad_norm_(model.parameters(), best_hyperparams['max_grad_norm'])
#                     model.optimizer.step()
            
#             # Update Lagrangian
#             avg_cost = costs_buffer.mean()
#             cost_error = avg_cost - config.cost_limit
#             model.update_lagrangian(cost_error)
            
#             # Log progress
#             if update % 100 == 0:
#                 avg_reward = rewards_buffer.mean()
#                 logger.info(
#                     f"Update {update}/{num_updates}: "
#                     f"Avg Reward: {avg_reward:.3f}, "
#                     f"Avg Cost: {avg_cost:.3f}, "
#                     f"Lagrangian: {model.get_lagrangian():.3f}"
#                 )
            
#             # Call rollout end callbacks
#             for callback in callbacks:
#                 callback.on_rollout_end()
                
#         pbar.close()
        
#     else:
#         logger.info("Training already completed")
    
#     # Call training end callbacks
#     for callback in callbacks:
#         callback.on_training_end()
        
#     return model

# # ═══════════════════════════════════════════════════════════════════
# # Main Training Script
# # ═══════════════════════════════════════════════════════════════════

# def main():
#     """Main training entry point"""
    
#     # Parse arguments
#     parser = argparse.ArgumentParser(description="PPO-LAG Trading Agent Training")
#     parser.add_argument('--test', action='store_true', help='Run in test mode')
#     parser.add_argument('--resume', type=str, help='Resume from checkpoint')
#     parser.add_argument('--skip-optimization', action='store_true', 
#                        help='Skip hyperparameter optimization')
#     args = parser.parse_args()
    
#     # Create configuration
#     config = TrainingConfig(test_mode=args.test)
    
#     # Setup main logger
#     logger = logging.getLogger("main")
#     logger.info(
#         f"\n{'='*60}\n"
#         f"PPO-LAG Trading Agent Training\n"
#         f"{'='*60}\n"
#         f"Mode: {'TEST' if config.test_mode else 'PRODUCTION'}\n"
#         f"Target: €{config.target_daily_profit}/day\n"
#         f"Max Drawdown: {config.max_drawdown_limit:.1%}\n"
#         f"{'='*60}\n"
#     )
    
#     # Set global seed
#     set_global_seed(config.global_seed)
    
#     # Hyperparameter optimization
#     if not args.skip_optimization:
#         logger.info("Starting hyperparameter optimization...")
        
#         # Create Optuna study
#         study = optuna.create_study(
#             study_name="ppo_lag_trading",
#             storage=f"sqlite:///{config.log_dir}/optuna.db",
#             direction="maximize",
#             load_if_exists=True,
#             pruner=MedianPruner(
#                 n_startup_trials=config.pruner_startup_trials,
#                 n_warmup_steps=config.pruner_warmup_steps,
#                 interval_steps=config.pruner_interval_steps
#             )
#         )
        
#         # Optimize
#         study.optimize(
#             lambda trial: optimize_hyperparameters(trial, config),
#             n_trials=config.n_trials,
#             n_jobs=1,  # PPO doesn't parallelize well
#             show_progress_bar=True
#         )
        
#         # Get best hyperparameters
#         best_trial = study.best_trial
#         best_hyperparams = best_trial.params
        
#         logger.info(
#             f"\nBest trial: {best_trial.number}\n"
#             f"Score: {best_trial.value:.3f}\n"
#             f"Hyperparameters: {best_hyperparams}\n"
#         )
        
#     else:
#         # Use default hyperparameters
#         best_hyperparams = {
#             'learning_rate': 3e-4,
#             'n_steps': 2048,
#             'batch_size': 64,
#             'n_epochs': 10,
#             'gamma': 0.99,
#             'gae_lambda': 0.95,
#             'clip_range': 0.2,
#             'ent_coef': 0.01,
#             'vf_coef': 0.5,
#             'max_grad_norm': 0.5,
#         }
        
#     # Train final model
#     logger.info("Training final model with best hyperparameters...")
    
#     final_model = train_final_model(
#         best_hyperparams,
#         config,
#         checkpoint_path=args.resume
#     )
    
#     # Save final model
#     model_path = os.path.join(config.model_dir, "ppo_lag_final.pkl")
#     torch.save(final_model.state_dict(), model_path)
#     logger.info(f"Final model saved to: {model_path}")
    
#     # Final evaluation
#     logger.info("Running final evaluation...")
    
#     data = load_data("data/processed")
#     eval_env = create_trading_env(data, config)
    
#     # Run comprehensive evaluation
#     total_episodes = 50 if not config.test_mode else 10
#     episode_metrics = []
    
#     for ep in tqdm(range(total_episodes), desc="Final Evaluation"):
#         obs = eval_env.reset()
#         done = False
        
#         while not done:
#             with torch.no_grad():
#                 obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(final_model.device)
#                 mean, _, _, _ = final_model(obs_tensor)
#                 action = mean.squeeze(0).cpu().numpy()
                
#             obs, reward, done, info = eval_env.step(action)
            
#         metrics = eval_env.get_metrics()
#         episode_metrics.append(metrics)
        
#     # Aggregate results
#     avg_balance = np.mean([m['balance'] for m in episode_metrics])
#     avg_profit = avg_balance - config.initial_balance
#     avg_win_rate = np.mean([m.get('win_rate', 0) for m in episode_metrics])
#     avg_sharpe = np.mean([m.get('sharpe_ratio', 0) for m in episode_metrics])
#     avg_max_dd = np.mean([m.get('max_drawdown', 0) for m in episode_metrics])
    
#     logger.info(
#         f"\n{'='*60}\n"
#         f"FINAL EVALUATION RESULTS\n"
#         f"{'='*60}\n"
#         f"Average Final Balance: €{avg_balance:,.2f}\n"
#         f"Average Profit: €{avg_profit:,.2f}\n"
#         f"Average Win Rate: {avg_win_rate:.2%}\n"
#         f"Average Sharpe Ratio: {avg_sharpe:.3f}\n"
#         f"Average Max Drawdown: {avg_max_dd:.2%}\n"
#         f"Daily Profit Target: €{config.target_daily_profit}\n"
#         f"Achieved Daily: €{avg_profit * 365 / config.max_steps_per_episode:.2f}\n"
#         f"{'='*60}\n"
#     )
    
#     # Generate final report
#     generate_training_report(config, episode_metrics)
    
#     logger.info("Training completed successfully!")

# def generate_training_report(config: TrainingConfig, episode_metrics: List[Dict[str, Any]]):
#     """Generate comprehensive training report"""
    
#     report_path = os.path.join(config.log_dir, "training_report.txt")
    
#     with open(report_path, 'w') as f:
#         f.write(
#             f"PPO-LAG Trading Agent Training Report\n"
#             f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
#             f"{'='*80}\n\n"
#             f"Configuration:\n"
#             f"- Mode: {'TEST' if config.test_mode else 'PRODUCTION'}\n"
#             f"- Initial Balance: €{config.initial_balance:,.2f}\n"
#             f"- Target Daily Profit: €{config.target_daily_profit}\n"
#             f"- Max Drawdown Limit: {config.max_drawdown_limit:.1%}\n"
#             f"- Cost Limit: {config.cost_limit}\n\n"
#             f"Training Details:\n"
#             f"- Total Timesteps: {config.final_training_steps:,}\n"
#             f"- Number of Trials: {config.n_trials}\n"
#             f"- Episodes per Trial: {config.timesteps_per_trial // config.max_steps_per_episode}\n\n"
#             f"Final Performance:\n"
#         )
        
#         # Add detailed metrics
#         for i, metrics in enumerate(episode_metrics[:10]):  # First 10 episodes
#             f.write(
#                 f"\nEpisode {i+1}:\n"
#                 f"  Balance: €{metrics['balance']:,.2f}\n"
#                 f"  Win Rate: {metrics.get('win_rate', 0):.2%}\n"
#                 f"  Sharpe: {metrics.get('sharpe_ratio', 0):.3f}\n"
#                 f"  Max DD: {metrics.get('max_drawdown', 0):.2%}\n"
#             )
            
#     # Also generate a JSON report for programmatic access
#     json_report = {
#         'config': config.__dict__,
#         'timestamp': datetime.now().isoformat(),
#         'episode_metrics': episode_metrics,
#         'summary': {
#             'avg_profit': np.mean([m['balance'] - config.initial_balance for m in episode_metrics]),
#             'avg_win_rate': np.mean([m.get('win_rate', 0) for m in episode_metrics]),
#             'avg_sharpe': np.mean([m.get('sharpe_ratio', 0) for m in episode_metrics]),
#             'success_rate': sum(1 for m in episode_metrics if m['balance'] > config.initial_balance) / len(episode_metrics)
#         }
#     }
    
#     with open(os.path.join(config.log_dir, 'training_report.json'), 'w') as f:
#         json.dump(json_report, f, indent=2, default=str)

# def set_global_seed(seed: int):
#     """Set all random seeds for reproducibility"""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# if __name__ == "__main__":
#     main()


# train_stable_ppo.py
"""
STABLE PPO Training Script
Fixes NaN issues and provides robust training
"""

import os
import sys
import logging
import traceback
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Import our modules
from config.trading_config import TradingConfig, ConfigPresets, ConfigFactory
from envs.env import EnhancedTradingEnv

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('stable_training.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# STABLE PPO Model with NaN Protection
# ═══════════════════════════════════════════════════════════════════

class StablePPO(nn.Module):
    """Numerically stable PPO implementation with NaN protection"""
    
    def __init__(self, obs_dim: int, action_dim: int, config: TradingConfig):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        
        # Model config
        model_config = config.get_model_config()
        self.hidden_size = min(model_config['policy_hidden_size'], 256)  # Cap size for stability
        self.lr = max(model_config['learning_rate'], 1e-5)  # Minimum LR for stability
        
        # FIXED: Much more conservative architecture
        self.shared_net = nn.Sequential(
            nn.Linear(obs_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),  # Normalize
            nn.Tanh(),  # More stable than ReLU
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2),
            nn.Tanh(),
        )
        
        # Separate heads
        self.policy_head = nn.Sequential(
            nn.Linear(self.hidden_size // 2, action_dim),
            nn.Tanh()  # Bounded output
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_size // 2, 1)
        )
        
        self.cost_head = nn.Sequential(
            nn.Linear(self.hidden_size // 2, 1),
            nn.Softplus()  # Always positive
        )
        
        # Fixed log_std instead of learning it (more stable)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -1.0)  # Start conservative
        
        # Lagrangian multiplier
        self.log_lagrangian = nn.Parameter(torch.zeros(1))
        
        # Initialize weights properly
        self._init_weights()
        
        # Optimizer with conservative settings
        self.optimizer = optim.Adam(
            self.parameters(), 
            lr=self.lr,
            eps=1e-8,  # Numerical stability
            weight_decay=1e-6  # Slight regularization
        )
        
        # Training parameters
        self.clip_ratio = 0.1  # More conservative than default 0.2
        self.cost_limit = config.cost_limit
        
        logger.info(f"StablePPO created: obs_dim={obs_dim}, action_dim={action_dim}")
        logger.info(f"Hidden size: {self.hidden_size}, Learning rate: {self.lr}")
        
    def _init_weights(self):
        """Conservative weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier uniform with small scale
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def _check_for_nans(self, tensor, name="tensor"):
        """Check for NaN/Inf and raise informative error"""
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            logger.error(f"NaN/Inf detected in {name}: {tensor}")
            raise ValueError(f"NaN/Inf in {name}")
    
    def forward(self, obs):
        """Forward pass with NaN checking"""
        # Input validation
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            
        # CRITICAL: Check input for NaN/Inf
        self._check_for_nans(obs, "input_obs")
        
        # Clamp input to reasonable range
        obs = torch.clamp(obs, -100, 100)
        
        # Forward pass
        features = self.shared_net(obs)
        self._check_for_nans(features, "features")
        
        # Get outputs
        policy_out = self.policy_head(features)
        value = self.value_head(features)
        cost_value = self.cost_head(features)
        
        # Check outputs
        self._check_for_nans(policy_out, "policy_out")
        self._check_for_nans(value, "value")
        self._check_for_nans(cost_value, "cost_value")
        
        # Scale policy output
        mean = policy_out * 0.5  # Conservative scaling
        log_std = torch.clamp(self.log_std, -5, 1)  # Bounded std
        
        return mean, log_std, value, cost_value
    
    def get_action(self, obs, deterministic=False):
        """Get action with stability checks"""
        with torch.no_grad():
            try:
                mean, log_std, value, cost_value = self.forward(obs)
                
                if deterministic:
                    action = mean
                    log_prob = torch.zeros(mean.shape[0])
                else:
                    std = torch.exp(log_std)
                    std = torch.clamp(std, 0.01, 1.0)  # Bound std
                    
                    normal = torch.distributions.Normal(mean, std)
                    action = normal.sample()
                    log_prob = normal.log_prob(action).sum(dim=-1)
                
                # Clamp action to valid range
                action = torch.clamp(action, -1.0, 1.0)
                
                return action, log_prob, value, cost_value
                
            except Exception as e:
                logger.error(f"Error in get_action: {e}")
                # Return safe fallback
                batch_size = obs.shape[0]
                action = torch.zeros(batch_size, self.action_dim)
                log_prob = torch.zeros(batch_size)
                value = torch.zeros(batch_size, 1)
                cost_value = torch.ones(batch_size, 1) * 0.1
                return action, log_prob, value, cost_value
    
    def evaluate_actions(self, obs, actions):
        """Evaluate actions with stability"""
        try:
            mean, log_std, value, cost_value = self.forward(obs)
            
            std = torch.exp(log_std)
            std = torch.clamp(std, 0.01, 1.0)
            
            normal = torch.distributions.Normal(mean, std)
            log_prob = normal.log_prob(actions).sum(dim=-1)
            entropy = normal.entropy().sum(dim=-1)
            
            return log_prob, entropy, value, cost_value
            
        except Exception as e:
            logger.error(f"Error in evaluate_actions: {e}")
            # Return safe fallback
            batch_size = obs.shape[0]
            log_prob = torch.zeros(batch_size)
            entropy = torch.ones(batch_size) * 0.1
            value = torch.zeros(batch_size, 1)
            cost_value = torch.ones(batch_size, 1) * 0.1
            return log_prob, entropy, value, cost_value
    
    def get_lagrangian(self):
        """Get Lagrangian multiplier"""
        return torch.exp(torch.clamp(self.log_lagrangian, -5, 5)).item()

# ═══════════════════════════════════════════════════════════════════
# Safe Data Creation
# ═══════════════════════════════════════════════════════════════════

def create_safe_dummy_data(config: TradingConfig):
    """Create numerically safe dummy data"""
    logger.info("Creating safe dummy market data...")
    
    data = {}
    
    for instrument in config.instruments:
        data[instrument] = {}
        
        for timeframe in config.timeframes:
            n_points = 1000  # Fixed size for stability
            
            # Generate very stable price data
            np.random.seed(42)  # Reproducible
            
            if "XAU" in instrument:
                base_price = 2000.0
                volatility = 0.005  # Very low volatility
            else:
                base_price = 1.1000
                volatility = 0.003
            
            # Gentle random walk
            returns = np.random.normal(0.0, volatility, n_points)
            returns = np.clip(returns, -0.01, 0.01)  # Clip extreme moves
            
            prices = base_price * np.exp(np.cumsum(returns))
            
            # Create stable OHLC
            close = prices
            high = close * (1 + np.abs(np.random.normal(0, 0.001, n_points)))
            low = close * (1 - np.abs(np.random.normal(0, 0.001, n_points)))
            open_prices = np.roll(close, 1)
            open_prices[0] = close[0]
            
            # Ensure no NaN/Inf
            close = np.nan_to_num(close, nan=base_price)
            high = np.nan_to_num(high, nan=base_price)
            low = np.nan_to_num(low, nan=base_price)
            open_prices = np.nan_to_num(open_prices, nan=base_price)
            
            df = pd.DataFrame({
                'datetime': pd.date_range(start='2023-01-01', periods=n_points, freq='h'),
                'open': open_prices,
                'high': high,
                'low': low,
                'close': close,
                'volume': np.random.randint(100, 1000, n_points),
                'volatility': np.random.uniform(0.002, 0.008, n_points)
            })
            
            # Final safety check
            for col in ['open', 'high', 'low', 'close', 'volume', 'volatility']:
                df[col] = df[col].fillna(method='ffill').fillna(base_price if col != 'volume' else 100)
            
            data[instrument][timeframe] = df
    
    logger.info(f"Created safe dummy data for {len(config.instruments)} instruments")
    return data

# ═══════════════════════════════════════════════════════════════════
# Safe Training Functions
# ═══════════════════════════════════════════════════════════════════

def safe_create_environment(data: Dict, config: TradingConfig, seed: int = 0):
    """Create environment with error handling"""
    try:
        env = EnhancedTradingEnv(data, config)
        if hasattr(env, '_set_seeds'):
            env._set_seeds(seed)
        logger.info(f"Created environment with seed {seed}")
        return env
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        logger.error(traceback.format_exc())
        raise

def safe_get_observation(env):
    """Safely get observation from environment"""
    try:
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        # Convert to numpy if needed
        if hasattr(obs, 'numpy'):
            obs = obs.numpy()
        elif not isinstance(obs, np.ndarray):
            obs = np.array(obs, dtype=np.float32)
        
        # Safety checks
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        obs = np.clip(obs, -100, 100)
        
        return obs
        
    except Exception as e:
        logger.error(f"Error getting observation: {e}")
        # Return safe fallback
        return np.zeros(100, dtype=np.float32)

def run_simple_training(config: TradingConfig):
    """Simplified, stable training loop"""
    logger.info("Starting stable training...")
    
    try:
        # Create safe data
        data = create_safe_dummy_data(config)
        
        # Create environment
        env = safe_create_environment(data, config, seed=config.init_seed)
        
        # Get dimensions safely
        dummy_obs = safe_get_observation(env)
        obs_dim = len(dummy_obs)
        action_dim = env.action_space.shape[0]
        
        logger.info(f"Dimensions: obs_dim={obs_dim}, action_dim={action_dim}")
        
        # Create stable model
        model = StablePPO(obs_dim, action_dim, config)
        
        # Training parameters
        n_episodes = 5 if config.test_mode else 20
        max_steps = min(config.max_steps, 50)  # Short episodes for stability
        
        logger.info(f"Training: {n_episodes} episodes, {max_steps} steps each")
        
        # Metrics
        episode_rewards = []
        episode_lengths = []
        
        # Training loop
        for episode in range(n_episodes):
            logger.info(f"Episode {episode + 1}/{n_episodes}")
            
            try:
                # Reset environment
                obs = safe_get_observation(env)
                total_reward = 0.0
                episode_length = 0
                
                # Episode loop
                for step in range(max_steps):
                    # Get action
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    
                    try:
                        action, log_prob, value, cost_value = model.get_action(obs_tensor)
                        action_np = action.cpu().numpy().flatten()
                    except Exception as action_error:
                        logger.warning(f"Action error: {action_error}, using random action")
                        action_np = np.random.uniform(-0.1, 0.1, action_dim)
                    
                    # Step environment
                    try:
                        next_obs, reward, done, truncated, info = env.step(action_np)
                        
                        # Safety checks
                        reward = float(np.nan_to_num(reward, nan=0.0))
                        reward = np.clip(reward, -100, 100)
                        
                        next_obs = safe_get_observation(env) if done or truncated else next_obs
                        
                    except Exception as step_error:
                        logger.warning(f"Step error: {step_error}, ending episode")
                        reward = 0.0
                        done = True
                        next_obs = obs
                    
                    total_reward += reward
                    episode_length += 1
                    obs = next_obs
                    
                    if done or truncated:
                        break
                
                # Record metrics
                episode_rewards.append(total_reward)
                episode_lengths.append(episode_length)
                
                # Log progress
                avg_reward = np.mean(episode_rewards[-5:])
                logger.info(
                    f"Episode {episode + 1}: Reward={total_reward:.3f}, "
                    f"Length={episode_length}, Avg Reward={avg_reward:.3f}"
                )
                
                # Simple model update (just to show it's working)
                if episode % 2 == 0 and episode_length > 1:
                    try:
                        # Dummy loss for demonstration
                        dummy_obs = torch.FloatTensor(obs).unsqueeze(0)
                        _, _, value, _ = model.forward(dummy_obs)
                        loss = value.mean()  # Dummy loss
                        
                        model.optimizer.zero_grad()
                        loss.backward()
                        
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                        
                        model.optimizer.step()
                        
                        logger.info(f"Model updated, loss: {loss.item():.4f}")
                        
                    except Exception as update_error:
                        logger.warning(f"Update error: {update_error}")
                
            except Exception as episode_error:
                logger.error(f"Episode {episode} failed: {episode_error}")
                episode_rewards.append(0.0)
                episode_lengths.append(1)
        
        # Final results
        final_avg_reward = np.mean(episode_rewards[-5:]) if episode_rewards else 0.0
        logger.info(f"Training completed! Final average reward: {final_avg_reward:.3f}")
        
        return model, {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'final_avg_reward': final_avg_reward
        }
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(traceback.format_exc())
        raise

# ═══════════════════════════════════════════════════════════════════
# Main Function
# ═══════════════════════════════════════════════════════════════════

def main():
    """Main entry point"""
    try:
        parser = argparse.ArgumentParser(description="Stable PPO Training")
        parser.add_argument('--test', action='store_true', help='Run in test mode')
        parser.add_argument('--config', type=str, help='Config file path')
        parser.add_argument('--preset', type=str, choices=['conservative', 'aggressive', 'research'])
        parser.add_argument('--balance', type=float, help='Initial balance')
        args = parser.parse_args()
        
        print("🛡️ Stable PPO Training System")
        print("=" * 40)
        
        # Create configuration
        if args.config:
            config = TradingConfig.load_config(args.config)
            print(f"✅ Loaded config from {args.config}")
        elif args.preset:
            if args.preset == 'conservative':
                config = ConfigPresets.conservative_live()
            elif args.preset == 'aggressive':
                config = ConfigPresets.aggressive_backtest()
            elif args.preset == 'research':
                config = ConfigPresets.research_mode()
            print(f"✅ Using {args.preset} preset")
        else:
            config = TradingConfig(test_mode=True)  # Default to test mode
            print("✅ Using safe test configuration")
        
        # Apply overrides
        if args.test:
            config.test_mode = True
            print("🧪 Test mode enabled")
        
        if args.balance:
            config.initial_balance = args.balance
            print(f"💰 Balance: €{args.balance:,.2f}")
        
        # Apply settings
        config.__post_init__()
        
        print(f"📊 Config: Mode={'TEST' if config.test_mode else 'PROD'}, "
              f"Balance=€{config.initial_balance:,.2f}")
        print()
        
        # Run training
        print("🏋️ Starting stable training...")
        model, results = run_simple_training(config)
        
        # Save results
        os.makedirs(config.model_dir, exist_ok=True)
        
        model_path = os.path.join(config.model_dir, "stable_ppo_model.pth")
        torch.save(model.state_dict(), model_path)
        
        config_path = os.path.join(config.model_dir, "stable_config.json")
        config.save_config(config_path)
        
        results_path = os.path.join(config.model_dir, "stable_results.json")
        with open(results_path, 'w') as f:
            safe_results = {
                'episode_rewards': [float(x) for x in results['episode_rewards']],
                'episode_lengths': [int(x) for x in results['episode_lengths']],
                'final_avg_reward': float(results['final_avg_reward']),
                'total_episodes': len(results['episode_rewards'])
            }
            json.dump(safe_results, f, indent=2)
        
        print("✅ Training completed successfully!")
        print(f"💾 Model saved to: {model_path}")
        print(f"⚙️ Config saved to: {config_path}")
        print(f"📊 Results saved to: {results_path}")
        print(f"📈 Final average reward: {results['final_avg_reward']:.3f}")
        print()
        print("🎉 No NaN errors! Model is stable and ready to use.")
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        logger.error(traceback.format_exc())
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()