# train/train_ppo_stable.py - FIXED VERSION
"""
Consolidated PPO Training Script - Windows Compatible
FIXED: Better error handling and Windows multiprocessing support
"""

import os
import platform
log_dirs = [
    'logs/training',
    'logs/regime', 
    'logs/strategy',
    'logs/checkpoints'
]

for log_dir in log_dirs:
    os.makedirs(log_dir, exist_ok=True)
import sys
import logging
import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

# Custom imports
from envs.env import EnhancedTradingEnv, TradingConfig
from utils.data_utils import load_data

# Configure logging with better Windows support
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/training/ppo_training.log', encoding='utf-8')
    ]
)
logger = logging.getLogger("PPOTraining")

# ═══════════════════════════════════════════════════════════════════
# Windows-compatible environment creation
# ═══════════════════════════════════════════════════════════════════

def test_env_creation(data: Dict, config: TradingConfig) -> bool:
    """Test if environment can be created successfully"""
    try:
        env = EnhancedTradingEnv(data, config)
        env.reset()
        env.close()
        return True
    except Exception as e:
        logger.error(f"Environment creation test failed: {e}")
        return False

def make_env(data: Dict, config: TradingConfig, rank: int = 0, seed: int = 0):
    """Create a single environment instance with error handling"""
    def _init():
        try:
            env = EnhancedTradingEnv(data, config)
            env = Monitor(env, filename=f"logs/training/monitor_{rank}.csv")
            env.seed(seed + rank)
            return env
        except Exception as e:
            logger.error(f"Failed to create environment {rank}: {e}")
            raise
    
    set_random_seed(seed)
    return _init

def create_envs(data: Dict, config: TradingConfig, n_envs: int = 4, seed: int = 42):
    """Create vectorized environments with Windows compatibility"""
    
    # Test environment creation first
    if not test_env_creation(data, config):
        logger.error("Environment creation test failed!")
        raise RuntimeError("Cannot create environment")
    
    # On Windows or if only 1 env requested, use DummyVecEnv
    use_dummy = (platform.system() == "Windows" or n_envs == 1)
    
    if use_dummy:
        logger.info(f"Using DummyVecEnv with {n_envs} environments")
        env = DummyVecEnv([make_env(data, config, i, seed) for i in range(n_envs)])
    else:
        logger.info(f"Using SubprocVecEnv with {n_envs} environments")
        try:
            # Try subprocess first
            env = SubprocVecEnv([make_env(data, config, i, seed) for i in range(n_envs)])
        except Exception as e:
            logger.warning(f"SubprocVecEnv failed: {e}, falling back to DummyVecEnv")
            env = DummyVecEnv([make_env(data, config, i, seed) for i in range(n_envs)])
    
    return env

# ═══════════════════════════════════════════════════════════════════
# Enhanced Callbacks with better error handling
# ═══════════════════════════════════════════════════════════════════

class DetailedLoggingCallback(BaseCallback):
    """Enhanced callback for detailed training metrics with error handling"""
    
    def __init__(self, log_freq: int = 100, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_metrics = []
        
    def _on_step(self) -> bool:
        try:
            # Log every log_freq steps
            if self.n_calls % self.log_freq == 0:
                # Get info from all environments
                infos = self.locals.get("infos", [])
                
                for info in infos:
                    if info and "episode" in info:
                        self.episode_rewards.append(info["episode"]["r"])
                        self.episode_lengths.append(info["episode"]["l"])
                        
                        # Extract custom metrics safely
                        metrics = {
                            "reward": info["episode"]["r"],
                            "length": info["episode"]["l"],
                            "balance": info.get("balance", 0),
                            "drawdown": info.get("drawdown", 0),
                            "sharpe": info.get("sharpe_ratio", 0),
                            "win_rate": info.get("win_rate", 0),
                            "total_trades": info.get("total_trades", 0),
                        }
                        self.episode_metrics.append(metrics)
                        
                        # Log to tensorboard safely
                        if len(self.episode_rewards) > 0:
                            self.logger.record("rollout/ep_rew_mean", np.mean(self.episode_rewards[-100:]))
                            self.logger.record("rollout/ep_len_mean", np.mean(self.episode_lengths[-100:]))
                            self.logger.record("metrics/balance", metrics["balance"])
                            self.logger.record("metrics/drawdown", metrics["drawdown"])
                            self.logger.record("metrics/sharpe_ratio", metrics["sharpe"])
                            self.logger.record("metrics/win_rate", metrics["win_rate"])
                            
                            if self.verbose > 0:
                                logger.info(
                                    f"Episode {len(self.episode_rewards)} | "
                                    f"Reward: {metrics['reward']:.2f} | "
                                    f"Balance: ${metrics['balance']:.2f} | "
                                    f"Drawdown: {metrics['drawdown']:.2%} | "
                                    f"Sharpe: {metrics['sharpe']:.3f}"
                                )
        except Exception as e:
            logger.warning(f"Error in logging callback: {e}")
        
        return True

class ModuleHealthCallback(BaseCallback):
    """Monitor health of all trading modules with error handling"""
    
    def __init__(self, check_freq: int = 1000):
        super().__init__()
        self.check_freq = check_freq
        
    def _on_step(self) -> bool:
        try:
            if self.n_calls % self.check_freq == 0:
                # Get environment safely
                env = None
                if hasattr(self.training_env, 'envs') and len(self.training_env.envs) > 0:
                    env = self.training_env.envs[0]
                    
                    if hasattr(env, 'unwrapped'):
                        env = env.unwrapped
                        
                    # Check module health
                    if hasattr(env, 'pipeline') and env.pipeline:
                        for module_name, module in env.pipeline._module_map.items():
                            if hasattr(module, 'get_health_status'):
                                try:
                                    health = module.get_health_status()
                                    self.logger.record(f"health/{module_name}", 
                                                     1.0 if health.get('status') == 'OK' else 0.0)
                                    
                                    if health.get('status') != 'OK':
                                        logger.warning(f"Module {module_name} unhealthy: {health}")
                                except Exception as e:
                                    logger.warning(f"Health check failed for {module_name}: {e}")
                    
                    # Check voting committee
                    if hasattr(env, 'arbiter') and env.arbiter:
                        try:
                            consensus = getattr(env.arbiter, 'last_consensus', 0)
                            self.logger.record("voting/consensus", consensus)
                            
                            weights = getattr(env.arbiter, 'weights', [])
                            for i, w in enumerate(weights):
                                self.logger.record(f"voting/weight_{i}", w)
                        except Exception as e:
                            logger.warning(f"Voting system check failed: {e}")
        except Exception as e:
            logger.warning(f"Error in module health callback: {e}")
        
        return True

class SafeCheckpointCallback(BaseCallback):
    """Save checkpoints with full system state and error handling"""
    
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "ppo_model"):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix
        self.save_path.mkdir(parents=True, exist_ok=True)
        
    def _on_step(self) -> bool:
        try:
            if self.n_calls % self.save_freq == 0:
                # Save model
                model_path = self.save_path / f"{self.name_prefix}_{self.n_calls}.zip"
                self.model.save(model_path)
                
                # Save additional metadata
                metadata = {
                    "timesteps": self.n_calls,
                    "timestamp": datetime.now().isoformat(),
                    "performance": {
                        "episodes": len(getattr(self, 'episode_rewards', [])),
                        "mean_reward": np.mean(getattr(self, 'episode_rewards', [0])[-100:]),
                    }
                }
                
                metadata_path = self.save_path / f"{self.name_prefix}_{self.n_calls}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                    
                logger.info(f"Checkpoint saved: {model_path}")
                
                # Clean old checkpoints (keep last 5)
                try:
                    checkpoints = sorted(self.save_path.glob(f"{self.name_prefix}_*.zip"))
                    if len(checkpoints) > 5:
                        for old_checkpoint in checkpoints[:-5]:
                            old_checkpoint.unlink()
                            # Also remove metadata
                            old_metadata = old_checkpoint.with_suffix('').with_suffix('.json')
                            if old_metadata.exists():
                                old_metadata.unlink()
                except Exception as e:
                    logger.warning(f"Failed to clean old checkpoints: {e}")
                    
        except Exception as e:
            logger.error(f"Error in checkpoint callback: {e}")
        
        return True

# ═══════════════════════════════════════════════════════════════════
# PPO Configuration - Same as before
# ═══════════════════════════════════════════════════════════════════

def create_ppo_model(env, config: Dict[str, Any]):
    """Create PPO model with optimal hyperparameters"""
    
    # Network architecture
    policy_kwargs = dict(
        net_arch=[dict(pi=[256, 128], vf=[256, 128])],
        activation_fn=nn.Tanh,  # More stable than ReLU
        normalize_images=False,
    )
    
    # Create PPO model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.get("learning_rate", 3e-4),
        n_steps=config.get("n_steps", 2048),
        batch_size=config.get("batch_size", 64),
        n_epochs=config.get("n_epochs", 10),
        gamma=config.get("gamma", 0.99),
        gae_lambda=config.get("gae_lambda", 0.95),
        clip_range=config.get("clip_range", 0.2),
        clip_range_vf=config.get("clip_range_vf", None),
        ent_coef=config.get("ent_coef", 0.01),
        vf_coef=config.get("vf_coef", 0.5),
        max_grad_norm=config.get("max_grad_norm", 0.5),
        target_kl=config.get("target_kl", 0.01),
        tensorboard_log="logs/tensorboard",
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=config.get("seed", 42),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    return model

# ═══════════════════════════════════════════════════════════════════
# Training Function with better error handling
# ═══════════════════════════════════════════════════════════════════

def train_ppo(args):
    """Main training function with enhanced error handling"""
    
    logger.info("=" * 60)
    logger.info("PPO Training System - Production Mode (Windows Compatible)")
    logger.info("=" * 60)
    logger.info(f"Platform: {platform.system()}")
    logger.info(f"Python: {sys.version}")
    logger.info(f"PyTorch: {torch.__version__}")
    
    # Create configuration
    config = TradingConfig(
        initial_balance=10000.0,
        max_steps=200,
        debug=args.debug,
        live_mode=False,
    )
    
    # Load market data
    logger.info("Loading market data...")
    try:
        data = load_data("data/processed")
        
        if not data:
            logger.error("No data found in data/processed")
            return
            
        logger.info(f"Loaded data for {len(data)} instruments")
        
        # Validate data briefly
        for instrument, timeframes in data.items():
            for tf, df in timeframes.items():
                if df.empty:
                    logger.warning(f"Empty data for {instrument}/{tf}")
                else:
                    logger.info(f"{instrument}/{tf}: {len(df)} bars")
                    
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    
    # Create environments with Windows compatibility
    logger.info("Creating training environments...")
    try:
        # Use fewer environments on Windows to avoid issues
        n_envs = 1 if platform.system() == "Windows" else min(4, os.cpu_count())
        logger.info(f"Using {n_envs} parallel environments")
        
        train_env = create_envs(data, config, n_envs=n_envs, seed=args.seed)
        eval_env = create_envs(data, config, n_envs=1, seed=args.seed + 1000)
        
        logger.info("Environments created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create environments: {e}")
        return
    
    # Create model
    logger.info("Creating PPO model...")
    try:
        model_config = {
            "learning_rate": args.lr,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "gamma": args.gamma,
            "clip_range": args.clip_range,
            "ent_coef": args.ent_coef,
            "vf_coef": args.vf_coef,
            "max_grad_norm": args.max_grad_norm,
            "target_kl": args.target_kl,
            "seed": args.seed,
        }
        
        model = create_ppo_model(train_env, model_config)
        logger.info("PPO model created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        train_env.close()
        eval_env.close()
        return
    
    # Setup callbacks
    callbacks = [
        DetailedLoggingCallback(log_freq=100),
        ModuleHealthCallback(check_freq=1000),
        SafeCheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path="checkpoints",
            name_prefix="ppo_stable"
        ),
        EvalCallback(
            eval_env,
            best_model_save_path="models/best",
            log_path="logs/eval",
            eval_freq=args.eval_freq,
            deterministic=True,
            render=False,
            n_eval_episodes=5,  # Reduced for faster evaluation
        ),
    ]
    
    # Start training
    logger.info("Starting training...")
    logger.info(f"Total timesteps: {args.timesteps:,}")
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            log_interval=10,
            tb_log_name="ppo_stable",
            reset_num_timesteps=True,
            progress_bar=True,
        )
        
        # Save final model
        final_model_path = "models/ppo_final_model.zip"
        model.save(final_model_path)
        logger.info(f"Training completed! Model saved to {final_model_path}")
        
        # Save training summary
        summary = {
            "training_completed": datetime.now().isoformat(),
            "total_timesteps": args.timesteps,
            "config": model_config,
            "final_model_path": final_model_path,
            "platform": platform.system(),
        }
        
        with open("models/training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
            
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        model.save("models/ppo_interrupted_model.zip")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.exception(e)
        
    finally:
        logger.info("Cleaning up...")
        try:
            train_env.close()
            eval_env.close()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

# ═══════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PPO Training for AI Trading System")
    
    # Training duration
    parser.add_argument("--timesteps", type=int, default=50000,  # Reduced default
                       help="Total training timesteps")
    
    # PPO hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size")
    parser.add_argument("--n_epochs", type=int, default=10,
                       help="Number of epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")
    parser.add_argument("--n_steps", type=int, default=1024,  # Reduced for faster training
                       help="Number of steps per update")
    parser.add_argument("--clip_range", type=float, default=0.2,
                       help="PPO clip range")
    parser.add_argument("--ent_coef", type=float, default=0.01,
                       help="Entropy coefficient")
    parser.add_argument("--vf_coef", type=float, default=0.5,
                       help="Value function coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                       help="Maximum gradient norm")
    parser.add_argument("--target_kl", type=float, default=0.01,
                       help="Target KL divergence")
    
    # Training settings
    parser.add_argument("--checkpoint_freq", type=int, default=5000,  # More frequent
                       help="Checkpoint frequency")
    parser.add_argument("--eval_freq", type=int, default=2500,  # More frequent
                       help="Evaluation frequency")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create directories
    directories = [
        "logs/training", "logs/eval", "logs/tensorboard",
        "checkpoints", "models/best", "data/processed"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Run training
    train_ppo(args)

if __name__ == "__main__":
    main()