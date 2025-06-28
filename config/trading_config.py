# train/train_ppo_fixed.py
"""
FIXED PPO Training Script - Compatible with Refactored Environment
Integrates centralized configuration and reward system fixes
"""

from html import parser
import os
import platform
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

# FIXED: Updated imports for refactored structure
try:
    from envs import EnhancedTradingEnv, TradingConfig
    from envs.config import ConfigPresets, ConfigFactory
    print("✅ Successfully imported refactored environment")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you have the refactored environment structure")
    sys.exit(1)

try:
    from utils.data_utils import load_data
    print("✅ Data utils imported")
except ImportError:
    print("⚠️  Data utils not found, will create fallback")
    def load_data(path):
        """Fallback data loader"""
        return {}

# Create necessary directories early
log_dirs = [
    'logs/training', 'logs/regime', 'logs/strategy', 'logs/checkpoints',
    'logs/reward', 'logs/tensorboard', 'checkpoints', 'models/best'
]
for log_dir in log_dirs:
    os.makedirs(log_dir, exist_ok=True)

# Configure logging with UTF-8 support
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
# REWARD SYSTEM FIXES
# ═══════════════════════════════════════════════════════════════════

def apply_reward_system_fixes(env):
    """
    Apply the reward system fixes we identified from the logs
    """
    try:
        if hasattr(env, 'reward_shaper'):
            rs = env.reward_shaper
            
            logger.info("🔧 Applying reward system fixes...")
            
            # 1. Reset negative history that's causing issues
            if hasattr(rs, '_reward_history'):
                rs._reward_history.clear()
            if hasattr(rs, '_pnl_history'):
                rs._pnl_history.clear()
            
            # 2. Fix harsh penalties
            if hasattr(rs, 'no_trade_penalty_weight'):
                rs.no_trade_penalty_weight = 0.02  # Reduced from 0.05
            if hasattr(rs, 'dd_pen_weight'):
                rs.dd_pen_weight = 1.0  # Reduced from 2.0
            if hasattr(rs, 'sharpe_bonus_weight'):
                rs.sharpe_bonus_weight = 0.1  # Reduced from 0.3
            
            # 3. Increase positive incentives
            if hasattr(rs, 'win_bonus_weight'):
                rs.win_bonus_weight = 1.5  # Increased
            if hasattr(rs, 'min_trade_bonus'):
                rs.min_trade_bonus = 1.0  # Increased
            
            # 4. Add baseline positivity
            rs._baseline_bonus = 0.1
            
            # 5. Add enhanced shape_reward method
            original_shape_reward = rs.shape_reward
            
            def enhanced_shape_reward(*args, **kwargs):
                reward = original_shape_reward(*args, **kwargs)
                
                # Add bootstrapping bonus for first 50 trades
                if rs._total_trades < 50:
                    bootstrap_bonus = 0.2 * (1 - rs._total_trades / 50)
                    reward += bootstrap_bonus
                
                # Add baseline bonus
                reward += getattr(rs, '_baseline_bonus', 0.1)
                
                # Ensure positive exploration early on
                if rs._total_trades < 10:
                    reward += 0.15  # Extra exploration bonus
                
                return reward
            
            rs.shape_reward = enhanced_shape_reward
            
            logger.info("✅ Reward system fixes applied successfully!")
            logger.info(f"   - No trade penalty: {getattr(rs, 'no_trade_penalty_weight', 'N/A')}")
            logger.info(f"   - Drawdown penalty: {getattr(rs, 'dd_pen_weight', 'N/A')}")
            logger.info(f"   - Win bonus weight: {getattr(rs, 'win_bonus_weight', 'N/A')}")
            
        else:
            logger.warning("⚠️  No reward_shaper found in environment")
            
    except Exception as e:
        logger.error(f"❌ Failed to apply reward fixes: {e}")

# ═══════════════════════════════════════════════════════════════════
# ENVIRONMENT CREATION - FIXED FOR NEW STRUCTURE
# ═══════════════════════════════════════════════════════════════════

def test_env_creation(data: Dict, config: TradingConfig) -> bool:
    """Test if environment can be created successfully"""
    try:
        logger.info("Testing environment creation...")
        env = EnhancedTradingEnv(data, config)
        
        # Apply reward fixes immediately
        apply_reward_system_fixes(env)
        
        # Test reset
        obs, info = env.reset()
        logger.info(f"Environment test passed - obs shape: {obs.shape}")
        
        # Test a few steps
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            logger.info(f"Test step {i+1}: reward={reward:.4f}")
            
        env.close()
        return True
        
    except Exception as e:
        logger.error(f"Environment creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def make_env(data: Dict, config: TradingConfig, rank: int = 0, seed: int = 0):
    """Create a single environment instance with error handling"""
    def _init():
        try:
            # Create environment
            env = EnhancedTradingEnv(data, config)
            
            # Apply reward system fixes
            apply_reward_system_fixes(env)
            
            # Add monitoring
            env = Monitor(env, filename=f"logs/training/monitor_{rank}.csv")
            
            # Set seed
            env.seed(seed + rank)
            
            logger.info(f"Environment {rank} created successfully")
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
    
    # Force single environment on Windows for stability
    if platform.system() == "Windows":
        n_envs = 1
        logger.info("Windows detected - using single environment for stability")
    
    # Always use DummyVecEnv for reliability
    logger.info(f"Creating {n_envs} environments using DummyVecEnv")
    env = DummyVecEnv([make_env(data, config, i, seed) for i in range(n_envs)])
    
    return env

# ═══════════════════════════════════════════════════════════════════
# ENHANCED CALLBACKS
# ═══════════════════════════════════════════════════════════════════

class RewardMonitoringCallback(BaseCallback):
    """Monitor reward system health"""
    
    def __init__(self, check_freq: int = 500):
        super().__init__()
        self.check_freq = check_freq
        self.episode_rewards = []
        
    def _on_step(self) -> bool:
        try:
            if self.n_calls % self.check_freq == 0:
                # Get environment
                if hasattr(self.training_env, 'envs') and len(self.training_env.envs) > 0:
                    env = self.training_env.envs[0]
                    if hasattr(env, 'unwrapped'):
                        env = env.unwrapped
                        
                    # Check reward system health
                    if hasattr(env, 'reward_shaper'):
                        rs = env.reward_shaper
                        
                        # Calculate metrics
                        if hasattr(rs, '_reward_history') and rs._reward_history:
                            avg_reward = np.mean(rs._reward_history)
                            min_reward = np.min(rs._reward_history)
                            max_reward = np.max(rs._reward_history)
                            
                            # Log to tensorboard
                            self.logger.record("reward_health/avg_reward", avg_reward)
                            self.logger.record("reward_health/min_reward", min_reward)
                            self.logger.record("reward_health/max_reward", max_reward)
                            
                            # Check for issues
                            if avg_reward < -0.2:
                                logger.warning(f"⚠️  Very negative average reward: {avg_reward:.4f}")
                            elif avg_reward > -0.05:
                                logger.info(f"✅ Healthy average reward: {avg_reward:.4f}")
                                
                        # Check win rate
                        if hasattr(rs, '_total_trades') and rs._total_trades > 0:
                            win_rate = rs._winning_trades / rs._total_trades
                            self.logger.record("reward_health/win_rate", win_rate)
                            
                            if win_rate < 0.35:
                                logger.warning(f"⚠️  Low win rate: {win_rate:.2%}")
                            elif win_rate > 0.45:
                                logger.info(f"✅ Good win rate: {win_rate:.2%}")
                                
        except Exception as e:
            logger.warning(f"Error in reward monitoring: {e}")
            
        return True

class DetailedLoggingCallback(BaseCallback):
    """Enhanced callback for detailed training metrics"""
    
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
                            "trades": info.get("trades", 0),
                            "consensus": info.get("consensus", 0),
                        }
                        self.episode_metrics.append(metrics)
                        
                        # Log to tensorboard
                        if len(self.episode_rewards) > 0:
                            self.logger.record("episode/reward_mean", np.mean(self.episode_rewards[-20:]))
                            self.logger.record("episode/reward_std", np.std(self.episode_rewards[-20:]))
                            self.logger.record("episode/length_mean", np.mean(self.episode_lengths[-20:]))
                            self.logger.record("trading/balance", metrics["balance"])
                            self.logger.record("trading/drawdown", metrics["drawdown"])
                            self.logger.record("trading/trades", metrics["trades"])
                            self.logger.record("trading/consensus", metrics["consensus"])
                            
                            if self.verbose > 0:
                                recent_reward = np.mean(self.episode_rewards[-5:])
                                logger.info(
                                    f"📊 Episode {len(self.episode_rewards)} | "
                                    f"Reward: {metrics['reward']:.3f} (avg: {recent_reward:.3f}) | "
                                    f"Balance: €{metrics['balance']:.2f} | "
                                    f"Trades: {metrics['trades']} | "
                                    f"Consensus: {metrics['consensus']:.2f}"
                                )
                                
        except Exception as e:
            logger.warning(f"Error in logging callback: {e}")
        
        return True

# ═══════════════════════════════════════════════════════════════════
# PPO MODEL CREATION
# ═══════════════════════════════════════════════════════════════════

def create_ppo_model(env, config: TradingConfig):
    """Create PPO model with configuration from TradingConfig"""
    
    # Get model configuration
    model_config = config.get_model_config()
    
    # Network architecture
    policy_kwargs = dict(
        net_arch=[
            dict(
                pi=[config.policy_hidden_size, config.policy_hidden_size // 2], 
                vf=[config.value_hidden_size, config.value_hidden_size // 2]
            )
        ],
        activation_fn=nn.Tanh,
        normalize_images=False,
    )
    
    # Create PPO model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        clip_range_vf=config.clip_range_vf,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        tensorboard_log=config.tensorboard_dir,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=config.init_seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    logger.info(f"PPO model created with device: {model.device}")
    return model

# ═══════════════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════════

def train_ppo(config: TradingConfig):
    """Main training function"""
    
    logger.info("=" * 60)
    logger.info("ENHANCED PPO TRAINING - REFACTORED VERSION")
    logger.info("=" * 60)
    logger.info(f"Platform: {platform.system()}")
    logger.info(f"Mode: {'LIVE' if config.live_mode else ('TEST' if config.test_mode else 'PRODUCTION')}")
    logger.info(f"Configuration:\n{config}")
    
    # Load market data
    logger.info("📊 Loading market data...")
    try:
        data = load_data(config.data_dir)
        
        if not data:
            logger.error("❌ No data found! Creating dummy data for testing...")
            # Create dummy data for testing
            import pandas as pd
            dummy_data = {}
            for instrument in config.instruments:
                dummy_data[instrument] = {}
                for tf in config.timeframes:
                    # Create dummy OHLCV data
                    dummy_df = pd.DataFrame({
                        'open': np.random.randn(1000).cumsum() + 1.2000,
                        'high': np.random.randn(1000).cumsum() + 1.2010,
                        'low': np.random.randn(1000).cumsum() + 1.1990,
                        'close': np.random.randn(1000).cumsum() + 1.2000,
                        'volume': np.random.randint(100, 1000, 1000),
                        'volatility': np.random.uniform(0.01, 0.05, 1000),
                    })
                    dummy_data[instrument][tf] = dummy_df
            data = dummy_data
            logger.info("✅ Created dummy data for testing")
            
        logger.info(f"✅ Loaded data for {len(data)} instruments:")
        for instrument, timeframes in data.items():
            for tf, df in timeframes.items():
                if not df.empty:
                    logger.info(f"   {instrument}/{tf}: {len(df)} bars")
                    
    except Exception as e:
        logger.error(f"❌ Failed to load data: {e}")
        return
    
    # Create environments
    logger.info("🔧 Creating training environments...")
    try:
        train_env = create_envs(data, config, n_envs=config.num_envs, seed=config.init_seed)
        eval_env = create_envs(data, config, n_envs=1, seed=config.init_seed + 1000)
        
        logger.info("✅ Environments created successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to create environments: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create model
    logger.info("🤖 Creating PPO model...")
    try:
        model = create_ppo_model(train_env, config)
        logger.info("✅ PPO model created successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to create model: {e}")
        train_env.close()
        eval_env.close()
        return
    
    # Setup callbacks
    callbacks = [
        RewardMonitoringCallback(check_freq=500),
        DetailedLoggingCallback(log_freq=config.log_interval),
        CheckpointCallback(
            save_freq=config.checkpoint_freq,
            save_path=config.checkpoint_dir,
            name_prefix="ppo_enhanced"
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(config.model_dir, "best"),
            log_path=os.path.join(config.log_dir, "eval"),
            eval_freq=config.eval_freq,
            deterministic=True,
            render=False,
            n_eval_episodes=config.n_eval_episodes,
        ),
    ]
    
    # Calculate total timesteps
    if config.test_mode:
        total_timesteps = config.final_training_steps
    else:
        total_timesteps = config.final_training_steps
    
    # Start training
    logger.info("🚀 Starting training...")
    logger.info(f"Total timesteps: {total_timesteps:,}")
    
    try:
        start_time = datetime.now()
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=config.log_interval,
            tb_log_name="ppo_enhanced",
            reset_num_timesteps=True,
            progress_bar=True,
        )
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        # Save final model
        final_model_path = os.path.join(config.model_dir, "ppo_final_model.zip")
        model.save(final_model_path)
        logger.info(f"✅ Training completed! Model saved to {final_model_path}")
        logger.info(f"⏱️  Training duration: {training_duration}")
        
        # Save training summary
        summary = {
            "training_completed": end_time.isoformat(),
            "training_duration": str(training_duration),
            "total_timesteps": total_timesteps,
            "config": config.get_model_config(),
            "final_model_path": final_model_path,
            "platform": platform.system(),
        }
        
        summary_path = os.path.join(config.model_dir, "training_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"📋 Training summary saved to {summary_path}")
            
    except KeyboardInterrupt:
        logger.info("⏹️  Training interrupted by user")
        interrupted_path = os.path.join(config.model_dir, "ppo_interrupted_model.zip")
        model.save(interrupted_path)
        logger.info(f"💾 Model saved to {interrupted_path}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        logger.info("🧹 Cleaning up...")
        try:
            train_env.close()
            eval_env.close()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

# ═══════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Enhanced PPO Training with Refactored Environment")
    
    # Mode selection
    parser.add_argument("--mode", choices=["test", "production", "live"], default="test",
                       help="Training mode")
    parser.add_argument("--preset", choices=["conservative", "aggressive", "research", "demo_online"], 
                    help="Use configuration preset")
    
    # Override parameters
    parser.add_argument("--timesteps", type=int, help="Override total timesteps")
    parser.add_argument("--balance", type=float, help="Override initial balance")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create configuration
    if args.preset:
        if args.preset == "conservative":
            config = ConfigPresets.conservative_live()
        elif args.preset == "aggressive":
            config = ConfigPresets.aggressive_backtest()
        elif args.preset == "research":
            config = ConfigPresets.research_mode()
        elif args.preset == "demo_online":
            config = ConfigPresets.demo_online()
    else:
        # Create based on mode
        if args.mode == "test":
            config = TradingConfig(test_mode=True)
        elif args.mode == "live":
            config = TradingConfig(live_mode=True)
        else:
            config = TradingConfig(test_mode=False, live_mode=False)
    
    # Apply overrides
    if args.timesteps:
        config.final_training_steps = args.timesteps
    if args.balance:
        config.initial_balance = args.balance
    if args.debug:
        config.debug = True
    
    # Save configuration
    config_path = os.path.join(config.log_dir, "training_config.json")
    config.save_config(config_path)
    logger.info(f"📝 Configuration saved to {config_path}")
    
    # Run training
    train_ppo(config)

if __name__ == "__main__":
    main()