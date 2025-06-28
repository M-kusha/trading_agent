# train/train_ppo_hybrid.py
"""
Hybrid PPO Trading Script - Switches between Offline and Live Demo Training
Can run offline pre-training OR live demo learning with the same script

═══════════════════════════════════════════════════════════════════
QUICK COMMAND REFERENCE
═══════════════════════════════════════════════════════════════════

🚀 LIVE DEMO TRADING (Real MT5 Demo Account):
   py train/train_ppo_hybrid.py --preset demo_online --timesteps 1000 --debug
   py train/train_ppo_hybrid.py --preset demo_online --auto-pretrained --timesteps 1000 --debug

📊 OFFLINE TRAINING (Historical Data):
   py train/train_ppo_hybrid.py --preset aggressive --timesteps 100000 --debug
   py train/train_ppo_hybrid.py --preset research --timesteps 50000 --debug

🔄 HYBRID WORKFLOW (Recommended):
   1. py train/train_ppo_hybrid.py --preset aggressive --timesteps 100000
   2. py train/train_ppo_hybrid.py --preset demo_online --auto-pretrained --timesteps 10000

🛠️ OTHER OPTIONS:
   py train/train_ppo_hybrid.py --preset conservative --balance 1000 --timesteps 5000
   py train/train_ppo_hybrid.py --mode live --balance 2000 --timesteps 2000
   py train/train_ppo_hybrid.py --preset demo_online --pretrained models/my_model.zip

⚠️ LIVE TRADING REQUIREMENTS:
   - MT5 running with DEMO account
   - EURUSD and XAUUSD in Market Watch
   - Sufficient demo balance ($2000+)
   - Automated trading enabled

🛑 TO STOP: Ctrl+C (saves model safely)

═══════════════════════════════════════════════════════════════════
"""

import os
import platform
import sys
import logging
import argparse
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

# Import MT5 for live trading
import MetaTrader5 as mt5

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
logger = logging.getLogger("HybridPPOTraining")

# ═══════════════════════════════════════════════════════════════════
# LIVE DATA COLLECTION
# ═══════════════════════════════════════════════════════════════════

def connect_mt5() -> bool:
    """Connect to MT5 for live data/trading"""
    try:
        if not mt5.initialize():
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False
        
        account_info = mt5.account_info()
        if account_info:
            logger.info(f"🔗 Connected to MT5 - Account: {account_info.login}, Server: {account_info.server}")
            logger.info(f"💰 Demo Balance: ${account_info.balance:.2f}")
        else:
            logger.warning("No account info available")
        
        return True
    except Exception as e:
        logger.error(f"MT5 connection failed: {e}")
        return False

def get_live_market_data(config: TradingConfig) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Get current market data from MT5 for live trading"""
    logger.info("🔴 LIVE MODE: Fetching real-time market data from MT5...")
    
    if not connect_mt5():
        raise RuntimeError("Cannot connect to MT5 for live data")
    
    # Symbol mapping
    symbol_map = {
        "EUR/USD": "EURUSD",
        "XAU/USD": "XAUUSD"
    }
    
    # Timeframe mapping
    tf_map = {
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1
    }
    
    live_data = {}
    
    for instrument in config.instruments:
        mt5_symbol = symbol_map.get(instrument, instrument.replace("/", ""))
        
        # Select symbol
        if not mt5.symbol_select(mt5_symbol, True):
            logger.error(f"Cannot select symbol: {mt5_symbol}")
            continue
            
        live_data[instrument] = {}
        
        for tf_name in config.timeframes:
            if tf_name not in tf_map:
                continue
                
            logger.info(f"📊 Fetching {instrument} {tf_name} live data...")
            
            # Get last 500 bars for context
            rates = mt5.copy_rates_from_pos(mt5_symbol, tf_map[tf_name], 0, 500)
            
            if rates is None or len(rates) == 0:
                logger.error(f"No live data for {instrument} {tf_name}")
                continue
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Rename columns to match expected format
            df = df.rename(columns={
                'time': 'time',
                'open': 'open',
                'high': 'high',
                'low': 'low', 
                'close': 'close',
                'tick_volume': 'volume'
            })
            
            # Add required technical indicators
            df['volatility'] = df['close'].pct_change().rolling(20).std().fillna(0.01)
            df['volatility'] = df['volatility'].clip(lower=1e-7)
            
            # Ensure numeric types
            numeric_cols = df.select_dtypes(include=['number']).columns
            df[numeric_cols] = df[numeric_cols].astype(np.float32)
            
            live_data[instrument][tf_name] = df
            
            logger.info(f"✅ {instrument} {tf_name}: {len(df)} bars, latest: {df['time'].iloc[-1]}")
    
    logger.info("🔴 Live market data ready for trading!")
    return live_data

def create_live_trading_data(config: TradingConfig) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Create a minimal live trading dataset"""
    logger.info("🔴 LIVE MODE: Creating live trading environment...")
    
    # For live trading, we need minimal historical context
    # The environment will get real-time updates during trading
    live_data = {}
    
    for instrument in config.instruments:
        live_data[instrument] = {}
        for tf in config.timeframes:
            # Create minimal DataFrame with current time
            current_time = datetime.now()
            
            # Dummy historical data for context (will be replaced by live data)
            df = pd.DataFrame({
                'time': [current_time - timedelta(hours=i) for i in range(100, 0, -1)],
                'open': [1.0800 + np.random.randn() * 0.001 for _ in range(100)],
                'high': [1.0801 + np.random.randn() * 0.001 for _ in range(100)],
                'low': [1.0799 + np.random.randn() * 0.001 for _ in range(100)],
                'close': [1.0800 + np.random.randn() * 0.001 for _ in range(100)],
                'volume': [1000.0 for _ in range(100)],
                'volatility': [0.01 for _ in range(100)],
            })
            
            # Convert to float32
            numeric_cols = df.select_dtypes(include=['number']).columns
            df[numeric_cols] = df[numeric_cols].astype(np.float32)
            
            live_data[instrument][tf] = df
    
    return live_data

# ═══════════════════════════════════════════════════════════════════
# SMART DATA LOADING (OFFLINE vs LIVE)
# ═══════════════════════════════════════════════════════════════════

def load_training_data(config: TradingConfig) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Smart data loading - switches between offline and live based on config"""
    
    if config.live_mode:
        logger.info("🔴 LIVE MODE DETECTED - Switching to live market data")
        try:
            return get_live_market_data(config)
        except Exception as e:
            logger.error(f"Live data failed: {e}")
            logger.info("Falling back to minimal live trading data...")
            return create_live_trading_data(config)
    else:
        logger.info("📊 OFFLINE MODE - Loading historical data")
        try:
            data = load_data(config.data_dir)
            if data:
                logger.info(f"✅ Loaded historical data for {len(data)} instruments")
                return data
            else:
                logger.warning("No historical data found, creating dummy data...")
                return create_dummy_data(config)
        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
            return create_dummy_data(config)

def create_dummy_data(config: TradingConfig) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Create dummy data for testing when no real data available"""
    logger.info("🧪 Creating dummy data for testing...")
    
    dummy_data = {}
    for instrument in config.instruments:
        dummy_data[instrument] = {}
        for tf in config.timeframes:
            # Create realistic dummy OHLCV data
            df = pd.DataFrame({
                'time': pd.date_range(start='2023-01-01', periods=1000, freq='1H'),
                'open': np.random.randn(1000).cumsum() + 1.2000,
                'high': np.random.randn(1000).cumsum() + 1.2010,
                'low': np.random.randn(1000).cumsum() + 1.1990,
                'close': np.random.randn(1000).cumsum() + 1.2000,
                'volume': np.random.randint(100, 1000, 1000),
                'volatility': np.random.uniform(0.01, 0.05, 1000),
            })
            
            # Convert to float32
            numeric_cols = df.select_dtypes(include=['number']).columns
            df[numeric_cols] = df[numeric_cols].astype(np.float32)
            
            dummy_data[instrument][tf] = df
            
    logger.info("✅ Dummy data created")
    return dummy_data

# ═══════════════════════════════════════════════════════════════════
# ENHANCED LIVE TRADING CALLBACKS
# ═══════════════════════════════════════════════════════════════════

class LiveTradingCallback(BaseCallback):
    """Special callback for live trading mode"""
    
    def __init__(self, config: TradingConfig):
        super().__init__()
        self.config = config
        self.live_trades = []
        
    def _on_step(self) -> bool:
        if not self.config.live_mode:
            return True
            
        try:
            # Get environment
            if hasattr(self.training_env, 'envs') and len(self.training_env.envs) > 0:
                env = self.training_env.envs[0]
                if hasattr(env, 'unwrapped'):
                    env = env.unwrapped
                    
                # Check for new live trades
                if hasattr(env, 'trades') and env.trades:
                    for trade in env.trades:
                        if trade not in self.live_trades:
                            self.live_trades.append(trade)
                            
                            # Log live trade
                            logger.info(
                                f"🔴 LIVE TRADE: {trade.get('instrument', 'Unknown')} "
                                f"Size: {trade.get('size', 0):.3f} "
                                f"PnL: ${trade.get('pnl', 0):.2f}"
                            )
                            
                            # Record to tensorboard
                            self.logger.record("live_trading/trade_count", len(self.live_trades))
                            self.logger.record("live_trading/trade_pnl", trade.get('pnl', 0))
                            
        except Exception as e:
            logger.warning(f"Error in live trading callback: {e}")
            
        return True

# ═══════════════════════════════════════════════════════════════════
# ENVIRONMENT CREATION (SAME AS BEFORE)
# ═══════════════════════════════════════════════════════════════════

def apply_reward_system_fixes(env):
    """Apply reward system fixes (same as before)"""
    try:
        if hasattr(env, 'reward_shaper'):
            rs = env.reward_shaper
            logger.info("🔧 Applying reward system fixes...")
            
            if hasattr(rs, '_reward_history'):
                rs._reward_history.clear()
            if hasattr(rs, '_pnl_history'):
                rs._pnl_history.clear()
            
            if hasattr(rs, 'no_trade_penalty_weight'):
                rs.no_trade_penalty_weight = 0.02
            if hasattr(rs, 'dd_pen_weight'):
                rs.dd_pen_weight = 1.0
            if hasattr(rs, 'sharpe_bonus_weight'):
                rs.sharpe_bonus_weight = 0.1
            
            if hasattr(rs, 'win_bonus_weight'):
                rs.win_bonus_weight = 1.5
            if hasattr(rs, 'min_trade_bonus'):
                rs.min_trade_bonus = 1.0
            
            rs._baseline_bonus = 0.1
            
            logger.info("✅ Reward system fixes applied!")
        else:
            logger.warning("⚠️  No reward_shaper found")
    except Exception as e:
        logger.error(f"❌ Failed to apply reward fixes: {e}")

def test_env_creation(data: Dict, config: TradingConfig) -> bool:
    """Test environment creation"""
    try:
        mode_str = "LIVE" if config.live_mode else "OFFLINE"
        logger.info(f"Testing {mode_str} environment creation...")
        
        env = EnhancedTradingEnv(data, config)
        apply_reward_system_fixes(env)
        
        obs, info = env.reset()
        logger.info(f"✅ {mode_str} environment test passed - obs shape: {obs.shape}")
        
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
    """Create environment instance"""
    def _init():
        try:
            env = EnhancedTradingEnv(data, config)
            apply_reward_system_fixes(env)
            env = Monitor(env, filename=f"logs/training/monitor_{rank}.csv")
            env.seed(seed + rank)
            
            mode_str = "LIVE" if config.live_mode else "OFFLINE"
            logger.info(f"✅ {mode_str} environment {rank} created")
            return env
        except Exception as e:
            logger.error(f"Failed to create environment {rank}: {e}")
            raise
    
    set_random_seed(seed)
    return _init

def create_envs(data: Dict, config: TradingConfig, n_envs: int = 1, seed: int = 42):
    """Create vectorized environments"""
    if not test_env_creation(data, config):
        logger.error("Environment creation test failed!")
        raise RuntimeError("Cannot create environment")
    
    # Force single environment for live trading
    if config.live_mode:
        n_envs = 1
        logger.info("🔴 LIVE MODE: Using single environment")
    elif platform.system() == "Windows":
        n_envs = 1
        logger.info("Windows detected - using single environment")
    
    logger.info(f"Creating {n_envs} environments using DummyVecEnv")
    env = DummyVecEnv([make_env(data, config, i, seed) for i in range(n_envs)])
    
    return env

# ═══════════════════════════════════════════════════════════════════
# PPO MODEL CREATION (SAME AS BEFORE)
# ═══════════════════════════════════════════════════════════════════

def create_ppo_model(env, config: TradingConfig):
    """Create PPO model"""
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
    
    device_str = "GPU" if torch.cuda.is_available() else "CPU"
    logger.info(f"🤖 PPO model created using {device_str}")
    return model

# ═══════════════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════════

def train_ppo_hybrid(config: TradingConfig, pretrained_model_path: Optional[str] = None):
    """Main hybrid training function with transfer learning support"""
    
    mode_str = "🔴 LIVE DEMO" if config.live_mode else "📊 OFFLINE"
    logger.info("=" * 60)
    logger.info(f"HYBRID PPO TRAINING - {mode_str} MODE")
    logger.info("=" * 60)
    logger.info(f"Platform: {platform.system()}")
    logger.info(f"Configuration:\n{config}")
    
    # Load appropriate data based on mode
    logger.info(f"{mode_str} Loading market data...")
    try:
        data = load_training_data(config)
        
        logger.info(f"✅ Data ready for {len(data)} instruments:")
        for instrument, timeframes in data.items():
            for tf, df in timeframes.items():
                if not df.empty:
                    latest_time = df['time'].iloc[-1] if 'time' in df.columns else "Unknown"
                    logger.info(f"   {instrument}/{tf}: {len(df)} bars (latest: {latest_time})")
    except Exception as e:
        logger.error(f"❌ Failed to load data: {e}")
        return
    
    # Create environments
    logger.info(f"🔧 Creating {mode_str} environments...")
    try:
        train_env = create_envs(data, config, n_envs=config.num_envs, seed=config.init_seed)
        eval_env = create_envs(data, config, n_envs=1, seed=config.init_seed + 1000)
        logger.info("✅ Environments created successfully")
    except Exception as e:
        logger.error(f"❌ Failed to create environments: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create or load model (TRANSFER LEARNING SUPPORT)
    logger.info("🤖 Setting up PPO model...")
    try:
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            logger.info(f"📥 Loading pretrained model from: {pretrained_model_path}")
            model = PPO.load(pretrained_model_path, env=train_env)
            logger.info("✅ Pretrained model loaded - continuing training!")
            
            # Optionally adjust learning rate for fine-tuning
            if config.live_mode:
                model.learning_rate = config.learning_rate * 0.5  # Slower learning for live
                logger.info(f"🔧 Adjusted learning rate for live mode: {model.learning_rate}")
        else:
            logger.info("🆕 Creating new PPO model...")
            model = create_ppo_model(train_env, config)
            logger.info("✅ New PPO model created")
            
    except Exception as e:
        logger.error(f"❌ Failed to setup model: {e}")
        train_env.close()
        eval_env.close()
        return
    
    # Setup callbacks (with live trading callback if needed)
    callbacks = [
        LiveTradingCallback(config) if config.live_mode else None,
    ]
    callbacks = [cb for cb in callbacks if cb is not None]  # Remove None
    
    # Add standard callbacks
    callbacks.extend([
        CheckpointCallback(
            save_freq=config.checkpoint_freq,
            save_path=config.checkpoint_dir,
            name_prefix="ppo_hybrid"
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
    ])
    
    # Start training
    total_timesteps = config.final_training_steps
    logger.info(f"🚀 Starting {mode_str} training...")
    logger.info(f"Total timesteps: {total_timesteps:,}")
    
    if config.live_mode:
        logger.info("🔴 WARNING: LIVE DEMO TRADING ACTIVE!")
        logger.info("🔴 Real trades will be placed in demo account!")
        
        # Give user a chance to cancel
        import time
        for i in range(5, 0, -1):
            logger.info(f"🔴 Starting live trading in {i} seconds... (Ctrl+C to cancel)")
            time.sleep(1)
    
    try:
        start_time = datetime.now()
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=config.log_interval,
            tb_log_name=f"ppo_{'live' if config.live_mode else 'offline'}",
            reset_num_timesteps=True,
            progress_bar=True,
        )
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        # Save final model (SINGLE MODEL APPROACH)
        final_model_path = os.path.join(config.model_dir, "ppo_trading_model.zip")
        model.save(final_model_path)
        
        phase = "LIVE" if config.live_mode else "OFFLINE"
        logger.info(f"✅ {phase} training completed!")
        logger.info(f"📁 Model saved to {final_model_path}")
        logger.info(f"⏱️  Training duration: {training_duration}")
        
        # Save training summary
        summary = {
            "training_completed": end_time.isoformat(),
            "training_duration": str(training_duration),
            "training_phase": "live" if config.live_mode else "offline",
            "was_pretrained": pretrained_model_path is not None,
            "pretrained_from": pretrained_model_path,
            "total_timesteps": total_timesteps,
            "final_model_path": final_model_path,
            "platform": platform.system(),
        }
        
        summary_path = os.path.join(config.model_dir, "training_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📋 Training summary saved to {summary_path}")
        
        # Next phase suggestion
        if not config.live_mode:
            logger.info("💡 Next: Run with --preset demo_online --pretrained to continue online!")
        
    except KeyboardInterrupt:
        logger.info("⏹️  Training interrupted by user")
        interrupted_path = os.path.join(config.model_dir, "ppo_trading_model_interrupted.zip")
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
            if config.live_mode:
                mt5.shutdown()
                logger.info("🔴 MT5 connection closed")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


def main():
    parser = argparse.ArgumentParser(description="Hybrid PPO Training - Single Model Transfer Learning")
    
    # Mode selection
    parser.add_argument("--mode", choices=["test", "production", "live"], default="test",
                       help="Training mode")
    parser.add_argument("--preset", choices=["conservative", "aggressive", "research", "demo_online"], 
                       help="Use configuration preset")
    
    # Transfer learning support
    parser.add_argument("--pretrained", type=str, default=None,
                       help="Path to pretrained model (for transfer learning)")
    parser.add_argument("--auto-pretrained", action="store_true",
                       help="Automatically use models/ppo_trading_model.zip if it exists")
    
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
    
    # Handle pretrained model
    pretrained_path = None
    if args.pretrained:
        pretrained_path = args.pretrained
        logger.info(f"🔄 Using specified pretrained model: {pretrained_path}")
    elif args.auto_pretrained:
        auto_path = "models/ppo_trading_model.zip"
        if os.path.exists(auto_path):
            pretrained_path = auto_path
            logger.info(f"🔄 Auto-detected pretrained model: {pretrained_path}")
        else:
            logger.info("💡 No pretrained model found, starting fresh training")
    
    # Save configuration
    config_path = os.path.join(config.log_dir, "training_config.json")
    config.save_config(config_path)
    logger.info(f"📝 Configuration saved to {config_path}")
    
    # Show mode clearly
    if config.live_mode:
        logger.info("🔴" * 20)
        logger.info("🔴 LIVE DEMO TRADING MODE ACTIVE!")
        if pretrained_path:
            logger.info("🔴 TRANSFER LEARNING: Continuing from offline model!")
        logger.info("🔴 Make sure MT5 is running with demo account!")
        logger.info("🔴" * 20)
    else:
        logger.info("📊 Offline training mode - using historical data")
        if pretrained_path:
            logger.info("🔄 Continuing training from existing model")
    
    # Run training
    train_ppo_hybrid(config, pretrained_path)

if __name__ == "__main__":
    main()