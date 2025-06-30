# train/train_ppo_hybrid.py
"""
Enhanced Hybrid PPO Trading Script with Real-time Metrics Broadcasting
Supports both Offline (CSV) and Online (MT5 Live) Training with WebSocket Updates
"""

import os
import platform
import sys
import logging
import argparse
import json
import pickle
import asyncio
import websockets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
import threading
import queue

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

from live.live_connector import LiveDataConnector, LiveTradingCallback

# Import environment and utilities
try:
    from envs import EnhancedTradingEnv, TradingConfig
    from envs.config import ConfigPresets, ConfigFactory
    print("✅ Successfully imported refactored environment")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

try:
    from utils.data_utils import load_data
except ImportError:
    def load_data(path):
        """Fallback data loader for CSV files"""
        data = {}
        if os.path.exists(path):
            for file in os.listdir(path):
                if file.endswith('.csv'):
                    instrument = file.replace('.csv', '')
                    df = pd.read_csv(os.path.join(path, file))
                    # Ensure required columns
                    required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
                    if all(col in df.columns for col in required_cols):
                        # Add volatility if not present
                        if 'volatility' not in df.columns:
                            df['volatility'] = df['close'].pct_change().rolling(20).std().fillna(0.01)
                        data[instrument] = {'H1': df}  # Assume H1 timeframe
        return data

# Create necessary directories
log_dirs = [
    'logs/training', 'logs/regime', 'logs/strategy', 'logs/checkpoints',
    'logs/reward', 'logs/tensorboard', 'checkpoints', 'models/best', 'metrics'
]
for log_dir in log_dirs:
    os.makedirs(log_dir, exist_ok=True)

# Configure logging
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
# METRICS BROADCASTING SYSTEM
# ═══════════════════════════════════════════════════════════════════

class MetricsBroadcaster:
    """Broadcasts training metrics via WebSocket to the backend"""
    
    def __init__(self, host='localhost', port=8001):
        self.host = host
        self.port = port
        self.metrics_queue = queue.Queue()
        self.ws_thread = None
        self.running = False
        self.websocket = None
        
    def start(self):
        """Start the WebSocket client thread"""
        self.running = True
        self.ws_thread = threading.Thread(target=self._run_websocket)
        self.ws_thread.daemon = True
        self.ws_thread.start()
        logger.info(f"📡 Metrics broadcaster started on ws://{self.host}:{self.port}")
        
    def stop(self):
        """Stop the WebSocket client"""
        self.running = False
        if self.ws_thread:
            self.ws_thread.join(timeout=5)
            
    def send_metrics(self, metrics: Dict[str, Any]):
        """Queue metrics for sending"""
        try:
            self.metrics_queue.put(metrics, block=False)
        except queue.Full:
            pass  # Skip if queue is full
            
    def _run_websocket(self):
        """WebSocket client loop"""
        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()
        
        while self.running:
            try:
                loop.run_until_complete(self._websocket_handler())
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if self.running:
                    asyncio.sleep(5)  # Reconnect after 5 seconds
                    
    async def _websocket_handler(self):
        """Handle WebSocket connection and message sending"""
        uri = f"ws://{self.host}:{self.port}/ws/training"
        
        try:
            async with websockets.connect(uri) as websocket:
                self.websocket = websocket
                logger.info("✅ Connected to metrics server")
                
                while self.running:
                    try:
                        # Get metrics from queue (non-blocking with timeout)
                        metrics = self.metrics_queue.get(timeout=0.1)
                        await websocket.send(json.dumps({
                            "type": "training_metrics",
                            "data": metrics,
                            "timestamp": datetime.now().isoformat()
                        }))
                    except queue.Empty:
                        continue
                    except Exception as e:
                        logger.error(f"Error sending metrics: {e}")
                        
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")

# Global metrics broadcaster
metrics_broadcaster = MetricsBroadcaster()

# ═══════════════════════════════════════════════════════════════════
# ENHANCED TRAINING CALLBACK WITH METRICS
# ═══════════════════════════════════════════════════════════════════

class EnhancedTrainingCallback(BaseCallback):
    """Enhanced callback that collects and broadcasts comprehensive training metrics"""
    
    def __init__(self, total_timesteps, config: TradingConfig, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.config = config
        self.start_time = datetime.now()
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.metrics_history = []
        
        # Performance tracking
        self.best_reward = -float('inf')
        self.current_episode_reward = 0
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        # Collect step metrics
        if self.n_calls % 100 == 0:  # Every 100 steps
            metrics = self._collect_metrics()
            self.metrics_history.append(metrics)
            
            # Broadcast metrics
            metrics_broadcaster.send_metrics(metrics)
            
            # Log to file for backup
            self._save_metrics_to_file(metrics)
            
        # Track episode rewards
        if self.locals.get('dones', [False])[0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.locals.get('episode_length', 0))
            self.episode_count += 1
            
            # Check for best episode
            if self.current_episode_reward > self.best_reward:
                self.best_reward = self.current_episode_reward
                logger.info(f"🏆 New best episode reward: {self.best_reward:.2f}")
                
            self.current_episode_reward = 0
        else:
            self.current_episode_reward += self.locals.get('rewards', [0])[0]
            
        return True
        
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive training metrics"""
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        progress = self.n_calls / self.total_timesteps
        
        # Get environment info
        env_info = {}
        if hasattr(self.training_env, 'envs') and self.training_env.envs:
            env = self.training_env.envs[0]
            if hasattr(env, 'unwrapped'):
                unwrapped_env = env.unwrapped
                if hasattr(unwrapped_env, 'get_metrics'):
                    env_info = unwrapped_env.get_metrics()
                    
        # Learning metrics from model
        learning_metrics = {}
        if hasattr(self.model, 'logger') and self.model.logger:
            learning_metrics = {
                'learning_rate': self.model.learning_rate,
                'clip_fraction': self.model.logger.name_to_value.get('train/clip_fraction', 0),
                'explained_variance': self.model.logger.name_to_value.get('train/explained_variance', 0),
                'policy_loss': self.model.logger.name_to_value.get('train/policy_loss', 0),
                'value_loss': self.model.logger.name_to_value.get('train/value_loss', 0),
                'entropy_loss': self.model.logger.name_to_value.get('train/entropy_loss', 0),
            }
            
        metrics = {
            # Training progress
            'timestep': self.n_calls,
            'total_timesteps': self.total_timesteps,
            'progress_pct': progress * 100,
            'episodes': self.episode_count,
            'elapsed_time': elapsed_time,
            'steps_per_second': self.n_calls / elapsed_time if elapsed_time > 0 else 0,
            'estimated_time_remaining': (elapsed_time / progress - elapsed_time) if progress > 0 else 0,
            
            # Performance metrics
            'episode_reward_mean': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'episode_reward_std': np.std(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'episode_length_mean': np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0,
            'best_episode_reward': self.best_reward,
            'current_episode_reward': self.current_episode_reward,
            
            # Learning metrics
            **learning_metrics,
            
            # Environment metrics
            'env_balance': env_info.get('balance', self.config.initial_balance),
            'env_total_pnl': env_info.get('total_pnl', 0),
            'env_win_rate': env_info.get('win_rate', 0),
            'env_sharpe_ratio': env_info.get('sharpe_ratio', 0),
            'env_max_drawdown': env_info.get('max_drawdown', 0),
            'env_total_trades': env_info.get('total_trades', 0),
            
            # System info
            'training_mode': 'LIVE' if self.config.live_mode else 'OFFLINE',
            'gpu_available': torch.cuda.is_available(),
            'device': str(self.model.device) if hasattr(self.model, 'device') else 'cpu',
        }
        
        return metrics
        
    def _save_metrics_to_file(self, metrics: Dict[str, Any]):
        """Save metrics to file for persistence"""
        metrics_file = f"metrics/training_metrics_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')

# ═══════════════════════════════════════════════════════════════════
# LIVE DATA COLLECTION (ENHANCED)
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
            logger.info(f"💰 Balance: ${account_info.balance:.2f}")
        
        return True
    except Exception as e:
        logger.error(f"MT5 connection failed: {e}")
        return False

def get_live_market_data(config: TradingConfig) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Get current market data from MT5 for live trading"""
    logger.info("🔴 LIVE MODE: Fetching real-time market data from MT5...")
    
    if not connect_mt5():
        raise RuntimeError("Cannot connect to MT5 for live data")
    
    symbol_map = {
        "EUR/USD": "EURUSD",
        "XAU/USD": "XAUUSD"
    }
    
    tf_map = {
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1
    }
    
    live_data = {}
    
    for instrument in config.instruments:
        mt5_symbol = symbol_map.get(instrument, instrument.replace("/", ""))
        
        if not mt5.symbol_select(mt5_symbol, True):
            logger.error(f"Cannot select symbol: {mt5_symbol}")
            continue
            
        live_data[instrument] = {}
        
        for tf_name in config.timeframes:
            if tf_name not in tf_map:
                continue
                
            logger.info(f"📊 Fetching {instrument} {tf_name} live data...")
            
            # Get last 1000 bars for training context
            rates = mt5.copy_rates_from_pos(mt5_symbol, tf_map[tf_name], 0, 1000)
            
            if rates is None or len(rates) == 0:
                logger.error(f"No live data for {instrument} {tf_name}")
                continue
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            df = df.rename(columns={
                'time': 'time',
                'open': 'open',
                'high': 'high',
                'low': 'low', 
                'close': 'close',
                'tick_volume': 'volume'
            })
            
            # Add technical indicators
            df['volatility'] = df['close'].pct_change().rolling(20).std().fillna(0.01)
            df['volatility'] = df['volatility'].clip(lower=1e-7)
            
            # Add more indicators for better training
            df['sma_20'] = df['close'].rolling(20).mean().fillna(df['close'])
            df['sma_50'] = df['close'].rolling(50).mean().fillna(df['close'])
            df['rsi'] = calculate_rsi(df['close'], 14)
            
            numeric_cols = df.select_dtypes(include=['number']).columns
            df[numeric_cols] = df[numeric_cols].astype(np.float32)
            
            live_data[instrument][tf_name] = df
            
            logger.info(f"✅ {instrument} {tf_name}: {len(df)} bars, latest: {df['time'].iloc[-1]}")
    
    logger.info("🔴 Live market data ready for training!")
    return live_data

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

# ═══════════════════════════════════════════════════════════════════
# SMART DATA LOADING (ENHANCED)
# ═══════════════════════════════════════════════════════════════════

def load_training_data(config: TradingConfig) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Smart data loading - switches between offline and live based on config"""
    
    if config.live_mode:
        logger.info("🔴 LIVE MODE DETECTED - Switching to live market data from MT5")
        try:
            data = get_live_market_data(config)
            if not data:
                raise ValueError("No live data received from MT5")
            return data
        except Exception as e:
            logger.error(f"Live data failed: {e}")
            raise
    else:
        logger.info("📊 OFFLINE MODE - Loading historical data from CSV files")
        try:
            # Try to load from CSV files
            if os.path.exists(config.data_dir):
                data = load_data(config.data_dir)
                if data:
                    logger.info(f"✅ Loaded historical data for {len(data)} instruments")
                    return data
                    
            # If no data found, create dummy data
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
            n_bars = 1000
            base_price = 1.2000 if "EUR" in instrument else 1800.0
            
            df = pd.DataFrame({
                'time': pd.date_range(start='2023-01-01', periods=n_bars, freq='1H'),
                'open': base_price + np.random.randn(n_bars).cumsum() * 0.001,
                'high': base_price + np.random.randn(n_bars).cumsum() * 0.001 + 0.001,
                'low': base_price + np.random.randn(n_bars).cumsum() * 0.001 - 0.001,
                'close': base_price + np.random.randn(n_bars).cumsum() * 0.001,
                'volume': np.random.randint(100, 1000, n_bars),
                'volatility': np.random.uniform(0.01, 0.05, n_bars),
            })
            
            # Ensure OHLC consistency
            df['high'] = df[['open', 'high', 'close']].max(axis=1)
            df['low'] = df[['open', 'low', 'close']].min(axis=1)
            
            numeric_cols = df.select_dtypes(include=['number']).columns
            df[numeric_cols] = df[numeric_cols].astype(np.float32)
            
            dummy_data[instrument][tf] = df
            
    logger.info("✅ Dummy data created")
    return dummy_data

# ═══════════════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION (ENHANCED)
# ═══════════════════════════════════════════════════════════════════

def train_ppo_hybrid(config: TradingConfig, pretrained_model_path: Optional[str] = None):
    """Main hybrid training function with enhanced metrics and real-time updates"""
    
    mode_str = "🔴 LIVE (MT5)" if config.live_mode else "📊 OFFLINE (CSV)"
    logger.info("=" * 60)
    logger.info(f"HYBRID PPO TRAINING - {mode_str} MODE")
    logger.info("=" * 60)
    logger.info(f"Configuration:\n{config}")
    
    # Start metrics broadcaster
    metrics_broadcaster.start()
    
    # Send initial status
    metrics_broadcaster.send_metrics({
        "status": "INITIALIZING",
        "mode": mode_str,
        "config": config.__dict__
    })
    
    try:
        # Load appropriate data based on mode
        logger.info(f"{mode_str} Loading market data...")
        data = load_training_data(config)
        
        logger.info(f"✅ Data ready for {len(data)} instruments:")
        for instrument, timeframes in data.items():
            for tf, df in timeframes.items():
                if not df.empty:
                    latest_time = df['time'].iloc[-1] if 'time' in df.columns else "Unknown"
                    logger.info(f"   {instrument}/{tf}: {len(df)} bars (latest: {latest_time})")
        
        # Create environments
        logger.info(f"🔧 Creating {mode_str} environments...")
        train_env = create_envs(data, config, n_envs=config.num_envs, seed=config.init_seed)
        eval_env = create_envs(data, config, n_envs=1, seed=config.init_seed + 1000)
        
        # Create or load model
        logger.info("🤖 Setting up PPO model...")
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            logger.info(f"📥 Loading pretrained model from: {pretrained_model_path}")
            model = PPO.load(pretrained_model_path, env=train_env)
            
            if config.live_mode:
                model.learning_rate = config.learning_rate * 0.5
                logger.info(f"🔧 Adjusted learning rate for live mode: {model.learning_rate}")
        else:
            logger.info("🆕 Creating new PPO model...")
            model = create_ppo_model(train_env, config)
        
        # Setup callbacks
        callbacks = [
            EnhancedTrainingCallback(
                total_timesteps=config.final_training_steps,
                config=config
            ),
            CheckpointCallback(
                save_freq=config.checkpoint_freq,
                save_path=config.checkpoint_dir,
                name_prefix=f"ppo_{config.live_mode and 'live' or 'offline'}"
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
        
        # Add live trading callback if in live mode
        if config.live_mode:
                # 1) Instantiate the connector with the two lists it needs:
                connector = LiveDataConnector(config.instruments, config.timeframes)
                # 2) Wrap it in your BaseCallback subclass:
                callbacks.append(LiveTradingCallback(connector))
        
        # Send training started status
        metrics_broadcaster.send_metrics({
            "status": "TRAINING_STARTED",
            "mode": mode_str,
            "total_timesteps": config.final_training_steps,
            "model_type": "PPO",
            "device": str(model.device)
        })
        
        # Start training
        logger.info(f"🚀 Starting {mode_str} training...")
        logger.info(f"Total timesteps: {config.final_training_steps:,}")
        
        if config.live_mode:
            logger.info("🔴 LIVE TRAINING WITH REAL MT5 DATA!")
            logger.info("🔴 Model will learn from real-time market conditions")
        
        start_time = datetime.now()
        
        model.learn(
            total_timesteps=config.final_training_steps,
            callback=callbacks,
            log_interval=config.log_interval,
            tb_log_name=f"ppo_{'live' if config.live_mode else 'offline'}",
            reset_num_timesteps=True,
            progress_bar=True,
        )
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        # Save final model
        final_model_path = os.path.join(config.model_dir, "ppo_trading_model.zip")
        model.save(final_model_path)
        
        # Send completion status
        metrics_broadcaster.send_metrics({
            "status": "TRAINING_COMPLETED",
            "mode": mode_str,
            "duration": str(training_duration),
            "final_model_path": final_model_path
        })
        
        logger.info(f"✅ {mode_str} training completed!")
        logger.info(f"📁 Model saved to {final_model_path}")
        logger.info(f"⏱️  Training duration: {training_duration}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        metrics_broadcaster.send_metrics({
            "status": "TRAINING_FAILED",
            "error": str(e)
        })
        raise
        
    finally:
        # Cleanup
        logger.info("🧹 Cleaning up...")
        try:
            train_env.close()
            eval_env.close()
            if config.live_mode:
                mt5.shutdown()
            metrics_broadcaster.stop()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

# ═══════════════════════════════════════════════════════════════════
# ENVIRONMENT CREATION (SAME AS BEFORE)
# ═══════════════════════════════════════════════════════════════════

def apply_reward_system_fixes(env):
    """Apply reward system fixes"""
    try:
        if hasattr(env, 'reward_shaper'):
            rs = env.reward_shaper
            logger.info("🔧 Applying reward system fixes...")
            
            if hasattr(rs, '_reward_history'):
                rs._reward_history.clear()
            if hasattr(rs, '_pnl_history'):
                rs._pnl_history.clear()
            
            rs.no_trade_penalty_weight = 0.02
            rs.dd_pen_weight = 1.0
            rs.sharpe_bonus_weight = 0.1
            rs.win_bonus_weight = 1.5
            rs.min_trade_bonus = 1.0
            rs._baseline_bonus = 0.1
            
            logger.info("✅ Reward system fixes applied!")
    except Exception as e:
        logger.error(f"❌ Failed to apply reward fixes: {e}")

def test_env_creation(data: Dict, config: TradingConfig) -> bool:
    """Test environment creation"""
    try:
        mode_str = "LIVE" if config.live_mode else "OFFLINE"
        logger.info(f"Testing {mode_str} environment creation...")
        
        env = EnhancedTradingEnv(data, config)
        apply_reward_system_fixes(env)
        
        # Add get_metrics method if not present
        if not hasattr(env, 'get_metrics'):
            def get_metrics(self):
                return {
                    'balance': self.balance,
                    'total_pnl': self.balance - self.initial_balance,
                    'win_rate': self.wins / max(self.total_trades, 1),
                    'total_trades': self.total_trades,
                    'sharpe_ratio': self.sharpe_ratio if hasattr(self, 'sharpe_ratio') else 0,
                    'max_drawdown': self.max_drawdown if hasattr(self, 'max_drawdown') else 0,
                }
            env.get_metrics = get_metrics.__get__(env, EnhancedTradingEnv)
        
        obs, info = env.reset()
        logger.info(f"✅ {mode_str} environment test passed - obs shape: {obs.shape}")
        
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
    
    if config.live_mode:
        n_envs = 1
        logger.info("🔴 LIVE MODE: Using single environment")
    elif platform.system() == "Windows":
        n_envs = 1
        logger.info("Windows detected - using single environment")
    
    logger.info(f"Creating {n_envs} environments using DummyVecEnv")
    env = DummyVecEnv([make_env(data, config, i, seed) for i in range(n_envs)])
    
    return env

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
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Enhanced Hybrid PPO Training")
    
    # Mode selection
    parser.add_argument("--mode", choices=["offline", "online", "test"], default="offline",
                       help="Training mode: offline (CSV), online (MT5 live), test")
    parser.add_argument("--preset", choices=["conservative", "aggressive", "research", "demo_online"], 
                       help="Use configuration preset")
    
    # Training parameters (passed from backend)
    parser.add_argument("--timesteps", type=int, help="Total timesteps")
    parser.add_argument("--lr", "--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--n_epochs", type=int, help="Number of epochs")
    parser.add_argument("--gamma", type=float, help="Discount factor")
    parser.add_argument("--n_steps", type=int, help="Number of steps")
    parser.add_argument("--clip_range", type=float, help="Clip range")
    parser.add_argument("--ent_coef", type=float, help="Entropy coefficient")
    parser.add_argument("--vf_coef", type=float, help="Value function coefficient")
    parser.add_argument("--max_grad_norm", type=float, help="Max gradient norm")
    parser.add_argument("--target_kl", type=float, help="Target KL divergence")
    parser.add_argument("--checkpoint_freq", type=int, help="Checkpoint frequency")
    parser.add_argument("--eval_freq", type=int, help="Evaluation frequency")
    parser.add_argument("--num_envs", type=int, help="Number of environments")
    
    # Data and model paths
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Directory with CSV files")
    parser.add_argument("--pretrained", type=str, help="Path to pretrained model")
    parser.add_argument("--auto-pretrained", action="store_true", help="Auto-load pretrained model")
    
    # Other options
    parser.add_argument("--balance", type=float, help="Initial balance")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create configuration based on mode
    if args.mode == "online":
        # Force online/live mode
        config = ConfigPresets.demo_online()
        config.live_mode = True
        logger.info("🔴 ONLINE MODE SELECTED - Will use MT5 live data")
    elif args.preset:
        # Use preset configuration
        if args.preset == "conservative":
            config = ConfigPresets.conservative_live()
        elif args.preset == "aggressive":
            config = ConfigPresets.aggressive_backtest()
        elif args.preset == "research":
            config = ConfigPresets.research_mode()
        elif args.preset == "demo_online":
            config = ConfigPresets.demo_online()
            config.live_mode = True
    else:
        # Create config based on mode
        if args.mode == "test":
            config = TradingConfig(test_mode=True, live_mode=False)
        else:  # offline
            config = TradingConfig(test_mode=False, live_mode=False)
    
    # Apply parameter overrides from command line
    if args.timesteps:
        config.final_training_steps = args.timesteps
    if args.lr:
        config.learning_rate = args.lr
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.n_epochs:
        config.n_epochs = args.n_epochs
    if args.gamma:
        config.gamma = args.gamma
    if args.n_steps:
        config.n_steps = args.n_steps
    if args.clip_range:
        config.clip_range = args.clip_range
    if args.ent_coef:
        config.ent_coef = args.ent_coef
    if args.vf_coef:
        config.vf_coef = args.vf_coef
    if args.max_grad_norm:
        config.max_grad_norm = args.max_grad_norm
    if args.target_kl:
        config.target_kl = args.target_kl
    if args.checkpoint_freq:
        config.checkpoint_freq = args.checkpoint_freq
    if args.eval_freq:
        config.eval_freq = args.eval_freq
    if args.num_envs:
        config.num_envs = args.num_envs
    if args.balance:
        config.initial_balance = args.balance
    if args.debug:
        config.debug = True
    if args.data_dir:
        config.data_dir = args.data_dir
    
    # Handle pretrained model
    pretrained_path = None
    if args.pretrained:
        pretrained_path = args.pretrained
    elif args.auto_pretrained:
        auto_path = "models/ppo_trading_model.zip"
        if os.path.exists(auto_path):
            pretrained_path = auto_path
    
    # Save configuration
    config_path = os.path.join(config.log_dir, "training_config.json")
    config.save_config(config_path)
    logger.info(f"📝 Configuration saved to {config_path}")
    
    # Show mode clearly
    if config.live_mode:
        logger.info("🔴" * 20)
        logger.info("🔴 LIVE TRAINING MODE WITH MT5 REAL-TIME DATA!")
        logger.info("🔴 Model will learn from actual market conditions!")
        if pretrained_path:
            logger.info("🔴 TRANSFER LEARNING: Continuing from offline model!")
        logger.info("🔴 Make sure MT5 is running and connected!")
        logger.info("🔴" * 20)
    else:
        logger.info("📊 OFFLINE TRAINING MODE - Using historical CSV data")
        logger.info(f"📊 Data directory: {config.data_dir}")
        if pretrained_path:
            logger.info("🔄 Continuing training from existing model")
    
    # Run training
    train_ppo_hybrid(config, pretrained_path)

if __name__ == "__main__":
    main()