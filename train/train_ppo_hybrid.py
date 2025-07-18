#!/usr/bin/env python3
"""
Complete Modern SmartInfoBus v4.0 Training Script
Production-ready with all fixes applied
"""

import os

# Suppress TensorFlow oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

import warnings
# Suppress specific warnings that clutter output
warnings.filterwarnings("ignore", message=".*dropout option adds dropout.*")
warnings.filterwarnings("ignore", message=".*The verbose parameter is deprecated.*")
warnings.filterwarnings("ignore", message=".*already in registry.*")
warnings.filterwarnings("ignore", message=".*Overriding environment.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*coroutine.*never awaited.*")
warnings.filterwarnings("ignore", message=".*The n_stack parameter is deprecated.*")
warnings.filterwarnings("ignore", message=".*Using DummyVecEnv.*")
warnings.filterwarnings("ignore", message=".*The env parameter is deprecated.*")
warnings.filterwarnings("ignore", message=".*Setting `log_interval`.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*You are using `torch.load`.*")
warnings.filterwarnings("ignore", message=".*`torch.nn.utils.clip_grad_norm` is deprecated.*")
warnings.filterwarnings("ignore", message=".*Using a non-tuple sequence.*")
warnings.filterwarnings("ignore", message=".*The `reset_num_timesteps` parameter.*")

import sys
import time
import platform
import logging
import argparse
import json
import asyncio
import threading
import queue
from datetime import datetime
from typing import Dict, Any, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Stable-Baselines3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

# ═══════════════════════════════════════════════════════════════════
# IMPORTS WITH MINIMAL LOGGING
# ═══════════════════════════════════════════════════════════════════

# Step 1: Core environment imports (REQUIRED)
try:
    # Suppress gymnasium registration warnings during import
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*already in registry.*")
        warnings.filterwarnings("ignore", message=".*Overriding environment.*")
        
        from envs.modern_env import ModernTradingEnv
        from envs.config import TradingConfig, ConfigPresets, ConfigFactory
except Exception as e:
    print(f"FATAL: Cannot import core environment modules: {e}")
    sys.exit(1)

# Step 2: Enhanced logging (with fallbacks)
RotatingLogger: Any = None
format_operator_message: Any = None

try:
    from modules.utils.audit_utils import RotatingLogger as _RotatingLogger, format_operator_message as _format_operator_message  # type: ignore
    ENHANCED_LOGGING = True
    RotatingLogger = _RotatingLogger
    format_operator_message = _format_operator_message
except Exception as e:
    ENHANCED_LOGGING = False
    
    class _FallbackRotatingLogger:
        def __init__(self, name=None, **kwargs):
            self.name = name or "FallbackLogger"
        def info(self, msg): pass  # Silent
        def warning(self, msg): pass
        def error(self, msg): print(f"ERROR: {msg}")
        def critical(self, msg): print(f"CRITICAL: {msg}")
    
    def _fallback_format_operator_message(icon="", message="", **kwargs):
        return f"{icon} {message}"
    
    RotatingLogger = _FallbackRotatingLogger  # type: ignore
    format_operator_message = _fallback_format_operator_message  # type: ignore

# Step 3: Import InfoBus
try:
    from modules.utils.info_bus import InfoBusManager, SmartInfoBus, create_info_bus, validate_info_bus
    SMARTINFOBUS_AVAILABLE = True
except Exception as e:
    # Fallback implementations
    class DummyInfoBusManager:
        @staticmethod
        def get_instance():
            return DummySmartBus()
    
    class DummySmartBus:
        def set(self, key, value, module=None, thesis=None): pass
        def get(self, key, module=None): return None
        def get_performance_metrics(self): return {}
    
    InfoBusManager = DummyInfoBusManager
    SMARTINFOBUS_AVAILABLE = False

# Step 4: Import monitoring modules with proper error handling
HealthMonitor: Any = None
PerformanceTracker: Any = None
IntegrationValidator: Any = None

try:
    from modules.monitoring.health_monitor import HealthMonitor as _HealthMonitor  # type: ignore
    HealthMonitor = _HealthMonitor
    HEALTH_MONITOR_AVAILABLE = True
    
    try:
        from modules.monitoring.performance_tracker import PerformanceTracker as _PerformanceTracker  # type: ignore
        PerformanceTracker = _PerformanceTracker
    except Exception:
        class _FallbackPerformanceTracker1:
            def __init__(self, orchestrator=None): pass
            def generate_performance_report(self): 
                return type('Report', (), {'module_metrics': {}})()
            def record_metric(self, *args, **kwargs): pass
        PerformanceTracker = _FallbackPerformanceTracker1  # type: ignore
    
    try:
        from modules.monitoring.integration_validator import IntegrationValidator as _IntegrationValidator  # type: ignore
        IntegrationValidator = _IntegrationValidator
    except Exception:
        class _FallbackIntegrationValidator1:
            def __init__(self, orchestrator=None): pass
            def validate_system(self): 
                return type('Report', (), {'integration_score': 100, 'issues': []})()
        IntegrationValidator = _FallbackIntegrationValidator1  # type: ignore
    
except Exception:
    # Fallback implementations
    class _DummyHealthMonitor:
        def __init__(self, **kwargs): pass
        def start(self): return True
        def stop(self): return True
        def check_system_health(self): 
            return {'overall_status': 'unknown', 'system': {}}
        def get_status(self): 
            return {'running': False}
    
    class _FallbackPerformanceTracker2:
        def __init__(self, orchestrator=None): pass
        def generate_performance_report(self): 
            return type('Report', (), {'module_metrics': {}})()
        def record_metric(self, *args, **kwargs): pass
    
    class _FallbackIntegrationValidator2:
        def __init__(self, orchestrator=None): pass
        def validate_system(self): 
            return type('Report', (), {'integration_score': 100, 'issues': []})()
    
    HealthMonitor = _DummyHealthMonitor  # type: ignore
    PerformanceTracker = _FallbackPerformanceTracker2  # type: ignore
    IntegrationValidator = _FallbackIntegrationValidator2  # type: ignore
    HEALTH_MONITOR_AVAILABLE = False

# Step 5: Import error handling
ErrorPinpointer: Any = None
create_error_handler: Any = None

try:
    from modules.core.error_pinpointer import ErrorPinpointer as _ErrorPinpointer, create_error_handler as _create_error_handler  # type: ignore
    ErrorPinpointer = _ErrorPinpointer
    create_error_handler = _create_error_handler
except Exception:
    class _FallbackErrorPinpointer:
        def analyze_error(self, e, context): return str(e)
    
    def _fallback_create_error_handler(name, pinpointer):
        class Handler:
            def handle_error(self, e, context): pass
        return Handler()
    
    ErrorPinpointer = _FallbackErrorPinpointer  # type: ignore
    create_error_handler = _fallback_create_error_handler  # type: ignore

# Step 6: Import enhanced callback
try:
    from train.enhanced_training_callback import ModernEnhancedTrainingCallback
    ENHANCED_CALLBACK_AVAILABLE = True
except Exception:
    ENHANCED_CALLBACK_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════
# DATA LOADING UTILITIES
# ═══════════════════════════════════════════════════════════════════

def load_data(data_dir: str = "data/processed") -> Dict[str, Dict[str, pd.DataFrame]]:
    """Enhanced data loader with validation"""
    data = {}
    
    if not os.path.exists(data_dir):
        print(f"[WARN]  Data directory not found: {data_dir}")
        return data
    
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            try:
                instrument = file.replace('.csv', '').replace('_', '/')
                df = pd.read_csv(os.path.join(data_dir, file))
                
                # Validate required columns
                required_cols = ['open', 'high', 'low', 'close']
                if all(col in df.columns for col in required_cols):
                    # Add volume if missing
                    if 'volume' not in df.columns:
                        df['volume'] = 1.0
                    
                    # Add volatility if missing
                    if 'volatility' not in df.columns:
                        df['volatility'] = df['close'].pct_change().rolling(20).std().fillna(0.01)
                    
                    # Ensure proper dtypes
                    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'volatility']
                    for col in numeric_cols:
                        if col in df.columns:
                            # Convert to numeric and handle NaN values
                            df[col] = df[col].astype(float).fillna(0).astype(np.float32)
                    
                    data[instrument] = {'H1': df}
                    print(f"[OK] Loaded {instrument}: {len(df)} bars")
                else:
                    print(f"[WARN]  Skipping {file}: missing required columns")
                    
            except Exception as e:
                print(f"[FAIL] Error loading {file}: {e}")
    
    return data

class FallbackSmartBus:
    """Fallback SmartInfoBus implementation"""
    def __init__(self):
        self._data = {}
        self._module_disabled = set()
        self._data_store = {}
    
    def set(self, key, value, module=None, thesis=None):
        self._data[key] = value
        self._data_store[key] = value
    
    def get(self, key, module=None):
        return self._data.get(key)
    
    def register_provider(self, module, keys):
        pass
    
    def register_consumer(self, module, keys):
        pass
    
    def get_performance_metrics(self):
        return {}

# ═══════════════════════════════════════════════════════════════════
# SIMPLIFIED TRAINING CALLBACK (FALLBACK)
# ═══════════════════════════════════════════════════════════════════

from stable_baselines3.common.callbacks import BaseCallback

class ModernTrainingCallback(BaseCallback):
    """
    FIXED: Modern training callback with SmartInfoBus integration and proper SB3 compatibility
    This version avoids the logger attribute conflict with BaseCallback
    """
    
    def __init__(self, total_timesteps: int, config: TradingConfig, verbose: int = 1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.config = config
        self.start_time = datetime.now()
        self.last_print_time = datetime.now()
        self.episode_count = 0
        self.episode_rewards = []
        self.best_reward = float('-inf')
        
        # FIXED: Use different attribute name to avoid conflict with BaseCallback.logger
        if ENHANCED_LOGGING:
            self.training_log = RotatingLogger(
                name="ModernTrainingCallback",
                log_path=f"logs/training/modern_training_{datetime.now().strftime('%Y%m%d')}.log",
                max_lines=2000,
                operator_mode=True
            )
        else:
            self.training_log = RotatingLogger(name="FallbackLogger")
        
        # SmartInfoBus integration
        if SMARTINFOBUS_AVAILABLE:
            self.smart_bus = InfoBusManager.get_instance()
        else:
            self.smart_bus = FallbackSmartBus()
        
        self.training_log.info(format_operator_message(
            message="Modern training callback initialized",
            icon="[ROCKET]",
            total_timesteps=total_timesteps,
            smartinfobus_v4=SMARTINFOBUS_AVAILABLE
        ))
        
        # Print initial status
        mode_str = "LIVE" if getattr(config, 'live_mode', False) else "OFFLINE"
        print(f"\n[ROCKET] MODERN TRAINING STARTED")
        print(f"[STATS] Total timesteps: {total_timesteps:,}")
        print(f"🔗 SmartInfoBus v4.0: {'ENABLED' if SMARTINFOBUS_AVAILABLE else 'FALLBACK'}")
        print(f"[CHART] Mode: {mode_str}")
        print("─" * 60)
    
    def _on_step(self) -> bool:
        """Enhanced step callback with progress tracking"""
        current_time = datetime.now()
        
        # Print progress every 10 seconds or every 1000 steps
        time_since_print = (current_time - self.last_print_time).total_seconds()
        if time_since_print >= 10 or self.n_calls % 1000 == 0:
            self._print_progress()
            self.last_print_time = current_time
        
        # Update metrics every 100 steps
        if self.n_calls % 100 == 0:
            self._update_metrics()
        
        # Check for episode completion
        if 'episode' in self.locals and 'dones' in self.locals:
            dones = self.locals['dones']
            if any(dones):
                self._handle_episode_completion()
        
        return True
    
    def _print_progress(self):
        """Print detailed progress to console"""
        elapsed = datetime.now() - self.start_time
        progress = (self.n_calls / self.total_timesteps) * 100
        
        # Calculate ETA
        if progress > 0:
            total_time_estimate = elapsed.total_seconds() / (progress / 100)
            remaining_time = total_time_estimate - elapsed.total_seconds()
            eta = f"{remaining_time/60:.1f}min" if remaining_time > 60 else f"{remaining_time:.0f}s"
        else:
            eta = "calculating..."
        
        # Get recent reward info
        recent_reward = "N/A"
        if self.episode_rewards:
            recent_reward = f"{self.episode_rewards[-1]:.2f}"
        
        # Print colorful progress
        print(f"\r[RELOAD] Step: {self.n_calls:,}/{self.total_timesteps:,} "
              f"({progress:.1f}%) | "
              f"[TIME] {elapsed.total_seconds()/60:.1f}min elapsed | "
              f"[CHART] Episodes: {self.episode_count} | "
              f"[MONEY] Last reward: {recent_reward} | "
              f"[TROPHY] Best: {self.best_reward:.2f} | "
              f"[WAIT] ETA: {eta}", end="", flush=True)
    
    def _update_metrics(self):
        """Update SmartInfoBus metrics"""
        progress = (self.n_calls / self.total_timesteps) * 100
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        metrics = {
            'step': self.n_calls,
            'total_timesteps': self.total_timesteps,
            'progress_pct': progress,
            'elapsed_time': elapsed,
            'episodes': self.episode_count,
            'best_reward': self.best_reward,
            'timestamp': datetime.now().isoformat(),
            'smartinfobus_v4': SMARTINFOBUS_AVAILABLE
        }
        
        # Update SmartInfoBus
        self.smart_bus.set(
            'training_progress',
            metrics,
            module='ModernTrainingCallback',
            thesis=f"Training progress: {progress:.1f}% complete, {self.episode_count} episodes"
        )
    
    def _handle_episode_completion(self):
        """Handle episode completion"""
        self.episode_count += 1
        
        # Try to get episode reward
        if hasattr(self.training_env, 'get_attr'):
            try:
                episode_rewards = self.training_env.get_attr('episode_reward')
                if episode_rewards and len(episode_rewards) > 0:
                    reward = episode_rewards[0]
                    self.episode_rewards.append(reward)
                    
                    if reward > self.best_reward:
                        self.best_reward = reward
                        print(f"\n[PARTY] NEW BEST REWARD: {reward:.2f} (Episode {self.episode_count})")
                    
                    # Print episode summary every 10 episodes
                    if self.episode_count % 10 == 0:
                        avg_reward = sum(self.episode_rewards[-10:]) / min(10, len(self.episode_rewards))
                        print(f"\n[STATS] Episode {self.episode_count}: Reward={reward:.2f}, "
                              f"Avg(10)={avg_reward:.2f}, Best={self.best_reward:.2f}")
            except Exception:
                pass  # Don't crash on reward extraction issues
    
    def _on_training_end(self):
        """Called when training ends"""
        duration = datetime.now() - self.start_time
        
        print(f"\n\n[OK] MODERN TRAINING COMPLETED!")
        print("─" * 60)
        print(f"[TIME]  Duration: {duration}")
        print(f"[STATS] Total steps: {self.n_calls:,}")
        print(f"🎬 Total episodes: {self.episode_count}")
        print(f"[TROPHY] Best reward: {self.best_reward:.2f}")
        if self.episode_rewards:
            print(f"[CHART] Average reward: {sum(self.episode_rewards)/len(self.episode_rewards):.2f}")
        print(f"🔗 SmartInfoBus v4.0: {'ACTIVE' if SMARTINFOBUS_AVAILABLE else 'FALLBACK'}")
        print("─" * 60)
        
        self.training_log.info(format_operator_message(
            message="Modern training completed",
            icon="[OK]",
            duration=str(duration),
            total_steps=self.n_calls,
            episodes=self.episode_count,
            best_reward=self.best_reward
        ))

# ═══════════════════════════════════════════════════════════════════
# ENVIRONMENT CREATION (FIXED)
# ═══════════════════════════════════════════════════════════════════

def test_environment_creation(data: Dict, config: TradingConfig) -> bool:
    """Test environment creation"""
    try:
        print("[TOOL] Testing environment creation...")
        env = ModernTradingEnv(data, config)
        
        # Test reset with seed
        obs, info = env.reset(seed=42)
        
        # Validate observation
        if not isinstance(obs, np.ndarray) or not np.all(np.isfinite(obs)):
            raise ValueError("Invalid observation")
        
        # Test step
        action = env.action_space.sample()
        env.step(action)
        
        print(f"[OK] Environment test passed (obs_shape: {obs.shape})")
        env.close()
        return True
        
    except Exception as e:
        print(f"[FAIL] Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_environments(data: Dict, config: TradingConfig, n_envs: int = 1, seed: int = 42):
    """Create training environments with proper compatibility"""
    
    # Test first
    if not test_environment_creation(data, config):
        raise RuntimeError("Environment creation test failed")
    
    # Force single environment for stability
    if config.live_mode or platform.system() == "Windows":
        n_envs = 1
        print(f"[TOOL] Using single environment for stability")
    
    print(f"🏗️ Creating {n_envs} environment(s)...")
    
    def make_env(rank: int):
        def _init():
            try:
                # Suppress gymnasium registration warnings
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*already in registry.*")
                    warnings.filterwarnings("ignore", message=".*Overriding environment.*")
                    
                    # Create environment
                    env = ModernTradingEnv(data, config)
                    
                    # Wrap with Monitor (using newer filename parameter)
                    Path("logs/training").mkdir(parents=True, exist_ok=True)
                    env = Monitor(env, filename=f"logs/training/monitor_{rank}.csv", 
                                info_keywords=())  # Fixed: empty tuple instead of deprecated default
                
                print(f"[OK] Environment {rank} created successfully")
                return env
                
            except Exception as e:
                print(f"[FAIL] Failed to create environment {rank}: {e}")
                raise
        
        set_random_seed(seed + rank)
        return _init
    
    # Create vectorized environment
    env = DummyVecEnv([make_env(i) for i in range(n_envs)])
    
    # Use newer seed method if available
    if hasattr(env, 'seed'):
        env.seed(seed)
    
    print(f"[OK] {n_envs} environment(s) ready")
    return env

def create_ppo_model(env, config: TradingConfig):
    """Create optimized PPO model"""
    
    policy_kwargs = dict(
        net_arch=dict(
            pi=[config.policy_hidden_size, config.policy_hidden_size // 2], 
            vf=[config.value_hidden_size, config.value_hidden_size // 2]
        ),
        activation_fn=nn.Tanh,
        normalize_images=False,
    )
    
    model_config = {
        "learning_rate": config.learning_rate,
        "n_steps": config.n_steps,
        "batch_size": config.batch_size,
        "n_epochs": config.n_epochs,
        "gamma": config.gamma,
        "gae_lambda": config.gae_lambda,
        "clip_range": config.clip_range,
        "ent_coef": config.ent_coef,
        "vf_coef": config.vf_coef,
        "max_grad_norm": config.max_grad_norm,
        "target_kl": config.target_kl,
        "verbose": 0,  # Changed from 1 to 0 to reduce deprecated verbose warnings
        "tensorboard_log": config.tensorboard_dir,
        "policy_kwargs": policy_kwargs,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": config.init_seed
    }
    
    device_str = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"[BOT] Creating PPO model on {device_str}")
    
    return PPO("MlpPolicy", env, **model_config)

# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def load_training_data(config: TradingConfig) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load training data with validation"""
    
    if config.live_mode:
        print("[RED] Live mode detected - using simulated data")
        return create_dummy_data(config)
    else:
        print(f"[STATS] Loading data from: {config.data_dir}")
        
        if os.path.exists(config.data_dir):
            data = load_data(config.data_dir)
            if data:
                # Validate data
                total_bars = sum(len(df) for inst_data in data.values() for df in inst_data.values())
                print(f"[OK] Loaded {len(data)} instruments ({total_bars:,} total bars)")
                
                # Print data summary
                for instrument, timeframes in data.items():
                    for tf, df in timeframes.items():
                        latest_price = df['close'].iloc[-1] if len(df) > 0 else 0
                        print(f"   {instrument}/{tf}: {len(df)} bars (latest: {latest_price:.5f})")
                
                return data
        
        print("[WARN]  No data found, creating dummy data")
        return create_dummy_data(config)

def create_dummy_data(config: TradingConfig) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Create realistic dummy data for testing"""
    
    print(f"🧪 Creating dummy data for {len(config.instruments)} instruments")
    
    dummy_data = {}
    
    for instrument in config.instruments:
        # Set realistic base prices
        if "EUR" in instrument:
            base_price = 1.1000
            volatility = 0.01
        elif "XAU" in instrument or "GOLD" in instrument:
            base_price = 1800.0
            volatility = 0.02
        else:
            base_price = 1.0000
            volatility = 0.015
        
        n_bars = 2000
        np.random.seed(42 + hash(instrument) % 1000)
        
        # Generate realistic price movements
        returns = np.random.normal(0, volatility, n_bars)
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Create OHLC data
        opens = prices[:-1]
        closes = prices[1:]
        
        # Add realistic spreads
        spreads = np.random.uniform(0.0005, 0.002, len(closes)) * closes
        highs = np.maximum(opens, closes) + spreads
        lows = np.minimum(opens, closes) - spreads
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': opens.astype(np.float32),
            'high': highs.astype(np.float32),
            'low': lows.astype(np.float32),
            'close': closes.astype(np.float32),
            'volume': np.random.randint(100, 2000, len(closes)).astype(np.float32),
            'volatility': np.abs(np.random.normal(volatility, volatility/4, len(closes))).astype(np.float32),
        })
        
        # Ensure OHLC consistency
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        dummy_data[instrument] = {'H1': df}
    
    total_bars = sum(len(df) for inst_data in dummy_data.values() for df in inst_data.values())
    print(f"[OK] Generated {len(dummy_data)} instruments ({total_bars:,} total bars)")
    
    return dummy_data

# ═══════════════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════════

def train_modern_ppo(config: TradingConfig, pretrained_model_path: Optional[str] = None):
    """Main training function - simplified and clean"""
    
    mode_str = "LIVE" if config.live_mode else "OFFLINE"
    print(f"PPO Training - {mode_str} Mode")

    # Initialize variables to avoid unbound errors
    train_env = None
    eval_env = None
    
    # Setup minimal logging
    if ENHANCED_LOGGING:
        training_log = RotatingLogger(
            name="ModernPPOTraining",
            log_dir="logs/rotate_logger/training",
            max_lines=1000,
            operator_mode=True
        )
    else:
        training_log = logging.getLogger("ModernPPOTraining")
    
    try:
        # Load data
        print("Loading market data...")
        data = load_training_data(config)
        
        # Create environments
        print("Creating environments...")
        train_env = create_environments(data, config, n_envs=1, seed=config.init_seed)
        eval_env = create_environments(data, config, n_envs=1, seed=config.init_seed + 1000)
        
        # Create or load model
        print("Setting up PPO model...")
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            print(f"Loading pretrained model: {pretrained_model_path}")
            model = PPO.load(pretrained_model_path, env=train_env)
            
            if config.live_mode:
                model.learning_rate = float(config.learning_rate) * 0.5
        else:
            print("Creating new PPO model...")
            model = create_ppo_model(train_env, config)
        
        # Setup callbacks
        if ENHANCED_CALLBACK_AVAILABLE:
            try:
                from train.enhanced_training_callback import ModernEnhancedTrainingCallback
                modern_callback = ModernEnhancedTrainingCallback(config.final_training_steps, config, verbose=1)
                print("Using enhanced training callback")
            except Exception:
                modern_callback = ModernTrainingCallback(config.final_training_steps, config, verbose=1)
        else:
            modern_callback = ModernTrainingCallback(config.final_training_steps, config, verbose=1)
        
        callbacks = [
            modern_callback,
            CheckpointCallback(
                save_freq=config.checkpoint_freq,
                save_path=config.checkpoint_dir,
                name_prefix=f"modern_ppo_{config.live_mode and 'live' or 'offline'}"
            ),
            EvalCallback(
                eval_env,
                best_model_save_path=os.path.join(config.model_dir, "best"),
                log_path=os.path.join("logs", "eval"),
                eval_freq=config.eval_freq,
                deterministic=True,
                render=False,
                n_eval_episodes=config.n_eval_episodes,
            ),
        ]
        
        # Start training
        print(f"Starting {mode_str} training...")
        print(f"Device: {model.device}")
        print(f"Training steps: {config.final_training_steps:,}")
        
        start_time = datetime.now()
        
        model.learn(
            total_timesteps=config.final_training_steps,
            callback=CallbackList(callbacks),
            tb_log_name=f"modern_ppo_{'live' if config.live_mode else 'offline'}",
            reset_num_timesteps=True,
            progress_bar=False,
        )
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        # Save final model
        final_model_path = os.path.join(config.model_dir, "modern_ppo_final.zip")
        model.save(final_model_path)
        
        # Final logging
        print(f"Training completed in {training_duration}")
        print(f"Model saved to {final_model_path}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        raise
        
    finally:
        # Cleanup
        try:
            if train_env is not None:
                train_env.close()
            if eval_env is not None:
                eval_env.close()
        except Exception:
            pass

# ═══════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def main():
    # Create only essential directories
    essential_dirs = ['models/best', 'checkpoints']
    for essential_dir in essential_dirs:
        os.makedirs(essential_dir, exist_ok=True)
    
    # Minimal logging setup
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
    
    parser = argparse.ArgumentParser(description="Modern PPO Training")
    
    # Mode selection
    parser.add_argument("--mode", choices=["offline", "online", "test"], default="offline",
                       help="Training mode")
    parser.add_argument("--preset", choices=["conservative", "aggressive", "research", "production"], 
                       help="Configuration preset")
    
    # Training parameters
    parser.add_argument("--timesteps", type=int, help="Total timesteps")
    parser.add_argument("--lr", "--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--balance", type=float, help="Initial balance")
    
    # Paths
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Data directory")
    parser.add_argument("--pretrained", type=str, help="Pretrained model path")
    parser.add_argument("--auto-pretrained", action="store_true", help="Auto-load pretrained model")
    
    # Options
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create configuration
    config = None  # Initialize config variable
    if args.mode == "online":
        config = ConfigPresets.conservative_live()
        config.live_mode = True
        print("ONLINE MODE SELECTED")
    elif args.preset:
        if args.preset == "conservative":
            config = ConfigPresets.conservative_live()
        elif args.preset == "aggressive":
            config = ConfigFactory.create_config("backtest", "aggressive")
        elif args.preset == "research":
            config = ConfigPresets.research_mode()
        elif args.preset == "production":
            config = ConfigPresets.production_backtest()
        else:
            config = TradingConfig(test_mode=(args.mode == "test"), live_mode=False)
    else:
        config = TradingConfig(test_mode=(args.mode == "test"), live_mode=False)
    
    # Apply overrides
    if args.timesteps:
        config.final_training_steps = args.timesteps
    if args.lr:
        config.learning_rate = args.lr
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.balance:
        config.initial_balance = args.balance
    if args.debug:
        config.debug = True
    if args.data_dir:
        config.data_dir = args.data_dir
    
    # Ensure reasonable defaults for testing
    if not args.timesteps and args.mode == "test":
        config.final_training_steps = 10000
    elif not args.timesteps:
        config.final_training_steps = max(50000, config.final_training_steps)
    
    # Handle pretrained model
    pretrained_path = None
    if args.pretrained:
        pretrained_path = args.pretrained
    elif args.auto_pretrained:
        auto_path = "models/modern_ppo_final.zip"
        if os.path.exists(auto_path):
            pretrained_path = auto_path
    
    # Save configuration (only if directory exists)
    try:
        os.makedirs("logs", exist_ok=True)
        config_path = "logs/modern_training_config.json"
        config.save_config(config_path)
    except Exception:
        pass  # Silent fail if can't save config
    
    # Display minimal configuration
    print(f"Training Mode: {args.mode.upper()}")
    print(f"Training Steps: {config.final_training_steps:,}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Initial Balance: ${config.initial_balance:,}")
    
    # Validate configuration
    if config.live_mode:
        print("LIVE TRADING MODE - Using simulated data")
    
    # Run training
    try:
        train_modern_ppo(config, pretrained_path)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("Starting Modern PPO Training System")
    
    # Check basic requirements
    if not torch.cuda.is_available():
        print("Note: Running on CPU")
    
    try:
        main()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        sys.exit(1)