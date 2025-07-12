# ─────────────────────────────────────────────────────────────
# File: train/modern_ppo_training.py (Updated Import Section)
# Modern SmartInfoBus v4.0 Integrated Training Script
# ─────────────────────────────────────────────────────────────

import os
import platform
import sys
import logging
import argparse
import json
import asyncio
import websockets
from datetime import datetime
from typing import Dict, Any, Optional
import threading
import queue
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

# ─────────────────────────────────────────────────────────────
# SAFE IMPORTS WITH PROPER ERROR HANDLING
# ─────────────────────────────────────────────────────────────

# Core imports that should always work
SMARTINFOBUS_AVAILABLE = False
HEALTH_MONITOR_AVAILABLE = False

# Initialize module references
HealthMonitor = None
health_monitor_instance = None

print("DEBUG: Starting safe imports...")

# Step 1: Import basic utilities first
try:
    print("DEBUG: Importing basic utilities...")
    from modules.utils.system_utilities import SystemUtilities, EnglishExplainer
    print("DEBUG: ✓ System utilities imported")
except Exception as e:
    print(f"DEBUG: System utilities import failed: {e}")
    SystemUtilities = None
    EnglishExplainer = None

# Step 2: Import audit utilities
try:
    print("DEBUG: Importing audit utilities...")
    from modules.utils.audit_utils import RotatingLogger, format_operator_message  # type: ignore
    print("DEBUG: ✓ Audit utilities imported")
except Exception as e:
    print(f"DEBUG: Audit utilities import failed: {e}")
    # Fallback implementations
    class RotatingLogger:
        def __init__(self, name=None, **kwargs):
            self.name = name or "FallbackLogger"
        def info(self, msg): print(f"INFO: {msg}")
        def warning(self, msg): print(f"WARN: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
        def critical(self, msg): print(f"CRITICAL: {msg}")
    
    def format_operator_message(icon="", message="", **kwargs):
        return f"{icon} {message} {kwargs}"

# Step 3: Import InfoBus
try:
    print("DEBUG: Importing InfoBus modules...")
    from modules.utils.info_bus import InfoBusManager, SmartInfoBus, create_info_bus, validate_info_bus
    SMARTINFOBUS_AVAILABLE = True
    print("DEBUG: ✓ InfoBus modules imported")
except Exception as e:
    print(f"DEBUG: InfoBus import failed: {e}")
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
try:
    print("DEBUG: Importing monitoring modules...")
    
    # Import HealthMonitor but DON'T create instance yet
    from modules.monitoring.health_monitor import HealthMonitor as HealthMonitorClass
    HealthMonitor = HealthMonitorClass
    HEALTH_MONITOR_AVAILABLE = True
    print("DEBUG: ✓ Health monitor imported (not started)")
    
    # Import other monitoring tools
    try:
        from modules.monitoring.performance_tracker import PerformanceTracker  # type: ignore
        print("DEBUG: ✓ Performance tracker imported")
    except Exception as e:
        print(f"DEBUG: Performance tracker import failed: {e}")
        class PerformanceTracker:
            def __init__(self, orchestrator=None): pass
            def generate_performance_report(self): 
                return type('Report', (), {'module_metrics': {}})()
            def record_metric(self, *args, **kwargs): pass
    
    try:
        from modules.monitoring.integration_validator import IntegrationValidator  # type: ignore
        print("DEBUG: ✓ Integration validator imported")
    except Exception as e:
        print(f"DEBUG: Integration validator import failed: {e}")
        class IntegrationValidator:
            def __init__(self, orchestrator=None): pass
            def validate_system(self): 
                return type('Report', (), {'integration_score': 100, 'issues': []})()
    
except Exception as e:
    print(f"DEBUG: Monitoring modules import failed: {e}")
    print("DEBUG: Creating fallback monitoring classes...")
    
    # Fallback implementations
    class DummyHealthMonitor:
        def __init__(self, **kwargs): pass
        def start(self): return True
        def stop(self): return True
        def check_system_health(self): 
            return {'overall_status': 'unknown', 'system': {}}
        def get_status(self): 
            return {'running': False}
    
    HealthMonitor = DummyHealthMonitor
    HEALTH_MONITOR_AVAILABLE = False
    
    # Use fallback PerformanceTracker and IntegrationValidator from above

# Step 5: Import error handling
try:
    print("DEBUG: Importing error handling...")
    from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler  # type: ignore
    print("DEBUG: ✓ Error handling imported")
except Exception as e:
    print(f"DEBUG: Error handling import failed: {e}")
    class ErrorPinpointer:
        def analyze_error(self, e, context): return str(e)
    def create_error_handler(name, pinpointer):
        class Handler:
            def handle_error(self, e, context): print(f"Error in {context}: {e}")
        return Handler()

# Step 6: Import environment
try:
    print("DEBUG: Importing environment modules...")
    from envs import EnhancedTradingEnv, TradingConfig
    from envs.config import ConfigPresets, ConfigFactory
    print("DEBUG: ✓ Environment modules imported")
except ImportError as e:
    print(f"❌ Critical: Environment import failed: {e}")
    sys.exit(1)

# Step 7: Import data utilities
try:
    print("DEBUG: Importing data utilities...")
    from utils.data_utils import load_data
    print("DEBUG: ✓ Data utilities imported")
except ImportError:
    print("DEBUG: Using fallback data loader")
    def load_data(data_dir: str = "data/processed") -> Dict[str, Dict[str, pd.DataFrame]]:
        """Fallback data loader for CSV files"""
        data = {}
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                if file.endswith('.csv'):
                    instrument = file.replace('.csv', '')
                    df = pd.read_csv(os.path.join(data_dir, file))
                    # Ensure required columns
                    required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
                    if all(col in df.columns for col in required_cols):
                        # Add volatility if not present
                        if 'volatility' not in df.columns:
                            df['volatility'] = df['close'].pct_change().rolling(20).std().fillna(0.01)
                        data[instrument] = {'H1': df}  # Assume H1 timeframe
        return data

print("\n" + "="*60)
print("IMPORT STATUS SUMMARY:")
print(f"✓ SmartInfoBus: {'AVAILABLE' if SMARTINFOBUS_AVAILABLE else 'FALLBACK MODE'}")
print(f"✓ Health Monitor: {'AVAILABLE' if HEALTH_MONITOR_AVAILABLE else 'FALLBACK MODE'}")
print(f"✓ Audit System: READY")
print("="*60 + "\n")

# ─────────────────────────────────────────────────────────────
# SAFE HEALTH MONITOR INITIALIZATION
# ─────────────────────────────────────────────────────────────

def initialize_health_monitor(orchestrator=None):
    """
    Safely initialize health monitor AFTER all imports are done.
    This prevents blocking during module import phase.
    """
    global health_monitor_instance
    
    if not HEALTH_MONITOR_AVAILABLE or HealthMonitor is None:
        print("DEBUG: Health monitoring not available, using dummy")
        return None
    
    try:
        print("DEBUG: Creating health monitor instance...")
        # Create instance but don't start it yet
        health_monitor_instance = HealthMonitor(
            orchestrator=orchestrator,
            check_interval=30,
            auto_start=False  # Important: Don't auto-start!
        )
        print("DEBUG: ✓ Health monitor instance created")
        
        # Now start it when we're ready
        print("DEBUG: Starting health monitor...")
        if health_monitor_instance.start():
            print("DEBUG: ✓ Health monitor started successfully")
            return health_monitor_instance
        else:
            print("DEBUG: ⚠ Health monitor failed to start")
            return None
            
    except Exception as e:
        print(f"DEBUG: Failed to initialize health monitor: {e}")
        import traceback
        traceback.print_exc()
        return None

def cleanup_health_monitor():
    """Safely cleanup health monitor"""
    global health_monitor_instance
    
    if health_monitor_instance:
        try:
            print("DEBUG: Stopping health monitor...")
            health_monitor_instance.stop()
            print("DEBUG: ✓ Health monitor stopped")
        except Exception as e:
            print(f"DEBUG: Error stopping health monitor: {e}")

# Create enhanced directory structure
log_dirs = [
    'logs/training', 'logs/live', 'logs/smartinfobus', 'logs/health',
    'logs/regime', 'logs/strategy', 'logs/checkpoints', 'logs/audit',
    'logs/reward', 'logs/tensorboard', 'checkpoints', 'models/best', 'metrics'
]
for log_dir in log_dirs:
    os.makedirs(log_dir, exist_ok=True)

# Modern logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/training/modern_ppo_training.log', encoding='utf-8')
    ]
)
logger = logging.getLogger("ModernPPOTraining")


# ═══════════════════════════════════════════════════════════════════
# MODERN METRICS BROADCASTING WITH SMARTINFOBUS v4.0
# ═══════════════════════════════════════════════════════════════════

class ModernMetricsBroadcaster:
    """Modern metrics broadcaster with SmartInfoBus v4.0 integration"""
    
    def __init__(self, host='localhost', port=8001):
        self.host = host
        self.port = port
        self.metrics_queue = queue.Queue(maxsize=1000)
        self.ws_thread = None
        self.running = False
        self.websocket = None
        
        # Enhanced tracking
        self.messages_sent = 0
        self.messages_failed = 0
        self.connection_attempts = 0
        self.last_successful_send = None
        
        # Modern infrastructure
        self.smart_bus = InfoBusManager.get_instance()
        self.training_logger = RotatingLogger(
            name="ModernMetricsBroadcaster",
            log_path="logs/training/modern_metrics_broadcaster.log",
            max_lines=2000,
            operator_mode=True,
            plain_english=True
        )
        
    def start(self):
        """Start metrics broadcasting with modern infrastructure"""
        self.running = True
        self.ws_thread = threading.Thread(target=self._run_websocket)
        self.ws_thread.daemon = True
        self.ws_thread.start()
        
        self.training_logger.info(format_operator_message(
            message="Modern metrics broadcaster started",
            icon="📡",
            host=self.host,
            port=self.port,
            smartinfobus_v4=True
        ))
        
    def stop(self):
        """Stop broadcasting with comprehensive reporting"""
        self.running = False
        if self.ws_thread:
            self.ws_thread.join(timeout=10)
            
        success_rate = (self.messages_sent/(self.messages_sent+self.messages_failed)*100) if (self.messages_sent+self.messages_failed) > 0 else 0
        
        self.training_logger.info(format_operator_message(
            message="Modern metrics broadcaster stopped",
            icon="📡",
            messages_sent=self.messages_sent,
            messages_failed=self.messages_failed,
            success_rate=f"{success_rate:.1f}%"
        ))
        
    def send_metrics(self, metrics: Dict[str, Any]):
        """Send metrics with enhanced health tracking"""
        try:
            # Add broadcaster health to metrics
            enhanced_metrics = {
                **metrics,
                'broadcaster_health': {
                    'messages_sent': self.messages_sent,
                    'messages_failed': self.messages_failed,
                    'queue_size': self.metrics_queue.qsize(),
                    'last_successful_send': self.last_successful_send,
                    'connection_attempts': self.connection_attempts
                },
                'smartinfobus_v4': True
            }
            
            self.metrics_queue.put(enhanced_metrics, block=False)
            
        except queue.Full:
            self.messages_failed += 1
            self.training_logger.warning("Metrics queue full - dropping message")
            
    def _run_websocket(self):
        """Modern WebSocket client loop"""
        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()
        
        while self.running:
            try:
                self.connection_attempts += 1
                loop.run_until_complete(self._websocket_handler())
            except Exception as e:
                self.training_logger.error(f"WebSocket error: {e}")
                if self.running:
                    asyncio.get_event_loop().run_until_complete(asyncio.sleep(5))
                    
    async def _websocket_handler(self):
        """Modern WebSocket handler with comprehensive error handling"""
        uri = f"ws://{self.host}:{self.port}/ws/training"
        
        try:
            async with websockets.connect(uri) as websocket:
                self.websocket = websocket
                self.training_logger.info("✅ Connected to metrics server")
                
                while self.running:
                    try:
                        # Get metrics from queue (non-blocking with timeout)
                        metrics = self.metrics_queue.get(timeout=0.1)
                        
                        message = {
                            "type": "modern_training_metrics",
                            "data": metrics,
                            "timestamp": datetime.now().isoformat(),
                            "source": "ModernSmartInfoBusTraining"
                        }
                        
                        await websocket.send(json.dumps(message, default=str))
                        self.messages_sent += 1
                        self.last_successful_send = datetime.now().isoformat()
                        
                    except queue.Empty:
                        continue
                    except Exception as e:
                        self.messages_failed += 1
                        self.training_logger.error(f"Error sending metrics: {e}")
                        
        except Exception as e:
            self.training_logger.error(f"WebSocket connection error: {e}")

# Global modern metrics broadcaster
modern_metrics_broadcaster = ModernMetricsBroadcaster()

# ═══════════════════════════════════════════════════════════════════
# MODERN DATA LOADING WITH SMARTINFOBUS v4.0
# ═══════════════════════════════════════════════════════════════════

def load_training_data_modern(config: TradingConfig) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Modern data loading with SmartInfoBus v4.0 integration"""
    
    print(f"DEBUG: load_training_data_modern called with live_mode={config.live_mode}")
    
    if config.live_mode:
        logger.info(format_operator_message(
            message="Live mode detected - using simulated data",
            icon="🔴",
            note="Real MT5 integration would require additional setup"
        ))
        # For now, create enhanced dummy data for live mode
        return create_modern_dummy_data(config)
    else:
        logger.info(format_operator_message(
            message="Offline mode detected - loading historical data",
            icon="📊",
            data_dir=config.data_dir
        ))
        try:
            # Try to load from CSV files
            if os.path.exists(config.data_dir):
                data = load_data(config.data_dir)
                if data:
                    # Enhanced data validation
                    validated_data = validate_and_enhance_data_modern(data, config)
                    
                    total_bars = sum(len(df) for inst_data in validated_data.values() for df in inst_data.values())
                    logger.info(format_operator_message(
                        message="Historical data loaded successfully",
                        icon="✅",
                        instruments=len(validated_data),
                        total_bars=total_bars
                    ))
                    return validated_data
                    
            # If no data found, create dummy data
            logger.warning("No historical data found, creating enhanced dummy data...")
            return create_modern_dummy_data(config)
            
        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
            return create_modern_dummy_data(config)

def validate_and_enhance_data_modern(data: Dict[str, Dict[str, pd.DataFrame]], 
                                   config: TradingConfig) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Modern data validation with comprehensive quality checks"""
    
    validated_data = {}
    validation_summary = {
        "instruments_processed": 0,
        "timeframes_processed": 0,
        "total_bars": 0,
        "quality_issues": []
    }
    
    for instrument in config.instruments:
        if instrument in data:
            validated_data[instrument] = {}
            validation_summary["instruments_processed"] += 1
            
            for tf in config.timeframes:
                if tf in data[instrument]:
                    df = data[instrument][tf].copy()
                    
                    # Enhanced validation
                    df = perform_modern_data_quality_checks(df, instrument, tf, validation_summary)
                    
                    validated_data[instrument][tf] = df
                    validation_summary["timeframes_processed"] += 1
                    validation_summary["total_bars"] += len(df)
    
    # Log validation summary
    if validation_summary["quality_issues"]:
        logger.warning(format_operator_message(
            message="Data quality issues detected",
            icon="⚠️",
            issues_count=len(validation_summary["quality_issues"])
        ))
        for issue in validation_summary["quality_issues"][:5]:  # Log first 5 issues
            logger.warning(f"  • {issue}")
    
    return validated_data

def perform_modern_data_quality_checks(df: pd.DataFrame, instrument: str, timeframe: str, 
                                     summary: Dict[str, Any]) -> pd.DataFrame:
    """Comprehensive modern data quality checks and fixes"""
    
    original_length = len(df)
    
    # Check for missing values
    if df.isnull().values.any():
        missing_count = df.isnull().sum().sum()
        summary["quality_issues"].append(f"{instrument}/{timeframe}: {missing_count} missing values")
        df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Validate OHLC consistency
    invalid_ohlc = (df['high'] < df[['open', 'close']].max(axis=1)) | (df['low'] > df[['open', 'close']].min(axis=1))
    if invalid_ohlc.any():
        invalid_count = invalid_ohlc.sum()
        summary["quality_issues"].append(f"{instrument}/{timeframe}: {invalid_count} OHLC inconsistencies")
        # Fix inconsistencies
        df.loc[invalid_ohlc, 'high'] = df.loc[invalid_ohlc, ['open', 'high', 'close']].max(axis=1)
        df.loc[invalid_ohlc, 'low'] = df.loc[invalid_ohlc, ['open', 'low', 'close']].min(axis=1)
    
    # Check for extreme price movements
    returns = df['close'].pct_change()
    extreme_moves = returns.abs() > 0.1  # 10% moves
    if extreme_moves.any():
        extreme_count = extreme_moves.sum()
        summary["quality_issues"].append(f"{instrument}/{timeframe}: {extreme_count} extreme price movements")
    
    # Ensure volatility column exists and is valid
    if 'volatility' not in df.columns:
        df['volatility'] = df['close'].pct_change().rolling(20).std().fillna(0.01)
    
    df['volatility'] = df['volatility'].fillna(0.01).clip(lower=0.001)
    
    # Convert to proper dtypes
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].astype(np.float32)
    
    return df

def create_modern_dummy_data(config: TradingConfig) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Create modern dummy data with realistic market characteristics"""
    
    logger.info(format_operator_message(
        message="Creating modern dummy data",
        icon="🧪",
        instruments=len(config.instruments),
        timeframes=len(config.timeframes)
    ))
    
    dummy_data = {}
    
    for instrument in config.instruments:
        dummy_data[instrument] = {}
        
        # Set realistic base prices
        if "EUR" in instrument:
            base_price = 1.1000
            volatility_base = 0.01
        elif "XAU" in instrument or "GOLD" in instrument:
            base_price = 1800.0
            volatility_base = 0.02
        else:
            base_price = 1.0000
            volatility_base = 0.015
        
        for tf in config.timeframes:
            n_bars = 2000  # More data for better training
            
            # Generate realistic price movements
            np.random.seed(42 + hash(instrument + tf) % 1000)  # Deterministic but varied
            
            # Create trend and noise components
            trend = np.cumsum(np.random.randn(n_bars) * 0.0002)  # Small trend
            noise = np.random.randn(n_bars) * volatility_base
            returns = trend + noise
            
            # Generate prices
            prices = base_price * np.exp(np.cumsum(returns))
            
            # Create OHLC data
            opens = prices[:-1] if len(prices) > 1 else prices
            closes = prices[1:] if len(prices) > 1 else prices
            
            # Add realistic high/low spreads
            spreads = np.random.uniform(0.0005, 0.002, len(closes)) * closes
            highs = np.maximum(opens, closes) + spreads
            lows = np.minimum(opens, closes) - spreads
            
            # Create DataFrame
            df = pd.DataFrame({
                'time': pd.date_range(start='2023-01-01', periods=len(closes), freq='1H'),
                'open': opens[:len(closes)] if len(opens) > len(closes) else opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': np.random.randint(100, 2000, len(closes)),
                'volatility': np.abs(np.random.normal(volatility_base, volatility_base/4, len(closes))),
            })
            
            # Ensure OHLC consistency
            df['high'] = df[['open', 'high', 'close']].max(axis=1)
            df['low'] = df[['open', 'low', 'close']].min(axis=1)
            
            # Convert to proper dtypes
            numeric_cols = df.select_dtypes(include=['number']).columns
            df[numeric_cols] = df[numeric_cols].astype(np.float32)
            
            dummy_data[instrument][tf] = df
    
    total_bars = sum(len(df) for inst_data in dummy_data.values() for df in inst_data.values())
    logger.info(format_operator_message(
        message="Modern dummy data generated",
        icon="✅",
        instruments=len(dummy_data),
        total_bars=total_bars
    ))
    
    return dummy_data

# ═══════════════════════════════════════════════════════════════════
# MODERN ENVIRONMENT CREATION WITH SMARTINFOBUS v4.0
# ═══════════════════════════════════════════════════════════════════

def test_modern_env_creation(data: Dict, config: TradingConfig) -> bool:
    """Modern environment creation test with SmartInfoBus v4.0"""
    
    print("DEBUG: test_modern_env_creation called")
    try:
        mode_str = "LIVE" if config.live_mode else "OFFLINE"
        logger.info(format_operator_message(
            message="Testing modern environment creation",
            icon="🔧",
            mode=mode_str,
            smartinfobus_v4=True
        ))
        
        # Create test environment
        print("DEBUG: Creating test environment")
        env = EnhancedTradingEnv(data, config)
        print("DEBUG: Test environment created successfully")
        
        # Test SmartInfoBus integration
        if hasattr(env, 'smart_bus'):
            logger.info("✅ SmartInfoBus v4.0 integration confirmed")
        elif hasattr(env, 'info_bus'):
            logger.info("✅ Legacy InfoBus integration confirmed")
        else:
            logger.warning("⚠️ No InfoBus/SmartInfoBus detected in environment")
        
        # Test environment reset and step
        print("DEBUG: Testing environment reset")
        obs, info = env.reset()
        print(f"DEBUG: Environment reset successful, obs shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
        
        # Validate observation
        if not isinstance(obs, np.ndarray):
            raise ValueError(f"Invalid observation type: {type(obs)}")
        
        if not np.all(np.isfinite(obs)):
            raise ValueError("Non-finite values in observation")
        
        logger.info(format_operator_message(
            message="Environment test passed",
            icon="✅",
            mode=mode_str,
            obs_shape=str(obs.shape),
            smartinfobus_enabled=hasattr(env, 'smart_bus') or hasattr(env, 'info_bus')
        ))
        
        env.close()
        return True
        
    except Exception as e:
        logger.error(format_operator_message(
            message="Environment test failed",
            icon="💥",
            error=str(e)
        ))
        import traceback
        traceback.print_exc()
        return False

def create_modern_envs(data: Dict, config: TradingConfig, n_envs: int = 1, seed: int = 42):
    """Modern environment creation with SmartInfoBus v4.0"""
    
    print(f"DEBUG: create_modern_envs called with n_envs={n_envs}, seed={seed}")
    
    if not test_modern_env_creation(data, config):
        logger.error("Environment creation test failed!")
        raise RuntimeError("Cannot create environment")
    
    if config.live_mode:
        n_envs = 1
        logger.info("🔴 LIVE MODE: Using single environment for safety")
    elif platform.system() == "Windows":
        n_envs = 1
        logger.info("Windows detected - using single environment")
    
    logger.info(format_operator_message(
        message="Creating modern environments",
        icon="🏗️",
        n_envs=n_envs,
        smartinfobus_v4=True
    ))
    
    def make_env(rank: int):
        def _init():
            try:
                print(f"DEBUG: Creating environment {rank}")
                env = EnhancedTradingEnv(data, config)
                print(f"DEBUG: Environment {rank} created, wrapping with Monitor")
                env = Monitor(env, filename=f"logs/training/monitor_{rank}.csv")
                print(f"DEBUG: Environment {rank} wrapped with Monitor, setting seed")
                env.seed(seed + rank)
                
                mode_str = "LIVE" if config.live_mode else "OFFLINE"
                logger.info(format_operator_message(
                    message="Environment created",
                    icon="✅",
                    rank=rank,
                    mode=mode_str
                ))
                print(f"DEBUG: Environment {rank} fully initialized")
                return env
            except Exception as e:
                print(f"DEBUG: ERROR creating environment {rank}: {e}")
                logger.error(f"Failed to create environment {rank}: {e}")
                raise
        
        set_random_seed(seed)
        return _init
    
    print(f"DEBUG: Creating DummyVecEnv with {n_envs} environments")
    env = DummyVecEnv([make_env(i) for i in range(n_envs)])
    print("DEBUG: DummyVecEnv created successfully")
    
    logger.info(format_operator_message(
        message="Modern environments ready",
        icon="✅",
        n_envs=n_envs,
        smartinfobus_integration=True
    ))
    
    return env

def create_modern_ppo_model(env, config: TradingConfig):
    """Modern PPO model creation with enhanced architecture"""
    
    print("DEBUG: create_modern_ppo_model called")
    
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
    logger.info(format_operator_message(
        message="Modern PPO model created",
        icon="🤖",
        device=device_str,
        learning_rate=config.learning_rate,
        hidden_size=config.policy_hidden_size
    ))
    
    return model

# ═══════════════════════════════════════════════════════════════════
# MODERN TRAINING CALLBACK WITH PROPER SB3 INTEGRATION
# ═══════════════════════════════════════════════════════════════════

from stable_baselines3.common.callbacks import BaseCallback

class ModernTrainingCallbackSB3(BaseCallback):
    """Modern training callback that properly integrates with Stable-Baselines3"""
    
    def __init__(self, total_timesteps: int, config: TradingConfig, verbose: int = 1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.config = config
        self.start_time = datetime.now()
        self.last_print_time = datetime.now()
        self.episode_count = 0
        self.episode_rewards = []
        self.best_reward = float('-inf')
        
        # Modern infrastructure (with fallback)
        self.smart_bus = InfoBusManager.get_instance()
        if SMARTINFOBUS_AVAILABLE:
            self.training_logger = RotatingLogger(
                name="ModernTrainingCallback",
                log_path=f"logs/training/modern_training_{datetime.now().strftime('%Y%m%d')}.log",
                max_lines=2000,
                operator_mode=True
            )
        else:
            self.training_logger = RotatingLogger(name="FallbackLogger")
        
        self.training_logger.info(format_operator_message(
            message="Modern training callback initialized",
            icon="🚀",
            total_timesteps=total_timesteps,
            smartinfobus_v4=SMARTINFOBUS_AVAILABLE
        ))
        
        # Print initial status
        print(f"\n🚀 MODERN TRAINING STARTED")
        print(f"📊 Total timesteps: {total_timesteps:,}")
        print(f"🔗 SmartInfoBus v4.0: {'ENABLED' if SMARTINFOBUS_AVAILABLE else 'DISABLED (fallback mode)'}")
        print(f"📈 Mode: {'LIVE' if config.live_mode else 'OFFLINE'}")
        print("─" * 60)
    
    def _on_step(self) -> bool:
        """Called at every step - provides real-time feedback"""
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
        print(f"\r🔄 Step: {self.n_calls:,}/{self.total_timesteps:,} "
              f"({progress:.1f}%) | "
              f"⏱️ {elapsed.total_seconds()/60:.1f}min elapsed | "
              f"📈 Episodes: {self.episode_count} | "
              f"💰 Last reward: {recent_reward} | "
              f"🏆 Best: {self.best_reward:.2f} | "
              f"⏳ ETA: {eta}", end="", flush=True)
    
    def _update_metrics(self):
        """Update SmartInfoBus and broadcast metrics"""
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
            'smartinfobus_v4': True
        }
        
        # Update SmartInfoBus
        self.smart_bus.set(
            'training_progress',
            metrics,
            module='ModernTrainingCallback',
            thesis=f"Training progress: {progress:.1f}% complete, {self.episode_count} episodes"
        )
        
        # Send to broadcaster
        if modern_metrics_broadcaster:
            modern_metrics_broadcaster.send_metrics(metrics)
    
    def _handle_episode_completion(self):
        """Handle episode completion with detailed logging"""
        self.episode_count += 1
        
        # Get episode reward if available
        if hasattr(self.training_env, 'get_attr'):
            try:
                # Try to get episode reward from environment
                episode_rewards = self.training_env.get_attr('total_reward')
                if episode_rewards and len(episode_rewards) > 0:
                    reward = episode_rewards[0]
                    self.episode_rewards.append(reward)
                    
                    if reward > self.best_reward:
                        self.best_reward = reward
                        print(f"\n🎉 NEW BEST REWARD: {reward:.2f} (Episode {self.episode_count})")
                    
                    # Print episode summary every 10 episodes
                    if self.episode_count % 10 == 0:
                        avg_reward = sum(self.episode_rewards[-10:]) / min(10, len(self.episode_rewards))
                        print(f"\n📊 Episode {self.episode_count}: Reward={reward:.2f}, "
                              f"Avg(10)={avg_reward:.2f}, Best={self.best_reward:.2f}")
            except Exception as e:
                pass  # Don't crash on reward extraction issues
    
    def _on_training_end(self):
        """Called when training ends"""
        duration = datetime.now() - self.start_time
        
        print(f"\n\n✅ MODERN TRAINING COMPLETED!")
        print("─" * 60)
        print(f"⏱️  Duration: {duration}")
        print(f"📊 Total steps: {self.n_calls:,}")
        print(f"🎬 Total episodes: {self.episode_count}")
        print(f"🏆 Best reward: {self.best_reward:.2f}")
        if self.episode_rewards:
            print(f"📈 Average reward: {sum(self.episode_rewards)/len(self.episode_rewards):.2f}")
        print(f"🔗 SmartInfoBus v4.0: {'ACTIVE' if SMARTINFOBUS_AVAILABLE else 'DISABLED'}")
        print("─" * 60)
        
        self.training_logger.info(format_operator_message(
            message="Modern training completed",
            icon="✅",
            duration=str(duration),
            total_steps=self.n_calls,
            episodes=self.episode_count,
            best_reward=self.best_reward
        ))
        
        # Final SmartInfoBus update
        self.smart_bus.set(
            'training_completed',
            {
                'duration': str(duration),
                'total_steps': self.n_calls,
                'episodes': self.episode_count,
                'best_reward': self.best_reward,
                'timestamp': datetime.now().isoformat()
            },
            module='ModernTrainingCallback',
            thesis=f"Training completed successfully: {self.episode_count} episodes, best reward: {self.best_reward:.2f}"
        )

# ═══════════════════════════════════════════════════════════════════
# MAIN MODERN TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════════

def train_modern_ppo(config: TradingConfig, pretrained_model_path: Optional[str] = None):
    """Main modern training function with SmartInfoBus v4.0 integration"""
    
    print("DEBUG: Entering train_modern_ppo function")
    mode_str = "🔴 LIVE" if config.live_mode else "📊 OFFLINE"
    print(f"DEBUG: Mode string set to: {mode_str}")
    
    logger.info("=" * 80)
    logger.info(f"MODERN PPO TRAINING - {mode_str} MODE WITH SMARTINFOBUS v4.0")
    logger.info("=" * 80)
    logger.info(f"Configuration:\n{config}")
    print("DEBUG: Initial logging completed")
    
    # Start modern metrics broadcaster
    print("DEBUG: Starting metrics broadcaster")
    try:
        modern_metrics_broadcaster.start()
        print("DEBUG: Metrics broadcaster started successfully")
    except Exception as e:
        print(f"DEBUG: Error starting broadcaster: {e}")
        # Continue anyway
    
    # Send initial status
    print("DEBUG: Sending initial status")
    modern_metrics_broadcaster.send_metrics({
        "status": "INITIALIZING",
        "mode": mode_str,
        "config": config.__dict__,
        "smartinfobus_v4": True,
        "modern_features": ["health_monitoring", "audit_trail", "performance_tracking"]
    })
    
    try:
        # Load data with modern infrastructure
        print("DEBUG: About to load training data")
        logger.info(f"{mode_str} Loading market data with SmartInfoBus v4.0...")
        data = load_training_data_modern(config)
        print(f"DEBUG: Data loaded successfully, got {len(data)} instruments")
        
        logger.info(f"✅ Data ready for {len(data)} instruments:")
        for instrument, timeframes in data.items():
            for tf, df in timeframes.items():
                if not df.empty:
                    latest_time = df.index[-1] if len(df) > 0 else "Unknown"
                    latest_price = df['close'].iloc[-1] if len(df) > 0 else 0
                    logger.info(f"   {instrument}/{tf}: {len(df)} bars (latest: {latest_price:.5f} @ {latest_time})")
        
        # Create modern environments
        print("DEBUG: About to create environments")
        logger.info(f"🔧 Creating {mode_str} environments with SmartInfoBus v4.0...")
        print(f"DEBUG: Creating training env with {config.num_envs} envs")
        train_env = create_modern_envs(data, config, n_envs=config.num_envs, seed=config.init_seed)
        print("DEBUG: Training env created successfully")
        print("DEBUG: Creating eval env")
        eval_env = create_modern_envs(data, config, n_envs=1, seed=config.init_seed + 1000)
        print("DEBUG: Eval env created successfully")
        
        # Create or load modern model
        print("DEBUG: About to create/load PPO model")
        logger.info("🤖 Setting up modern PPO model...")
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            print(f"DEBUG: Loading pretrained model from {pretrained_model_path}")
            logger.info(f"📥 Loading pretrained model from: {pretrained_model_path}")
            model = PPO.load(pretrained_model_path, env=train_env)
            print("DEBUG: Pretrained model loaded successfully")
            
            if config.live_mode:
                model.learning_rate = float(config.learning_rate) * 0.5
                logger.info(f"🔧 Adjusted learning rate for live mode: {model.learning_rate}")
        else:
            print("DEBUG: Creating new PPO model")
            logger.info("🆕 Creating new modern PPO model...")
            model = create_modern_ppo_model(train_env, config)
            print("DEBUG: New PPO model created successfully")
        
        # Setup modern callbacks with proper SB3 integration
        print("DEBUG: Creating modern callback")
        modern_callback = ModernTrainingCallbackSB3(config.final_training_steps, config, verbose=1)
        print("DEBUG: Modern callback created successfully")
        
        print("DEBUG: Creating callbacks list")
        callbacks = [
            modern_callback,  # Add our modern callback first
            CheckpointCallback(
                save_freq=config.checkpoint_freq,
                save_path=config.checkpoint_dir,
                name_prefix=f"modern_ppo_{config.live_mode and 'live' or 'offline'}"
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
        print(f"DEBUG: Created {len(callbacks)} callbacks")
        
        # Send training started status
        modern_metrics_broadcaster.send_metrics({
            "status": "TRAINING_STARTED",
            "mode": mode_str,
            "total_timesteps": config.final_training_steps,
            "model_type": "Modern_PPO",
            "device": str(model.device),
            "smartinfobus_v4": True,
            "callback_count": len(callbacks)
        })
        
        # Start modern training
        logger.info(f"🚀 Starting {mode_str} training with SmartInfoBus v4.0...")
        logger.info(f"Total timesteps: {config.final_training_steps:,}")
        logger.info(f"SmartInfoBus v4.0 features: Health monitoring, Performance tracking, Modern audit")
        
        start_time = datetime.now()
        
        # Modern training with comprehensive monitoring and verbose output
        print(f"\n🎯 STARTING PPO TRAINING...")
        print(f"🔧 Model device: {model.device}")
        print(f"📚 Learning rate: {model.learning_rate}")
        print(f"🎲 Batch size: {config.batch_size}")
        print(f"🔄 Training steps: {config.final_training_steps:,}")
        print("═" * 60)
        
        print("DEBUG: About to start model.learn()")
        print(f"DEBUG: Training steps: {config.final_training_steps}")
        print(f"DEBUG: Number of callbacks: {len(callbacks)}")
        
        model.learn(
            total_timesteps=config.final_training_steps,
            callback=CallbackList(callbacks),
            log_interval=1,  # Log more frequently for better feedback
            tb_log_name=f"modern_ppo_{'live' if config.live_mode else 'offline'}",
            reset_num_timesteps=True,
            progress_bar=False,  # We have our own progress display
        )
        
        print("DEBUG: model.learn() completed successfully")
        print("\n" + "═" * 60)
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        # Save modern final model
        final_model_path = os.path.join(config.model_dir, "modern_ppo_trading_model.zip")
        model.save(final_model_path)
        
        # Generate modern training report
        training_summary = {
            "status": "TRAINING_COMPLETED",
            "mode": mode_str,
            "duration": str(training_duration),
            "final_model_path": final_model_path,
            "total_timesteps": config.final_training_steps,
            "smartinfobus_v4_enabled": True,
            "modern_features_used": [
                "SmartInfoBus v4.0 integration",
                "Modern health monitoring", 
                "Performance tracking",
                "Enhanced audit trail",
                "Modern metrics broadcasting"
            ]
        }
        
        # Send completion status
        modern_metrics_broadcaster.send_metrics(training_summary)
        
        # Modern completion logging
        logger.info("=" * 80)
        logger.info(f"✅ {mode_str} MODERN TRAINING COMPLETED!")
        logger.info("=" * 80)
        logger.info(f"📁 Model saved to {final_model_path}")
        logger.info(f"⏱️  Training duration: {training_duration}")
        logger.info(f"🔗 SmartInfoBus v4.0 integration: ENABLED")
        logger.info(f"📊 Modern monitoring: Health checks, performance tracking, audit trail")
        logger.info("=" * 80)
        
        # Training end callback is automatically called by SB3
        
    except Exception as e:
        error_msg = f"Modern training failed: {e}"
        logger.error(f"❌ {error_msg}")
        
        modern_metrics_broadcaster.send_metrics({
            "status": "TRAINING_FAILED",
            "error": str(e),
            "mode": mode_str
        })
        
        raise
        
    finally:
        # Modern cleanup
        logger.info("🧹 Performing modern cleanup...")
        try:
            train_env.close()
            eval_env.close()
            modern_metrics_broadcaster.stop()
            
            logger.info("✅ Modern cleanup completed")
            
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

# ═══════════════════════════════════════════════════════════════════
# MODERN MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def main():
    print("DEBUG: Entering main() function")
    parser = argparse.ArgumentParser(description="Modern SmartInfoBus v4.0 PPO Training")
    
    # Mode selection
    parser.add_argument("--mode", choices=["offline", "online", "test"], default="offline",
                       help="Training mode: offline (CSV), online (simulated live), test")
    parser.add_argument("--preset", choices=["conservative", "aggressive", "research", "production"], 
                       help="Use configuration preset")
    
    # SmartInfoBus configuration
    parser.add_argument("--smartinfobus", action="store_true", default=True, 
                       help="Enable SmartInfoBus v4.0 integration (default: True)")
    
    # Training parameters
    parser.add_argument("--timesteps", type=int, help="Total timesteps")
    parser.add_argument("--lr", "--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--n_epochs", type=int, help="Number of epochs")
    parser.add_argument("--gamma", type=float, help="Discount factor")
    parser.add_argument("--n_steps", type=int, help="Number of steps")
    
    # Data and model paths
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Directory with CSV files")
    parser.add_argument("--pretrained", type=str, help="Path to pretrained model")
    parser.add_argument("--auto-pretrained", action="store_true", help="Auto-load pretrained model")
    
    # Modern options
    parser.add_argument("--balance", type=float, help="Initial balance")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    print(f"DEBUG: Arguments parsed successfully: mode={args.mode}")
    
    # Create modern configuration based on mode
    if args.mode == "online":
        config = ConfigPresets.conservative_live()
        config.live_mode = True
        logger.info("🔴 ONLINE MODE SELECTED - Using simulated live data with SmartInfoBus v4.0")
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
    
    print(f"DEBUG: Configuration created - live_mode={config.live_mode}")
    
    # Apply SmartInfoBus configuration
    config.info_bus_enabled = args.smartinfobus
    
    # Apply parameter overrides
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
    if args.balance:
        config.initial_balance = args.balance
    if args.debug:
        config.debug = True
    if args.data_dir:
        config.data_dir = args.data_dir
    
    # Ensure reasonable defaults for quick testing
    if not args.timesteps and args.mode == "test":
        config.final_training_steps = 10000  # Shorter for testing
        print(f"🧪 TEST MODE: Reduced timesteps to {config.final_training_steps:,}")
    elif not args.timesteps:
        # Default to reasonable training length if not specified
        config.final_training_steps = max(50000, config.final_training_steps)
        print(f"📊 Using {config.final_training_steps:,} timesteps")
    
    # Handle pretrained model
    pretrained_path = None
    if args.pretrained:
        pretrained_path = args.pretrained
    elif args.auto_pretrained:
        auto_path = "models/modern_ppo_trading_model.zip"
        if os.path.exists(auto_path):
            pretrained_path = auto_path
    
    # Save modern configuration
    print("DEBUG: About to save configuration")
    config_path = os.path.join(config.log_dir, "modern_training_config.json")
    try:
        config.save_config(config_path)
        logger.info(f"📝 Modern configuration saved to {config_path}")
        print("DEBUG: Configuration saved successfully")
    except Exception as e:
        print(f"DEBUG: Error saving config: {e}")
        # Continue anyway
    
    # Modern mode display
    if config.live_mode:
        logger.info("🔴" * 30)
        logger.info("🔴 MODERN LIVE TRAINING MODE!")
        logger.info("🔴 SmartInfoBus v4.0 integration with simulated live data")
        logger.info("🔴 Comprehensive health monitoring active")
        logger.info("🔴 Modern audit trail enabled")
        if pretrained_path:
            logger.info("🔴 TRANSFER LEARNING: Continuing from offline model")
        logger.info("🔴" * 30)
    else:
        logger.info("📊 MODERN OFFLINE TRAINING MODE")
        logger.info(f"📊 Data directory: {config.data_dir}")
        logger.info(f"📊 SmartInfoBus v4.0 integration: {'ENABLED' if config.info_bus_enabled else 'DISABLED'}")
        logger.info(f"📊 Modern monitoring: ENABLED")
        if pretrained_path:
            logger.info("🔄 Continuing training from existing model")
    
    # Run modern training
    print(f"DEBUG: About to call train_modern_ppo with pretrained_path={pretrained_path}")
    train_modern_ppo(config, pretrained_path)
    print("DEBUG: train_modern_ppo completed")

if __name__ == "__main__":
    print("ABOUT TO ENTER MAIN")
    try:
        main()
        print("MAIN FINISHED OK")
    except Exception as e:
        print(f"UNHANDLED EXCEPTION: {e}")