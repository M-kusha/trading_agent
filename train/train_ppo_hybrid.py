# ─────────────────────────────────────────────────────────────
# File: train/enhanced_ppo_training.py
# Complete InfoBus-Integrated Training Script with Comprehensive Module Health Monitoring
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

# Import MT5 for live trading
import MetaTrader5 as mt5

# Enhanced InfoBus-integrated components
from train.enhanced_training_callback import InfoBusTrainingCallback, ModuleHealthTracker
from live.live_connector import InfoBusLiveDataConnector, InfoBusLiveTradingCallback

# InfoBus and audit infrastructure
from modules.utils.info_bus import InfoBus, create_info_bus, validate_info_bus
from modules.utils.audit_utils import RotatingLogger, AuditTracker, format_operator_message, system_audit

# Import environment and utilities
try:
    from envs import EnhancedTradingEnv, TradingConfig
    from envs.config import ConfigPresets, ConfigFactory
    print("✅ Successfully imported InfoBus-integrated environment")
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

# Create enhanced directory structure
log_dirs = [
    'logs/training', 'logs/live', 'logs/infobus', 'logs/health',
    'logs/regime', 'logs/strategy', 'logs/checkpoints', 'logs/audit',
    'logs/reward', 'logs/tensorboard', 'checkpoints', 'models/best', 'metrics'
]
for log_dir in log_dirs:
    os.makedirs(log_dir, exist_ok=True)

# Enhanced logging configuration with InfoBus integration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/training/enhanced_ppo_training.log', encoding='utf-8')
    ]
)
logger = logging.getLogger("EnhancedPPOTraining")

# ═══════════════════════════════════════════════════════════════════
# ENHANCED METRICS BROADCASTING SYSTEM WITH INFOBUS
# ═══════════════════════════════════════════════════════════════════

class EnhancedMetricsBroadcaster:
    """Enhanced metrics broadcaster with InfoBus integration and health monitoring"""
    
    def __init__(self, host='localhost', port=8001):
        self.host = host
        self.port = port
        self.metrics_queue = queue.Queue(maxsize=1000)  # Larger queue
        self.ws_thread = None
        self.running = False
        self.websocket = None
        
        # Enhanced tracking
        self.messages_sent = 0
        self.messages_failed = 0
        self.connection_attempts = 0
        self.last_successful_send = None
        
        # InfoBus integration
        self.audit_tracker = AuditTracker("MetricsBroadcaster")
        self.logger = RotatingLogger(
            name="MetricsBroadcaster",
            log_path="logs/training/metrics_broadcaster.log",
            max_lines=2000,
            operator_mode=True
        )
        
    def start(self):
        """Enhanced start with health monitoring"""
        self.running = True
        self.ws_thread = threading.Thread(target=self._run_websocket)
        self.ws_thread.daemon = True
        self.ws_thread.start()
        
        self.logger.info(
            format_operator_message(
                "📡", "METRICS_BROADCASTER_STARTED",
                details=f"ws://{self.host}:{self.port}",
                context="broadcasting_startup"
            )
        )
        
        # Record audit event
        self.audit_tracker.record_event(
            "broadcaster_started",
            "MetricsBroadcaster",
            {"host": self.host, "port": self.port},
            severity="info"
        )
        
    def stop(self):
        """Enhanced stop with cleanup reporting"""
        self.running = False
        if self.ws_thread:
            self.ws_thread.join(timeout=10)
            
        # Log final statistics
        self.logger.info(
            format_operator_message(
                "📡", "METRICS_BROADCASTER_STOPPED",
                details=f"Sent: {self.messages_sent}, Failed: {self.messages_failed}",
                result=f"Success rate: {(self.messages_sent/(self.messages_sent+self.messages_failed)*100) if (self.messages_sent+self.messages_failed) > 0 else 0:.1f}%",
                context="broadcasting_shutdown"
            )
        )
        
    def send_metrics(self, metrics: Dict[str, Any]):
        """Enhanced metrics sending with health tracking"""
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
                }
            }
            
            self.metrics_queue.put(enhanced_metrics, block=False)
            
        except queue.Full:
            self.messages_failed += 1
            self.logger.warning("Metrics queue full - dropping message")
            
    def _run_websocket(self):
        """Enhanced WebSocket client loop with health monitoring"""
        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()
        
        while self.running:
            try:
                self.connection_attempts += 1
                loop.run_until_complete(self._websocket_handler())
            except Exception as e:
                self.logger.error(f"WebSocket error: {e}")
                if self.running:
                    asyncio.sleep(5)  # Reconnect after 5 seconds
                    
    async def _websocket_handler(self):
        """Enhanced WebSocket handler with comprehensive error handling"""
        uri = f"ws://{self.host}:{self.port}/ws/training"
        
        try:
            async with websockets.connect(uri) as websocket:
                self.websocket = websocket
                self.logger.info("✅ Connected to metrics server")
                
                while self.running:
                    try:
                        # Get metrics from queue (non-blocking with timeout)
                        metrics = self.metrics_queue.get(timeout=0.1)
                        
                        message = {
                            "type": "enhanced_training_metrics",
                            "data": metrics,
                            "timestamp": datetime.now().isoformat(),
                            "source": "InfoBusTraining"
                        }
                        
                        await websocket.send(json.dumps(message, default=str))
                        self.messages_sent += 1
                        self.last_successful_send = datetime.now().isoformat()
                        
                    except queue.Empty:
                        continue
                    except Exception as e:
                        self.messages_failed += 1
                        self.logger.error(f"Error sending metrics: {e}")
                        
        except Exception as e:
            self.logger.error(f"WebSocket connection error: {e}")

# Global enhanced metrics broadcaster
enhanced_metrics_broadcaster = EnhancedMetricsBroadcaster()

# ═══════════════════════════════════════════════════════════════════
# ENHANCED LIVE DATA COLLECTION WITH INFOBUS
# ═══════════════════════════════════════════════════════════════════

def connect_mt5_with_infobus() -> bool:
    """Enhanced MT5 connection with InfoBus monitoring"""
    
    logger.info(
        format_operator_message(
            "🔗", "ATTEMPTING_MT5_CONNECTION",
            details="Initializing MT5 connection",
            context="live_connection"
        )
    )
    
    try:
        if not mt5.initialize():
            error = mt5.last_error()
            logger.error(f"MT5 initialization failed: {error}")
            return False
        
        account_info = mt5.account_info()
        if account_info:
            logger.info(
                format_operator_message(
                    "✅", "MT5_CONNECTION_SUCCESSFUL",
                    details=f"Account: {account_info.login}",
                    result=f"Server: {account_info.server}, Balance: ${account_info.balance:.2f}",
                    context="live_connection"
                )
            )
            
            # Record successful connection
            system_audit.record_event(
                "mt5_connection_established",
                "MT5Connector",
                {
                    "account": account_info.login,
                    "server": account_info.server,
                    "balance": account_info.balance
                },
                severity="info"
            )
        
        return True
        
    except Exception as e:
        logger.error(f"MT5 connection failed: {e}")
        system_audit.record_event(
            "mt5_connection_failed",
            "MT5Connector",
            {"error": str(e)},
            severity="error"
        )
        return False

def get_live_market_data_with_infobus(config: TradingConfig) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Enhanced live market data collection with InfoBus integration"""
    
    logger.info(
        format_operator_message(
            "🔴", "FETCHING_LIVE_DATA",
            details="Real-time market data collection",
            result="InfoBus integration enabled",
            context="live_data"
        )
    )
    
    # Create InfoBus-integrated connector
    connector = InfoBusLiveDataConnector(
        instruments=[inst.replace("/", "") for inst in config.instruments],  # Convert to MT5 format
        timeframes=config.timeframes,
        config=config.get_module_config()
    )
    
    try:
        # Connect with InfoBus monitoring
        info_bus = create_info_bus(None)  # Create basic InfoBus
        connector.connect(info_bus)
        
        # Get historical data with InfoBus integration
        live_data = connector.get_historical_data_with_infobus(n_bars=1000, info_bus=info_bus)
        
        if not live_data:
            raise RuntimeError("No live data received from MT5")
        
        # Log data summary
        total_bars = sum(len(df) for inst_data in live_data.values() for df in inst_data.values())
        logger.info(
            format_operator_message(
                "✅", "LIVE_DATA_READY",
                details=f"{len(live_data)} instruments",
                result=f"{total_bars} total bars collected",
                context="live_data"
            )
        )
        
        # Enhanced data quality reporting
        for instrument, timeframes in live_data.items():
            for tf, df in timeframes.items():
                if not df.empty:
                    latest_time = df.index[-1] if len(df) > 0 else "Unknown"
                    latest_price = df['close'].iloc[-1] if len(df) > 0 else 0
                    logger.info(
                        format_operator_message(
                            "📊", "DATA_QUALITY_CHECK",
                            instrument=instrument,
                            details=f"{tf}: {len(df)} bars",
                            result=f"Latest: {latest_price:.5f} @ {latest_time}",
                            context="data_validation"
                        )
                    )
        
        # Cleanup
        connector.disconnect(info_bus)
        
        return live_data
        
    except Exception as e:
        logger.error(
            format_operator_message(
                "💥", "LIVE_DATA_FAILED",
                details=str(e),
                context="live_data_error"
            )
        )
        raise

# ═══════════════════════════════════════════════════════════════════
# ENHANCED DATA LOADING WITH INFOBUS INTEGRATION
# ═══════════════════════════════════════════════════════════════════

def load_training_data_with_infobus(config: TradingConfig) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Smart data loading with InfoBus monitoring and enhanced validation"""
    
    if config.live_mode:
        logger.info(
            format_operator_message(
                "🔴", "LIVE_MODE_DETECTED",
                details="Switching to real-time MT5 data",
                context="data_loading"
            )
        )
        try:
            return get_live_market_data_with_infobus(config)
        except Exception as e:
            logger.error(f"Live data loading failed: {e}")
            raise
    else:
        logger.info(
            format_operator_message(
                "📊", "OFFLINE_MODE_DETECTED",
                details="Loading historical CSV data",
                context="data_loading"
            )
        )
        try:
            # Try to load from CSV files
            if os.path.exists(config.data_dir):
                data = load_data(config.data_dir)
                if data:
                    # Enhanced data validation
                    validated_data = validate_and_enhance_data(data, config)
                    
                    total_bars = sum(len(df) for inst_data in validated_data.values() for df in inst_data.values())
                    logger.info(
                        format_operator_message(
                            "✅", "HISTORICAL_DATA_LOADED",
                            details=f"{len(validated_data)} instruments",
                            result=f"{total_bars} total bars",
                            context="data_loading"
                        )
                    )
                    return validated_data
                    
            # If no data found, create dummy data
            logger.warning("No historical data found, creating dummy data...")
            return create_enhanced_dummy_data(config)
            
        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
            return create_enhanced_dummy_data(config)

def validate_and_enhance_data(data: Dict[str, Dict[str, pd.DataFrame]], 
                            config: TradingConfig) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Enhanced data validation with InfoBus quality tracking"""
    
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
                    df = perform_data_quality_checks(df, instrument, tf, validation_summary)
                    
                    validated_data[instrument][tf] = df
                    validation_summary["timeframes_processed"] += 1
                    validation_summary["total_bars"] += len(df)
    
    # Log validation summary
    if validation_summary["quality_issues"]:
        logger.warning(
            format_operator_message(
                "⚠️", "DATA_QUALITY_ISSUES",
                details=f"{len(validation_summary['quality_issues'])} issues found",
                context="data_validation"
            )
        )
        for issue in validation_summary["quality_issues"][:5]:  # Log first 5 issues
            logger.warning(f"  • {issue}")
    
    # Record validation audit
    system_audit.record_event(
        "data_validation_completed",
        "DataLoader",
        validation_summary,
        severity="warning" if validation_summary["quality_issues"] else "info"
    )
    
    return validated_data

def perform_data_quality_checks(df: pd.DataFrame, instrument: str, timeframe: str, 
                               summary: Dict[str, Any]) -> pd.DataFrame:
    """Comprehensive data quality checks and fixes"""
    
    original_length = len(df)
    
    # Check for missing values
    if df.isnull().any().any():
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
    
    if len(df) != original_length:
        summary["quality_issues"].append(f"{instrument}/{timeframe}: Length changed from {original_length} to {len(df)}")
    
    return df

def create_enhanced_dummy_data(config: TradingConfig) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Create enhanced dummy data with realistic market characteristics"""
    
    logger.info(
        format_operator_message(
            "🧪", "CREATING_DUMMY_DATA",
            details="Generating realistic market data for testing",
            context="data_generation"
        )
    )
    
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
    logger.info(
        format_operator_message(
            "✅", "DUMMY_DATA_GENERATED",
            details=f"{len(dummy_data)} instruments",
            result=f"{total_bars} total bars with realistic characteristics",
            context="data_generation"
        )
    )
    
    return dummy_data

# ═══════════════════════════════════════════════════════════════════
# ENHANCED ENVIRONMENT CREATION WITH INFOBUS
# ═══════════════════════════════════════════════════════════════════

def test_env_creation_with_infobus(data: Dict, config: TradingConfig) -> bool:
    """Enhanced environment creation test with InfoBus monitoring"""
    
    try:
        mode_str = "LIVE" if config.live_mode else "OFFLINE"
        logger.info(
            format_operator_message(
                "🔧", "TESTING_ENVIRONMENT_CREATION",
                details=f"{mode_str} mode with InfoBus integration",
                context="env_testing"
            )
        )
        
        # Create test environment
        env = EnhancedTradingEnv(data, config)
        
        # Test InfoBus integration
        if hasattr(env, 'info_bus'):
            logger.info("✅ InfoBus integration confirmed")
        else:
            logger.warning("⚠️ InfoBus not detected in environment")
        
        # Test environment reset and step
        obs, info = env.reset()
        
        # Validate observation
        if not isinstance(obs, np.ndarray):
            raise ValueError(f"Invalid observation type: {type(obs)}")
        
        if not np.all(np.isfinite(obs)):
            raise ValueError("Non-finite values in observation")
        
        logger.info(
            format_operator_message(
                "✅", "ENVIRONMENT_TEST_PASSED",
                details=f"{mode_str} mode",
                result=f"Obs shape: {obs.shape}, InfoBus: {'enabled' if config.info_bus_enabled else 'disabled'}",
                context="env_testing"
            )
        )
        
        # Test module health if available
        if hasattr(env, 'pipeline') and hasattr(env.pipeline, 'modules'):
            module_count = len(env.pipeline.modules)
            logger.info(
                format_operator_message(
                    "📊", "MODULE_INTEGRATION_VERIFIED",
                    details=f"{module_count} modules detected",
                    context="env_testing"
                )
            )
        
        env.close()
        return True
        
    except Exception as e:
        logger.error(
            format_operator_message(
                "💥", "ENVIRONMENT_TEST_FAILED",
                details=str(e),
                context="env_testing_error"
            )
        )
        import traceback
        traceback.print_exc()
        return False

def create_envs_with_infobus(data: Dict, config: TradingConfig, n_envs: int = 1, seed: int = 42):
    """Enhanced environment creation with InfoBus integration"""
    
    if not test_env_creation_with_infobus(data, config):
        logger.error("Environment creation test failed!")
        raise RuntimeError("Cannot create environment")
    
    if config.live_mode:
        n_envs = 1
        logger.info("🔴 LIVE MODE: Using single environment for safety")
    elif platform.system() == "Windows":
        n_envs = 1
        logger.info("Windows detected - using single environment")
    
    logger.info(
        format_operator_message(
            "🏗️", "CREATING_ENVIRONMENTS",
            details=f"{n_envs} environments with InfoBus",
            context="env_creation"
        )
    )
    
    def make_env(rank: int):
        def _init():
            try:
                env = EnhancedTradingEnv(data, config)
                env = Monitor(env, filename=f"logs/training/monitor_{rank}.csv")
                env.seed(seed + rank)
                
                mode_str = "LIVE" if config.live_mode else "OFFLINE"
                logger.info(
                    format_operator_message(
                        "✅", "ENVIRONMENT_CREATED",
                        details=f"Rank {rank}, {mode_str} mode",
                        context="env_creation"
                    )
                )
                return env
            except Exception as e:
                logger.error(f"Failed to create environment {rank}: {e}")
                raise
        
        set_random_seed(seed)
        return _init
    
    env = DummyVecEnv([make_env(i) for i in range(n_envs)])
    
    logger.info(
        format_operator_message(
            "✅", "ENVIRONMENTS_READY",
            details=f"{n_envs} environments created",
            result="InfoBus integration enabled",
            context="env_creation"
        )
    )
    
    return env

def create_ppo_model_enhanced(env, config: TradingConfig):
    """Enhanced PPO model creation with better architecture"""
    
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
    logger.info(
        format_operator_message(
            "🤖", "PPO_MODEL_CREATED",
            details=f"Device: {device_str}",
            result=f"LR: {config.learning_rate}, Hidden: {config.policy_hidden_size}",
            context="model_creation"
        )
    )
    
    return model

# ═══════════════════════════════════════════════════════════════════
# MAIN ENHANCED TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════════

def train_enhanced_ppo(config: TradingConfig, pretrained_model_path: Optional[str] = None):
    """Main enhanced training function with comprehensive InfoBus integration"""
    
    mode_str = "🔴 LIVE (MT5)" if config.live_mode else "📊 OFFLINE (CSV)"
    logger.info("=" * 80)
    logger.info(f"ENHANCED PPO TRAINING - {mode_str} MODE WITH INFOBUS")
    logger.info("=" * 80)
    logger.info(f"Configuration:\n{config}")
    
    # Start enhanced metrics broadcaster
    enhanced_metrics_broadcaster.start()
    
    # Enhanced startup audit
    system_audit.record_event(
        "training_session_started",
        "EnhancedPPOTraining",
        {
            "mode": mode_str,
            "config": config.__dict__,
            "infobus_enabled": config.info_bus_enabled,
            "pretrained_model": pretrained_model_path is not None
        },
        severity="info"
    )
    
    # Send initial status
    enhanced_metrics_broadcaster.send_metrics({
        "status": "INITIALIZING",
        "mode": mode_str,
        "config": config.__dict__,
        "infobus_integration": True,
        "enhanced_features": ["module_health", "audit_trail", "live_integration"]
    })
    
    try:
        # Load appropriate data based on mode with InfoBus
        logger.info(f"{mode_str} Loading market data with InfoBus integration...")
        data = load_training_data_with_infobus(config)
        
        logger.info(f"✅ Data ready for {len(data)} instruments:")
        for instrument, timeframes in data.items():
            for tf, df in timeframes.items():
                if not df.empty:
                    latest_time = df.index[-1] if len(df) > 0 else "Unknown"
                    latest_price = df['close'].iloc[-1] if len(df) > 0 else 0
                    logger.info(f"   {instrument}/{tf}: {len(df)} bars (latest: {latest_price:.5f} @ {latest_time})")
        
        # Create enhanced environments
        logger.info(f"🔧 Creating {mode_str} environments with InfoBus...")
        train_env = create_envs_with_infobus(data, config, n_envs=config.num_envs, seed=config.init_seed)
        eval_env = create_envs_with_infobus(data, config, n_envs=1, seed=config.init_seed + 1000)
        
        # Create or load enhanced model
        logger.info("🤖 Setting up enhanced PPO model...")
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            logger.info(f"📥 Loading pretrained model from: {pretrained_model_path}")
            model = PPO.load(pretrained_model_path, env=train_env)
            
            if config.live_mode:
                model.learning_rate = config.learning_rate * 0.5
                logger.info(f"🔧 Adjusted learning rate for live mode: {model.learning_rate}")
        else:
            logger.info("🆕 Creating new enhanced PPO model...")
            model = create_ppo_model_enhanced(train_env, config)
        
        # Setup enhanced callbacks with InfoBus integration
        callbacks = [
            InfoBusTrainingCallback(
                total_timesteps=config.final_training_steps,
                config=config,
                metrics_broadcaster=enhanced_metrics_broadcaster
            ),
            CheckpointCallback(
                save_freq=config.checkpoint_freq,
                save_path=config.checkpoint_dir,
                name_prefix=f"enhanced_ppo_{config.live_mode and 'live' or 'offline'}"
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
        
        # Add enhanced live trading callback if in live mode
        if config.live_mode:
            logger.info("🔴 Adding InfoBus live trading callback...")
            live_connector = InfoBusLiveDataConnector(
                instruments=[inst.replace("/", "") for inst in config.instruments],
                timeframes=config.timeframes,
                config=config.get_module_config()
            )
            callbacks.append(InfoBusLiveTradingCallback(live_connector))
        
        # Send training started status
        enhanced_metrics_broadcaster.send_metrics({
            "status": "TRAINING_STARTED",
            "mode": mode_str,
            "total_timesteps": config.final_training_steps,
            "model_type": "Enhanced_PPO",
            "device": str(model.device),
            "infobus_features": ["health_monitoring", "audit_trail", "live_integration"],
            "callback_count": len(callbacks)
        })
        
        # Start enhanced training
        logger.info(f"🚀 Starting {mode_str} training with InfoBus integration...")
        logger.info(f"Total timesteps: {config.final_training_steps:,}")
        logger.info(f"InfoBus features: Health monitoring, Audit trail, Module integration")
        
        if config.live_mode:
            logger.info("🔴 LIVE TRAINING WITH REAL MT5 DATA AND INFOBUS!")
            logger.info("🔴 Enhanced monitoring and safety systems active")
        
        start_time = datetime.now()
        
        # Enhanced training with comprehensive monitoring
        model.learn(
            total_timesteps=config.final_training_steps,
            callback=CallbackList(callbacks),
            log_interval=config.log_interval,
            tb_log_name=f"enhanced_ppo_{'live' if config.live_mode else 'offline'}",
            reset_num_timesteps=True,
            progress_bar=True,
        )
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        # Save enhanced final model
        final_model_path = os.path.join(config.model_dir, "enhanced_ppo_trading_model.zip")
        model.save(final_model_path)
        
        # Generate comprehensive training report
        training_summary = {
            "status": "TRAINING_COMPLETED",
            "mode": mode_str,
            "duration": str(training_duration),
            "final_model_path": final_model_path,
            "total_timesteps": config.final_training_steps,
            "infobus_enabled": config.info_bus_enabled,
            "live_mode": config.live_mode,
            "instruments": config.instruments,
            "enhanced_features_used": [
                "InfoBus integration",
                "Module health monitoring", 
                "Enhanced audit trail",
                "Real-time metrics broadcasting",
                "Live trading integration" if config.live_mode else "Offline training"
            ]
        }
        
        # Send completion status
        enhanced_metrics_broadcaster.send_metrics(training_summary)
        
        # Enhanced completion logging
        logger.info("=" * 80)
        logger.info(f"✅ {mode_str} ENHANCED TRAINING COMPLETED!")
        logger.info("=" * 80)
        logger.info(f"📁 Model saved to {final_model_path}")
        logger.info(f"⏱️  Training duration: {training_duration}")
        logger.info(f"🔗 InfoBus integration: {'ENABLED' if config.info_bus_enabled else 'DISABLED'}")
        logger.info(f"📊 Enhanced monitoring: Health checks, audit trail, metrics")
        logger.info("=" * 80)
        
        # Record completion audit
        system_audit.record_event(
            "training_session_completed",
            "EnhancedPPOTraining",
            training_summary,
            severity="info"
        )
        
    except Exception as e:
        error_msg = f"Enhanced training failed: {e}"
        logger.error(f"❌ {error_msg}")
        
        enhanced_metrics_broadcaster.send_metrics({
            "status": "TRAINING_FAILED",
            "error": str(e),
            "mode": mode_str
        })
        
        # Record failure audit
        system_audit.record_event(
            "training_session_failed",
            "EnhancedPPOTraining",
            {"error": str(e), "mode": mode_str},
            severity="error"
        )
        
        raise
        
    finally:
        # Enhanced cleanup
        logger.info("🧹 Performing enhanced cleanup...")
        try:
            train_env.close()
            eval_env.close()
            if config.live_mode:
                mt5.shutdown()
            enhanced_metrics_broadcaster.stop()
            
            logger.info("✅ Enhanced cleanup completed")
            
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

# ═══════════════════════════════════════════════════════════════════
# ENHANCED MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Enhanced InfoBus-Integrated PPO Training")
    
    # Mode selection
    parser.add_argument("--mode", choices=["offline", "online", "test"], default="offline",
                       help="Training mode: offline (CSV), online (MT5 live), test")
    parser.add_argument("--preset", choices=["conservative", "aggressive", "research", "production"], 
                       help="Use configuration preset")
    
    # InfoBus configuration
    parser.add_argument("--infobus", action="store_true", default=True, 
                       help="Enable InfoBus integration (default: True)")
    parser.add_argument("--infobus-audit-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="InfoBus audit level")
    
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
    
    # Enhanced options
    parser.add_argument("--balance", type=float, help="Initial balance")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--health-checks", action="store_true", default=True, 
                       help="Enable module health monitoring (default: True)")
    
    args = parser.parse_args()
    
    # Create enhanced configuration based on mode
    if args.mode == "online":
        config = ConfigPresets.conservative_live()
        config.live_mode = True
        logger.info("🔴 ONLINE MODE SELECTED - Will use MT5 live data with InfoBus")
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
    
    # Apply InfoBus configuration
    config.info_bus_enabled = args.infobus
    config.info_bus_audit_level = args.infobus_audit_level
    
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
    
    # Handle pretrained model
    pretrained_path = None
    if args.pretrained:
        pretrained_path = args.pretrained
    elif args.auto_pretrained:
        auto_path = "models/enhanced_ppo_trading_model.zip"
        if os.path.exists(auto_path):
            pretrained_path = auto_path
    
    # Save enhanced configuration
    config_path = os.path.join(config.log_dir, "enhanced_training_config.json")
    config.save_config(config_path)
    logger.info(f"📝 Enhanced configuration saved to {config_path}")
    
    # Enhanced mode display
    if config.live_mode:
        logger.info("🔴" * 30)
        logger.info("🔴 ENHANCED LIVE TRAINING MODE!")
        logger.info("🔴 InfoBus integration with MT5 real-time data")
        logger.info("🔴 Comprehensive health monitoring active")
        logger.info("🔴 Enhanced audit trail enabled")
        if pretrained_path:
            logger.info("🔴 TRANSFER LEARNING: Continuing from offline model")
        logger.info("🔴 Make sure MT5 is running and connected")
        logger.info("🔴" * 30)
    else:
        logger.info("📊 ENHANCED OFFLINE TRAINING MODE")
        logger.info(f"📊 Data directory: {config.data_dir}")
        logger.info(f"📊 InfoBus integration: {'ENABLED' if config.info_bus_enabled else 'DISABLED'}")
        logger.info(f"📊 Health monitoring: {'ENABLED' if args.health_checks else 'DISABLED'}")
        if pretrained_path:
            logger.info("🔄 Continuing training from existing model")
    
    # Run enhanced training
    train_enhanced_ppo(config, pretrained_path)

if __name__ == "__main__":
    main()