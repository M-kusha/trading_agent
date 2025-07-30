"""
Modern SmartInfoBus Trading Environment
Zero-wiring architecture with automatic module discovery
Fixed for modern Gymnasium compatibility
"""
from __future__ import annotations

import copy
import warnings
from typing import Any, Dict, Optional, Tuple, List
import asyncio
import threading
import time

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

# Configuration
from .config import TradingConfig, MarketState, EpisodeMetrics

# SmartInfoBus infrastructure
try:
    from modules.utils.info_bus import InfoBusManager
    from modules.utils.audit_utils import RotatingLogger
    SMARTINFOBUS_AVAILABLE = True
except ImportError:
    InfoBusManager = None
    RotatingLogger = None
    SMARTINFOBUS_AVAILABLE = False

# Core module system
try:
    from modules.core.module_system import ModuleOrchestrator
    MODULE_SYSTEM_AVAILABLE = True
except ImportError:
    ModuleOrchestrator = None
    MODULE_SYSTEM_AVAILABLE = False

# Suppress warnings for cleaner logs
warnings.filterwarnings('ignore', category=RuntimeWarning)


class ModernTradingEnv(gym.Env):
    """
    Modern SmartInfoBus-integrated trading environment.
    Zero-wiring architecture - modules auto-discover and self-organize.
    Full Gymnasium v0.26+ compatibility
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        data_dict: Dict[str, Dict[str, pd.DataFrame]],
        config: Optional[TradingConfig] = None,
    ):
        super().__init__()

        # ═══════════════════════════════════════════════════════════
        # Core Configuration
        # ═══════════════════════════════════════════════════════════
        self.config = config or TradingConfig()
        self.current_step = 0
        
        # ═══════════════════════════════════════════════════════════
        # Logging (Initialize first to avoid circular dependencies)
        # ═══════════════════════════════════════════════════════════
        self.logger = self._create_logger()
        
        # ═══════════════════════════════════════════════════════════
        # SmartInfoBus & Module System (with timeout protection)
        # ═══════════════════════════════════════════════════════════
        
        self.smart_bus = None
        self.smart_bus_enabled = False
        self.orchestrator = None
        self.orchestrator_enabled = False
        
        # Initialize systems with timeout protection
        self._initialize_systems()

        # ═══════════════════════════════════════════════════════════
        # Market Data & State
        # ═══════════════════════════════════════════════════════════
        self.orig_data = data_dict
        self.data = copy.deepcopy(data_dict)
        self.instruments = list(self.data.keys())
        self._validate_data()
        
        # Market state
        initial_balance = self.config.initial_balance
        self.market_state = MarketState(
            balance=initial_balance,
            peak_balance=initial_balance,
            current_step=0,
            current_drawdown=0.0,
        )
        
        # Episode tracking
        self.episode_count = 0
        self.episode_metrics = EpisodeMetrics()
        
        # ═══════════════════════════════════════════════════════════
        # Action & Observation Spaces
        # ═══════════════════════════════════════════════════════════
        self.action_dim = 2 * len(self.instruments)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )
        
        # Dynamic observation space - let modules determine size
        self.observation_space = self._get_observation_space()
        
        # ═══════════════════════════════════════════════════════════
        # Initialize System
        # ═══════════════════════════════════════════════════════════
        self._setup_environment()
        
        # Log initialization status
        module_count = len(self.orchestrator.modules) if self.orchestrator else 0
        self.logger.info(
            f"🚀 MODERN_ENV_INITIALIZED: {len(self.instruments)} instruments, {module_count} modules - "
            f"{'Zero-wiring architecture active' if self.orchestrator_enabled else 'Fallback mode active'}"
        )

    def _create_logger(self):
        """Create logger with fallback"""
        try:
            if SMARTINFOBUS_AVAILABLE and RotatingLogger:
                return RotatingLogger(
                    name="ModernTradingEnv",
                    log_path="logs/modern_env.log",
                    max_lines=2000,
                    operator_mode=True
                )
            else:
                # Fallback logger
                import logging
                logger = logging.getLogger("ModernTradingEnv")
                if not logger.handlers:
                    handler = logging.StreamHandler()
                    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                    handler.setFormatter(formatter)
                    logger.addHandler(handler)
                    logger.setLevel(logging.INFO)
                return logger
        except Exception as e:
            print(f"Failed to create logger: {e}")
            # Create simple fallback logger
            class FallbackLogger:
                def info(self, msg): print(f"INFO: {msg}")
                def warning(self, msg): print(f"WARNING: {msg}")
                def error(self, msg): print(f"ERROR: {msg}")
                def debug(self, msg): print(f"DEBUG: {msg}")
            return FallbackLogger()

    def _initialize_systems(self):
        """Initialize SmartInfoBus and Module systems with timeout protection"""
        
        # Try to initialize SmartInfoBus with timeout
        if SMARTINFOBUS_AVAILABLE and InfoBusManager:
            try:
                def init_smart_bus():
                    try:
                        if InfoBusManager is not None:
                            self.smart_bus = InfoBusManager.get_instance()
                            self.smart_bus_enabled = True
                        else:
                            self.smart_bus = None
                            self.smart_bus_enabled = False
                    except Exception as e:
                        self.logger.warning(f"SmartInfoBus initialization failed: {e}")
                        self.smart_bus = None
                        self.smart_bus_enabled = False
                
                # Start SmartInfoBus initialization in background
                bus_thread = threading.Thread(target=init_smart_bus, daemon=True)
                bus_thread.start()
                bus_thread.join(timeout=1.0)  # 1 second timeout
                
                if bus_thread.is_alive():
                    self.logger.warning("SmartInfoBus initialization taking too long - using fallback mode")
                    self.smart_bus = None
                    self.smart_bus_enabled = False
                
            except Exception as e:
                self.logger.warning(f"Failed to start SmartInfoBus: {e}")
                self.smart_bus = None
                self.smart_bus_enabled = False
        
        # Create fallback SmartInfoBus if needed
        if not self.smart_bus_enabled:
            self.smart_bus = self._create_fallback_smart_bus()
            self.smart_bus_enabled = True
        
        # Try to initialize ModuleOrchestrator with timeout
        if MODULE_SYSTEM_AVAILABLE and ModuleOrchestrator:
            try:
                def init_orchestrator():
                    try:
                        if ModuleOrchestrator is not None:
                            self.orchestrator = ModuleOrchestrator()
                            self.orchestrator.initialize()
                            self.orchestrator_enabled = True
                        else:
                            self.orchestrator = None
                            self.orchestrator_enabled = False
                    except Exception as e:
                        self.logger.warning(f"ModuleOrchestrator initialization failed: {e}")
                        self.orchestrator = None
                        self.orchestrator_enabled = False
                
                # Start orchestrator initialization in background
                orchestrator_thread = threading.Thread(target=init_orchestrator, daemon=True)
                orchestrator_thread.start()
                orchestrator_thread.join(timeout=2.0)  # 2 second timeout
                
                if orchestrator_thread.is_alive():
                    self.logger.warning("ModuleOrchestrator initialization taking too long - continuing without it")
                    self.orchestrator = None
                    self.orchestrator_enabled = False
                
            except Exception as e:
                self.logger.warning(f"Failed to start ModuleOrchestrator: {e}")
                self.orchestrator = None
                self.orchestrator_enabled = False

    def _validate_data(self):
        """Validate market data"""
        required_columns = {"open", "high", "low", "close"}
        
        for inst in self.instruments:
            if inst not in self.data:
                raise ValueError(f"Missing data for instrument: {inst}")
                
            for tf, df in self.data[inst].items():
                missing = required_columns - set(df.columns)
                if missing:
                    raise ValueError(f"Missing columns for {inst}/{tf}: {missing}")
                
                # Add volume if missing
                if 'volume' not in df.columns:
                    self.data[inst][tf]['volume'] = 1.0
    
    def _get_observation_space(self) -> spaces.Box:
        """Get observation space from modules"""
        # Start with a reasonable default
        default_size = 256
        
        return spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(default_size,), 
            dtype=np.float32
        )
    
    def _setup_environment(self):
        """Setup environment for trading"""
        # Store basic environment data in SmartInfoBus
        if self.smart_bus:
            self.smart_bus.set(
                'environment_config',
                {
                    'instruments': self.instruments,
                    'initial_balance': self.config.initial_balance,
                    'action_dim': self.action_dim,
                    'max_steps': self.config.max_steps,
                },
                module='Environment',
                thesis="Environment configuration for module access"
            )
        
        # Initialize market data in SmartInfoBus
        self._store_market_data()
    
    def _store_market_data(self):
        """Store current market data in SmartInfoBus"""
        step = self.market_state.current_step
        
        for instrument in self.instruments:
            for timeframe in ['H1', 'H4', 'D1']:
                if timeframe in self.data[instrument]:
                    df = self.data[instrument][timeframe]
                    if step < len(df):
                        # Current price
                        current_price = float(df['close'].iloc[step])
                        if self.smart_bus:
                            self.smart_bus.set(
                                f'price_{instrument}_{timeframe}',
                                current_price,
                                module='Environment',
                                thesis=f"Current {instrument} price at step {step}"
                            )
                        
                        # OHLCV window
                        window_size = min(100, step + 1)
                        start_idx = max(0, step - window_size + 1)
                        end_idx = step + 1
                        
                        ohlcv_data = {
                            'open': df['open'].iloc[start_idx:end_idx].values,
                            'high': df['high'].iloc[start_idx:end_idx].values,
                            'low': df['low'].iloc[start_idx:end_idx].values,
                            'close': df['close'].iloc[start_idx:end_idx].values,
                            'volume': df['volume'].iloc[start_idx:end_idx].values,
                            'step': step,
                            'instrument': instrument,
                            'timeframe': timeframe
                        }
                        
                        if self.smart_bus:
                            self.smart_bus.set(
                                f'market_data_{instrument}_{timeframe}',
                                ohlcv_data,
                                module='Environment',
                                thesis=f"Market data window for {instrument} {timeframe}"
                            )

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment to initial state (Gymnasium v0.26+ compatible)"""
        super().reset(seed=seed)
        
        # Handle seeding for Gymnasium compatibility
        if seed is not None:
            np.random.seed(seed)
        
        self.logger.info(
            f"🔄 ENVIRONMENT_RESET: Episode {self.episode_count + 1}"
        )
        
        # Reset episode tracking
        self.episode_count += 1
        self.episode_metrics = EpisodeMetrics()
        
        # Reset data
        self.data = copy.deepcopy(self.orig_data)
        
        # Reset market state
        initial_balance = self.config.initial_balance
        self.market_state = MarketState(
            balance=initial_balance,
            peak_balance=initial_balance,
            current_step=self._select_starting_step(),
            current_drawdown=0.0,
        )
        self.current_step = self.market_state.current_step
        
        # Reset SmartInfoBus for new episode
        if self.smart_bus:
            self.smart_bus.set(
                'episode_info',
                {
                    'episode': self.episode_count,
                    'step': self.current_step,
                    'balance': self.market_state.balance,
                    'reset': True
                },
                module='Environment',
                thesis=f"Episode {self.episode_count} reset information"
            )
        
        # Store initial market data
        self._store_market_data()
        
        # Get initial observation from modules
        obs = self._get_observation()
        
        # Create reset info
        modules_active = len(self.orchestrator.modules) if self.orchestrator else 0
        info = {
            'episode': self.episode_count,
            'step': self.current_step,
            'balance': self.market_state.balance,
            'modules_active': modules_active,
            'reset': True
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step"""
        self.current_step += 1
        self.market_state.current_step = self.current_step
        
        # Store action in SmartInfoBus
        if self.smart_bus:
            self.smart_bus.set(
                'agent_action',
                action,
                module='Environment',
                thesis=f"Agent action at step {self.current_step}"
            )
        
        # Store updated market data
        self._store_market_data()
        
        # Store current market state
        if self.smart_bus:
            self.smart_bus.set(
                'market_state',
                {
                    'balance': self.market_state.balance,
                    'step': self.current_step,
                    'drawdown': self.market_state.current_drawdown,
                    'peak_balance': self.market_state.peak_balance
                },
                module='Environment',
                thesis="Current market state for modules"
            )
        
        # Execute modules through orchestrator (if available)
        if self.orchestrator_enabled and self.orchestrator:
            try:
                asyncio.run(self.orchestrator.execute_step({}))
            except Exception as e:
                self.logger.error(f"Module execution failed: {e}")
        
        # Get final trading decision from SmartInfoBus
        final_action = (self.smart_bus.get('final_trading_action', 'Environment') if self.smart_bus else None) or action
        
        # Execute trade and calculate reward
        reward = self._execute_step(final_action)
        
        # Get next observation
        obs = self._get_observation()
        
        # Check termination
        terminated, truncated = self._check_termination()
        
        # Create step info
        modules_executed = len(self.orchestrator.modules) if self.orchestrator else 0
        info = {
            'step': self.current_step,
            'balance': self.market_state.balance,
            'drawdown': self.market_state.current_drawdown,
            'reward': reward,
            'modules_executed': modules_executed,
            'terminated': terminated,
            'truncated': truncated
        }
        
        return obs, reward, terminated, truncated, info
    
    def _select_starting_step(self) -> int:
        """Select random starting step"""
        if not self.instruments:
            return 0
        
        # Find minimum data length across all instruments
        min_length = float('inf')
        for instrument in self.instruments:
            for timeframe, df in self.data[instrument].items():
                min_length = min(min_length, len(df))
        
        if min_length == float('inf') or min_length < 100:
            return 0
        
        # Random start between 50 and (length - max_steps - 50)
        max_start = max(50, int(min_length) - self.config.max_steps - 50)
        
        return np.random.randint(50, int(max_start))
    
    def _get_observation(self) -> np.ndarray:
        """Get observation from SmartInfoBus"""
        # Check if modules produced an observation
        obs = self.smart_bus.get('environment_observation', 'Environment') if self.smart_bus else None
        
        if obs is not None and isinstance(obs, np.ndarray):
            # Ensure observation matches expected size
            expected_size = self.observation_space.shape[0] if self.observation_space.shape else 256
            if obs.size != expected_size:
                if obs.size < expected_size:
                    # Pad with zeros
                    padded_obs = np.zeros(expected_size, dtype=np.float32)
                    padded_obs[:obs.size] = obs.flatten()
                    return padded_obs
                else:
                    # Truncate
                    return obs.flatten()[:expected_size]
            return obs.flatten()
        
        # Fallback: create basic observation
        return self._create_fallback_observation()
    
    def _create_fallback_observation(self) -> np.ndarray:
        """Create MULTI-TIMEFRAME observation if modules don't provide one"""
        features = []
        
        # Market state features
        features.extend([
            self.market_state.balance / 10000.0,  # Normalized balance
            self.market_state.current_drawdown,
            float(self.current_step) / 1000.0,   # Normalized step
        ])
        
        # Multi-timeframe price features for each instrument
        for instrument in self.instruments:
            # Add features from ALL timeframes for comprehensive analysis
            for timeframe in ['H1', 'H4', 'D1']:
                if timeframe in self.data[instrument]:
                    df = self.data[instrument][timeframe]
                    if self.current_step < len(df):
                        current_bar = {
                            'open': df['open'].iloc[self.current_step],
                            'high': df['high'].iloc[self.current_step],
                            'low': df['low'].iloc[self.current_step],
                            'close': df['close'].iloc[self.current_step],
                            'volume': df['volume'].iloc[self.current_step]
                        }
                        
                        # Add normalized OHLCV features
                        base_price = current_bar['close']
                        features.extend([
                            current_bar['close'] / 10000.0,  # Normalized close
                            (current_bar['high'] - current_bar['low']) / base_price if base_price > 0 else 0,  # HL range %
                            (current_bar['close'] - current_bar['open']) / base_price if base_price > 0 else 0,  # OC change %
                            current_bar['volume'] / 1000.0,  # Normalized volume
                        ])
                        
                        # Add momentum features (if enough history)
                        if self.current_step >= 5:
                            prev_close = df['close'].iloc[self.current_step - 5]
                            momentum = (current_bar['close'] - prev_close) / prev_close if prev_close > 0 else 0
                            features.append(momentum)
                        else:
                            features.append(0.0)
                            
                        # Add volatility feature (20-period if available)
                        if self.current_step >= 20:
                            recent_closes = df['close'].iloc[self.current_step-19:self.current_step+1]
                            volatility = recent_closes.std() / recent_closes.mean() if recent_closes.mean() > 0 else 0
                            features.append(volatility)
                        else:
                            features.append(0.01)  # Default volatility
                            
                    else:
                        # Fill with zeros if no data
                        features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                else:
                    # Fill with zeros if timeframe not available
                    features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Add cross-timeframe analysis features
        # Compare H1 vs H4 vs D1 trends for each instrument
        for instrument in self.instruments:
            h1_trend = h4_trend = d1_trend = 0.0
            
            # Calculate simple trend for each timeframe
            for tf_name, trend_var in [('H1', 'h1_trend'), ('H4', 'h4_trend'), ('D1', 'd1_trend')]:
                if tf_name in self.data[instrument] and self.current_step >= 5:
                    df = self.data[instrument][tf_name]
                    if self.current_step < len(df):
                        current_price = df['close'].iloc[self.current_step]
                        past_price = df['close'].iloc[max(0, self.current_step - 5)]
                        trend = (current_price - past_price) / past_price if past_price > 0 else 0
                        locals()[trend_var] = trend
            
            # Add trend alignment features
            features.extend([
                h1_trend,
                h4_trend, 
                d1_trend,
                1.0 if (h1_trend > 0 and h4_trend > 0 and d1_trend > 0) else 0.0,  # All timeframes bullish
                1.0 if (h1_trend < 0 and h4_trend < 0 and d1_trend < 0) else 0.0,  # All timeframes bearish
            ])
        
        # Pad to expected size
        obs_array = np.array(features, dtype=np.float32)
        expected_size = self.observation_space.shape[0] if self.observation_space.shape else 256
        
        if obs_array.size < expected_size:
            padded_obs = np.zeros(expected_size, dtype=np.float32)
            padded_obs[:obs_array.size] = obs_array
            return padded_obs
        elif obs_array.size > expected_size:
            return obs_array[:expected_size]
        else:
            return obs_array
        
        return obs_array[:expected_size]
    
    def _execute_step(self, action: np.ndarray) -> float:
        """Execute trading step and return reward"""
        # Get trading result from SmartInfoBus
        trading_result = self.smart_bus.get('trading_result', 'Environment') if self.smart_bus else None
        
        if trading_result:
            pnl = trading_result.get('pnl', 0.0)
            
            # Update balance
            self.market_state.balance += pnl
            
            # Update peak and drawdown
            if self.market_state.balance > self.market_state.peak_balance:
                self.market_state.peak_balance = self.market_state.balance
                self.market_state.current_drawdown = 0.0
            else:
                drawdown = (self.market_state.peak_balance - self.market_state.balance) / self.market_state.peak_balance
                self.market_state.current_drawdown = drawdown
            
            return pnl / 100.0  # Normalized reward
        
        # Default: small negative reward for no action
        return -0.01
    
    def _check_termination(self) -> Tuple[bool, bool]:
        """Check if episode should terminate"""
        # Check drawdown limit
        if self.market_state.current_drawdown > self.config.max_drawdown:
            return True, False  # Terminated due to drawdown
        
        # Check max steps
        if self.current_step >= self.config.max_steps:
            return False, True  # Truncated due to time limit
        
        # Check balance
        if self.market_state.balance <= 0:
            return True, False  # Terminated due to bankruptcy
        
        return False, False
    
    def _create_fallback_smart_bus(self):
        """Create a fallback SmartInfoBus implementation"""
        class FallbackSmartBus:
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
        
        return FallbackSmartBus()
    
    def get_smartinfobus_status(self) -> Dict[str, Any]:
        """Get SmartInfoBus system status"""
        modules_active = len(self.orchestrator.modules) if self.orchestrator else 0
        return {
            'performance_metrics': self.smart_bus.get_performance_metrics() if self.smart_bus else {},
            'modules_active': modules_active,
            'modules_disabled': list(self.smart_bus._module_disabled) if self.smart_bus else [],
            'data_keys': len(self.smart_bus._data_store) if self.smart_bus else 0,
            'current_step': self.current_step,
            'episode': self.episode_count
        }
    
    def render(self, mode: str = "human"):
        """Render environment"""
        if mode == "human":
            print(f"Step: {self.current_step}, Balance: ${self.market_state.balance:.2f}, "
                  f"Drawdown: {self.market_state.current_drawdown:.1%}")
    
    def close(self):
        """Close environment"""
        self.logger.info("Environment closed")

    # ════════════════════════════════════════════════════════════════════
    # Legacy Compatibility Methods (for Stable-Baselines3 compatibility)
    # ════════════════════════════════════════════════════════════════════
    
    def seed(self, seed: Optional[int] = None):
        """
        Legacy seed method for compatibility with older environments.
        This is deprecated in favor of passing seed to reset().
        """
        if seed is not None:
            np.random.seed(seed)
        return [seed]
    
    @property
    def unwrapped(self):
        """Return unwrapped environment (for compatibility)"""
        return self