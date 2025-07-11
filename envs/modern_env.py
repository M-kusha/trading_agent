"""
Modern SmartInfoBus Trading Environment
Zero-wiring architecture with automatic module discovery
"""
from __future__ import annotations

import copy
import warnings
from typing import Any, Dict, Optional, Tuple, List
import asyncio

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

# Configuration
from .config import TradingConfig, MarketState, EpisodeMetrics

# SmartInfoBus infrastructure
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message

# Core module system
from modules.core.module_system import ModuleOrchestrator

# Suppress warnings for cleaner logs
warnings.filterwarnings('ignore', category=RuntimeWarning)


class ModernTradingEnv(gym.Env):
    """
    Modern SmartInfoBus-integrated trading environment.
    Zero-wiring architecture - modules auto-discover and self-organize.
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
        # SmartInfoBus & Module System
        # ═══════════════════════════════════════════════════════════
        self.smart_bus = InfoBusManager.get_instance()
        self.orchestrator = ModuleOrchestrator()
        
        # ═══════════════════════════════════════════════════════════
        # Logging
        # ═══════════════════════════════════════════════════════════
        self.logger = RotatingLogger(
            name="ModernTradingEnv",
            log_path="logs/modern_env.log",
            max_lines=2000,
            operator_mode=True
        )
        
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
        
        self.logger.info(
            format_operator_message(
                "🚀", "MODERN_ENV_INITIALIZED",
                details=f"{len(self.instruments)} instruments, {len(self.orchestrator.modules)} modules",
                result="Zero-wiring architecture active",
                context="system_startup"
            )
        )
    
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
                        
                        self.smart_bus.set(
                            f'market_data_{instrument}_{timeframe}',
                            ohlcv_data,
                            module='Environment',
                            thesis=f"Market data window for {instrument} {timeframe}"
                        )
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.logger.info(
            format_operator_message(
                "🔄", "ENVIRONMENT_RESET",
                details=f"Episode {self.episode_count + 1}",
                context="episode_management"
            )
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
        
        # Module reset handled automatically by orchestrator during execution
        
        # Get initial observation from modules
        obs = self._get_observation()
        
        # Create reset info
        info = {
            'episode': self.episode_count,
            'step': self.current_step,
            'balance': self.market_state.balance,
            'modules_active': len(self.orchestrator.modules),
            'reset': True
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step"""
        self.current_step += 1
        self.market_state.current_step = self.current_step
        
        # Store action in SmartInfoBus
        self.smart_bus.set(
            'agent_action',
            action,
            module='Environment',
            thesis=f"Agent action at step {self.current_step}"
        )
        
        # Store updated market data
        self._store_market_data()
        
        # Store current market state
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
        
        # Execute modules through orchestrator
        try:
            asyncio.run(self.orchestrator.execute_step({}))
        except Exception as e:
            self.logger.error(f"Module execution failed: {e}")
        
        # Get final trading decision from SmartInfoBus
        final_action = self.smart_bus.get('final_trading_action', 'Environment') or action
        
        # Execute trade and calculate reward
        reward = self._execute_step(final_action)
        
        # Get next observation
        obs = self._get_observation()
        
        # Check termination
        terminated, truncated = self._check_termination()
        
        # Create step info
        info = {
            'step': self.current_step,
            'balance': self.market_state.balance,
            'drawdown': self.market_state.current_drawdown,
            'reward': reward,
            'modules_executed': len(self.orchestrator.modules),
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
        obs = self.smart_bus.get('environment_observation', 'Environment')
        
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
        """Create fallback observation if modules don't provide one"""
        features = []
        
        # Market state features
        features.extend([
            self.market_state.balance / 10000.0,  # Normalized balance
            self.market_state.current_drawdown,
            float(self.current_step) / 1000.0,   # Normalized step
        ])
        
        # Price features for each instrument
        for instrument in self.instruments:
            if 'H1' in self.data[instrument]:
                df = self.data[instrument]['H1']
                if self.current_step < len(df):
                    close_price = df['close'].iloc[self.current_step]
                    features.extend([
                        close_price / 10000.0,  # Normalized price
                        df['volume'].iloc[self.current_step] / 1000.0,  # Normalized volume
                    ])
                else:
                    features.extend([0.0, 0.0])
            else:
                features.extend([0.0, 0.0])
        
        # Pad to expected size
        obs_array = np.array(features, dtype=np.float32)
        expected_size = self.observation_space.shape[0] if self.observation_space.shape else 256
        
        if obs_array.size < expected_size:
            padded_obs = np.zeros(expected_size, dtype=np.float32)
            padded_obs[:obs_array.size] = obs_array
            return padded_obs
        
        return obs_array[:expected_size]
    
    def _execute_step(self, action: np.ndarray) -> float:
        """Execute trading step and return reward"""
        # Get trading result from SmartInfoBus
        trading_result = self.smart_bus.get('trading_result', 'Environment')
        
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
    
    def get_smartinfobus_status(self) -> Dict[str, Any]:
        """Get SmartInfoBus system status"""
        return {
            'performance_metrics': self.smart_bus.get_performance_metrics(),
            'modules_active': len(self.orchestrator.modules),
            'modules_disabled': list(self.smart_bus._module_disabled),
            'data_keys': len(self.smart_bus._data_store),
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