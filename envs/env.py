# envs/env.py
"""
State-of-the-art PPO Trading Environment
FIXED: Proper data flow, committee wiring, and live trade recording
"""
from __future__ import annotations

import copy
import warnings
from typing import Any, Dict, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

# Local imports - configuration and utilities
from .config import TradingConfig, MarketState, EpisodeMetrics
from .shared_utils import UnifiedRiskManager

# Import all method implementations
from .env_initialization import (
    _setup_logging, _set_seeds, seed, _validate_data, _initialize_modules,
    _initialize_arbiter, _initialize_dependent_modules, _create_pipeline,
    _get_stable_observation_space, _create_dummy_input
)
from .env_trading import (
    _validate_actions, _apply_meta_rl, _pass_risk_checks, _get_committee_decision,
    _calculate_consensus, _pass_consensus_check, _execute_trades, _execute_single_trade,
    _execute_simulated_trade, _execute_live_trade, _calculate_position_size,
    _round_lot_size, _create_no_trade_step, step
)
from .env_observation import (
    _sanitize_observation, _get_full_observation, _get_next_observation
)
from .env_memory import (
    _feed_memory_modules, _feed_memory_compressor_step_by_step,
    _get_current_market_context, _update_memory_compressor,
    _record_episode_in_replay_analyzer
)
from .env_utils import (
    _get_initial_balance, _select_starting_step, _reset_all_modules,
    _prime_risk_system, _select_strategy_genome, _get_current_volatility,
    _get_instrument_volatility, _get_price_history, _get_recent_returns,
    get_instrument_correlations, _calculate_reward, _check_termination,
    _finalize_step, _create_reset_info, _create_step_info, _update_mode_manager,
    _handle_episode_end, _save_checkpoints, set_module_enabled, get_state,
    set_state, render, close
)

from modules.auditing.explanation_auditor import TradeExplanationAuditor
from modules.strategy.strategy import MetaRLController
from modules.strategy.voting import ConsensusDetector, CollusionAuditor, TimeHorizonAligner
from modules.strategy.voting_wrappers import MetaRLExpert

# Suppress warnings for cleaner logs
warnings.filterwarnings('ignore', category=RuntimeWarning)


class EnhancedTradingEnv(gym.Env):
    """State-of-the-art trading environment with robust module integration"""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        data_dict: Dict[str, Dict[str, pd.DataFrame]],
        config: Optional[TradingConfig] = None,
    ):
        super().__init__()

        # ─── Core placeholders ───────────────────────────────────────
        self.committee = []                  # filled by arbiter
        self._obs_cache = {}                 # for cached observations
        self.current_step = 0
        self.module_enabled = defaultdict(lambda: True)

        # ─── Config, logging, seeds ─────────────────────────────────
        self.config = config or TradingConfig()
        self._setup_logging()
        self._set_seeds(self.config.init_seed)

        # ─── Market & episode state ─────────────────────────────────
        initial_bal = self._get_initial_balance()
        self.market_state = MarketState(
            balance=initial_bal,
            peak_balance=initial_bal,
            current_step=0,
            current_drawdown=0.0,
        )
        self.episode_count = 0
        self.episode_metrics = EpisodeMetrics()

        # ─── Data & instruments ──────────────────────────────────────
        self.orig_data = data_dict
        self.data = copy.deepcopy(data_dict)
        self.instruments = list(self.data.keys())
        self._validate_data()

        # ─── Action space ───────────────────────────────────────────
        self.action_dim = 2 * len(self.instruments)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )
        self.trade_auditor = TradeExplanationAuditor(
            history_len=200, debug=self.config.debug
        )

        # ─── Stub out meta_rl so arbiter creation won't break ────────
        self.meta_rl = None

        # ─── Consensus & horizon‐aligner for arbiter ────────────────
        self.consensus = ConsensusDetector(0)  # will resize after arbiter
        self.haligner = TimeHorizonAligner([1, 4, 24, 96])
        # (collusion isn't consumed by arbiter, but env.step() might use it)
        self.collusion = CollusionAuditor(4, 3, debug=self.config.debug)

        # ─── Initialize all core modules ────────────────────────────
        self._initialize_modules()

        # ─── Wire up strategy arbiter & pipeline ────────────────────
        self._initialize_arbiter()            # uses self.consensus & self.haligner
        self._initialize_dependent_modules()  # builds self.pipeline

        # ─── Now that committee is known, resize consensus properly ──
        self.consensus.resize(len(self.committee))

        # ─── Create stable observation space (pipeline must exist) ──
        self.observation_space = self._get_stable_observation_space()
        obs_dim = self.observation_space.shape[0]

        # ─── Instantiate the real MetaRL controller ─────────────────
        self.meta_rl = MetaRLController(
            obs_dim, self.action_dim, debug=self.config.debug
        )

        # ─── Replace the stub MetaRLExpert inside the arbiter ───────
        for idx, member in enumerate(self.arbiter.members):
            if isinstance(member, MetaRLExpert):
                self.arbiter.members[idx] = MetaRLExpert(self.meta_rl, self)
                break

        # ─── Final trading‐state fields ─────────────────────────────
        self.trades = []
        self.current_genome = None
        self._last_actions = np.zeros(self.action_dim, dtype=np.float32)
        self._last_reward = 0.0
        self.info_bus = None
        self.point_value = {
            "EUR/USD": 100000,
            "XAU/USD": 100,
            "EURUSD": 100000,
            "XAUUSD": 100,
        }

        self.logger.info(
            f"Environment initialized with {len(self.instruments)} instruments, "
            f"action_dim={self.action_dim}, obs_dim={obs_dim}"
        )

    # ═══════════════════════════════════════════════════════════════════
    #  Main Interface Methods
    # ═══════════════════════════════════════════════════════════════════

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        if seed is not None:
            self._set_seeds(seed)
            
        self.logger.info(f"Resetting environment for episode {self.episode_count + 1}")
        
        # Reset episode tracking
        self.episode_count += 1
        self.episode_metrics = EpisodeMetrics()
        
        # Reset data
        self.data = copy.deepcopy(self.orig_data)
        
        # Reset market state
        initial_balance = self._get_initial_balance()
        self.market_state = MarketState(
            balance=initial_balance,
            peak_balance=initial_balance,
            current_step=self._select_starting_step(),
            current_drawdown=0.0,
            last_trade_step={inst: -999 for inst in self.instruments}
        )
        self.current_step = self.market_state.current_step  # FIXED: Sync step counters
        
        # Clear caches
        if not hasattr(self, '_obs_cache') or not isinstance(self._obs_cache, dict):
            self._obs_cache = {}
        else:
            self._obs_cache.clear()
        
        # Reset all modules
        self._reset_all_modules()
        
        # Prime risk system with historical data
        self._prime_risk_system()
        
        # Select new strategy genome
        self._select_strategy_genome()
        
        # Get initial observation
        obs = self._get_full_observation(self._create_dummy_input())
        obs = self._sanitize_observation(obs)
        
        # Initialize meta-RL embedding
        if hasattr(self.meta_rl, 'last_embedding'):
            self.meta_rl.last_embedding = np.zeros_like(obs)
            
        info = self._create_reset_info()
        
        return obs, info

    # Import all method implementations
    _setup_logging = _setup_logging
    _set_seeds = _set_seeds
    seed = seed
    _validate_data = _validate_data
    _initialize_modules = _initialize_modules
    _initialize_arbiter = _initialize_arbiter
    _initialize_dependent_modules = _initialize_dependent_modules
    _create_pipeline = _create_pipeline
    _get_stable_observation_space = _get_stable_observation_space
    _create_dummy_input = _create_dummy_input
    
    # Trading methods
    _validate_actions = _validate_actions
    _apply_meta_rl = _apply_meta_rl
    _pass_risk_checks = _pass_risk_checks
    _get_committee_decision = _get_committee_decision
    _calculate_consensus = _calculate_consensus
    _pass_consensus_check = _pass_consensus_check
    _execute_trades = _execute_trades
    _execute_single_trade = _execute_single_trade
    _execute_simulated_trade = _execute_simulated_trade
    _execute_live_trade = _execute_live_trade
    _calculate_position_size = _calculate_position_size
    _round_lot_size = _round_lot_size
    _create_no_trade_step = _create_no_trade_step
    step = step
    
    # Observation methods
    _sanitize_observation = _sanitize_observation
    _get_full_observation = _get_full_observation
    _get_next_observation = _get_next_observation
    
    # Memory methods
    _feed_memory_modules = _feed_memory_modules
    _feed_memory_compressor_step_by_step = _feed_memory_compressor_step_by_step
    _get_current_market_context = _get_current_market_context
    _update_memory_compressor = _update_memory_compressor
    _record_episode_in_replay_analyzer = _record_episode_in_replay_analyzer
    
    # Utility methods
    _get_initial_balance = _get_initial_balance
    _select_starting_step = _select_starting_step
    _reset_all_modules = _reset_all_modules
    _prime_risk_system = _prime_risk_system
    _select_strategy_genome = _select_strategy_genome
    _get_current_volatility = _get_current_volatility
    _get_instrument_volatility = _get_instrument_volatility
    _get_price_history = _get_price_history
    _get_recent_returns = _get_recent_returns
    get_instrument_correlations = get_instrument_correlations
    _calculate_reward = _calculate_reward
    _check_termination = _check_termination
    _finalize_step = _finalize_step
    _create_reset_info = _create_reset_info
    _create_step_info = _create_step_info
    _update_mode_manager = _update_mode_manager
    _handle_episode_end = _handle_episode_end
    _save_checkpoints = _save_checkpoints
    set_module_enabled = set_module_enabled
    get_state = get_state
    set_state = set_state
    render = render
    close = close