# envs/env.py
"""
Enhanced InfoBus-Integrated Trading Environment
Maintains all existing class names and interfaces while adding InfoBus infrastructure
"""
from __future__ import annotations

import copy
import warnings
from typing import Any, Dict, Optional, Tuple, List
from collections import defaultdict

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

# Enhanced configuration and utilities with InfoBus
from .config import TradingConfig, MarketState, EpisodeMetrics
from .shared_utils import TradingPipeline, UnifiedRiskManager

# InfoBus infrastructure
from modules.utils.info_bus import InfoBus, create_info_bus, validate_info_bus
from modules.utils.audit_utils import RotatingLogger, AuditTracker, format_operator_message

# Import all enhanced method implementations with InfoBus integration
from .env_initialization import (
    _setup_logging, _set_seeds, seed, _validate_data, _initialize_modules,
    _initialize_arbiter, _initialize_dependent_modules, _create_pipeline,
    _get_stable_observation_space, _create_dummy_input, DummyExplanationGenerator
)

from .env_trading import (
    _validate_actions, _apply_meta_rl, _pass_risk_checks, _get_committee_decision,
    _calculate_consensus, _pass_consensus_check, _execute_trades, _execute_single_trade,
    _execute_simulated_trade, _execute_live_trade, _calculate_position_size,
    _round_lot_size, _create_no_trade_step, step
)

from .env_observation import (
    _sanitize_observation, _get_full_observation, _get_next_observation,
    _get_fallback_observation, _create_info_bus_for_step
)

from .env_memory import (
    _feed_memory_modules, _get_current_market_context, _update_memory_compressor,
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

from .env_trading import (
    _extract_current_market_data, _update_regime_modules, _create_voting_context,
    _get_legacy_observation_for_voting, _blend_timeframe_actions
)

# Core modules for voting
from modules.auditing.trade_explanation_auditor import TradeExplanationAuditor
from modules.meta.metar_rl_controller import MetaRLController
from modules.voting.collusion_auditor import CollusionAuditor
from modules.voting.consensus_detector import ConsensusDetector
from modules.voting.time_horizon_aligner import TimeHorizonAligner
from modules.voting.voting_wrappers import MetaRLExpert

# Suppress warnings for cleaner logs
warnings.filterwarnings('ignore', category=RuntimeWarning)


class EnhancedTradingEnv(gym.Env):
    """
    Enhanced InfoBus-integrated trading environment with comprehensive module integration
    Maintains backward compatibility while providing InfoBus infrastructure
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        data_dict: Dict[str, Dict[str, pd.DataFrame]],
        config: Optional[TradingConfig] = None,
    ):
        super().__init__()

        # ─── Enhanced Core Infrastructure ──────────────────────────────
        self.config = config or TradingConfig()
        self.committee = []                  # Will be filled by arbiter
        self._obs_cache = {}                 # Enhanced observation caching
        self.current_step = 0
        self.module_enabled = defaultdict(lambda: True)
        
        # InfoBus state
        self.info_bus: Optional[InfoBus] = None
        self.last_info_bus_quality = None

        # ─── Enhanced Logging & Auditing ─────────────────────────────── 
        self._setup_logging()
        self._set_seeds(self.config.init_seed)
        
        # Audit system
        self.audit_tracker = AuditTracker("TradingEnvironment")

        # ─── Enhanced Market & Episode State ────────────────────────────
        initial_bal = self._get_initial_balance()
        self.market_state = MarketState(
            balance=initial_bal,
            peak_balance=initial_bal,
            current_step=0,
            current_drawdown=0.0,
        )
        self.episode_count = 0
        self.episode_metrics = EpisodeMetrics()

        # ─── Enhanced Data & Instruments ─────────────────────────────────
        self.orig_data = data_dict
        self.data = copy.deepcopy(data_dict)
        self.instruments = list(self.data.keys())
        self._validate_data()

        # ─── Enhanced Action Space ───────────────────────────────────────
        self.action_dim = 2 * len(self.instruments)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )
        
        # Trade auditor with InfoBus
        self.trade_auditor = TradeExplanationAuditor(
            config=self.config.get_module_config()
        )

        # ─── Initialize Meta-RL Controller (Placeholder) ────────────────
        self.meta_rl = None  # Will be properly initialized after observation space

        # ─── Enhanced Consensus & Voting Infrastructure ─────────────────
        self.consensus = ConsensusDetector(0)  # Will resize after arbiter
        self.haligner = TimeHorizonAligner([1, 4, 24, 96])
        self.collusion = CollusionAuditor(4, 3, config=self.config.get_module_config())

        # ─── Initialize Enhanced Core Modules ───────────────────────────
        self._initialize_modules()

        # ─── Enhanced Strategy Arbiter & Pipeline ──────────────────────
        self._initialize_arbiter()
        self._initialize_dependent_modules()

        # ─── Resize Consensus for Final Committee Size ─────────────────
        self.consensus.resize(len(self.committee))

        # ─── Enhanced Observation Space ─────────────────────────────────
        self.observation_space = self._get_stable_observation_space()
        obs_dim = self.observation_space.shape[0]

        # ─── Initialize Enhanced MetaRL Controller ──────────────────────
        self.meta_rl = MetaRLController(
            obs_dim, self.action_dim, 
            config=self.config.get_module_config()
        )

        # ─── Update MetaRLExpert in Arbiter ─────────────────────────────
        for idx, member in enumerate(self.arbiter.members):
            if isinstance(member, MetaRLExpert):
                self.arbiter.members[idx] = MetaRLExpert(self.meta_rl, self)
                break

        # ─── Enhanced Trading State Fields ──────────────────────────────
        self.trades = []
        self.current_genome = None
        self._last_actions = np.zeros(self.action_dim, dtype=np.float32)
        self._last_reward = 0.0
        self.point_value = {
            "EUR/USD": 100000, "XAU/USD": 100, "EURUSD": 100000, "XAUUSD": 100,
        }

        # ─── Log Successful Initialization ──────────────────────────────
        self.logger.info(
            format_operator_message(
                "🚀", "ENVIRONMENT_INITIALIZED",
                details=f"{len(self.instruments)} instruments, action_dim={self.action_dim}, obs_dim={obs_dim}",
                result=f"InfoBus {'ENABLED' if self.config.info_bus_enabled else 'DISABLED'}",
                context="system_startup"
            )
        )

    # ═══════════════════════════════════════════════════════════════════
    # Enhanced Main Interface Methods
    # ═══════════════════════════════════════════════════════════════════

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Enhanced reset with comprehensive InfoBus integration"""
        super().reset(seed=seed)
        
        if seed is not None:
            self._set_seeds(seed)
            
        self.logger.info(
            format_operator_message(
                "🔄", "ENVIRONMENT_RESET",
                details=f"Starting episode {self.episode_count + 1}",
                context="episode_management"
            )
        )
        
        # Reset episode tracking
        self.episode_count += 1
        self.episode_metrics = EpisodeMetrics()
        
        # Reset data
        self.data = copy.deepcopy(self.orig_data)
        
        # Enhanced market state reset
        initial_balance = self._get_initial_balance()
        self.market_state = MarketState(
            balance=initial_balance,
            peak_balance=initial_balance,
            current_step=self._select_starting_step(),
            current_drawdown=0.0,
            last_trade_step={inst: -999 for inst in self.instruments}
        )
        self.current_step = self.market_state.current_step
        
        # Clear enhanced caches
        self._obs_cache.clear()
        self.info_bus = None
        self.last_info_bus_quality = None
        
        # Reset all enhanced modules
        self._reset_all_modules()
        
        # Prime enhanced risk system
        self._prime_risk_system()
        
        # Select new strategy genome
        self._select_strategy_genome()
        
        # Create initial InfoBus and get observation
        info_bus = create_info_bus(self, step=self.market_state.current_step)
        obs = self._get_full_observation(info_bus)
        obs = self._sanitize_observation(obs)
        
        # Initialize meta-RL embedding
        if hasattr(self.meta_rl, 'last_embedding'):
            self.meta_rl.last_embedding = np.zeros_like(obs)
            
        # Enhanced reset info
        info = self._create_reset_info()
        info['info_bus_quality'] = validate_info_bus(info_bus)
        
        return obs, info

    # ═══════════════════════════════════════════════════════════════════
    # Method Implementations (Enhanced with InfoBus)
    # ═══════════════════════════════════════════════════════════════════

    # Initialization methods
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
    
    # Enhanced trading methods with InfoBus
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

    _extract_current_market_data = _extract_current_market_data
    _update_regime_modules = _update_regime_modules
    _create_voting_context = _create_voting_context
    _get_legacy_observation_for_voting = _get_legacy_observation_for_voting
    _blend_timeframe_actions = _blend_timeframe_actions
    
    # Enhanced observation methods with InfoBus
    _sanitize_observation = _sanitize_observation
    _get_full_observation = _get_full_observation
    _get_next_observation = _get_next_observation
    _get_fallback_observation = _get_fallback_observation
    _create_info_bus_for_step = _create_info_bus_for_step
    
    # Enhanced memory methods with InfoBus
    _feed_memory_modules = _feed_memory_modules
    _get_current_market_context = _get_current_market_context
    _update_memory_compressor = _update_memory_compressor
    _record_episode_in_replay_analyzer = _record_episode_in_replay_analyzer
    
    # Enhanced utility methods with InfoBus
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

    # ═══════════════════════════════════════════════════════════════════
    # Enhanced InfoBus-Specific Methods
    # ═══════════════════════════════════════════════════════════════════

    def get_info_bus_status(self) -> Dict[str, Any]:
        """Get current InfoBus system status"""
        
        status = {
            'enabled': self.config.info_bus_enabled,
            'last_quality': self.last_info_bus_quality,
            'pipeline_size': getattr(self.pipeline, 'expected_size', None),
            'module_count': len(self.pipeline.modules) if hasattr(self, 'pipeline') else 0,
        }
        
        if hasattr(self.pipeline, 'module_performance'):
            status['module_performance'] = dict(self.pipeline.module_performance)
        
        return status

    def get_enhanced_metrics(self) -> Dict[str, Any]:
        """Get enhanced metrics including InfoBus data"""
        
        metrics = {
            'episode': self.episode_count,
            'step': self.market_state.current_step,
            'balance': self.market_state.balance,
            'drawdown': self.market_state.current_drawdown,
            'trades': len(self.episode_metrics.trades),
            'info_bus_status': self.get_info_bus_status(),
        }
        
        # Add module health if available
        if hasattr(self, 'pipeline') and hasattr(self.pipeline, 'get_performance_summary'):
            metrics['pipeline_performance'] = self.pipeline.get_performance_summary()
        
        return metrics

    def force_info_bus_refresh(self):
        """Force refresh of InfoBus data (for debugging)"""
        
        if not self.config.info_bus_enabled:
            self.logger.warning("InfoBus is disabled - cannot refresh")
            return
        
        try:
            # Create fresh InfoBus
            self.info_bus = create_info_bus(self, step=self.market_state.current_step)
            self.last_info_bus_quality = validate_info_bus(self.info_bus)
            
            self.logger.info(
                format_operator_message(
                    "🔄", "INFOBUS_REFRESHED",
                    details=f"Quality: {self.last_info_bus_quality.is_valid}",
                    context="debugging"
                )
            )
            
        except Exception as e:
            self.logger.error(f"InfoBus refresh failed: {e}")

    # ═══════════════════════════════════════════════════════════════════
    # Backward Compatibility Methods
    # ═══════════════════════════════════════════════════════════════════

    def get_observation_legacy(self, price_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Legacy observation method for backward compatibility"""
        
        # Create InfoBus from legacy data
        info_bus = create_info_bus(self, step=self.market_state.current_step)
        
        # Add legacy price data
        for inst, prices in price_data.items():
            if inst in self.instruments:
                info_bus[f'price_history_{inst}'] = prices
        
        return self._get_full_observation(info_bus)

    def feed_memory_legacy(self, trades: List[Dict], actions: np.ndarray, obs: np.ndarray):
        """Legacy memory feeding for backward compatibility"""
        
        info_bus = create_info_bus(self, step=self.market_state.current_step)
        info_bus['recent_trades'] = trades
        info_bus['raw_actions'] = actions
        info_bus['observation'] = obs
        
        self._feed_memory_modules(info_bus)


# ═══════════════════════════════════════════════════════════════════
# Module Integration Checklist & Summary
# ═══════════════════════════════════════════════════════════════════

INTEGRATED_MODULES_CHECKLIST = {
    # ✅ Fully Integrated with InfoBus
    'auditing': ['TradeExplanationAuditor', 'TradeThesisTracker'],
    'core': ['InfoBus', 'Module', 'ModuleConfig', 'Mixins'],
    'external': ['All external data sources'],
    'features': ['AdvancedFeatureEngine', 'MultiScaleFeatureEngine'],
    'market': ['MarketThemeDetector', 'FractalRegimeConfirmation', 'LiquidityHeatmapLayer', 
              'TimeAwareRiskScaling', 'RegimePerformanceMatrix'],
    'memory': ['NeuralMemoryArchitect', 'MistakeMemory', 'MemoryCompressor', 
              'HistoricalReplayAnalyzer', 'PlaybookMemory', 'MemoryBudgetOptimizer'],
    'models': ['RNNWorldModel'],
    'position': ['PositionManager'],
    'reward': ['RiskAdjustedReward'],
    'risk': ['ActiveTradeMonitor', 'CorrelatedRiskController', 'DrawdownRescue',
            'ExecutionQualityMonitor', 'AnomalyDetector', 'PortfolioRiskSystem',
            'ComplianceModule', 'DynamicRiskController'],
    'simulation': ['OpponentSimulator', 'RoleCoach', 'ShadowSimulator'],
    'trading_modes': ['TradingModeManager'],
    'visualisation': ['VisualizationInterface', 'TradeMapVisualizer'],
    'voting': ['TimeHorizonAligner', 'AlternativeRealitySampler', 'CollusionAuditor',
              'ConsensusDetector', 'StrategyArbiter', 'VotingWrappers'],
    'strategy': ['PlaybookClusterer', 'StrategyIntrospector', 'CurriculumPlannerPlus',
                'StrategyGenomePool', 'BiasAuditor', 'OpponentModeEnhancer',
                'ThesisEvolutionEngine', 'ExplanationGenerator'],
    'utils': ['InfoBus', 'AuditUtils', 'All utility functions'],
    'meta': ['MetaAgent', 'MetaCognitivePlanner', 'MetaRLController'],
    
    # ✅ Now Integrated with InfoBus
    'environment': ['TradingConfig', 'TradingPipeline', 'UnifiedRiskManager',
                   'EnhancedTradingEnv', 'All env methods'],
}

def get_integration_summary() -> str:
    """Get comprehensive integration summary"""
    
    total_modules = sum(len(modules) for modules in INTEGRATED_MODULES_CHECKLIST.values())
    
    return f"""
🎯 INFUBUS INTEGRATION COMPLETE
═══════════════════════════════════════════════════════════════════

✅ INTEGRATED MODULES: {total_modules} modules across {len(INTEGRATED_MODULES_CHECKLIST)} categories

🏗️ INFRASTRUCTURE ADDED:
• Enhanced InfoBus integration throughout environment
• Operator-centric logging with 2000-line rotation
• Comprehensive audit tracking
• Backward compatibility maintained
• Enhanced error handling and recovery

🔧 ENVIRONMENT ENHANCEMENTS:
• InfoBus-integrated observation pipeline
• Enhanced committee decision making
• Comprehensive memory module integration
• Real-time risk management with InfoBus
• Enhanced trading execution with context

🛡️ RELIABILITY IMPROVEMENTS:
• Robust error handling with graceful degradation
• Enhanced data validation and sanitization
• Comprehensive logging and audit trails
• Performance monitoring and optimization
• Memory management and resource control

📊 MONITORING CAPABILITIES:
• Real-time module performance tracking
• InfoBus quality validation
• Enhanced consensus and voting analysis
• Comprehensive trade and risk monitoring
• Detailed episode and system metrics

🎯 RESULT: Complete InfoBus integration while maintaining all existing
   interfaces and ensuring seamless backward compatibility.
"""


if __name__ == "__main__":
    print(get_integration_summary())