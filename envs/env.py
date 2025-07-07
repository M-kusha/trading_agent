# envs/env.py
"""
Enhanced SmartInfoBus-Integrated Trading Environment
Now with Module Orchestrator for zero-wiring architecture
"""
from __future__ import annotations

import copy
import warnings
from typing import Any, Dict, Optional, Tuple, List
from collections import defaultdict
import asyncio

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

# Enhanced configuration and utilities with SmartInfoBus
from .config import TradingConfig, MarketState, EpisodeMetrics
from .shared_utils import TradingPipeline, UnifiedRiskManager

# SmartInfoBus infrastructure
from modules.utils.info_bus import SmartInfoBus, InfoBus, InfoBusManager, create_info_bus, validate_info_bus
from modules.utils.audit_utils import RotatingLogger, SmartInfoBusAuditTracker, format_operator_message

# Module Orchestrator for zero-wiring
from modules.core.module_orchestrator import ModuleOrchestrator
from modules.core.state_manager import StateManager

# Import all enhanced method implementations
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
    Enhanced SmartInfoBus-integrated trading environment with Module Orchestrator.
    Zero-wiring architecture with automatic module discovery and execution.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        data_dict: Dict[str, Dict[str, pd.DataFrame]],
        config: Optional[TradingConfig] = None,
    ):
        super().__init__()

        # ─── Enhanced Core Infrastructure with SmartInfoBus ────────────
        self.config = config or TradingConfig()
        self.committee = []  # Will be filled by orchestrator
        self._obs_cache = {}
        self.current_step = 0
        self.module_enabled = defaultdict(lambda: True)
        
        # SmartInfoBus state
        self.smart_bus = InfoBusManager.get_instance()
        self.info_bus: Optional[InfoBus] = None
        self.last_info_bus_quality = None
        
        # Module Orchestrator for zero-wiring
        self.orchestrator: Optional[ModuleOrchestrator] = None
        self.state_manager = StateManager()

        # ─── Enhanced Logging & Auditing with SmartInfoBus ──────────── 
        self._setup_logging()
        self._set_seeds(self.config.init_seed)
        
        # Enhanced audit system
        self.audit_tracker = SmartInfoBusAuditTracker("TradingEnvironment")

        # ─── Market & Episode State ──────────────────────────────────────
        initial_bal = self._get_initial_balance()
        self.market_state = MarketState(
            balance=initial_bal,
            peak_balance=initial_bal,
            current_step=0,
            current_drawdown=0.0,
        )
        self.episode_count = 0
        self.episode_metrics = EpisodeMetrics()

        # ─── Data & Instruments ──────────────────────────────────────────
        self.orig_data = data_dict
        self.data = copy.deepcopy(data_dict)
        self.instruments = list(self.data.keys())
        self._validate_data()

        # ─── Action Space ────────────────────────────────────────────────
        self.action_dim = 2 * len(self.instruments)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )
        
        # Initialize trade auditor
        self.trade_auditor = TradeExplanationAuditor(
            config=self.config.get_module_config()
        )

        # ─── Initialize Module Orchestrator ──────────────────────────────
        self._initialize_orchestrator()

        # ─── Legacy module initialization for compatibility ──────────────
        self.meta_rl = None  # Will be found by orchestrator
        
        # Consensus & Voting Infrastructure
        self.consensus = ConsensusDetector(0)  # Will resize after discovery
        self.haligner = TimeHorizonAligner([1, 4, 24, 96])
        self.collusion = CollusionAuditor(4, 3, config=self.config.get_module_config())

        # ─── Discover and Initialize All Modules ─────────────────────────
        self._discover_all_modules()

        # ─── Get observation space from modules ──────────────────────────
        self.observation_space = self._get_stable_observation_space()
        obs_dim = self.observation_space.shape[0]

# ─── Trading State Fields ────────────────────────────────────────
        self.trades = []
        self.current_genome = None
        self._last_actions = np.zeros(self.action_dim, dtype=np.float32)
        self._last_reward = 0.0
        self.point_value = {
            "EUR/USD": 100000, "XAU/USD": 100, "EURUSD": 100000, "XAUUSD": 100,
        }

        # ─── Log Successful Initialization ───────────────────────────────
        self.logger.info(
            format_operator_message(
                "🚀", "ENVIRONMENT_INITIALIZED",
                details=f"{len(self.instruments)} instruments, {len(self.orchestrator.modules)} modules",
                result=f"SmartInfoBus ACTIVE",
                context="system_startup"
            )
        )

    # ═══════════════════════════════════════════════════════════════════
    # SmartInfoBus Module Orchestration
    # ═══════════════════════════════════════════════════════════════════

    def _initialize_orchestrator(self):
        """Initialize Module Orchestrator for zero-wiring"""
        self.orchestrator = ModuleOrchestrator(self.smart_bus)
        
        # Subscribe to orchestrator events
        self.smart_bus.subscribe('module_disabled', self._on_module_disabled)
        self.smart_bus.subscribe('performance_warning', self._on_performance_warning)
        
        self.logger.info(
            format_operator_message(
                "🎯", "ORCHESTRATOR_INITIALIZED",
                details="Zero-wiring architecture active",
                context="module_management"
            )
        )
    
    def _discover_all_modules(self):
        """Discover and initialize all decorated modules"""
        self.logger.info("🔍 Starting module discovery...")
        
        # Orchestrator auto-discovers on init
        module_count = len(self.orchestrator.modules)
        voting_count = len(self.orchestrator.voting_members)
        
        # Update committee with voting members
        self.committee = [
            self.orchestrator.modules[name] 
            for name in self.orchestrator.voting_members
        ]
        
        # Resize consensus detector
        if self.committee:
            self.consensus.resize(len(self.committee))
        
        # Find MetaRL controller if present
        for name, module in self.orchestrator.modules.items():
            if 'MetaRL' in name:
                self.meta_rl = module
                break
        
        self.logger.info(
            format_operator_message(
                "✅", "MODULES_DISCOVERED",
                details=f"{module_count} modules, {voting_count} voting members",
                result=f"Execution order: {' → '.join(self.orchestrator.execution_order[:5])}...",
                context="module_discovery"
            )
        )
    
    def _on_module_disabled(self, event: Dict[str, Any]):
        """Handle module disabled event"""
        module = event['module']
        failures = event['failures']
        
        self.logger.error(
            format_operator_message(
                "🚨", "MODULE_DISABLED",
                instrument=module,
                details=f"After {failures} failures",
                context="circuit_breaker"
            )
        )
        
        # Update internal tracking
        self.module_enabled[module] = False
    
    def _on_performance_warning(self, event: Dict[str, Any]):
        """Handle performance warning event"""
        module = event['module']
        latency = event['avg_latency_ms']
        
        self.logger.warning(
            format_operator_message(
                "⚠️", "PERFORMANCE_DEGRADATION",
                instrument=module,
                details=f"Avg latency: {latency:.0f}ms",
                context="performance"
            )
        )

    # ═══════════════════════════════════════════════════════════════════
    # Enhanced Main Interface Methods with SmartInfoBus
    # ═══════════════════════════════════════════════════════════════════

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Enhanced reset with SmartInfoBus orchestration"""
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
        
        # Reset market state
        initial_balance = self._get_initial_balance()
        self.market_state = MarketState(
            balance=initial_balance,
            peak_balance=initial_balance,
            current_step=self._select_starting_step(),
            current_drawdown=0.0,
            last_trade_step={inst: -999 for inst in self.instruments}
        )
        self.current_step = self.market_state.current_step
        
        # Clear caches
        self._obs_cache.clear()
        self.info_bus = None
        self.last_info_bus_quality = None
        
        # Reset all modules through orchestrator
        self._reset_all_modules_orchestrated()
        
        # Prime risk system
        self._prime_risk_system()
        
        # Select new strategy genome
        self._select_strategy_genome()
        
        # Create initial InfoBus and get observation
        self.info_bus = create_info_bus(self, step=self.market_state.current_step)
        
        # Execute initial module step through orchestrator
        self._execute_orchestrated_step(self.info_bus)
        
        # Get observation after modules have processed
        obs = self._get_full_observation(self.info_bus)
        obs = self._sanitize_observation(obs)
        
        # Initialize meta-RL embedding if available
        if self.meta_rl and hasattr(self.meta_rl, 'last_embedding'):
            self.meta_rl.last_embedding = np.zeros_like(obs)
            
        # Enhanced reset info
        info = self._create_reset_info()
        info['info_bus_quality'] = validate_info_bus(self.info_bus)
        info['orchestrator_status'] = {
            'modules_active': len([m for m in self.orchestrator.modules if self.smart_bus.is_module_enabled(m)]),
            'modules_disabled': list(self.smart_bus._module_disabled)
        }
        
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Enhanced step with SmartInfoBus orchestration"""
        # Store raw action in SmartInfoBus
        self.smart_bus.set(
            'raw_actions',
            action,
            module='Environment',
            thesis=f"Agent action at step {self.current_step}"
        )
        
        # Create InfoBus for this step
        self.info_bus = self._create_info_bus_for_step()
        self.info_bus['raw_actions'] = action
        
        # Execute orchestrated module processing
        asyncio.run(self._execute_orchestrated_step_async(action))
        
        # Get final actions after module processing
        final_actions = self._get_final_actions()
        
        # Execute trades
        executed_trades = self._execute_trades(final_actions)
        
        # Calculate reward
        reward = self._calculate_reward(executed_trades)
        
        # Get next observation
        obs = self._get_next_observation()
        obs = self._sanitize_observation(obs)
        
        # Check termination
        terminated, truncated = self._check_termination()
        
        # Create step info with SmartInfoBus data
        info = self._create_enhanced_step_info(executed_trades, final_actions)
        
        # Finalize step
        self._finalize_step(reward, terminated or truncated)
        
        return obs, reward, terminated, truncated, info

    # ═══════════════════════════════════════════════════════════════════
    # SmartInfoBus Orchestration Methods
    # ═══════════════════════════════════════════════════════════════════

    def _reset_all_modules_orchestrated(self):
        """Reset all modules through orchestrator"""
        reset_count = 0
        failed_resets = []
        
        for module_name, module in self.orchestrator.modules.items():
            try:
                module.reset()
                reset_count += 1
            except Exception as e:
                self.logger.error(f"Failed to reset {module_name}: {e}")
                failed_resets.append(module_name)
        
        # Re-enable previously disabled modules
        for module in list(self.smart_bus._module_disabled):
            self.smart_bus.reset_module_failures(module)
        
        self.logger.info(
            format_operator_message(
                "🔄", "MODULES_RESET",
                details=f"{reset_count} modules reset successfully",
                result=f"Failed: {len(failed_resets)}",
                context="reset"
            )
        )
    
    def _execute_orchestrated_step(self, info_bus: InfoBus):
        """Execute all modules in dependency order synchronously"""
        # Store market data in SmartInfoBus
        self._store_market_data_in_smartbus()
        
        # Execute modules in order
        for module_name in self.orchestrator.execution_order:
            if not self.smart_bus.is_module_enabled(module_name):
                continue
            
            module = self.orchestrator.modules.get(module_name)
            if module and hasattr(module, 'step'):
                try:
                    module.step(info_bus)
                except Exception as e:
                    self.logger.error(f"Module {module_name} failed: {e}")
                    self.smart_bus.record_module_failure(module_name, str(e))
    
    async def _execute_orchestrated_step_async(self, action: np.ndarray):
        """Execute modules asynchronously through orchestrator"""
        # Prepare market data
        market_data = self._extract_current_market_data()
        
        # Execute through orchestrator
        try:
            results = await self.orchestrator.execute_step(market_data)
            
            # Log execution summary
            self.logger.debug(
                f"Orchestrator executed {len(results)} modules in "
                f"{self.smart_bus.get_performance_metrics()['module_latencies']}"
            )
            
        except Exception as e:
            self.logger.error(f"Orchestrator execution failed: {e}")
            # Fall back to synchronous execution
            self._execute_orchestrated_step(self.info_bus)
    
    def _store_market_data_in_smartbus(self):
        """Store current market data in SmartInfoBus"""
        step = self.market_state.current_step
        
        for instrument in self.instruments:
            for timeframe in ['H1', 'H4', 'D1']:
                if timeframe in self.data[instrument]:
                    df = self.data[instrument][timeframe]
                    if step < len(df):
                        # Store current price
                        self.smart_bus.set(
                            f'price_{instrument}_{timeframe}',
                            float(df['close'].iloc[step]),
                            module='Environment',
                            thesis=f"Market price for {instrument} at step {step}"
                        )
                        
                        # Store OHLCV data
                        window_size = 100
                        start_idx = max(0, step - window_size + 1)
                        end_idx = step + 1
                        
                        ohlcv_data = {
                            'open': df['open'].iloc[start_idx:end_idx].values,
                            'high': df['high'].iloc[start_idx:end_idx].values,
                            'low': df['low'].iloc[start_idx:end_idx].values,
                            'close': df['close'].iloc[start_idx:end_idx].values,
                            'volume': df['volume'].iloc[start_idx:end_idx].values if 'volume' in df else None
                        }
                        
                        self.smart_bus.set(
                            f'ohlcv_{instrument}_{timeframe}',
                            ohlcv_data,
                            module='Environment',
                            thesis=f"OHLCV data for {instrument} {timeframe}"
                        )
    
    def _get_final_actions(self) -> np.ndarray:
        """Get final actions after module processing"""
        # Check if consensus voting produced actions
        consensus_actions = self.smart_bus.get('consensus_actions', 'Environment')
        if consensus_actions is not None:
            return consensus_actions
        
        # Check if meta-RL produced actions
        meta_actions = self.smart_bus.get('meta_rl_actions', 'Environment')
        if meta_actions is not None:
            return meta_actions
        
        # Fall back to raw actions
        return self.info_bus.get('raw_actions', np.zeros(self.action_dim))
    
    def _create_enhanced_step_info(self, executed_trades: List[Dict], 
                                  final_actions: np.ndarray) -> Dict[str, Any]:
        """Create step info with SmartInfoBus enhancements"""
        info = self._create_step_info()
        
        # Add SmartInfoBus metrics
        info['smart_bus_metrics'] = self.smart_bus.get_performance_metrics()
        
        # Add module execution summary
        info['module_execution'] = {
            'execution_order': self.orchestrator.execution_order[:10],  # First 10
            'disabled_modules': list(self.smart_bus._module_disabled),
            'avg_latencies': {
                module: np.mean(list(timings)) if timings else 0
                for module, timings in self.smart_bus._latency_history.items()
                if timings
            }
        }
        
        # Add thesis summaries if available
        theses = []
        for key in ['risk_assessment', 'trading_summary', 'strategy_selection']:
            data_with_thesis = self.smart_bus.get_with_thesis(key, 'Environment')
            if data_with_thesis:
                _, thesis = data_with_thesis
                if thesis and thesis != "No thesis provided":
                    theses.append({
                        'topic': key,
                        'summary': thesis.split('\n')[0]  # First line
                    })
        
        if theses:
            info['decision_theses'] = theses
        
        # Add dependency graph summary
        dep_graph = self.smart_bus.get_dependency_graph()
        info['module_dependencies'] = {
            'total_modules': len(dep_graph),
            'max_dependencies': max(len(deps) for deps in dep_graph.values()) if dep_graph else 0
        }
        
        return info

    # ═══════════════════════════════════════════════════════════════════
    # Hot-Reload Support
    # ═══════════════════════════════════════════════════════════════════

    def hot_reload_module(self, module_name: str) -> bool:
        """Hot-reload a module preserving state"""
        if module_name not in self.orchestrator.modules:
            self.logger.error(f"Module {module_name} not found")
            return False
        
        try:
            success = self.state_manager.reload_module(module_name, self.orchestrator)
            
            if success:
                self.logger.info(
                    format_operator_message(
                        "🔄", "MODULE_HOT_RELOADED",
                        instrument=module_name,
                        context="hot_reload"
                    )
                )
                
                # Re-enable if it was disabled
                if module_name in self.smart_bus._module_disabled:
                    self.smart_bus.reset_module_failures(module_name)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Hot-reload failed for {module_name}: {e}")
            return False
    
    def get_module_states(self) -> Dict[str, Dict[str, Any]]:
        """Get all module states for persistence"""
        states = {}
        
        for name, module in self.orchestrator.modules.items():
            try:
                states[name] = module.get_state()
            except Exception as e:
                self.logger.error(f"Failed to get state for {name}: {e}")
                states[name] = {'error': str(e)}
        
        return states
    
    def set_module_states(self, states: Dict[str, Dict[str, Any]]):
        """Restore module states"""
        restored = 0
        
        for name, state in states.items():
            if name in self.orchestrator.modules:
                try:
                    self.orchestrator.modules[name].set_state(state)
                    restored += 1
                except Exception as e:
                    self.logger.error(f"Failed to restore state for {name}: {e}")
        
        self.logger.info(f"Restored state for {restored}/{len(states)} modules")

    # ═══════════════════════════════════════════════════════════════════
    # Enhanced Monitoring & Visualization Support
    # ═══════════════════════════════════════════════════════════════════

    def get_smartinfobus_status(self) -> Dict[str, Any]:
        """Get comprehensive SmartInfoBus status"""
        return {
            'performance_metrics': self.smart_bus.get_performance_metrics(),
            'dependency_graph': self.smart_bus.get_dependency_graph(),
            'execution_order': self.orchestrator.execution_order,
            'module_health': {
                name: {
                    'enabled': self.smart_bus.is_module_enabled(name),
                    'failures': self.smart_bus._module_failures.get(name, 0),
                    'avg_latency': np.mean(list(self.smart_bus._latency_history.get(name, []))) if name in self.smart_bus._latency_history else 0
                }
                for name in self.orchestrator.modules
            },
            'data_flow': self._get_data_flow_summary()
        }
    
    def _get_data_flow_summary(self) -> Dict[str, Any]:
        """Get data flow summary from SmartInfoBus"""
        summary = {
            'total_keys': len(self.smart_bus._data_store),
            'providers_count': len(self.smart_bus._providers),
            'consumers_count': len(self.smart_bus._consumers),
            'event_log_size': len(self.smart_bus._event_log)
        }
        
        # Most active data keys
        access_counts = defaultdict(int)
        for module, accesses in self.smart_bus._access_patterns.items():
            for key, count in accesses.items():
                if key.startswith('read:'):
                    actual_key = key[5:]  # Remove 'read:' prefix
                    access_counts[actual_key] += count
        
        summary['most_accessed'] = sorted(
            access_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        return summary
    
    def generate_plain_english_report(self) -> str:
        """Generate plain English system report"""
        return self.audit_tracker.generate_plain_english_summary(self.info_bus)

    # ═══════════════════════════════════════════════════════════════════
    # Method Implementations (Enhanced for SmartInfoBus)
    # ═══════════════════════════════════════════════════════════════════

    # Most methods remain the same, using the imported implementations
    _setup_logging = _setup_logging
    _set_seeds = _set_seeds
    seed = seed
    _validate_data = _validate_data
    _initialize_modules = _initialize_modules  # Legacy method kept for compatibility
    _initialize_arbiter = _initialize_arbiter  # Legacy method kept for compatibility
    _initialize_dependent_modules = _initialize_dependent_modules  # Legacy
    _create_pipeline = _create_pipeline  # Legacy
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
    
    # Other methods remain unchanged
    _extract_current_market_data = _extract_current_market_data
    _update_regime_modules = _update_regime_modules
    _create_voting_context = _create_voting_context
    _get_legacy_observation_for_voting = _get_legacy_observation_for_voting
    _blend_timeframe_actions = _blend_timeframe_actions
    
    # Observation methods
    _sanitize_observation = _sanitize_observation
    _get_full_observation = _get_full_observation
    _get_next_observation = _get_next_observation
    _get_fallback_observation = _get_fallback_observation
    _create_info_bus_for_step = _create_info_bus_for_step
    
    # Memory methods
    _feed_memory_modules = _feed_memory_modules
    _get_current_market_context = _get_current_market_context
    _update_memory_compressor = _update_memory_compressor
    _record_episode_in_replay_analyzer = _record_episode_in_replay_analyzer
    
    # Utility methods
    _get_initial_balance = _get_initial_balance
    _select_starting_step = _select_starting_step
    _reset_all_modules = _reset_all_modules  # Legacy version kept
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
    # Backward Compatibility Methods
    # ═══════════════════════════════════════════════════════════════════

    def get_info_bus_status(self) -> Dict[str, Any]:
        """Legacy method - redirects to SmartInfoBus status"""
        return self.get_smartinfobus_status()
    
    def get_enhanced_metrics(self) -> Dict[str, Any]:
        """Legacy method - enhanced with SmartInfoBus data"""
        metrics = {
            'episode': self.episode_count,
            'step': self.market_state.current_step,
            'balance': self.market_state.balance,
            'drawdown': self.market_state.current_drawdown,
            'trades': len(self.episode_metrics.trades),
            'smartinfobus_status': self.get_smartinfobus_status(),
        }
        
        return metrics
    
    def force_info_bus_refresh(self):
        """Legacy method - creates new InfoBus"""
        self.info_bus = create_info_bus(self, step=self.market_state.current_step)
        self.last_info_bus_quality = validate_info_bus(self.info_bus)
        
        self.logger.info(
            format_operator_message(
                "🔄", "INFOBUS_REFRESHED",
                details=f"Quality: {self.last_info_bus_quality.is_valid}",
                context="debugging"
            )
        )