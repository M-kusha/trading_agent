# ─────────────────────────────────────────────────────────────
# File: modules/simulation/shadow_simulator.py
# Enhanced Shadow Simulator with InfoBus integration
# ─────────────────────────────────────────────────────────────

import numpy as np
import datetime
import copy
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import deque, defaultdict

from modules.core.core import Module, ModuleConfig, audit_step
from modules.core.mixins import AnalysisMixin, StateManagementMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor, InfoBusUpdater, extract_standard_context
from modules.utils.audit_utils import RotatingLogger, AuditTracker, format_operator_message, system_audit


class ShadowSimulator(Module, AnalysisMixin, StateManagementMixin):
    """
    Enhanced shadow simulator with InfoBus integration.
    Provides intelligent forward-looking trade simulation with context-aware
    strategy variations and comprehensive performance analysis.
    """

    # Simulation strategies
    SIMULATION_STRATEGIES = {
        "greedy": "Aggressive profit maximization",
        "conservative": "Risk-minimized approach",
        "adaptive": "Context-aware strategy",
        "contrarian": "Counter-trend positions",
        "momentum": "Trend-following strategy",
        "balanced": "Risk-reward balanced",
        "volatility_play": "Volatility exploitation",
        "regime_specific": "Market regime optimized"
    }

    # Enhanced default configuration
    ENHANCED_DEFAULTS = {
        "horizon": 5,
        "strategy": "adaptive",
        "confidence_threshold": 0.6,
        "risk_scaling": True,
        "regime_awareness": True,
        "volatility_adjustment": True,
        "session_sensitivity": 0.8,
        "learning_rate": 0.1,
        "simulation_depth": 3,
        "scenario_count": 5
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        horizon: int = 5,
        strategy: str = "adaptive",
        debug: bool = False,
        **kwargs
    ):
        # Initialize with enhanced config
        enhanced_config = ModuleConfig(
            debug=debug,
            max_history=kwargs.get('max_history', 100),
            audit_enabled=kwargs.get('audit_enabled', True),
            **kwargs
        )
        super().__init__(enhanced_config)
        
        # Initialize mixins
        self._initialize_analysis_state()
        
        # Merge configuration with enhanced defaults
        self.sim_config = copy.deepcopy(self.ENHANCED_DEFAULTS)
        if config:
            self.sim_config.update(config)
        
        # Core parameters
        self.horizon = int(horizon)
        self.strategy = strategy if strategy in self.SIMULATION_STRATEGIES else "adaptive"
        self.confidence_threshold = float(self.sim_config["confidence_threshold"])
        self.risk_scaling = bool(self.sim_config["risk_scaling"])
        self.regime_awareness = bool(self.sim_config["regime_awareness"])
        self.volatility_adjustment = bool(self.sim_config["volatility_adjustment"])
        self.session_sensitivity = float(self.sim_config["session_sensitivity"])
        self.learning_rate = float(self.sim_config["learning_rate"])
        self.simulation_depth = int(self.sim_config["simulation_depth"])
        self.scenario_count = int(self.sim_config["scenario_count"])
        
        # Enhanced state tracking
        self.simulation_history = deque(maxlen=100)
        self.scenario_results = deque(maxlen=50)
        self.strategy_performance = defaultdict(lambda: deque(maxlen=30))
        self.prediction_accuracy = deque(maxlen=50)
        
        # Market context awareness
        self.market_regime = "normal"
        self.volatility_regime = "medium"
        self.market_session = "unknown"
        self.current_volatility = 0.01
        
        # Adaptive parameters
        self.adaptive_horizon = self.horizon
        self.strategy_weights = {strategy: 1.0 for strategy in self.SIMULATION_STRATEGIES}
        self.confidence_score = 0.5
        
        # Simulation statistics
        self.simulation_stats = {
            "total_simulations": 0,
            "scenarios_evaluated": 0,
            "avg_scenario_score": 0.0,
            "prediction_accuracy": 0.0,
            "strategy_effectiveness": 0.0,
            "horizon_optimization": 0.0
        }
        
        # Performance analytics
        self.simulation_analytics = defaultdict(list)
        self.regime_performance = defaultdict(lambda: defaultdict(list))
        self.session_performance = defaultdict(lambda: defaultdict(list))
        
        # Learning and adaptation
        self.learning_history = deque(maxlen=50)
        self.strategy_evolution = deque(maxlen=20)
        self.scenario_templates = {}
        
        # Environment backup for safe simulation
        self.env_backup_state = None
        self.simulation_environments = {}
        
        # Setup enhanced logging with rotation
        self.logger = RotatingLogger(
            "ShadowSimulator",
            "logs/simulation/shadow_simulator.log",
            max_lines=2000,
            operator_mode=debug
        )
        
        # Audit system
        self.audit_tracker = AuditTracker("ShadowSimulator")
        
        self.log_operator_info(
            "🔮 Enhanced Shadow Simulator initialized",
            horizon=self.horizon,
            strategy=self.strategy,
            confidence_threshold=f"{self.confidence_threshold:.1%}",
            regime_awareness=self.regime_awareness,
            scenarios=self.scenario_count
        )

    def reset(self) -> None:
        """Enhanced reset with comprehensive state cleanup"""
        super().reset()
        self._reset_analysis_state()
        
        # Reset simulation history
        self.simulation_history.clear()
        self.scenario_results.clear()
        self.prediction_accuracy.clear()
        
        # Reset strategy performance
        for strategy in self.strategy_performance:
            self.strategy_performance[strategy].clear()
        
        # Reset market context
        self.market_regime = "normal"
        self.volatility_regime = "medium"
        self.market_session = "unknown"
        self.current_volatility = 0.01
        
        # Reset adaptive parameters
        self.adaptive_horizon = self.horizon
        self.strategy_weights = {strategy: 1.0 for strategy in self.SIMULATION_STRATEGIES}
        self.confidence_score = 0.5
        
        # Reset statistics
        self.simulation_stats = {
            "total_simulations": 0,
            "scenarios_evaluated": 0,
            "avg_scenario_score": 0.0,
            "prediction_accuracy": 0.0,
            "strategy_effectiveness": 0.0,
            "horizon_optimization": 0.0
        }
        
        # Reset analytics
        self.simulation_analytics.clear()
        self.regime_performance.clear()
        self.session_performance.clear()
        
        # Reset learning
        self.learning_history.clear()
        self.strategy_evolution.clear()
        
        # Clear environment backups
        self.env_backup_state = None
        self.simulation_environments.clear()
        
        self.log_operator_info("🔄 Shadow Simulator reset - all state cleared")

    @audit_step
    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration"""
        
        if not info_bus:
            self.log_operator_warning("No InfoBus provided - using fallback mode")
            self._process_legacy_step(**kwargs)
            return
        
        # Extract comprehensive context
        context = extract_standard_context(info_bus)
        
        # Update market context awareness
        self._update_market_context(context, info_bus)
        
        # Extract simulation data from InfoBus
        simulation_data = self._extract_simulation_data_from_info_bus(info_bus)
        
        # Perform multi-scenario simulation
        simulation_results = self._perform_multi_scenario_simulation(simulation_data, context)
        
        # Analyze simulation effectiveness
        self._analyze_simulation_effectiveness(simulation_results, context)
        
        # Update adaptive parameters
        self._update_adaptive_parameters(simulation_results, context)
        
        # Update InfoBus with results
        self._update_info_bus(info_bus, simulation_results)
        
        # Record audit for significant simulations
        self._record_simulation_audit(info_bus, context, simulation_results)
        
        # Update performance metrics
        self._update_simulation_performance_metrics()

    def _extract_simulation_data_from_info_bus(self, info_bus: InfoBus) -> Dict[str, Any]:
        """Extract simulation data from InfoBus"""
        
        data = {}
        
        try:
            # Get current market state
            prices = info_bus.get('prices', {})
            data['current_prices'] = prices
            
            # Get current positions
            positions = InfoBusExtractor.get_positions(info_bus)
            data['current_positions'] = positions
            
            # Get market features
            features = info_bus.get('features', {})
            data['market_features'] = features
            
            # Get risk snapshot
            risk_data = info_bus.get('risk', {})
            data['risk_state'] = risk_data
            
            # Get recent trading activity
            recent_trades = info_bus.get('recent_trades', [])
            data['recent_trades'] = recent_trades
            
            # Get pending orders
            pending_orders = info_bus.get('pending_orders', [])
            data['pending_orders'] = pending_orders
            
            # Get committee votes for decision context
            votes = info_bus.get('votes', [])
            data['committee_votes'] = votes
            
            # Get market context
            market_context = info_bus.get('market_context', {})
            data['market_context'] = market_context
            
            # Extract volatility
            volatilities = market_context.get('volatility', {})
            if volatilities:
                self.current_volatility = np.mean(list(volatilities.values()))
            else:
                self.current_volatility = 0.01
            
            data['current_volatility'] = self.current_volatility
            
            # Get environment reference for simulation
            data['environment'] = info_bus.get('environment')
            
        except Exception as e:
            self.log_operator_warning(f"Simulation data extraction failed: {e}")
            # Provide safe defaults
            data = {
                'current_prices': {},
                'current_positions': [],
                'market_features': {},
                'risk_state': {},
                'recent_trades': [],
                'pending_orders': [],
                'committee_votes': [],
                'market_context': {},
                'current_volatility': 0.01,
                'environment': None
            }
        
        return data

    def _update_market_context(self, context: Dict[str, Any], info_bus: InfoBus) -> None:
        """Update market context awareness"""
        
        try:
            # Update regime tracking
            old_regime = self.market_regime
            self.market_regime = context.get('regime', 'unknown')
            self.volatility_regime = context.get('volatility_level', 'medium')
            self.market_session = context.get('session', 'unknown')
            
            # Adapt simulation parameters for regime change
            if self.market_regime != old_regime and self.regime_awareness:
                self._adapt_simulation_for_regime(old_regime, self.market_regime)
                
                self.log_operator_info(
                    f"📊 Regime change detected: {old_regime} → {self.market_regime}",
                    adaptation="Simulation parameters updated",
                    session=self.market_session
                )
            
        except Exception as e:
            self.log_operator_warning(f"Market context update failed: {e}")

    def _adapt_simulation_for_regime(self, old_regime: str, new_regime: str) -> None:
        """Adapt simulation parameters for regime change"""
        
        try:
            # Adjust horizon based on regime
            regime_horizon_multipliers = {
                'volatile': 0.7,  # Shorter horizon in volatile markets
                'trending': 1.3,  # Longer horizon in trending markets
                'ranging': 1.0,   # Normal horizon in ranging markets
                'unknown': 1.0
            }
            
            multiplier = regime_horizon_multipliers.get(new_regime, 1.0)
            self.adaptive_horizon = int(self.horizon * multiplier)
            self.adaptive_horizon = max(1, min(self.adaptive_horizon, 20))  # Bounds
            
            # Adjust strategy weights
            if new_regime == 'volatile':
                self.strategy_weights['conservative'] *= 1.5
                self.strategy_weights['greedy'] *= 0.7
            elif new_regime == 'trending':
                self.strategy_weights['momentum'] *= 1.4
                self.strategy_weights['contrarian'] *= 0.8
            elif new_regime == 'ranging':
                self.strategy_weights['contrarian'] *= 1.3
                self.strategy_weights['momentum'] *= 0.8
            
            # Normalize weights
            total_weight = sum(self.strategy_weights.values())
            if total_weight > 0:
                for strategy in self.strategy_weights:
                    self.strategy_weights[strategy] /= total_weight
            
        except Exception as e:
            self.log_operator_warning(f"Regime adaptation failed: {e}")

    def _perform_multi_scenario_simulation(self, simulation_data: Dict[str, Any], 
                                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive multi-scenario simulation"""
        
        results = {
            'scenarios': [],
            'best_scenario': None,
            'worst_scenario': None,
            'avg_score': 0.0,
            'confidence': 0.0,
            'recommendations': [],
            'context': context.copy()
        }
        
        try:
            # Get environment for simulation
            env = simulation_data.get('environment')
            if env is None:
                self.log_operator_warning("No environment available for simulation")
                return results
            
            # Backup environment state
            self._backup_environment_state(env)
            
            # Generate scenarios
            scenarios = self._generate_simulation_scenarios(simulation_data, context)
            
            # Simulate each scenario
            scenario_results = []
            for i, scenario in enumerate(scenarios):
                try:
                    scenario_result = self._simulate_scenario(env, scenario, simulation_data, context)
                    scenario_results.append(scenario_result)
                    
                except Exception as e:
                    self.log_operator_warning(f"Scenario {i} simulation failed: {e}")
                    continue
                finally:
                    # Always restore environment
                    self._restore_environment_state(env)
            
            # Analyze results
            if scenario_results:
                results['scenarios'] = scenario_results
                results['best_scenario'] = max(scenario_results, key=lambda s: s.get('score', 0))
                results['worst_scenario'] = min(scenario_results, key=lambda s: s.get('score', 0))
                results['avg_score'] = np.mean([s.get('score', 0) for s in scenario_results])
                results['confidence'] = self._calculate_simulation_confidence(scenario_results)
                results['recommendations'] = self._generate_simulation_recommendations(scenario_results, context)
            
            # Update statistics
            self.simulation_stats['total_simulations'] += 1
            self.simulation_stats['scenarios_evaluated'] += len(scenario_results)
            self.simulation_stats['avg_scenario_score'] = results['avg_score']
            
            # Store results
            self.scenario_results.append(results)
            
        except Exception as e:
            self.log_operator_error(f"Multi-scenario simulation failed: {e}")
            results['error'] = str(e)
        
        return results

    def _backup_environment_state(self, env) -> None:
        """Backup environment state for safe simulation"""
        
        try:
            if hasattr(env, 'get_state'):
                self.env_backup_state = env.get_state()
            else:
                # Fallback: backup key attributes
                self.env_backup_state = {
                    'current_step': getattr(env, 'current_step', 0),
                    'balance': getattr(env, 'balance', 10000),
                    'positions': copy.deepcopy(getattr(env, 'positions', {})),
                    'market_state': copy.deepcopy(getattr(env, 'market_state', {}))
                }
        except Exception as e:
            self.log_operator_warning(f"Environment backup failed: {e}")
            self.env_backup_state = None

    def _restore_environment_state(self, env) -> None:
        """Restore environment state after simulation"""
        
        try:
            if self.env_backup_state is None:
                return
            
            if hasattr(env, 'set_state'):
                env.set_state(self.env_backup_state)
            else:
                # Fallback: restore key attributes
                for attr, value in self.env_backup_state.items():
                    if hasattr(env, attr):
                        setattr(env, attr, value)
        except Exception as e:
            self.log_operator_warning(f"Environment restore failed: {e}")

    def _generate_simulation_scenarios(self, simulation_data: Dict[str, Any], 
                                      context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate diverse simulation scenarios"""
        
        scenarios = []
        
        try:
            # Get current committee votes for action guidance
            votes = simulation_data.get('committee_votes', [])
            base_actions = self._extract_base_actions_from_votes(votes)
            
            # Generate scenarios for each strategy
            strategies_to_test = list(self.SIMULATION_STRATEGIES.keys())
            if len(strategies_to_test) > self.scenario_count:
                # Select top strategies by weight
                strategies_to_test = sorted(
                    strategies_to_test,
                    key=lambda s: self.strategy_weights.get(s, 0),
                    reverse=True
                )[:self.scenario_count]
            
            for strategy in strategies_to_test:
                scenario = {
                    'strategy': strategy,
                    'description': self.SIMULATION_STRATEGIES[strategy],
                    'base_actions': base_actions,
                    'modified_actions': self._modify_actions_for_strategy(base_actions, strategy, context),
                    'horizon': self._get_strategy_horizon(strategy, context),
                    'risk_scaling': self._get_strategy_risk_scaling(strategy, context),
                    'context_adjustments': self._get_strategy_context_adjustments(strategy, context)
                }
                scenarios.append(scenario)
            
            # Add adaptive scenario if not already included
            if 'adaptive' not in strategies_to_test:
                adaptive_scenario = {
                    'strategy': 'adaptive',
                    'description': 'Context-adaptive strategy',
                    'base_actions': base_actions,
                    'modified_actions': self._create_adaptive_actions(base_actions, context),
                    'horizon': self.adaptive_horizon,
                    'risk_scaling': True,
                    'context_adjustments': self._get_adaptive_context_adjustments(context)
                }
                scenarios.append(adaptive_scenario)
            
        except Exception as e:
            self.log_operator_warning(f"Scenario generation failed: {e}")
            # Fallback: create basic scenario
            scenarios = [{
                'strategy': 'balanced',
                'description': 'Basic balanced strategy',
                'base_actions': {},
                'modified_actions': {},
                'horizon': self.horizon,
                'risk_scaling': True,
                'context_adjustments': {}
            }]
        
        return scenarios

    def _extract_base_actions_from_votes(self, votes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract base actions from committee votes"""
        
        try:
            if not votes:
                return {}
            
            # Aggregate votes by action
            action_sums = defaultdict(float)
            confidence_sums = defaultdict(float)
            vote_counts = defaultdict(int)
            
            for vote in votes:
                action = vote.get('action', [])
                confidence = vote.get('confidence', 0.5)
                
                if isinstance(action, (list, np.ndarray)) and len(action) > 0:
                    for i, act in enumerate(action):
                        action_sums[i] += float(act) * confidence
                        confidence_sums[i] += confidence
                        vote_counts[i] += 1
                elif isinstance(action, (int, float)):
                    action_sums[0] += float(action) * confidence
                    confidence_sums[0] += confidence
                    vote_counts[0] += 1
            
            # Calculate weighted averages
            base_actions = {}
            for i in action_sums:
                if vote_counts[i] > 0:
                    base_actions[i] = action_sums[i] / confidence_sums[i] if confidence_sums[i] > 0 else 0.0
            
            return base_actions
            
        except Exception as e:
            self.log_operator_warning(f"Base action extraction failed: {e}")
            return {}

    def _modify_actions_for_strategy(self, base_actions: Dict[str, Any], 
                                    strategy: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Modify actions based on strategy"""
        
        try:
            modified_actions = copy.deepcopy(base_actions)
            
            # Strategy-specific modifications
            if strategy == "greedy":
                # Amplify existing actions
                for key in modified_actions:
                    modified_actions[key] *= 1.5
            
            elif strategy == "conservative":
                # Reduce action magnitude
                for key in modified_actions:
                    modified_actions[key] *= 0.6
            
            elif strategy == "contrarian":
                # Reverse actions
                for key in modified_actions:
                    modified_actions[key] *= -0.8
            
            elif strategy == "momentum":
                # Amplify in trend direction
                trend_multiplier = 1.3 if context.get('regime') == 'trending' else 1.0
                for key in modified_actions:
                    modified_actions[key] *= trend_multiplier
            
            elif strategy == "balanced":
                # Moderate actions
                for key in modified_actions:
                    modified_actions[key] *= 0.8
            
            elif strategy == "volatility_play":
                # Adjust for volatility
                vol_multiplier = 1.5 if context.get('volatility_level') in ['high', 'extreme'] else 0.7
                for key in modified_actions:
                    modified_actions[key] *= vol_multiplier
            
            elif strategy == "regime_specific":
                # Regime-specific adjustments
                regime = context.get('regime', 'unknown')
                regime_multipliers = {
                    'volatile': 0.7,
                    'trending': 1.2,
                    'ranging': 0.9,
                    'unknown': 1.0
                }
                multiplier = regime_multipliers.get(regime, 1.0)
                for key in modified_actions:
                    modified_actions[key] *= multiplier
            
            # Apply bounds
            for key in modified_actions:
                modified_actions[key] = np.clip(modified_actions[key], -1.0, 1.0)
            
            return modified_actions
            
        except Exception as e:
            self.log_operator_warning(f"Action modification failed: {e}")
            return base_actions

    def _create_adaptive_actions(self, base_actions: Dict[str, Any], 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Create adaptive actions based on context"""
        
        try:
            adaptive_actions = copy.deepcopy(base_actions)
            
            # Adapt based on market regime
            regime = context.get('regime', 'unknown')
            vol_level = context.get('volatility_level', 'medium')
            session = context.get('session', 'unknown')
            
            # Regime adaptation
            if regime == 'volatile':
                for key in adaptive_actions:
                    adaptive_actions[key] *= 0.75  # More conservative
            elif regime == 'trending':
                for key in adaptive_actions:
                    if adaptive_actions[key] > 0:
                        adaptive_actions[key] *= 1.2  # Amplify trend direction
            
            # Volatility adaptation
            vol_multipliers = {
                'low': 1.1,
                'medium': 1.0,
                'high': 0.8,
                'extreme': 0.6
            }
            vol_mult = vol_multipliers.get(vol_level, 1.0)
            for key in adaptive_actions:
                adaptive_actions[key] *= vol_mult
            
            # Session adaptation
            if self.session_sensitivity > 0:
                session_multipliers = {
                    'asian': 0.9,
                    'european': 1.0,
                    'american': 1.1,
                    'rollover': 0.7
                }
                session_mult = session_multipliers.get(session, 1.0)
                for key in adaptive_actions:
                    adaptive_actions[key] *= session_mult * self.session_sensitivity + (1 - self.session_sensitivity)
            
            # Apply bounds
            for key in adaptive_actions:
                adaptive_actions[key] = np.clip(adaptive_actions[key], -1.0, 1.0)
            
            return adaptive_actions
            
        except Exception as e:
            self.log_operator_warning(f"Adaptive action creation failed: {e}")
            return base_actions

    def _get_strategy_horizon(self, strategy: str, context: Dict[str, Any]) -> int:
        """Get strategy-specific horizon"""
        
        base_horizon = self.adaptive_horizon
        
        strategy_multipliers = {
            'greedy': 0.8,          # Shorter horizon for quick gains
            'conservative': 1.2,     # Longer horizon for stability
            'momentum': 1.3,         # Longer horizon for trend capture
            'contrarian': 0.9,       # Shorter horizon for quick reversals
            'balanced': 1.0,         # Standard horizon
            'volatility_play': 0.7,  # Short horizon for vol trades
            'regime_specific': 1.0,  # Context dependent
            'adaptive': 1.0          # Context dependent
        }
        
        multiplier = strategy_multipliers.get(strategy, 1.0)
        
        # Additional context adjustments
        if context.get('volatility_level') == 'extreme':
            multiplier *= 0.8  # Shorter horizon in extreme volatility
        elif context.get('regime') == 'trending':
            multiplier *= 1.1  # Slightly longer in trends
        
        horizon = int(base_horizon * multiplier)
        return max(1, min(horizon, 15))  # Bounds

    def _get_strategy_risk_scaling(self, strategy: str, context: Dict[str, Any]) -> bool:
        """Get strategy-specific risk scaling setting"""
        
        risk_scaling_strategies = {
            'greedy': False,         # No risk scaling for aggressive approach
            'conservative': True,    # Always use risk scaling
            'momentum': True,        # Risk scaling for trend following
            'contrarian': True,      # Risk scaling for counter-trend
            'balanced': True,        # Balanced approach uses scaling
            'volatility_play': False, # No scaling for vol strategies
            'regime_specific': True, # Context dependent
            'adaptive': True         # Context dependent
        }
        
        return risk_scaling_strategies.get(strategy, self.risk_scaling)

    def _get_strategy_context_adjustments(self, strategy: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get strategy-specific context adjustments"""
        
        adjustments = {
            'strategy': strategy,
            'regime_sensitivity': 1.0,
            'volatility_sensitivity': 1.0,
            'session_sensitivity': 1.0
        }
        
        # Strategy-specific sensitivities
        if strategy == 'conservative':
            adjustments['volatility_sensitivity'] = 1.5  # More sensitive to volatility
        elif strategy == 'momentum':
            adjustments['regime_sensitivity'] = 1.3      # More sensitive to regime
        elif strategy == 'contrarian':
            adjustments['regime_sensitivity'] = 0.8      # Less sensitive to regime
        elif strategy == 'volatility_play':
            adjustments['volatility_sensitivity'] = 2.0  # Highly sensitive to volatility
        
        return adjustments

    def _get_adaptive_context_adjustments(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get adaptive context adjustments"""
        
        return {
            'strategy': 'adaptive',
            'regime_sensitivity': self.regime_awareness * 1.0,
            'volatility_sensitivity': self.volatility_adjustment * 1.0,
            'session_sensitivity': self.session_sensitivity,
            'confidence_threshold': self.confidence_threshold,
            'learning_rate': self.learning_rate
        }

    def _simulate_scenario(self, env, scenario: Dict[str, Any], 
                          simulation_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate a single scenario"""
        
        result = {
            'scenario': scenario.copy(),
            'trades': [],
            'final_balance': 0.0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'trade_count': 0,
            'score': 0.0,
            'success_rate': 0.0,
            'risk_metrics': {},
            'timeline': []
        }
        
        try:
            strategy = scenario['strategy']
            modified_actions = scenario['modified_actions']
            horizon = scenario['horizon']
            
            initial_balance = getattr(env, 'balance', 10000)
            trades = []
            timeline = []
            
            # Simulate forward steps
            for step in range(horizon):
                # Check if we can continue
                if not self._can_continue_simulation(env):
                    break
                
                step_result = {
                    'step': step,
                    'timestamp': datetime.datetime.now().isoformat(),
                    'balance': getattr(env, 'balance', initial_balance),
                    'actions_taken': [],
                    'trades_executed': []
                }
                
                # Execute actions based on strategy
                step_trades = self._execute_scenario_step(env, modified_actions, strategy, context)
                
                if step_trades:
                    trades.extend(step_trades)
                    step_result['trades_executed'] = step_trades
                
                timeline.append(step_result)
                
                # Advance environment
                if hasattr(env, 'step'):
                    try:
                        env.step()
                    except Exception as e:
                        self.log_operator_warning(f"Environment step failed: {e}")
                        break
            
            # Calculate results
            final_balance = getattr(env, 'balance', initial_balance)
            total_pnl = final_balance - initial_balance
            
            # Calculate performance metrics
            result.update({
                'trades': trades,
                'final_balance': final_balance,
                'total_pnl': total_pnl,
                'trade_count': len(trades),
                'timeline': timeline
            })
            
            # Calculate score
            result['score'] = self._calculate_scenario_score(result, scenario, context)
            
            # Calculate success rate
            if trades:
                successful_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
                result['success_rate'] = successful_trades / len(trades)
            else:
                result['success_rate'] = 0.0
            
            # Calculate risk metrics
            result['risk_metrics'] = self._calculate_scenario_risk_metrics(result, initial_balance)
            
        except Exception as e:
            self.log_operator_warning(f"Scenario simulation failed: {e}")
            result['error'] = str(e)
            result['score'] = 0.0
        
        return result

    def _can_continue_simulation(self, env) -> bool:
        """Check if simulation can continue"""
        
        try:
            # Check if environment has data left
            if hasattr(env, 'current_step') and hasattr(env, 'data'):
                for instrument_data in env.data.values():
                    for timeframe_data in instrument_data.values():
                        if env.current_step >= len(timeframe_data):
                            return False
            
            # Check balance
            balance = getattr(env, 'balance', 10000)
            if balance <= 0:
                return False
            
            return True
            
        except Exception:
            return False

    def _execute_scenario_step(self, env, modified_actions: Dict[str, Any], 
                              strategy: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute one step of scenario simulation"""
        
        trades = []
        
        try:
            # Get instruments from environment
            instruments = getattr(env, 'instruments', ['EUR/USD', 'XAU/USD'])
            
            # Execute trades based on modified actions
            for i, instrument in enumerate(instruments):
                if i * 2 in modified_actions and (i * 2 + 1) in modified_actions:
                    action_0 = modified_actions[i * 2]
                    action_1 = modified_actions[i * 2 + 1]
                    
                    # Execute trade if action is significant
                    if abs(action_0) > 0.1 or abs(action_1) > 0.1:
                        trade = self._execute_simulated_trade(env, instrument, action_0, action_1, strategy)
                        if trade:
                            trades.append(trade)
            
        except Exception as e:
            self.log_operator_warning(f"Scenario step execution failed: {e}")
        
        return trades

    def _execute_simulated_trade(self, env, instrument: str, action_0: float, 
                                action_1: float, strategy: str) -> Optional[Dict[str, Any]]:
        """Execute a simulated trade"""
        
        try:
            # Use environment's trade execution if available
            if hasattr(env, '_execute_trade'):
                trade = env._execute_trade(instrument, action_0, action_1)
                if trade:
                    trade['simulation_strategy'] = strategy
                    trade['simulation_timestamp'] = datetime.datetime.now().isoformat()
                return trade
            
            # Fallback: create synthetic trade
            current_price = self._get_current_price(env, instrument)
            if current_price is None:
                return None
            
            # Determine position size (simplified)
            position_size = abs(action_0) * 0.1  # 10% of action magnitude
            side = 'BUY' if action_0 > 0 else 'SELL'
            
            # Simulate basic P&L
            estimated_pnl = action_1 * position_size * 100  # Simplified
            
            return {
                'instrument': instrument,
                'side': side,
                'size': position_size,
                'price': current_price,
                'pnl': estimated_pnl,
                'timestamp': datetime.datetime.now().isoformat(),
                'simulation_strategy': strategy,
                'action_0': action_0,
                'action_1': action_1
            }
            
        except Exception as e:
            self.log_operator_warning(f"Simulated trade execution failed: {e}")
            return None

    def _get_current_price(self, env, instrument: str) -> Optional[float]:
        """Get current price for instrument"""
        
        try:
            # Try to get from environment data
            if hasattr(env, 'data') and instrument in env.data:
                timeframe_data = env.data[instrument]
                if 'D1' in timeframe_data:
                    df = timeframe_data['D1']
                    current_step = getattr(env, 'current_step', 0)
                    if current_step < len(df):
                        return float(df.iloc[current_step]['close'])
            
            # Fallback prices
            fallback_prices = {
                'EUR/USD': 1.0850,
                'XAU/USD': 2000.0,
                'GBP/USD': 1.2650,
                'USD/JPY': 150.0
            }
            
            return fallback_prices.get(instrument, 1.0)
            
        except Exception:
            return 1.0

    def _calculate_scenario_score(self, result: Dict[str, Any], 
                                 scenario: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate scenario performance score"""
        
        try:
            # Base score from PnL
            total_pnl = result.get('total_pnl', 0.0)
            pnl_score = np.tanh(total_pnl / 1000.0)  # Normalize PnL
            
            # Success rate component
            success_rate = result.get('success_rate', 0.0)
            
            # Risk-adjusted score
            risk_metrics = result.get('risk_metrics', {})
            max_drawdown = risk_metrics.get('max_drawdown', 0.0)
            risk_penalty = abs(max_drawdown) * 2.0  # Penalize drawdown
            
            # Trade frequency component
            trade_count = result.get('trade_count', 0)
            horizon = scenario.get('horizon', 1)
            trade_frequency = trade_count / max(horizon, 1)
            
            # Optimal frequency is around 0.5-1.0 trades per step
            if trade_frequency < 0.5:
                frequency_score = trade_frequency * 2.0
            elif trade_frequency <= 1.0:
                frequency_score = 1.0
            else:
                frequency_score = max(0.0, 2.0 - trade_frequency)
            
            # Strategy-specific adjustments
            strategy = scenario.get('strategy', 'balanced')
            strategy_bonus = self.strategy_weights.get(strategy, 1.0)
            
            # Combined score
            score = (
                pnl_score * 0.4 +
                success_rate * 0.3 +
                frequency_score * 0.2 +
                strategy_bonus * 0.1 -
                risk_penalty
            )
            
            # Apply bounds
            return float(np.clip(score, -1.0, 1.0))
            
        except Exception as e:
            self.log_operator_warning(f"Scenario scoring failed: {e}")
            return 0.0

    def _calculate_scenario_risk_metrics(self, result: Dict[str, Any], 
                                        initial_balance: float) -> Dict[str, Any]:
        """Calculate risk metrics for scenario"""
        
        try:
            timeline = result.get('timeline', [])
            
            if not timeline:
                return {'max_drawdown': 0.0, 'volatility': 0.0, 'sharpe': 0.0}
            
            # Extract balance progression
            balances = [step.get('balance', initial_balance) for step in timeline]
            balances.insert(0, initial_balance)  # Add initial balance
            
            # Calculate drawdown
            running_max = np.maximum.accumulate(balances)
            drawdown = (np.array(balances) - running_max) / running_max
            max_drawdown = float(np.min(drawdown))
            
            # Calculate returns
            returns = np.diff(balances) / balances[:-1]
            
            # Calculate volatility
            volatility = float(np.std(returns)) if len(returns) > 1 else 0.0
            
            # Calculate Sharpe ratio
            if volatility > 0:
                sharpe = float(np.mean(returns) / volatility * np.sqrt(252))
            else:
                sharpe = 0.0
            
            return {
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'sharpe': sharpe,
                'final_balance': float(balances[-1]),
                'total_return': float((balances[-1] - initial_balance) / initial_balance)
            }
            
        except Exception as e:
            self.log_operator_warning(f"Risk metrics calculation failed: {e}")
            return {'max_drawdown': 0.0, 'volatility': 0.0, 'sharpe': 0.0}

    def _calculate_simulation_confidence(self, scenario_results: List[Dict[str, Any]]) -> float:
        """Calculate confidence in simulation results"""
        
        try:
            if not scenario_results:
                return 0.0
            
            scores = [result.get('score', 0) for result in scenario_results]
            
            # Confidence based on score consistency and magnitude
            score_mean = np.mean(scores)
            score_std = np.std(scores)
            
            # Higher confidence if scores are consistently good and low variance
            consistency_factor = max(0.0, 1.0 - score_std * 2.0)
            magnitude_factor = (1.0 + np.tanh(score_mean)) / 2.0  # 0 to 1
            
            confidence = (consistency_factor + magnitude_factor) / 2.0
            
            # Boost confidence if multiple scenarios agree
            positive_scenarios = sum(1 for score in scores if score > 0)
            agreement_factor = positive_scenarios / len(scores)
            
            final_confidence = confidence * 0.7 + agreement_factor * 0.3
            
            return float(np.clip(final_confidence, 0.0, 1.0))
            
        except Exception as e:
            self.log_operator_warning(f"Confidence calculation failed: {e}")
            return 0.5

    def _generate_simulation_recommendations(self, scenario_results: List[Dict[str, Any]], 
                                           context: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on simulation results"""
        
        recommendations = []
        
        try:
            if not scenario_results:
                recommendations.append("⚠️ No simulation results available for recommendations.")
                return recommendations
            
            # Find best performing scenario
            best_scenario = max(scenario_results, key=lambda s: s.get('score', 0))
            best_strategy = best_scenario['scenario']['strategy']
            best_score = best_scenario.get('score', 0)
            
            # Overall recommendation
            if best_score > 0.5:
                recommendations.append(f"✅ Strong signal: {best_strategy} strategy shows {best_score:.1%} confidence")
            elif best_score > 0.2:
                recommendations.append(f"⚡ Moderate signal: {best_strategy} strategy shows potential")
            elif best_score > 0:
                recommendations.append(f"⚠️ Weak signal: Consider {best_strategy} strategy with caution")
            else:
                recommendations.append("🚫 No positive scenarios identified - consider staying neutral")
            
            # Risk recommendations
            best_risk = best_scenario.get('risk_metrics', {})
            max_dd = best_risk.get('max_drawdown', 0)
            
            if abs(max_dd) > 0.1:
                recommendations.append(f"⚠️ High risk detected: {abs(max_dd):.1%} drawdown expected")
            elif abs(max_dd) > 0.05:
                recommendations.append(f"⚡ Moderate risk: {abs(max_dd):.1%} drawdown possible")
            
            # Context-specific recommendations
            regime = context.get('regime', 'unknown')
            vol_level = context.get('volatility_level', 'medium')
            
            if regime == 'volatile' and best_strategy in ['greedy', 'momentum']:
                recommendations.append("💥 Volatile market: Consider more conservative approach despite signal")
            elif regime == 'trending' and best_strategy == 'contrarian':
                recommendations.append("📈 Trending market: Contrarian signal may be premature")
            
            if vol_level == 'extreme':
                recommendations.append("🌪️ Extreme volatility: Reduce position sizes regardless of strategy")
            
            # Comparative recommendations
            strategy_scores = {}
            for result in scenario_results:
                strategy = result['scenario']['strategy']
                score = result.get('score', 0)
                strategy_scores[strategy] = score
            
            top_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            
            if len(top_strategies) > 1:
                top_names = [strategy.replace('_', ' ').title() for strategy, _ in top_strategies]
                recommendations.append(f"📊 Top strategies: {', '.join(top_names)}")
            
        except Exception as e:
            self.log_operator_warning(f"Recommendation generation failed: {e}")
            recommendations.append("⚠️ Unable to generate specific recommendations")
        
        return recommendations[:5]  # Limit to top 5

    def _analyze_simulation_effectiveness(self, simulation_results: Dict[str, Any], 
                                        context: Dict[str, Any]) -> None:
        """Analyze simulation effectiveness"""
        
        try:
            scenarios = simulation_results.get('scenarios', [])
            avg_score = simulation_results.get('avg_score', 0.0)
            confidence = simulation_results.get('confidence', 0.0)
            
            # Update effectiveness tracking
            self.simulation_stats['prediction_accuracy'] = confidence
            self.simulation_stats['strategy_effectiveness'] = avg_score
            
            # Track strategy performance
            for scenario in scenarios:
                strategy = scenario['scenario']['strategy']
                score = scenario.get('score', 0)
                self.strategy_performance[strategy].append(score)
            
            # Update regime and session performance
            regime = context.get('regime', 'unknown')
            session = context.get('session', 'unknown')
            
            if regime != 'unknown':
                self.regime_performance[regime]['avg_scores'].append(avg_score)
                self.regime_performance[regime]['confidences'].append(confidence)
            
            if session != 'unknown':
                self.session_performance[session]['avg_scores'].append(avg_score)
                self.session_performance[session]['confidences'].append(confidence)
            
        except Exception as e:
            self.log_operator_warning(f"Effectiveness analysis failed: {e}")

    def _update_adaptive_parameters(self, simulation_results: Dict[str, Any], 
                                   context: Dict[str, Any]) -> None:
        """Update adaptive parameters based on simulation results"""
        
        try:
            # Update strategy weights based on performance
            scenarios = simulation_results.get('scenarios', [])
            
            for scenario in scenarios:
                strategy = scenario['scenario']['strategy']
                score = scenario.get('score', 0)
                
                # Update weight with learning rate
                current_weight = self.strategy_weights.get(strategy, 1.0)
                performance_factor = (score + 1.0) / 2.0  # Convert from [-1,1] to [0,1]
                
                new_weight = (
                    current_weight * (1 - self.learning_rate) +
                    performance_factor * self.learning_rate
                )
                
                self.strategy_weights[strategy] = new_weight
            
            # Normalize weights
            total_weight = sum(self.strategy_weights.values())
            if total_weight > 0:
                for strategy in self.strategy_weights:
                    self.strategy_weights[strategy] /= total_weight
            
            # Update confidence score
            new_confidence = simulation_results.get('confidence', 0.5)
            self.confidence_score = (
                self.confidence_score * (1 - self.learning_rate) +
                new_confidence * self.learning_rate
            )
            
            # Store learning data
            self.learning_history.append({
                'timestamp': datetime.datetime.now().isoformat(),
                'avg_score': simulation_results.get('avg_score', 0.0),
                'confidence': new_confidence,
                'strategy_weights': self.strategy_weights.copy(),
                'context': context.copy()
            })
            
        except Exception as e:
            self.log_operator_warning(f"Adaptive parameter update failed: {e}")

    def _update_info_bus(self, info_bus: InfoBus, simulation_results: Dict[str, Any]) -> None:
        """Update InfoBus with simulation results"""
        
        # Add module data
        InfoBusUpdater.add_module_data(info_bus, 'shadow_simulator', {
            'simulation_strategy': self.strategy,
            'horizon': self.horizon,
            'adaptive_horizon': self.adaptive_horizon,
            'confidence_score': self.confidence_score,
            'simulation_stats': self.simulation_stats.copy(),
            'strategy_weights': self.strategy_weights.copy(),
            'simulation_results': {
                'avg_score': simulation_results.get('avg_score', 0.0),
                'confidence': simulation_results.get('confidence', 0.0),
                'scenarios_evaluated': len(simulation_results.get('scenarios', [])),
                'best_strategy': simulation_results.get('best_scenario', {}).get('scenario', {}).get('strategy', 'unknown')
            },
            'recommendations': simulation_results.get('recommendations', []),
            'market_context': {
                'regime': self.market_regime,
                'volatility_regime': self.volatility_regime,
                'session': self.market_session
            }
        })
        
        # Add simulation predictions to InfoBus
        best_scenario = simulation_results.get('best_scenario', {})
        if best_scenario:
            if 'predictions' not in info_bus:
                info_bus['predictions'] = {}
            
            info_bus['predictions']['shadow_simulation'] = {
                'strategy': best_scenario.get('scenario', {}).get('strategy', 'unknown'),
                'expected_pnl': best_scenario.get('total_pnl', 0.0),
                'confidence': simulation_results.get('confidence', 0.0),
                'trade_count_estimate': best_scenario.get('trade_count', 0),
                'risk_estimate': best_scenario.get('risk_metrics', {}).get('max_drawdown', 0.0)
            }
        
        # Add alerts for significant simulation results
        confidence = simulation_results.get('confidence', 0.0)
        avg_score = simulation_results.get('avg_score', 0.0)
        
        if confidence > 0.8 and avg_score > 0.5:
            InfoBusUpdater.add_alert(
                info_bus,
                f"High confidence simulation: {confidence:.1%} confidence, {avg_score:.1%} score",
                severity="info",
                module="ShadowSimulator"
            )
        elif confidence < 0.3:
            InfoBusUpdater.add_alert(
                info_bus,
                f"Low simulation confidence: {confidence:.1%} - consider caution",
                severity="warning",
                module="ShadowSimulator"
            )

    def _record_simulation_audit(self, info_bus: InfoBus, context: Dict[str, Any], 
                                simulation_results: Dict[str, Any]) -> None:
        """Record comprehensive audit trail"""
        
        # Only audit when simulations are performed or periodically
        should_audit = (
            len(simulation_results.get('scenarios', [])) > 0 or
            info_bus.get('step_idx', 0) % 50 == 0
        )
        
        if should_audit:
            audit_data = {
                'simulation_config': {
                    'strategy': self.strategy,
                    'horizon': self.horizon,
                    'adaptive_horizon': self.adaptive_horizon,
                    'scenario_count': self.scenario_count,
                    'confidence_threshold': self.confidence_threshold
                },
                'simulation_results': {
                    'scenarios_evaluated': len(simulation_results.get('scenarios', [])),
                    'avg_score': simulation_results.get('avg_score', 0.0),
                    'confidence': simulation_results.get('confidence', 0.0),
                    'best_strategy': simulation_results.get('best_scenario', {}).get('scenario', {}).get('strategy', 'unknown')
                },
                'adaptive_parameters': {
                    'strategy_weights': self.strategy_weights.copy(),
                    'confidence_score': self.confidence_score,
                    'learning_rate': self.learning_rate
                },
                'context': context.copy(),
                'statistics': self.simulation_stats.copy(),
                'recommendations': simulation_results.get('recommendations', [])
            }
            
            confidence = simulation_results.get('confidence', 0.0)
            severity = "info" if confidence > 0.6 else "warning" if confidence < 0.3 else "info"
            
            self.audit_tracker.record_event(
                event_type="shadow_simulation",
                module="ShadowSimulator",
                details=audit_data,
                severity=severity
            )

    def _update_simulation_performance_metrics(self) -> None:
        """Update performance metrics"""
        
        # Update performance metrics
        self._update_performance_metric('total_simulations', self.simulation_stats['total_simulations'])
        self._update_performance_metric('scenarios_evaluated', self.simulation_stats['scenarios_evaluated'])
        self._update_performance_metric('prediction_accuracy', self.simulation_stats['prediction_accuracy'])
        self._update_performance_metric('strategy_effectiveness', self.simulation_stats['strategy_effectiveness'])
        self._update_performance_metric('confidence_score', self.confidence_score)
        self._update_performance_metric('adaptive_horizon', self.adaptive_horizon)

    def _process_legacy_step(self, **kwargs) -> None:
        """Process legacy step parameters for backward compatibility"""
        
        try:
            # Legacy simulation if environment and actions provided
            env = kwargs.get('env')
            actions = kwargs.get('actions')
            
            if env is not None and actions is not None:
                # Perform legacy simulation
                simulated_trades = self.simulate_legacy(env, actions)
                self.simulation_stats['total_simulations'] += 1
                
        except Exception as e:
            self.log_operator_error(f"Legacy step processing failed: {e}")

    # ================== PUBLIC INTERFACE METHODS ==================

    def simulate_legacy(self, env, actions: np.ndarray) -> List[Dict[str, Any]]:
        """Legacy simulation interface for backward compatibility"""
        
        try:
            # Backup environment
            self._backup_environment_state(env)
            
            trades = []
            
            # Simple forward simulation
            for step in range(self.horizon):
                if not self._can_continue_simulation(env):
                    break
                
                # Execute actions based on strategy
                instruments = getattr(env, 'instruments', ['EUR/USD', 'XAU/USD'])
                
                for i, instrument in enumerate(instruments):
                    if len(actions) > i * 2 + 1:
                        action_0 = actions[i * 2]
                        action_1 = actions[i * 2 + 1]
                        
                        # Apply strategy modification
                        if self.strategy == "greedy":
                            action_0 *= 1.5
                            action_1 *= 1.5
                        elif self.strategy == "conservative":
                            action_0 *= 0.6
                            action_1 *= 0.6
                        elif self.strategy == "random":
                            action_0 = np.random.uniform(-1, 1)
                            action_1 = np.random.uniform(-1, 1)
                        
                        # Execute trade
                        trade = self._execute_simulated_trade(env, instrument, action_0, action_1, self.strategy)
                        if trade:
                            trades.append(trade)
                
                # Advance environment
                if hasattr(env, 'step'):
                    env.step()
            
            # Restore environment
            self._restore_environment_state(env)
            
            return trades
            
        except Exception as e:
            self.log_operator_error(f"Legacy simulation failed: {e}")
            return []

    def get_observation_components(self) -> np.ndarray:
        """Return simulation features for observation"""
        
        try:
            strategy_idx = float(list(self.SIMULATION_STRATEGIES.keys()).index(self.strategy))
            recent_score = self.simulation_stats.get('strategy_effectiveness', 0.0)
            
            return np.array([
                float(self.horizon),
                float(self.adaptive_horizon),
                strategy_idx / len(self.SIMULATION_STRATEGIES),
                float(self.confidence_score),
                float(self.confidence_threshold),
                float(recent_score),
                float(self.simulation_stats.get('prediction_accuracy', 0.0)),
                float(min(1.0, self.simulation_stats.get('total_simulations', 0) / 100.0))
            ], dtype=np.float32)
            
        except Exception as e:
            self.log_operator_error(f"Observation generation failed: {e}")
            return np.array([5.0, 5.0, 0.0, 0.5, 0.6, 0.0, 0.0, 0.0], dtype=np.float32)

    def get_shadow_simulation_report(self) -> str:
        """Generate operator-friendly simulation report"""
        
        # Confidence status
        if self.confidence_score > 0.8:
            confidence_status = "✅ High"
        elif self.confidence_score > 0.6:
            confidence_status = "⚡ Good"
        elif self.confidence_score > 0.4:
            confidence_status = "⚠️ Moderate"
        else:
            confidence_status = "🚨 Low"
        
        # Strategy weights (top 3)
        top_strategies = sorted(
            self.strategy_weights.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        strategy_lines = []
        for strategy, weight in top_strategies:
            strategy_name = strategy.replace('_', ' ').title()
            strategy_lines.append(f"  📊 {strategy_name}: {weight:.1%}")
        
        # Recent learning progress
        learning_lines = []
        for entry in list(self.learning_history)[-3:]:
            timestamp = entry['timestamp'][:19]
            avg_score = entry['avg_score']
            confidence = entry['confidence']
            learning_lines.append(f"  📈 {timestamp}: Score {avg_score:.1%}, Confidence {confidence:.1%}")
        
        # Recent scenarios
        scenario_lines = []
        for result in list(self.scenario_results)[-3:]:
            scenarios = result.get('scenarios', [])
            avg_score = result.get('avg_score', 0.0)
            confidence = result.get('confidence', 0.0)
            scenario_lines.append(f"  🔮 {len(scenarios)} scenarios: Score {avg_score:.1%}, Confidence {confidence:.1%}")
        
        return f"""
🔮 SHADOW SIMULATOR
═══════════════════════════════════════
🎯 Strategy: {self.strategy.title().replace('_', ' ')}
📊 Confidence: {confidence_status} ({self.confidence_score:.1%})
🔭 Horizon: Base {self.horizon} | Adaptive {self.adaptive_horizon}
⚙️ Scenarios: {self.scenario_count} per simulation

📈 SIMULATION CONFIGURATION
• Confidence Threshold: {self.confidence_threshold:.1%}
• Risk Scaling: {'✅ Enabled' if self.risk_scaling else '❌ Disabled'}
• Regime Awareness: {'✅ Enabled' if self.regime_awareness else '❌ Disabled'}
• Volatility Adjustment: {'✅ Enabled' if self.volatility_adjustment else '❌ Disabled'}
• Session Sensitivity: {self.session_sensitivity:.1%}
• Learning Rate: {self.learning_rate:.1%}
• Simulation Depth: {self.simulation_depth}

📊 PERFORMANCE STATISTICS
• Total Simulations: {self.simulation_stats['total_simulations']:,}
• Scenarios Evaluated: {self.simulation_stats['scenarios_evaluated']:,}
• Prediction Accuracy: {self.simulation_stats['prediction_accuracy']:.1%}
• Strategy Effectiveness: {self.simulation_stats['strategy_effectiveness']:.1%}
• Avg Scenario Score: {self.simulation_stats['avg_scenario_score']:.1%}

🎯 STRATEGY PERFORMANCE (Top 3)
{chr(10).join(strategy_lines) if strategy_lines else "  📭 No strategy data available"}

🔧 ADAPTIVE PARAMETERS
• Current Strategy: {self.strategy.title().replace('_', ' ')}
• Adaptive Horizon: {self.adaptive_horizon} steps
• Confidence Score: {self.confidence_score:.1%}
• Market Regime: {self.market_regime.title()}
• Volatility Level: {self.volatility_regime.title()}
• Market Session: {self.market_session.title()}

📚 LEARNING PROGRESS
{chr(10).join(learning_lines) if learning_lines else "  📭 No recent learning data"}

🔮 RECENT SIMULATIONS
{chr(10).join(scenario_lines) if scenario_lines else "  📭 No recent simulations"}

💡 AVAILABLE STRATEGIES
• Greedy: Aggressive profit maximization
• Conservative: Risk-minimized approach
• Adaptive: Context-aware strategy (Current)
• Contrarian: Counter-trend positions
• Momentum: Trend-following strategy
• Balanced: Risk-reward balanced
• Volatility Play: Volatility exploitation
• Regime Specific: Market regime optimized

🎯 EFFECTIVENESS METRICS
• Prediction Accuracy: {self.simulation_stats['prediction_accuracy']:.1%}
• Strategy Effectiveness: {self.simulation_stats['strategy_effectiveness']:.1%}
• Learning Progress: {len(self.learning_history)} sessions tracked
• Adaptation Rate: {self.learning_rate:.1%} per session
        """

    # ================== EVOLUTIONARY METHODS ==================

    def mutate(self, std: float = 1.0) -> None:
        """Mutate simulation parameters"""
        
        old_horizon = self.horizon
        old_strategy = self.strategy
        
        # Mutate horizon
        self.horizon = max(1, int(self.horizon + np.random.randint(-2, 3)))
        self.adaptive_horizon = self.horizon
        
        # Mutate strategy occasionally
        if np.random.random() < 0.3:
            self.strategy = np.random.choice(list(self.SIMULATION_STRATEGIES.keys()))
        
        # Mutate other parameters
        if np.random.random() < 0.1:
            self.confidence_threshold = np.clip(
                self.confidence_threshold + np.random.normal(0, 0.1),
                0.1, 0.9
            )
        
        self.log_operator_info(
            f"🧬 Simulation mutation applied",
            horizon=f"{old_horizon} → {self.horizon}",
            strategy=f"{old_strategy} → {self.strategy}" if old_strategy != self.strategy else "unchanged"
        )

    def crossover(self, other: "ShadowSimulator") -> "ShadowSimulator":
        """Create offspring through crossover"""
        
        # Select parameters from parents
        horizon = self.horizon if np.random.random() < 0.5 else other.horizon
        strategy = self.strategy if np.random.random() < 0.5 else other.strategy
        confidence_threshold = (self.confidence_threshold + other.confidence_threshold) / 2
        
        # Create offspring
        offspring = ShadowSimulator(
            horizon=horizon,
            strategy=strategy,
            debug=self.config.debug
        )
        
        # Mix other parameters
        offspring.confidence_threshold = confidence_threshold
        offspring.risk_scaling = self.risk_scaling if np.random.random() < 0.5 else other.risk_scaling
        offspring.regime_awareness = self.regime_awareness if np.random.random() < 0.5 else other.regime_awareness
        offspring.scenario_count = int((self.scenario_count + other.scenario_count) / 2)
        
        # Mix strategy weights
        for strategy_name in self.strategy_weights:
            offspring.strategy_weights[strategy_name] = (
                self.strategy_weights[strategy_name] + 
                other.strategy_weights.get(strategy_name, 1.0)
            ) / 2
        
        self.log_operator_info(
            f"🔬 Simulation crossover created offspring",
            horizon=horizon,
            strategy=strategy,
            confidence_threshold=f"{confidence_threshold:.1%}"
        )
        
        return offspring

    # ================== STATE MANAGEMENT ==================

    def get_state(self) -> Dict[str, Any]:
        """Get complete state for serialization"""
        return {
            "config": {
                "horizon": self.horizon,
                "strategy": self.strategy,
                "confidence_threshold": self.confidence_threshold,
                "risk_scaling": self.risk_scaling,
                "regime_awareness": self.regime_awareness,
                "volatility_adjustment": self.volatility_adjustment,
                "scenario_count": self.scenario_count
            },
            "adaptive_parameters": {
                "adaptive_horizon": self.adaptive_horizon,
                "confidence_score": self.confidence_score,
                "strategy_weights": self.strategy_weights.copy()
            },
            "market_context": {
                "regime": self.market_regime,
                "volatility_regime": self.volatility_regime,
                "session": self.market_session,
                "current_volatility": self.current_volatility
            },
            "statistics": self.simulation_stats.copy(),
            "history": {
                "scenario_results": list(self.scenario_results)[-10:],
                "learning_history": list(self.learning_history)[-10:],
                "prediction_accuracy": list(self.prediction_accuracy)[-20:]
            }
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Load state from serialization"""
        
        # Load config
        config = state.get("config", {})
        self.horizon = int(config.get("horizon", self.horizon))
        self.strategy = config.get("strategy", self.strategy)
        self.confidence_threshold = float(config.get("confidence_threshold", self.confidence_threshold))
        self.risk_scaling = bool(config.get("risk_scaling", self.risk_scaling))
        self.regime_awareness = bool(config.get("regime_awareness", self.regime_awareness))
        self.volatility_adjustment = bool(config.get("volatility_adjustment", self.volatility_adjustment))
        self.scenario_count = int(config.get("scenario_count", self.scenario_count))
        
        # Load adaptive parameters
        adaptive = state.get("adaptive_parameters", {})
        self.adaptive_horizon = int(adaptive.get("adaptive_horizon", self.horizon))
        self.confidence_score = float(adaptive.get("confidence_score", 0.5))
        self.strategy_weights.update(adaptive.get("strategy_weights", {}))
        
        # Load market context
        context = state.get("market_context", {})
        self.market_regime = context.get("regime", "normal")
        self.volatility_regime = context.get("volatility_regime", "medium")
        self.market_session = context.get("session", "unknown")
        self.current_volatility = float(context.get("current_volatility", 0.01))
        
        # Load statistics
        self.simulation_stats.update(state.get("statistics", {}))
        
        # Load history
        history = state.get("history", {})
        
        scenario_results = history.get("scenario_results", [])
        self.scenario_results.clear()
        for result in scenario_results:
            self.scenario_results.append(result)
            
        learning_history = history.get("learning_history", [])
        self.learning_history.clear()
        for entry in learning_history:
            self.learning_history.append(entry)
            
        prediction_accuracy = history.get("prediction_accuracy", [])
        self.prediction_accuracy.clear()
        for accuracy in prediction_accuracy:
            self.prediction_accuracy.append(accuracy)

    # ================== LEGACY COMPATIBILITY ==================

    def step(self, **kwargs) -> None:
        """Legacy step interface for backward compatibility"""
        self._process_legacy_step(**kwargs)

    def simulate(self, env, actions: np.ndarray) -> List[Dict[str, Any]]:
        """Legacy simulate interface for backward compatibility"""
        return self.simulate_legacy(env, actions)