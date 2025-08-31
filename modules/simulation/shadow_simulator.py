# ─────────────────────────────────────────────────────────────
# File: modules/simulation/shadow_simulator.py
# ShadowSimulator — Contract-Strict & Robust (v3.0.0)
# - Provides (per contracts.py):
#   forward_projections, scenario_analysis, scenario_recommendations,
#   shadow_predictions, shadow_simulation, simulation_confidence,
#   simulation_predictions, strategy_simulations
# - Requires:
#   votes, environment, market_context, market_data, pending_orders,
#   positions, prices, recent_trades, risk_metrics, trading_performance
# - Thesis required by meta: returns `_thesis`
# - No undeclared bus writes
# ─────────────────────────────────────────────────────────────

from modules.contracts import module_args
import numpy as np
import datetime
import time
import copy
from typing import Dict, Any, List, Optional, Tuple
from collections import deque, defaultdict

# Modern imports
from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusStateMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.performance_tracker import PerformanceTracker


@module(**module_args(
    "ShadowSimulator",
    description="Intelligent forward-looking trade simulation with context-aware strategy variations",
    error_handling=True,
    hot_reload=True,
    timeout_ms=3000,
))
class ShadowSimulator(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    Modern shadow simulator with comprehensive SmartInfoBus integration.
    Contract-strict: consumes all declared required keys (directly or defensively)
    and publishes all declared provided keys. Returns `_thesis` (explainable).
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
        horizon: int = 5,
        strategy: str = "adaptive",
        debug: bool = False,
        **kwargs
    ):
        # Initialize BaseModule
        super().__init__(**kwargs)

        # Initialize mixins
        self._initialize_trading_state()

        # Build sim_config from defaults, then update if config override is provided
        self.sim_config = copy.deepcopy(self.ENHANCED_DEFAULTS)
        if 'config' in kwargs and isinstance(kwargs['config'], dict):
            self.sim_config.update(kwargs['config'])

        # Core parameters
        self.horizon               = int(horizon)
        self.strategy              = strategy if strategy in self.SIMULATION_STRATEGIES else "adaptive"
        self.confidence_threshold  = float(self.sim_config["confidence_threshold"])
        self.risk_scaling          = bool(self.sim_config["risk_scaling"])
        self.regime_awareness      = bool(self.sim_config["regime_awareness"])
        self.volatility_adjustment = bool(self.sim_config["volatility_adjustment"])
        self.session_sensitivity   = float(self.sim_config["session_sensitivity"])
        self.learning_rate         = float(self.sim_config["learning_rate"])
        self.simulation_depth      = int(self.sim_config["simulation_depth"])
        self.scenario_count        = int(self.sim_config["scenario_count"])

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

        # Circuit breaker and error handling
        self.error_count = 0
        self.circuit_breaker_threshold = 5
        self.is_disabled = False

        # Initialize advanced systems
        self._initialize_advanced_systems()

        self.logger.info(format_operator_message(
            icon="🔮",
            message="Enhanced Shadow Simulator initialized",
            horizon=self.horizon,
            strategy=self.strategy,
            confidence_threshold=f"{self.confidence_threshold:.1%}",
            regime_awareness=self.regime_awareness,
            scenarios=self.scenario_count
        ))

    def _initialize_advanced_systems(self):
        """Initialize all modern system components"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="ShadowSimulator",
            log_path="logs/simulation/shadow_simulator.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("ShadowSimulator", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()

    def _initialize(self):
        """Satisfy BaseModule abstract initializer (no-op: real setup in __init__)."""
        return None

    def reset(self) -> None:
        """Enhanced reset with comprehensive state cleanup"""
        super().reset()

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

        # Reset error state
        self.error_count = 0
        self.is_disabled = False

        self.logger.info(format_operator_message(
            icon="[RELOAD]",
            message="Shadow Simulator reset - all state cleared"
        ))

    # ─────────────────────────────────────────────────────────────
    # Helpers (contract robustness)
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_action_keys(d: Dict[Any, Any]) -> Dict[str, float]:
        """Convert action dict keys to strings to avoid int/str mismatches."""
        out: Dict[str, float] = {}
        for k, v in (d or {}).items():
            try:
                out[str(k)] = float(v)
            except Exception:
                # best-effort coercion
                try:
                    out[str(k)] = float(v) if v is not None else 0.0
                except Exception:
                    out[str(k)] = 0.0
        return out

    @staticmethod
    def _get_action_pair(modified_actions: Dict[str, float], idx: int) -> Optional[Tuple[float, float]]:
        """
        Fetch a robust (a0, a1) pair for instrument idx.
        Prefers keys "2*idx" and "2*idx+1"; falls back to "idx" and "idx+1"; finally first two keys.
        """
        k0, k1 = str(idx * 2), str(idx * 2 + 1)
        if k0 in modified_actions and k1 in modified_actions:
            return modified_actions[k0], modified_actions[k1]
        k0b, k1b = str(idx), str(idx + 1)
        if k0b in modified_actions and k1b in modified_actions:
            return modified_actions[k0b], modified_actions[k1b]
        # fallback: try any two keys
        if len(modified_actions) >= 2:
            keys = list(modified_actions.keys())
            return modified_actions[keys[0]], modified_actions[keys[1]]
        return None

    # ─────────────────────────────────────────────────────────────
    # Core processing
    # ─────────────────────────────────────────────────────────────

    async def process(self, **inputs) -> Dict[str, Any]:
        """Modern async processing with comprehensive simulation (contract-strict)."""
        start_time = time.time()

        try:
            # Circuit breaker check
            if self.is_disabled:
                return self._generate_disabled_response()

            # Get comprehensive simulation data from SmartInfoBus (satisfies requires)
            simulation_data = await self._extract_simulation_data_from_smart_bus()

            # Update market context awareness
            await self._update_market_context(simulation_data)

            # Perform multi-scenario simulation
            simulation_results = await self._perform_multi_scenario_simulation(simulation_data)

            # Analyze simulation effectiveness
            self._analyze_simulation_effectiveness(simulation_results)

            # Update adaptive parameters
            self._update_adaptive_parameters(simulation_results)

            # Compose contract-compliant payload
            best_scenario = simulation_results.get('best_scenario', {}) or {}
            thesis = (
                f"Shadow simulation completed: {len(simulation_results.get('scenarios', []))} "
                f"scenarios, confidence {simulation_results.get('confidence', 0.0):.1%}"
            )

            # Build shadow_predictions from best scenario trades
            best_trades = best_scenario.get('trades', []) or []
            shadow_predictions = [
                {
                    'instrument': t.get('instrument'),
                    'side': t.get('side'),
                    'size': t.get('size', 0.0),
                    'expected_pnl': t.get('pnl', 0.0),
                    'strategy': t.get('simulation_strategy', best_scenario.get('scenario', {}).get('strategy', 'unknown')),
                    'timestamp': t.get('timestamp')
                }
                for t in best_trades
            ]

            # Forward projections mirror the best scenario timeline
            forward_projections = [
                {
                    'step': step.get('step'),
                    'balance': step.get('balance'),
                    'actions_taken': step.get('actions_taken', []),
                    'trades_executed': step.get('trades_executed', [])
                }
                for step in best_scenario.get('timeline', []) or []
            ]

            # Scenario analysis summary
            scenario_analysis = {
                'avg_score': simulation_results.get('avg_score', 0.0),
                'confidence': simulation_results.get('confidence', 0.0),
                'best_strategy': best_scenario.get('scenario', {}).get('strategy', 'unknown'),
                'best_score': best_scenario.get('score', 0.0),
                'worst_score': (simulation_results.get('worst_scenario') or {}).get('score', 0.0),
                'scenarios_evaluated': len(simulation_results.get('scenarios', [])),
            }

            # Scenario recommendations and strategy simulations
            scenario_recommendations = simulation_results.get('recommendations', []) or []
            strategy_simulations = [
                {
                    'strategy': s.get('scenario', {}).get('strategy', 'unknown'),
                    'score': s.get('score', 0.0),
                    'trade_count': s.get('trade_count', 0)
                }
                for s in simulation_results.get('scenarios', [])
            ]

            # Simulation confidence and predictions summary
            simulation_confidence = float(simulation_results.get('confidence', 0.0))
            simulation_predictions = {
                'strategy': best_scenario.get('scenario', {}).get('strategy', 'unknown'),
                'expected_pnl': best_scenario.get('total_pnl', 0.0),
                'confidence': simulation_confidence,
                'trade_count_estimate': best_scenario.get('trade_count', 0),
                'risk_estimate': (best_scenario.get('risk_metrics', {}) or {}).get('max_drawdown', 0.0)
            }

            # Shadow simulation summary (mirrors bus payload)
            shadow_simulation = {
                'simulation_strategy': self.strategy,
                'horizon': self.horizon,
                'adaptive_horizon': self.adaptive_horizon,
                'confidence_score': self.confidence_score,
                'simulation_stats': self.simulation_stats.copy(),
                'strategy_weights': self.strategy_weights.copy(),
                'simulation_results': {
                    'avg_score': simulation_results.get('avg_score', 0.0),
                    'confidence': simulation_confidence,
                    'scenarios_evaluated': len(simulation_results.get('scenarios', [])),
                    'best_strategy': best_scenario.get('scenario', {}).get('strategy', 'unknown')
                },
                'recommendations': scenario_recommendations,
                'market_context': {
                    'regime': self.market_regime,
                    'volatility_regime': self.volatility_regime,
                    'session': self.market_session
                }
            }

            # Merge into a single, contract-compliant result payload
            result_payload = {
                **simulation_results,
                'forward_projections': forward_projections,
                'scenario_analysis': scenario_analysis,
                'scenario_recommendations': scenario_recommendations,
                'shadow_predictions': shadow_predictions,
                'shadow_simulation': shadow_simulation,
                'simulation_confidence': simulation_confidence,
                'simulation_predictions': simulation_predictions,
                'strategy_simulations': strategy_simulations,
                '_thesis': thesis,
            }

            # Update SmartInfoBus with ALL provided keys (contract-strict)
            await self._update_smartinfobus_comprehensive(result_payload)

            # Record performance metrics
            processing_time = (time.time() - start_time) * 1000
            self.performance_tracker.record_metric('ShadowSimulator', 'process_time', processing_time, True)

            # Reset error count on successful processing
            self.error_count = 0

            return result_payload

        except Exception as e:
            return await self._handle_processing_error(e, start_time)

    async def _extract_simulation_data_from_smart_bus(self) -> Dict[str, Any]:
        """Extract simulation data from SmartInfoBus, satisfying all `requires` keys."""
        data: Dict[str, Any] = {}

        try:
            sb = self.smart_bus

            # Required keys (primary reads)
            data['current_prices']     = sb.get('prices', 'ShadowSimulator') or {}
            data['current_positions']  = sb.get('positions', 'ShadowSimulator') or []
            data['market_data']        = sb.get('market_data', 'ShadowSimulator') or {}
            data['risk_state']         = sb.get('risk_metrics', 'ShadowSimulator') or {}
            data['recent_trades']      = sb.get('recent_trades', 'ShadowSimulator') or []
            data['pending_orders']     = sb.get('pending_orders', 'ShadowSimulator') or []
            data['environment']        = sb.get('environment', 'ShadowSimulator')  # may be None
            data['market_context']     = sb.get('market_context', 'ShadowSimulator') or {}
            data['trading_performance']= sb.get('trading_performance', 'ShadowSimulator') or {}

            # votes (contract) with fallback to committee_votes for backward compat
            votes = sb.get('votes', 'ShadowSimulator')
            if votes is None:
                votes = sb.get('committee_votes', 'ShadowSimulator')
            data['votes'] = votes or []

            # Derived context
            mc = data['market_context']
            data['regime']           = mc.get('regime', 'unknown')
            data['session']          = mc.get('session', 'unknown')
            data['volatility_level'] = mc.get('volatility_level', 'medium')

            # Extract volatility (optional)
            volatilities = mc.get('volatility', {})
            self.current_volatility = float(np.mean(list(volatilities.values()))) if volatilities else 0.01
            data['current_volatility'] = self.current_volatility

            # (Optional) features snapshot
            data['market_features'] = data['market_data'].get('features', {})

        except Exception as e:
            self.logger.warning(f"Simulation data extraction failed: {e}")
            # Provide safe defaults (still satisfies requires by keys present)
            data = {
                'current_prices': {},
                'current_positions': [],
                'market_data': {},
                'risk_state': {},
                'recent_trades': [],
                'pending_orders': [],
                'votes': [],
                'market_context': {},
                'regime': 'unknown',
                'session': 'unknown',
                'volatility_level': 'medium',
                'current_volatility': 0.01,
                'environment': None,
                'trading_performance': {}
            }

        return data

    async def _update_market_context(self, simulation_data: Dict[str, Any]) -> None:
        """Update market context awareness"""
        try:
            old_regime = self.market_regime
            self.market_regime = simulation_data.get('regime', 'unknown')
            self.volatility_regime = simulation_data.get('volatility_level', 'medium')
            self.market_session = simulation_data.get('session', 'unknown')

            if self.market_regime != old_regime and self.regime_awareness:
                self._adapt_simulation_for_regime(old_regime, self.market_regime)
                self.logger.info(format_operator_message(
                    icon="[STATS]",
                    message=f"Regime change detected: {old_regime} → {self.market_regime}",
                    adaptation="Simulation parameters updated",
                    session=self.market_session
                ))
        except Exception as e:
            self.logger.warning(f"Market context update failed: {e}")

    def _adapt_simulation_for_regime(self, old_regime: str, new_regime: str) -> None:
        """Adapt simulation parameters for regime change"""
        try:
            regime_horizon_multipliers = {
                'volatile': 0.7,   # Shorter horizon in volatile markets
                'trending': 1.3,   # Longer horizon in trending markets
                'ranging': 1.0,
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
                for s in self.strategy_weights:
                    self.strategy_weights[s] /= total_weight
        except Exception as e:
            self.logger.warning(f"Regime adaptation failed: {e}")

    async def _perform_multi_scenario_simulation(self, simulation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive multi-scenario simulation"""
        results = {
            'scenarios': [],
            'best_scenario': None,
            'worst_scenario': None,
            'avg_score': 0.0,
            'confidence': 0.0,
            'recommendations': [],
            'context': {
                'regime': self.market_regime,
                'session': self.market_session,
                'volatility_level': self.volatility_regime
            }
        }

        try:
            # Generate scenarios
            scenarios = self._generate_simulation_scenarios(simulation_data)

            # Simulate each scenario
            scenario_results: List[Dict[str, Any]] = []
            for i, scenario in enumerate(scenarios):
                try:
                    scenario_result = await self._simulate_scenario(scenario, simulation_data)
                    scenario_results.append(scenario_result)
                except Exception as e:
                    self.logger.warning(f"Scenario {i} simulation failed: {e}")
                    continue

            # Analyze results
            if scenario_results:
                results['scenarios'] = scenario_results
                results['best_scenario'] = max(scenario_results, key=lambda s: s.get('score', 0))
                results['worst_scenario'] = min(scenario_results, key=lambda s: s.get('score', 0))
                results['avg_score'] = float(np.mean([s.get('score', 0) for s in scenario_results]))
                results['confidence'] = self._calculate_simulation_confidence(scenario_results)
                results['recommendations'] = self._generate_simulation_recommendations(scenario_results, simulation_data)

            # Update statistics
            self.simulation_stats['total_simulations'] += 1
            self.simulation_stats['scenarios_evaluated'] += len(scenario_results)
            self.simulation_stats['avg_scenario_score'] = results['avg_score']

            # Store results
            self.scenario_results.append(results)

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "multi_scenario_simulation")
            self.logger.error(f"Multi-scenario simulation failed: {error_context}")
            results['error'] = str(error_context)

        return results

    def _generate_simulation_scenarios(self, simulation_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate diverse simulation scenarios"""
        scenarios: List[Dict[str, Any]] = []

        try:
            votes = simulation_data.get('votes', [])
            base_actions = self._extract_base_actions_from_votes(votes)  # normalized keys

            # Generate scenarios for each strategy (limited by scenario_count)
            strategies_to_test = list(self.SIMULATION_STRATEGIES.keys())
            if len(strategies_to_test) > self.scenario_count:
                strategies_to_test = sorted(
                    strategies_to_test,
                    key=lambda s: self.strategy_weights.get(s, 0),
                    reverse=True
                )[:self.scenario_count]

            for strategy in strategies_to_test:
                modified = self._modify_actions_for_strategy(base_actions, strategy, simulation_data)
                scenario = {
                    'strategy': strategy,
                    'description': self.SIMULATION_STRATEGIES[strategy],
                    'base_actions': base_actions,
                    'modified_actions': modified,
                    'horizon': self._get_strategy_horizon(strategy, simulation_data),
                    'risk_scaling': self._get_strategy_risk_scaling(strategy, simulation_data),
                    'context_adjustments': self._get_strategy_context_adjustments(strategy, simulation_data)
                }
                scenarios.append(scenario)

            # Add adaptive scenario if not already included
            if 'adaptive' not in strategies_to_test:
                adaptive_scenario = {
                    'strategy': 'adaptive',
                    'description': 'Context-adaptive strategy',
                    'base_actions': base_actions,
                    'modified_actions': self._create_adaptive_actions(base_actions, simulation_data),
                    'horizon': self.adaptive_horizon,
                    'risk_scaling': True,
                    'context_adjustments': self._get_adaptive_context_adjustments(simulation_data)
                }
                scenarios.append(adaptive_scenario)

        except Exception as e:
            self.logger.warning(f"Scenario generation failed: {e}")
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

    def _extract_base_actions_from_votes(self, votes: List[Dict[str, Any]]) -> Dict[str, float]:
        """Extract base actions from committee votes (normalized key strings)."""
        try:
            if not votes:
                return {}

            action_sums = defaultdict(float)
            confidence_sums = defaultdict(float)

            for vote in votes:
                action = vote.get('action', [])
                confidence = float(vote.get('confidence', 0.5))

                if isinstance(action, (list, tuple, np.ndarray)) and len(action) > 0:
                    for i, act in enumerate(action):
                        action_sums[i] += float(act) * confidence
                        confidence_sums[i] += confidence
                elif isinstance(action, (int, float)):
                    action_sums[0] += float(action) * confidence
                    confidence_sums[0] += confidence

            # Weighted averages → normalize keys to strings
            base_actions = {}
            for i, s in action_sums.items():
                denom = confidence_sums[i] if confidence_sums[i] > 0 else 1.0
                base_actions[str(i)] = s / denom

            return base_actions

        except Exception as e:
            self.logger.warning(f"Base action extraction failed: {e}")
            return {}

    def _modify_actions_for_strategy(self, base_actions: Dict[str, float],
                                    strategy: str, simulation_data: Dict[str, Any]) -> Dict[str, float]:
        """Modify actions based on strategy (keys remain strings)."""
        try:
            modified_actions = copy.deepcopy(base_actions)

            if strategy == "greedy":
                for k in modified_actions:
                    modified_actions[k] *= 1.5
            elif strategy == "conservative":
                for k in modified_actions:
                    modified_actions[k] *= 0.6
            elif strategy == "contrarian":
                for k in modified_actions:
                    modified_actions[k] *= -0.8
            elif strategy == "momentum":
                trend_multiplier = 1.3 if simulation_data.get('regime') == 'trending' else 1.0
                for k in modified_actions:
                    modified_actions[k] *= trend_multiplier
            elif strategy == "balanced":
                for k in modified_actions:
                    modified_actions[k] *= 0.8
            elif strategy == "volatility_play":
                vol_multiplier = 1.5 if simulation_data.get('volatility_level') in ['high', 'extreme'] else 0.7
                for k in modified_actions:
                    modified_actions[k] *= vol_multiplier
            elif strategy == "regime_specific":
                regime = simulation_data.get('regime', 'unknown')
                regime_multipliers = {
                    'volatile': 0.7,
                    'trending': 1.2,
                    'ranging': 0.9,
                    'unknown': 1.0
                }
                mult = regime_multipliers.get(regime, 1.0)
                for k in modified_actions:
                    modified_actions[k] *= mult

            # Bounds
            for k in modified_actions:
                modified_actions[k] = float(np.clip(modified_actions[k], -1.0, 1.0))

            return modified_actions

        except Exception as e:
            self.logger.warning(f"Action modification failed: {e}")
            return base_actions

    def _create_adaptive_actions(self, base_actions: Dict[str, float],
                                 simulation_data: Dict[str, Any]) -> Dict[str, float]:
        """Create adaptive actions based on context (keys remain strings)."""
        try:
            adaptive_actions = copy.deepcopy(base_actions)

            regime = simulation_data.get('regime', 'unknown')
            vol_level = simulation_data.get('volatility_level', 'medium')
            session = simulation_data.get('session', 'unknown')

            # Regime adaptation
            if regime == 'volatile':
                for k in adaptive_actions:
                    adaptive_actions[k] *= 0.75
            elif regime == 'trending':
                for k in adaptive_actions:
                    if adaptive_actions[k] > 0:
                        adaptive_actions[k] *= 1.2

            # Volatility adaptation
            vol_multipliers = {'low': 1.1, 'medium': 1.0, 'high': 0.8, 'extreme': 0.6}
            vm = vol_multipliers.get(vol_level, 1.0)
            for k in adaptive_actions:
                adaptive_actions[k] *= vm

            # Session adaptation
            if self.session_sensitivity > 0:
                session_multipliers = {'asian': 0.9, 'european': 1.0, 'american': 1.1, 'rollover': 0.7}
                sm = session_multipliers.get(session, 1.0)
                for k in adaptive_actions:
                    adaptive_actions[k] *= sm * self.session_sensitivity + (1 - self.session_sensitivity)

            # Bounds
            for k in adaptive_actions:
                adaptive_actions[k] = float(np.clip(adaptive_actions[k], -1.0, 1.0))

            return adaptive_actions

        except Exception as e:
            self.logger.warning(f"Adaptive action creation failed: {e}")
            return base_actions

    def _get_strategy_horizon(self, strategy: str, simulation_data: Dict[str, Any]) -> int:
        """Get strategy-specific horizon"""
        base_horizon = self.adaptive_horizon
        strategy_multipliers = {
            'greedy': 0.8, 'conservative': 1.2, 'momentum': 1.3, 'contrarian': 0.9,
            'balanced': 1.0, 'volatility_play': 0.7, 'regime_specific': 1.0, 'adaptive': 1.0
        }
        mult = strategy_multipliers.get(strategy, 1.0)
        if simulation_data.get('volatility_level') == 'extreme':
            mult *= 0.8
        elif simulation_data.get('regime') == 'trending':
            mult *= 1.1
        horizon = int(base_horizon * mult)
        return max(1, min(horizon, 15))

    def _get_strategy_risk_scaling(self, strategy: str, simulation_data: Dict[str, Any]) -> bool:
        """Get strategy-specific risk scaling setting"""
        risk_scaling_strategies = {
            'greedy': False, 'conservative': True, 'momentum': True, 'contrarian': True,
            'balanced': True, 'volatility_play': False, 'regime_specific': True, 'adaptive': True
        }
        return bool(risk_scaling_strategies.get(strategy, self.risk_scaling))

    def _get_strategy_context_adjustments(self, strategy: str, simulation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get strategy-specific context adjustments"""
        adjustments = {
            'strategy': strategy,
            'regime_sensitivity': 1.0,
            'volatility_sensitivity': 1.0,
            'session_sensitivity': 1.0
        }
        if strategy == 'conservative':
            adjustments['volatility_sensitivity'] = 1.5
        elif strategy == 'momentum':
            adjustments['regime_sensitivity'] = 1.3
        elif strategy == 'contrarian':
            adjustments['regime_sensitivity'] = 0.8
        elif strategy == 'volatility_play':
            adjustments['volatility_sensitivity'] = 2.0
        return adjustments

    def _get_adaptive_context_adjustments(self, simulation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get adaptive context adjustments"""
        return {
            'strategy': 'adaptive',
            'regime_sensitivity': float(self.regime_awareness),
            'volatility_sensitivity': float(self.volatility_adjustment),
            'session_sensitivity': self.session_sensitivity,
            'confidence_threshold': self.confidence_threshold,
            'learning_rate': self.learning_rate
        }

    async def _simulate_scenario(self, scenario: Dict[str, Any],
                                 simulation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate a single scenario with evolving balance timeline"""
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
            modified_actions = self._normalize_action_keys(scenario.get('modified_actions', {}))
            horizon = int(scenario.get('horizon', 1))

            # Simulate using available data or synthetic environment
            initial_balance = float(simulation_data.get('risk_state', {}).get('balance', 10000.0))
            trades: List[Dict[str, Any]] = []
            timeline: List[Dict[str, Any]] = []
            running_balance = initial_balance

            for step in range(horizon):
                step_result = {
                    'step': step,
                    'timestamp': datetime.datetime.now().isoformat(),
                    'balance': running_balance,
                    'actions_taken': [],
                    'trades_executed': []
                }

                # Execute trades based on strategy/actions
                step_trades = self._execute_scenario_step(modified_actions, strategy, simulation_data)
                if step_trades:
                    trades.extend(step_trades)
                    step_result['trades_executed'] = step_trades
                    # accumulate pnl for this step
                    step_pnl = float(sum(t.get('pnl', 0.0) for t in step_trades))
                    running_balance += step_pnl
                    step_result['balance'] = running_balance

                timeline.append(step_result)

            # Calculate results
            final_balance = running_balance
            total_pnl = sum(trade.get('pnl', 0.0) for trade in trades)

            result.update({
                'trades': trades,
                'final_balance': final_balance,
                'total_pnl': total_pnl,
                'trade_count': len(trades),
                'timeline': timeline
            })

            # Calculate success rate
            if trades:
                successful_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
                result['success_rate'] = successful_trades / len(trades)
            else:
                result['success_rate'] = 0.0

            # Risk metrics & score
            result['risk_metrics'] = self._calculate_scenario_risk_metrics(result, initial_balance)
            result['score'] = self._calculate_scenario_score(result, scenario, simulation_data)

        except Exception as e:
            self.logger.warning(f"Scenario simulation failed: {e}")
            result['error'] = str(e)
            result['score'] = 0.0

        return result

    def _execute_scenario_step(self, modified_actions: Dict[str, float],
                               strategy: str, simulation_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute one step of scenario simulation"""
        trades: List[Dict[str, Any]] = []
        try:
            current_prices = simulation_data.get('current_prices', {})
            instruments = list(current_prices.keys()) or ['EUR/USD', 'XAU/USD']

            for idx, instrument in enumerate(instruments[:2]):  # Limit to 2 instruments
                pair = self._get_action_pair(modified_actions, idx)
                if not pair:
                    continue
                action_0, action_1 = pair
                # Execute trade if action is significant
                if abs(action_0) > 0.1 or abs(action_1) > 0.1:
                    trade = self._execute_simulated_trade(instrument, action_0, action_1, strategy, simulation_data)
                    if trade:
                        trades.append(trade)

        except Exception as e:
            self.logger.warning(f"Scenario step execution failed: {e}")

        return trades

    def _execute_simulated_trade(self, instrument: str, action_0: float,
                                 action_1: float, strategy: str, simulation_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute a simulated trade (simple synthetic PnL)."""
        try:
            current_prices = simulation_data.get('current_prices', {})
            current_price = current_prices.get(instrument)
            if current_price is None:
                fallback_prices = {
                    'EUR/USD': 1.0850,
                    'XAU/USD': 2000.0,
                    'GBP/USD': 1.2650,
                    'USD/JPY': 150.0
                }
                current_price = fallback_prices.get(instrument, 1.0)

            # Determine position size (simplified)
            position_size = abs(float(action_0)) * 0.1  # 10% of action magnitude
            side = 'BUY' if action_0 > 0 else 'SELL'

            # Simulate basic P&L using action_1 as performance indicator
            estimated_pnl = float(action_1) * position_size * 100.0  # Simplified

            return {
                'instrument': instrument,
                'side': side,
                'size': position_size,
                'price': current_price,
                'pnl': estimated_pnl,
                'timestamp': datetime.datetime.now().isoformat(),
                'simulation_strategy': strategy,
                'action_0': float(action_0),
                'action_1': float(action_1)
            }

        except Exception as e:
            self.logger.warning(f"Simulated trade execution failed: {e}")
            return None

    def _calculate_scenario_score(self, result: Dict[str, Any],
                                  scenario: Dict[str, Any], simulation_data: Dict[str, Any]) -> float:
        """Calculate scenario performance score"""
        try:
            total_pnl = float(result.get('total_pnl', 0.0))
            pnl_score = float(np.tanh(total_pnl / 1000.0))  # Normalize PnL

            success_rate = float(result.get('success_rate', 0.0))

            risk_metrics = result.get('risk_metrics', {})
            max_drawdown = float(risk_metrics.get('max_drawdown', 0.0))
            risk_penalty = abs(max_drawdown) * 2.0  # Penalize drawdown

            trade_count = int(result.get('trade_count', 0))
            horizon = int(scenario.get('horizon', 1))
            trade_frequency = trade_count / max(horizon, 1)

            if trade_frequency < 0.5:
                frequency_score = trade_frequency * 2.0
            elif trade_frequency <= 1.0:
                frequency_score = 1.0
            else:
                frequency_score = max(0.0, 2.0 - trade_frequency)

            strategy = scenario.get('strategy', 'balanced')
            strategy_bonus = float(self.strategy_weights.get(strategy, 1.0))

            score = (
                pnl_score * 0.4 +
                success_rate * 0.3 +
                float(frequency_score) * 0.2 +
                strategy_bonus * 0.1 -
                risk_penalty
            )
            return float(np.clip(score, -1.0, 1.0))

        except Exception as e:
            self.logger.warning(f"Scenario scoring failed: {e}")
            return 0.0

    def _calculate_scenario_risk_metrics(self, result: Dict[str, Any],
                                         initial_balance: float) -> Dict[str, Any]:
        """Calculate risk metrics for scenario"""
        try:
            timeline = result.get('timeline', [])
            if not timeline:
                return {'max_drawdown': 0.0, 'volatility': 0.0, 'sharpe': 0.0}

            balances = [float(step.get('balance', initial_balance)) for step in timeline]
            balances.insert(0, float(initial_balance))  # Add initial balance

            running_max = np.maximum.accumulate(balances)
            drawdown = (np.array(balances) - running_max) / running_max
            max_drawdown = float(np.min(drawdown))

            returns = np.diff(balances) / balances[:-1]
            volatility = float(np.std(returns)) if len(returns) > 1 else 0.0
            sharpe = float((np.mean(returns) / volatility * np.sqrt(252))) if volatility > 0 else 0.0

            return {
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'sharpe': sharpe,
                'final_balance': float(balances[-1]),
                'total_return': float((balances[-1] - initial_balance) / initial_balance)
            }

        except Exception as e:
            self.logger.warning(f"Risk metrics calculation failed: {e}")
            return {'max_drawdown': 0.0, 'volatility': 0.0, 'sharpe': 0.0}

    def _calculate_simulation_confidence(self, scenario_results: List[Dict[str, Any]]) -> float:
        """Calculate confidence in simulation results"""
        try:
            if not scenario_results:
                return 0.0
            scores = [float(result.get('score', 0)) for result in scenario_results]
            score_mean = float(np.mean(scores))
            score_std = float(np.std(scores))
            consistency = max(0.0, 1.0 - score_std * 2.0)
            magnitude = (1.0 + np.tanh(score_mean)) / 2.0
            base_conf = (consistency + magnitude) / 2.0
            positive = sum(1 for s in scores if s > 0)
            agreement = positive / len(scores)
            final_confidence = base_conf * 0.7 + agreement * 0.3
            return float(np.clip(final_confidence, 0.0, 1.0))
        except Exception as e:
            self.logger.warning(f"Confidence calculation failed: {e}")
            return 0.5

    def _generate_simulation_recommendations(self, scenario_results: List[Dict[str, Any]],
                                             simulation_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on simulation results"""
        recommendations: List[str] = []
        try:
            if not scenario_results:
                return ["[WARN] No simulation results available for recommendations."]

            best_scenario = max(scenario_results, key=lambda s: s.get('score', 0))
            best_strategy = best_scenario['scenario']['strategy']
            best_score = float(best_scenario.get('score', 0))

            if best_score > 0.5:
                recommendations.append(f"[OK] Strong signal: {best_strategy} strategy shows {best_score:.1%} confidence")
            elif best_score > 0.2:
                recommendations.append(f"[FAST] Moderate signal: {best_strategy} strategy shows potential")
            elif best_score > 0:
                recommendations.append(f"[WARN] Weak signal: Consider {best_strategy} strategy with caution")
            else:
                recommendations.append("🚫 No positive scenarios identified - consider staying neutral")

            best_risk = best_scenario.get('risk_metrics', {}) or {}
            max_dd = float(best_risk.get('max_drawdown', 0))
            if abs(max_dd) > 0.1:
                recommendations.append(f"[WARN] High risk detected: {abs(max_dd):.1%} drawdown expected")
            elif abs(max_dd) > 0.05:
                recommendations.append(f"[FAST] Moderate risk: {abs(max_dd):.1%} drawdown possible")

            regime = simulation_data.get('regime', 'unknown')
            vol_level = simulation_data.get('volatility_level', 'medium')
            if regime == 'volatile' and best_strategy in ['greedy', 'momentum']:
                recommendations.append("[CRASH] Volatile market: Consider more conservative approach despite signal")
            elif regime == 'trending' and best_strategy == 'contrarian':
                recommendations.append("[CHART] Trending market: Contrarian signal may be premature")
            if vol_level == 'extreme':
                recommendations.append("🌪️ Extreme volatility: Reduce position sizes regardless of strategy")

            strategy_scores = {r['scenario']['strategy']: float(r.get('score', 0)) for r in scenario_results}
            top = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            if len(top) > 1:
                top_names = [s.replace('_', ' ').title() for s, _ in top]
                recommendations.append(f"[STATS] Top strategies: {', '.join(top_names)}")

        except Exception as e:
            self.logger.warning(f"Recommendation generation failed: {e}")
            recommendations.append("[WARN] Unable to generate specific recommendations")

        return recommendations[:5]

    def _analyze_simulation_effectiveness(self, simulation_results: Dict[str, Any]) -> None:
        """Analyze simulation effectiveness"""
        try:
            scenarios = simulation_results.get('scenarios', [])
            avg_score = float(simulation_results.get('avg_score', 0.0))
            confidence = float(simulation_results.get('confidence', 0.0))

            self.simulation_stats['prediction_accuracy'] = confidence
            self.simulation_stats['strategy_effectiveness'] = avg_score

            for scenario in scenarios:
                strategy = scenario['scenario']['strategy']
                score = float(scenario.get('score', 0))
                self.strategy_performance[strategy].append(score)

            regime = self.market_regime
            session = self.market_session
            if regime != 'unknown':
                self.regime_performance[regime]['avg_scores'].append(avg_score)
                self.regime_performance[regime]['confidences'].append(confidence)
            if session != 'unknown':
                self.session_performance[session]['avg_scores'].append(avg_score)
                self.session_performance[session]['confidences'].append(confidence)

        except Exception as e:
            self.logger.warning(f"Effectiveness analysis failed: {e}")

    def _update_adaptive_parameters(self, simulation_results: Dict[str, Any]) -> None:
        """Update adaptive parameters based on simulation results"""
        try:
            scenarios = simulation_results.get('scenarios', [])
            for scenario in scenarios:
                strategy = scenario['scenario']['strategy']
                score = float(scenario.get('score', 0))
                current_weight = float(self.strategy_weights.get(strategy, 1.0))
                performance_factor = (score + 1.0) / 2.0  # [-1,1] -> [0,1]
                new_weight = current_weight * (1 - self.learning_rate) + performance_factor * self.learning_rate
                self.strategy_weights[strategy] = new_weight

            total_weight = sum(self.strategy_weights.values())
            if total_weight > 0:
                for s in self.strategy_weights:
                    self.strategy_weights[s] /= total_weight

            new_confidence = float(simulation_results.get('confidence', 0.5))
            self.confidence_score = self.confidence_score * (1 - self.learning_rate) + new_confidence * self.learning_rate

            self.learning_history.append({
                'timestamp': datetime.datetime.now().isoformat(),
                'avg_score': float(simulation_results.get('avg_score', 0.0)),
                'confidence': new_confidence,
                'strategy_weights': self.strategy_weights.copy(),
                'context': simulation_results.get('context', {}).copy()
            })

        except Exception as e:
            self.logger.warning(f"Adaptive parameter update failed: {e}")

    async def _update_smartinfobus_comprehensive(self, simulation_results: Dict[str, Any]):
        """Publish ALL provided keys to SmartInfoBus (contract-strict)."""
        try:
            thesis = simulation_results.get('_thesis') or (
                f"Shadow simulation completed: {len(simulation_results.get('scenarios', []))} "
                f"scenarios, confidence {simulation_results.get('confidence', 0):.1%}"
            )

            # Core objects
            self.smart_bus.set('shadow_simulation', simulation_results.get('shadow_simulation', {}), module='ShadowSimulator', thesis=thesis)
            self.smart_bus.set('simulation_predictions', simulation_results.get('simulation_predictions', {}), module='ShadowSimulator', thesis=thesis)

            # Provided list outputs
            self.smart_bus.set('shadow_predictions', simulation_results.get('shadow_predictions', []), module='ShadowSimulator', thesis=thesis)
            self.smart_bus.set('forward_projections', simulation_results.get('forward_projections', []), module='ShadowSimulator', thesis=thesis)
            self.smart_bus.set('strategy_simulations', simulation_results.get('strategy_simulations', []), module='ShadowSimulator', thesis=thesis)
            self.smart_bus.set('scenario_recommendations', simulation_results.get('scenario_recommendations', []), module='ShadowSimulator', thesis=thesis)

            # Scalars / dicts
            self.smart_bus.set('scenario_analysis', simulation_results.get('scenario_analysis', {}), module='ShadowSimulator', thesis=thesis)
            self.smart_bus.set('simulation_confidence', float(simulation_results.get('simulation_confidence', 0.0)), module='ShadowSimulator', thesis=thesis)

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "smartinfobus_update")
            self.logger.warning(f"SmartInfoBus update failed: {error_context}")

    async def _handle_processing_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle processing errors with intelligent recovery"""
        self.error_count += 1
        error_context = self.error_pinpointer.analyze_error(error, "ShadowSimulator")

        if self.error_count >= self.circuit_breaker_threshold:
            self.is_disabled = True
            self.logger.error(format_operator_message(
                icon="[ALERT]",
                message="ShadowSimulator disabled due to repeated errors",
                error_count=self.error_count,
                threshold=self.circuit_breaker_threshold
            ))

        thesis = "ShadowSimulator encountered an error; safe defaults returned"
        return {
            'scenarios': [],
            'best_scenario': None,
            'worst_scenario': None,
            'avg_score': 0.0,
            'confidence': 0.0,
            'forward_projections': [],
            'scenario_analysis': {
                'avg_score': 0.0,
                'confidence': 0.0,
                'best_strategy': 'unknown',
                'best_score': 0.0,
                'worst_score': 0.0,
                'scenarios_evaluated': 0,
            },
            'scenario_recommendations': [],
            'shadow_predictions': [],
            'shadow_simulation': {
                'simulation_strategy': self.strategy,
                'horizon': self.horizon,
                'adaptive_horizon': self.adaptive_horizon,
                'confidence_score': self.confidence_score,
                'simulation_stats': self.simulation_stats.copy(),
                'strategy_weights': self.strategy_weights.copy(),
                'simulation_results': {
                    'avg_score': 0.0,
                    'confidence': 0.0,
                    'scenarios_evaluated': 0,
                    'best_strategy': 'unknown'
                },
                'recommendations': [],
                'market_context': {
                    'regime': self.market_regime,
                    'volatility_regime': self.volatility_regime,
                    'session': self.market_session
                }
            },
            'simulation_confidence': 0.0,
            'simulation_predictions': {
                'strategy': 'unknown',
                'expected_pnl': 0.0,
                'confidence': 0.0,
                'trade_count_estimate': 0,
                'risk_estimate': 0.0
            },
            'strategy_simulations': [],
            'error': str(error_context),
            'status': 'error',
            '_thesis': thesis,
        }

    def _generate_disabled_response(self) -> Dict[str, Any]:
        """Generate response when module is disabled (still contract-compliant)."""
        return {
            'scenarios': [],
            'best_scenario': None,
            'worst_scenario': None,
            'avg_score': 0.0,
            'confidence': 0.0,
            'forward_projections': [],
            'scenario_analysis': {
                'avg_score': 0.0,
                'confidence': 0.0,
                'best_strategy': 'unknown',
                'best_score': 0.0,
                'worst_score': 0.0,
                'scenarios_evaluated': 0,
            },
            'scenario_recommendations': [],
            'shadow_predictions': [],
            'shadow_simulation': {
                'simulation_strategy': self.strategy,
                'horizon': self.horizon,
                'adaptive_horizon': self.adaptive_horizon,
                'confidence_score': self.confidence_score,
                'simulation_stats': self.simulation_stats.copy(),
                'strategy_weights': self.strategy_weights.copy(),
                'simulation_results': {
                    'avg_score': 0.0,
                    'confidence': 0.0,
                    'scenarios_evaluated': 0,
                    'best_strategy': 'unknown'
                },
                'recommendations': [],
                'market_context': {
                    'regime': self.market_regime,
                    'volatility_regime': self.volatility_regime,
                    'session': self.market_session
                }
            },
            'simulation_confidence': 0.0,
            'simulation_predictions': {
                'strategy': 'unknown',
                'expected_pnl': 0.0,
                'confidence': 0.0,
                'trade_count_estimate': 0,
                'risk_estimate': 0.0
            },
            'strategy_simulations': [],
            'status': 'disabled',
            'reason': 'circuit_breaker_triggered',
            '_thesis': 'ShadowSimulator is temporarily disabled due to repeated errors (circuit breaker). Safe defaults returned.'
        }

    # ================== PUBLIC INTERFACE METHODS ==================

    def simulate_legacy(self, env, actions: np.ndarray) -> List[Dict[str, Any]]:
        """Legacy simulation interface for backward compatibility"""
        try:
            trades: List[Dict[str, Any]] = []
            for step in range(self.horizon):
                instruments = getattr(env, 'instruments', ['EUR/USD', 'XAU/USD'])
                for i, instrument in enumerate(instruments):
                    if len(actions) > i * 2 + 1:
                        action_0 = float(actions[i * 2])
                        action_1 = float(actions[i * 2 + 1])
                        if self.strategy == "greedy":
                            action_0 *= 1.5; action_1 *= 1.5
                        elif self.strategy == "conservative":
                            action_0 *= 0.6; action_1 *= 0.6
                        elif self.strategy == "random":
                            action_0 = float(np.random.uniform(-1, 1))
                            action_1 = float(np.random.uniform(-1, 1))
                        trade = self._execute_simulated_trade(instrument, action_0, action_1, self.strategy, {})
                        if trade:
                            trades.append(trade)
            return trades
        except Exception as e:
            self.logger.error(f"Legacy simulation failed: {e}")
            return []

    def get_observation_components(self) -> np.ndarray:
        """Return simulation features for observation"""
        try:
            strategy_idx = float(list(self.SIMULATION_STRATEGIES.keys()).index(self.strategy))
            recent_score = float(self.simulation_stats.get('strategy_effectiveness', 0.0))
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
            self.logger.error(f"Observation generation failed: {e}")
            return np.array([5.0, 5.0, 0.0, 0.5, 0.6, 0.0, 0.0, 0.0], dtype=np.float32)

    def get_shadow_simulation_report(self) -> str:
        """Generate operator-friendly simulation report"""
        if self.confidence_score > 0.8:      confidence_status = "[OK] High"
        elif self.confidence_score > 0.6:    confidence_status = "[FAST] Good"
        elif self.confidence_score > 0.4:    confidence_status = "[WARN] Moderate"
        else:                                 confidence_status = "[ALERT] Low"

        top_strategies = sorted(self.strategy_weights.items(), key=lambda x: x[1], reverse=True)[:3]
        strategy_lines = [f"  [STATS] {k.replace('_', ' ').title()}: {v:.1%}" for k, v in top_strategies]

        learning_lines = []
        for entry in list(self.learning_history)[-3:]:
            timestamp = entry['timestamp'][:19]
            learning_lines.append(f"  [CHART] {timestamp}: Score {entry['avg_score']:.1%}, Confidence {entry['confidence']:.1%}")

        scenario_lines = []
        for result in list(self.scenario_results)[-3:]:
            scenarios = result.get('scenarios', [])
            scenario_lines.append(f"  🔮 {len(scenarios)} scenarios: Score {result.get('avg_score', 0.0):.1%}, Confidence {result.get('confidence', 0.0):.1%}")

        return f"""
🔮 SHADOW SIMULATOR
═══════════════════════════════════════
[TARGET] Strategy: {self.strategy.title().replace('_', ' ')}
[STATS] Confidence: {confidence_status} ({self.confidence_score:.1%})
🔭 Horizon: Base {self.horizon} | Adaptive {self.adaptive_horizon}
⚙️ Scenarios: {self.scenario_count} per simulation
[TOOL] Status: {'[ALERT] Disabled' if self.is_disabled else '[OK] Healthy'}

[CHART] SIMULATION CONFIGURATION
• Confidence Threshold: {self.confidence_threshold:.1%}
• Risk Scaling: {'[OK] Enabled' if self.risk_scaling else '[FAIL] Disabled'}
• Regime Awareness: {'[OK] Enabled' if self.regime_awareness else '[FAIL] Disabled'}
• Volatility Adjustment: {'[OK] Enabled' if self.volatility_adjustment else '[FAIL] Disabled'}
• Session Sensitivity: {self.session_sensitivity:.1%}
• Learning Rate: {self.learning_rate:.1%}
• Simulation Depth: {self.simulation_depth}

[STATS] PERFORMANCE STATISTICS
• Total Simulations: {self.simulation_stats['total_simulations']:,}
• Scenarios Evaluated: {self.simulation_stats['scenarios_evaluated']:,}
• Prediction Accuracy: {self.simulation_stats['prediction_accuracy']:.1%}
• Strategy Effectiveness: {self.simulation_stats['strategy_effectiveness']:.1%}
• Avg Scenario Score: {self.simulation_stats['avg_scenario_score']:.1%}
• Error Count: {self.error_count}

[TARGET] STRATEGY PERFORMANCE (Top 3)
{chr(10).join(strategy_lines) if strategy_lines else "  📭 No strategy data available"}

[TOOL] ADAPTIVE PARAMETERS
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
• Greedy, Conservative, Adaptive, Contrarian, Momentum, Balanced, Volatility Play, Regime Specific

[TARGET] EFFECTIVENESS METRICS
• Prediction Accuracy: {self.simulation_stats['prediction_accuracy']:.1%}
• Strategy Effectiveness: {self.simulation_stats['strategy_effectiveness']:.1%}
• Learning Progress: {len(self.learning_history)} sessions tracked
• Adaptation Rate: {self.learning_rate:.1%} per session
"""

    # ================== STATE MANAGEMENT ==================

    def get_state(self) -> Dict[str, Any]:
        """Get complete state for hot-reload and persistence"""
        return {
            'module_info': {
                'name': 'ShadowSimulator',
                'version': '3.0.0',
                'last_updated': datetime.datetime.now().isoformat()
            },
            'configuration': {
                'horizon': self.horizon,
                'strategy': self.strategy,
                'confidence_threshold': self.confidence_threshold,
                'risk_scaling': self.risk_scaling,
                'regime_awareness': self.regime_awareness,
                'volatility_adjustment': self.volatility_adjustment,
                'scenario_count': self.scenario_count
            },
            'adaptive_parameters': {
                'adaptive_horizon': self.adaptive_horizon,
                'confidence_score': self.confidence_score,
                'strategy_weights': self.strategy_weights.copy()
            },
            'market_context': {
                'regime': self.market_regime,
                'volatility_regime': self.volatility_regime,
                'session': self.market_session,
                'current_volatility': self.current_volatility
            },
            'system_state': {
                'statistics': self.simulation_stats.copy(),
                'error_count': self.error_count,
                'is_disabled': self.is_disabled
            },
            'history': {
                'scenario_results': list(self.scenario_results)[-10:],
                'learning_history': list(self.learning_history)[-10:],
                'prediction_accuracy': list(self.prediction_accuracy)[-20:]
            }
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set state for hot-reload and persistence"""
        try:
            config = state.get("configuration", {})
            self.horizon = int(config.get("horizon", self.horizon))
            self.strategy = config.get("strategy", self.strategy)
            self.confidence_threshold = float(config.get("confidence_threshold", self.confidence_threshold))
            self.risk_scaling = bool(config.get("risk_scaling", self.risk_scaling))
            self.regime_awareness = bool(config.get("regime_awareness", self.regime_awareness))
            self.volatility_adjustment = bool(config.get("volatility_adjustment", self.volatility_adjustment))
            self.scenario_count = int(config.get("scenario_count", self.scenario_count))

            adaptive = state.get("adaptive_parameters", {})
            self.adaptive_horizon = int(adaptive.get("adaptive_horizon", self.horizon))
            self.confidence_score = float(adaptive.get("confidence_score", 0.5))
            self.strategy_weights.update(adaptive.get("strategy_weights", {}))

            context = state.get("market_context", {})
            self.market_regime = context.get("regime", "normal")
            self.volatility_regime = context.get("volatility_regime", "medium")
            self.market_session = context.get("session", "unknown")
            self.current_volatility = float(context.get("current_volatility", 0.01))

            system_state = state.get("system_state", {})
            self.simulation_stats.update(system_state.get("statistics", {}))
            self.error_count = system_state.get("error_count", 0)
            self.is_disabled = system_state.get("is_disabled", False)

            history = state.get("history", {})
            self.scenario_results.clear()
            for result in history.get("scenario_results", []):
                self.scenario_results.append(result)
            self.learning_history.clear()
            for entry in history.get("learning_history", []):
                self.learning_history.append(entry)
            self.prediction_accuracy.clear()
            for accuracy in history.get("prediction_accuracy", []):
                self.prediction_accuracy.append(accuracy)

            self.logger.info(format_operator_message(
                icon="[RELOAD]",
                message="ShadowSimulator state restored",
                simulations=self.simulation_stats.get('total_simulations', 0),
                scenarios=len(self.scenario_results),
                confidence=f"{self.confidence_score:.1%}"
            ))

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "state_restoration")
            self.logger.error(f"State restoration failed: {error_context}")

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status for monitoring"""
        return {
            'module_name': 'ShadowSimulator',
            'status': 'disabled' if self.is_disabled else 'healthy',
            'error_count': self.error_count,
            'circuit_breaker_threshold': self.circuit_breaker_threshold,
            'total_simulations': self.simulation_stats['total_simulations'],
            'scenarios_evaluated': self.simulation_stats['scenarios_evaluated'],
            'prediction_accuracy': self.simulation_stats['prediction_accuracy'],
            'strategy_effectiveness': self.simulation_stats['strategy_effectiveness'],
            'confidence_score': self.confidence_score,
            'current_strategy': self.strategy
        }

    # ================== LEGACY COMPATIBILITY ==================

    def step(self, **kwargs) -> None:
        """Legacy step interface for backward compatibility"""
        try:
            env = kwargs.get('env')
            actions = kwargs.get('actions')
            if env is not None and actions is not None:
                simulated_trades = self.simulate_legacy(env, actions)
                self.simulation_stats['total_simulations'] += 1
        except Exception as e:
            self.logger.error(f"Legacy step processing failed: {e}")

    def simulate(self, env, actions: np.ndarray) -> List[Dict[str, Any]]:
        """Legacy simulate interface for backward compatibility"""
        return self.simulate_legacy(env, actions)
