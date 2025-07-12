# ─────────────────────────────────────────────────────────────
# File: modules/simulation/opponent_simulator.py
# Enhanced Opponent Simulator with Modern Architecture
# ─────────────────────────────────────────────────────────────

import numpy as np
import datetime
import time
import copy
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import deque, defaultdict

# Modern imports
from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusStateMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.performance_tracker import PerformanceTracker


@module(
    name="OpponentSimulator",
    version="3.0.0",
    category="simulation",
    provides=[
        "market_perturbations", "simulation_effects", "opponent_analysis", "market_noise",
        "adversarial_scenarios", "simulation_statistics", "perturbation_history"
    ],
    requires=[
        "market_data", "prices", "historical_prices", "positions", "market_context",
        "volatility", "regime_data", "session_data"
    ],
    description="Intelligent market opponent behavior simulation with context-aware perturbations",
    thesis_required=True,
    health_monitoring=True,
    performance_tracking=True,
    error_handling=True,
    timeout_ms=150,
    priority=5,
    explainable=True,
    hot_reload=True
)
class OpponentSimulator(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    Modern opponent simulator with comprehensive SmartInfoBus integration.
    Simulates market opponent behavior by applying intelligent perturbations
    to market data based on market conditions and regime.
    """
    
    # Type annotations for instance attributes
    mode: str
    intensity: float
    context_sensitivity: float
    volatility_scaling: bool
    session_aware: bool
    rng: np.random.RandomState

    # Simulation modes
    SIMULATION_MODES = {
        "random": "Random noise injection",
        "adversarial": "Counter-trend perturbations",
        "trend_follow": "Momentum amplification",
        "volatility_spike": "Volatility clustering",
        "liquidity_drain": "Reduced liquidity simulation",
        "news_shock": "Event-driven price shocks",
        "regime_shift": "Market regime transitions"
    }

    # Enhanced default configuration
    ENHANCED_DEFAULTS = {
        "mode": "random",
        "intensity": 1.0,
        "adaptation_rate": 0.1,
        "context_sensitivity": 0.8,
        "regime_multiplier": 1.5,
        "volatility_scaling": True,
        "session_aware": True,
        "max_perturbation": 0.05,
        "noise_decay": 0.95
    }

    def __init__(
        self,
        mode: str = "random",
        intensity: float = 1.0,
        debug: bool = False,
        seed: Optional[int] = None,
        **kwargs
    ):
        # Initialize BaseModule
        super().__init__(**kwargs)
        
        # Initialize mixins
        self._initialize_trading_state()

        # Merge defaults + any user-supplied overrides
        self.sim_config = copy.deepcopy(self.ENHANCED_DEFAULTS)
        if 'config' in kwargs and isinstance(kwargs['config'], dict):
            self.sim_config.update(kwargs['config'])

        # Core parameters
        self.mode              = mode if mode in self.SIMULATION_MODES else "random"
        self.intensity         = float(intensity)
        self.adaptation_rate   = float(self.sim_config["adaptation_rate"])
        self.context_sensitivity = float(self.sim_config["context_sensitivity"])
        self.regime_multiplier = float(self.sim_config["regime_multiplier"])
        self.volatility_scaling = bool(self.sim_config["volatility_scaling"])
        self.session_aware     = bool(self.sim_config["session_aware"])
        self.max_perturbation  = float(self.sim_config["max_perturbation"])
        self.noise_decay       = float(self.sim_config["noise_decay"])

        # Reproducible RNG
        self.rng = np.random.RandomState(seed)
        
        # Enhanced state tracking
        self.simulation_history = deque(maxlen=100)
        self.perturbation_effects = deque(maxlen=50)
        self.regime_adaptations = deque(maxlen=20)
        
        # Market context awareness
        self.market_regime = "normal"
        self.volatility_regime = "medium"
        self.market_session = "unknown"
        self.current_volatility = 0.01
        
        # Simulation statistics
        self.simulation_stats = {
            "total_simulations": 0,
            "perturbations_applied": 0,
            "avg_perturbation_size": 0.0,
            "regime_adaptations": 0,
            "effectiveness_score": 0.0
        }
        
        # Adaptive parameters
        self.adaptive_intensity = self.intensity
        self.session_multipliers = {
            "asian": 1.2,
            "european": 1.0,
            "american": 1.1,
            "rollover": 1.3
        }
        
        # Performance analytics
        self.simulation_analytics = defaultdict(list)
        self.regime_performance = defaultdict(lambda: defaultdict(list))
        
        # Circuit breaker and error handling
        self.error_count = 0
        self.circuit_breaker_threshold = 5
        self.is_disabled = False

        # Initialize advanced systems
        self._initialize_advanced_systems()
        
        self.logger.info(format_operator_message(
            icon="🎮",
            message="Enhanced Opponent Simulator initialized",
            mode=self.mode,
            intensity=f"{self.intensity:.2f}",
            context_sensitivity=f"{self.context_sensitivity:.1%}",
            volatility_scaling=self.volatility_scaling,
            session_aware=self.session_aware
        ))

    def _initialize_advanced_systems(self):
        """Initialize all modern system components"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="OpponentSimulator",
            log_path="logs/simulation/opponent_simulator.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("OpponentSimulator", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()

    def reset(self) -> None:
        """Enhanced reset with comprehensive state cleanup"""
        super().reset()
        
        # Reset simulation history
        self.simulation_history.clear()
        self.perturbation_effects.clear()
        self.regime_adaptations.clear()
        
        # Reset market context
        self.market_regime = "normal"
        self.volatility_regime = "medium"
        self.market_session = "unknown"
        self.current_volatility = 0.01
        
        # Reset statistics
        self.simulation_stats = {
            "total_simulations": 0,
            "perturbations_applied": 0,
            "avg_perturbation_size": 0.0,
            "regime_adaptations": 0,
            "effectiveness_score": 0.0
        }
        
        # Reset adaptive parameters
        self.adaptive_intensity = self.intensity
        
        # Reset analytics
        self.simulation_analytics.clear()
        self.regime_performance.clear()
        
        # Reset error state
        self.error_count = 0
        self.is_disabled = False
        
        self.logger.info(format_operator_message(
            icon="🔄",
            message="Opponent Simulator reset - all state cleared"
        ))

    async def process(self) -> Dict[str, Any]:
        """Modern async processing with comprehensive simulation"""
        start_time = time.time()
        
        try:
            # Circuit breaker check
            if self.is_disabled:
                return self._generate_disabled_response()
            
            # Get comprehensive market data from SmartInfoBus
            market_data = await self._extract_market_data_from_smart_bus()
            
            # Update market context awareness
            await self._update_market_context(market_data)
            
            # Apply intelligent perturbations
            simulation_results = await self._apply_context_aware_simulation(market_data)
            
            # Update adaptive parameters
            self._update_adaptive_parameters(simulation_results)
            
            # Analyze simulation effectiveness
            self._analyze_simulation_effectiveness(simulation_results)
            
            # Update SmartInfoBus with results
            await self._update_smartinfobus_comprehensive(simulation_results)
            
            # Record performance metrics
            processing_time = (time.time() - start_time) * 1000
            self.performance_tracker.record_metric('OpponentSimulator', 'process_time', processing_time, True)
            
            # Reset error count on successful processing
            self.error_count = 0
            
            return simulation_results
            
        except Exception as e:
            return await self._handle_processing_error(e, start_time)

    async def _extract_market_data_from_smart_bus(self) -> Dict[str, Any]:
        """Extract market data for simulation from SmartInfoBus"""
        
        data = {}
        
        try:
            # Get current prices
            prices = self.smart_bus.get('prices', 'OpponentSimulator') or {}
            data['prices'] = prices
            
            # Get market features
            market_data = self.smart_bus.get('market_data', 'OpponentSimulator') or {}
            data['features'] = market_data.get('features', {})
            
            # Extract volatility data
            market_context = self.smart_bus.get('market_context', 'OpponentSimulator') or {}
            volatilities = market_context.get('volatility', {})
            
            if volatilities:
                self.current_volatility = np.mean(list(volatilities.values()))
            else:
                self.current_volatility = 0.01
                
            data['volatility'] = self.current_volatility
            
            # Get historical data if available
            data['historical_prices'] = self.smart_bus.get('historical_prices', 'OpponentSimulator') or {}
            
            # Get position information for impact calculation
            positions = self.smart_bus.get('positions', 'OpponentSimulator') or []
            data['positions'] = positions
            
            # Get regime and session info
            data['regime'] = market_context.get('regime', 'unknown')
            data['session'] = market_context.get('session', 'unknown')
            data['volatility_level'] = market_context.get('volatility_level', 'medium')
            
        except Exception as e:
            self.logger.warning(f"Market data extraction failed: {e}")
            # Provide safe defaults
            data = {
                'prices': {},
                'features': {},
                'volatility': 0.01,
                'historical_prices': {},
                'positions': [],
                'regime': 'unknown',
                'session': 'unknown',
                'volatility_level': 'medium'
            }
        
        return data

    async def _update_market_context(self, market_data: Dict[str, Any]) -> None:
        """Update market context awareness"""
        
        try:
            # Update regime tracking
            old_regime = self.market_regime
            self.market_regime = market_data.get('regime', 'unknown')
            self.volatility_regime = market_data.get('volatility_level', 'medium')
            self.market_session = market_data.get('session', 'unknown')
            
            # Track regime changes for adaptation
            if self.market_regime != old_regime:
                self.regime_adaptations.append({
                    'timestamp': datetime.datetime.now().isoformat(),
                    'from_regime': old_regime,
                    'to_regime': self.market_regime,
                    'adaptation_applied': True
                })
                
                self.simulation_stats["regime_adaptations"] += 1
                
                self.logger.info(format_operator_message(
                    icon="📊",
                    message=f"Regime change detected: {old_regime} → {self.market_regime}",
                    adaptation="Simulation parameters updated",
                    session=self.market_session
                ))
            
        except Exception as e:
            self.logger.warning(f"Market context update failed: {e}")

    async def _apply_context_aware_simulation(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply intelligent context-aware market simulation"""
        
        results = {
            'mode': self.mode,
            'perturbations_applied': [],
            'effectiveness_metrics': {},
            'context_adjustments': {}
        }
        
        try:
            # Calculate context-adjusted intensity
            adjusted_intensity = self._calculate_context_adjusted_intensity(market_data)
            
            # Apply mode-specific simulation
            perturbations = self._generate_mode_specific_perturbations(
                market_data, adjusted_intensity
            )
            
            # Apply perturbations to market data
            simulated_effects = self._apply_perturbations(market_data, perturbations)
            
            # Track results
            results['perturbations_applied'] = perturbations
            results['simulated_effects'] = simulated_effects
            results['adjusted_intensity'] = adjusted_intensity
            results['context_adjustments'] = self._get_context_adjustments(market_data)
            
            # Update statistics
            self.simulation_stats["total_simulations"] += 1
            self.simulation_stats["perturbations_applied"] += len(perturbations)
            
            if perturbations:
                avg_size = np.mean([abs(p.get('magnitude', 0)) for p in perturbations])
                self.simulation_stats["avg_perturbation_size"] = float(avg_size)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "context_aware_simulation")
            self.logger.error(f"Context-aware simulation failed: {error_context}")
            results['error'] = str(error_context)
        
        return results

    def _calculate_context_adjusted_intensity(self, market_data: Dict[str, Any]) -> float:
        """Calculate context-adjusted simulation intensity"""
        
        try:
            base_intensity = self.adaptive_intensity
            
            # Regime adjustments
            regime = market_data.get('regime', 'unknown')
            if regime == 'volatile':
                base_intensity *= self.regime_multiplier
            elif regime == 'trending':
                base_intensity *= 0.8  # Less perturbation in trending markets
            elif regime == 'ranging':
                base_intensity *= 1.2  # More perturbation in ranging markets
            
            # Volatility adjustments
            if self.volatility_scaling:
                vol_level = market_data.get('volatility_level', 'medium')
                vol_multipliers = {
                    'low': 0.7,
                    'medium': 1.0,
                    'high': 1.4,
                    'extreme': 1.8
                }
                base_intensity *= vol_multipliers.get(vol_level, 1.0)
            
            # Session adjustments
            if self.session_aware:
                session = market_data.get('session', 'unknown')
                base_intensity *= self.session_multipliers.get(session, 1.0)
            
            # Apply context sensitivity
            context_factor = 1.0 + (base_intensity - self.intensity) * self.context_sensitivity
            final_intensity = self.intensity * context_factor
            
            # Apply limits
            return float(np.clip(final_intensity, 0.1, self.max_perturbation * 100))
            
        except Exception as e:
            self.logger.warning(f"Intensity calculation failed: {e}")
            return self.intensity

    def _generate_mode_specific_perturbations(self, market_data: Dict[str, Any], 
                                            intensity: float) -> List[Dict[str, Any]]:
        """Generate perturbations based on simulation mode"""
        
        perturbations = []
        
        try:
            prices = market_data.get('prices', {})
            volatility = market_data.get('volatility', 0.01)
            
            for instrument, price in prices.items():
                if price <= 0:
                    continue
                
                perturbation = {
                    'instrument': instrument,
                    'original_price': price,
                    'timestamp': datetime.datetime.now().isoformat(),
                    'mode': self.mode
                }
                
                if self.mode == "random":
                    # Gaussian noise with volatility scaling
                    noise = self.rng.normal(0, volatility * intensity)
                    magnitude = noise
                    perturbation['type'] = 'gaussian_noise'
                    perturbation['rationale'] = f"Random noise scaled by volatility ({volatility:.4f}) and intensity ({intensity:.2f})"
                
                elif self.mode == "adversarial":
                    # Counter-trend perturbations
                    trend_direction = self._estimate_trend_direction(instrument, market_data)
                    magnitude = -trend_direction * volatility * intensity
                    perturbation['type'] = 'counter_trend'
                    perturbation['trend_direction'] = trend_direction
                    perturbation['rationale'] = f"Adversarial counter-trend (trend: {trend_direction:.2f})"
                
                elif self.mode == "trend_follow":
                    # Momentum amplification
                    momentum = self._calculate_momentum(instrument, market_data)
                    magnitude = momentum * volatility * intensity
                    perturbation['type'] = 'momentum_amplification'
                    perturbation['momentum'] = momentum
                    perturbation['rationale'] = f"Trend following momentum amplification ({momentum:.2f})"
                
                elif self.mode == "volatility_spike":
                    # Volatility clustering
                    spike_intensity = self.rng.exponential(volatility * intensity)
                    magnitude = self.rng.choice([-1, 1]) * spike_intensity
                    perturbation['type'] = 'volatility_spike'
                    perturbation['spike_intensity'] = spike_intensity
                    perturbation['rationale'] = f"Volatility spike simulation ({spike_intensity:.4f})"
                
                elif self.mode == "liquidity_drain":
                    # Simulated liquidity reduction effects
                    liquidity_impact = volatility * intensity * 2.0  # Amplified impact
                    magnitude = self.rng.normal(0, liquidity_impact)
                    perturbation['type'] = 'liquidity_impact'
                    perturbation['rationale'] = f"Liquidity drainage simulation"
                
                elif self.mode == "news_shock":
                    # Event-driven shocks
                    if self.rng.random() < 0.1:  # 10% chance of shock
                        shock_magnitude = self.rng.choice([-1, 1]) * volatility * intensity * 3.0
                        magnitude = shock_magnitude
                        perturbation['type'] = 'news_shock'
                        perturbation['shock_magnitude'] = shock_magnitude
                        perturbation['rationale'] = f"News shock event simulation"
                    else:
                        magnitude = 0
                        perturbation['type'] = 'no_shock'
                        perturbation['rationale'] = "No news shock this step"
                
                elif self.mode == "regime_shift":
                    # Market regime transition effects
                    regime = market_data.get('regime', 'unknown')
                    if regime == 'volatile':
                        magnitude = self.rng.normal(0, volatility * intensity * 2.0)
                    elif regime == 'trending':
                        trend = self._estimate_trend_direction(instrument, market_data)
                        magnitude = trend * volatility * intensity * 0.5
                    else:
                        magnitude = self.rng.normal(0, volatility * intensity)
                    
                    perturbation['type'] = 'regime_shift'
                    perturbation['regime'] = regime
                    perturbation['rationale'] = f"Regime-specific perturbation ({regime})"
                
                else:
                    # Default to random
                    magnitude = self.rng.normal(0, volatility * intensity)
                    perturbation['type'] = 'default_random'
                    perturbation['rationale'] = "Default random perturbation"
                
                # Apply magnitude limits
                magnitude = np.clip(magnitude, -self.max_perturbation, self.max_perturbation)
                
                perturbation['magnitude'] = float(magnitude)
                perturbation['simulated_price'] = price + magnitude
                perturbation['relative_change'] = magnitude / price if price > 0 else 0
                
                if abs(magnitude) > 1e-8:  # Only add non-zero perturbations
                    perturbations.append(perturbation)
                    
        except Exception as e:
            self.logger.error(f"Perturbation generation failed: {e}")
        
        return perturbations

    def _estimate_trend_direction(self, instrument: str, market_data: Dict[str, Any]) -> float:
        """Estimate trend direction for an instrument"""
        
        try:
            # Try to get historical prices
            historical = market_data.get('historical_prices', {}).get(instrument, [])
            if len(historical) >= 3:
                recent_prices = historical[-3:]
                trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
                return float(np.tanh(trend * 10))  # Bounded between -1 and 1
            
            # Fallback to random
            return self.rng.choice([-1, 0, 1]) * 0.5
            
        except Exception:
            return 0.0

    def _calculate_momentum(self, instrument: str, market_data: Dict[str, Any]) -> float:
        """Calculate momentum for an instrument"""
        
        try:
            # Try to get historical prices
            historical = market_data.get('historical_prices', {}).get(instrument, [])
            if len(historical) >= 5:
                returns = np.diff(historical[-5:]) / historical[-5:-1]
                momentum = np.mean(returns)
                return float(np.tanh(momentum * 20))  # Bounded momentum
            
            # Fallback to random
            return self.rng.uniform(-0.5, 0.5)
            
        except Exception:
            return 0.0

    def _apply_perturbations(self, market_data: Dict[str, Any], 
                           perturbations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply perturbations and calculate effects"""
        
        effects = {
            'total_perturbations': len(perturbations),
            'instruments_affected': set(),
            'total_impact': 0.0,
            'max_impact': 0.0,
            'impact_distribution': defaultdict(list)
        }
        
        try:
            for perturbation in perturbations:
                instrument = perturbation['instrument']
                magnitude = perturbation['magnitude']
                
                effects['instruments_affected'].add(instrument)
                effects['total_impact'] += abs(magnitude)
                effects['max_impact'] = max(effects['max_impact'], abs(magnitude))
                effects['impact_distribution'][instrument].append(magnitude)
                
                # Store perturbation effect
                self.perturbation_effects.append({
                    'timestamp': perturbation['timestamp'],
                    'instrument': instrument,
                    'magnitude': magnitude,
                    'mode': self.mode,
                    'context': market_data.get('regime', 'unknown')
                })
            
            effects['instruments_affected'] = list(effects['instruments_affected'])
            effects['avg_impact'] = effects['total_impact'] / len(perturbations) if perturbations else 0.0
            
        except Exception as e:
            self.logger.warning(f"Perturbation application failed: {e}")
            effects['error'] = str(e)
        
        return effects

    def _get_context_adjustments(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get context-based adjustments applied"""
        
        return {
            'regime': market_data.get('regime', 'unknown'),
            'volatility_level': market_data.get('volatility_level', 'medium'),
            'session': market_data.get('session', 'unknown'),
            'regime_multiplier': self.regime_multiplier if market_data.get('regime') == 'volatile' else 1.0,
            'session_multiplier': self.session_multipliers.get(market_data.get('session', 'unknown'), 1.0),
            'volatility_scaling': self.volatility_scaling,
            'context_sensitivity': self.context_sensitivity
        }

    def _update_adaptive_parameters(self, simulation_results: Dict[str, Any]) -> None:
        """Update adaptive parameters based on simulation results"""
        
        try:
            # Decay adaptive intensity towards base intensity
            self.adaptive_intensity = (
                self.adaptive_intensity * self.noise_decay + 
                self.intensity * (1 - self.noise_decay)
            )
            
            # Adapt based on effectiveness
            effectiveness = simulation_results.get('effectiveness_metrics', {}).get('score', 0.5)
            if effectiveness < 0.3:  # Poor effectiveness
                self.adaptive_intensity *= 1.1  # Increase intensity
            elif effectiveness > 0.8:  # High effectiveness
                self.adaptive_intensity *= 0.95  # Slightly decrease intensity
            
            # Apply bounds
            self.adaptive_intensity = np.clip(self.adaptive_intensity, 0.1, 10.0)
            
        except Exception as e:
            self.logger.warning(f"Adaptive parameter update failed: {e}")

    def _analyze_simulation_effectiveness(self, simulation_results: Dict[str, Any]) -> None:
        """Analyze simulation effectiveness"""
        
        try:
            perturbations = simulation_results.get('perturbations_applied', [])
            effects = simulation_results.get('simulated_effects', {})
            
            # Calculate effectiveness score
            if perturbations:
                impact_variance = np.var([abs(p.get('magnitude', 0)) for p in perturbations])
                coverage = len(effects.get('instruments_affected', [])) / max(len(perturbations), 1)
                effectiveness = min(1.0, impact_variance * 10 + coverage * 0.5)
            else:
                effectiveness = 0.0
            
            # Update effectiveness tracking
            self.simulation_stats["effectiveness_score"] = float(effectiveness)
            simulation_results['effectiveness_metrics'] = {
                'score': effectiveness,
                'impact_variance': impact_variance if perturbations else 0.0,
                'coverage': coverage if perturbations else 0.0
            }
            
            # Track regime-specific performance
            regime = self.market_regime
            if regime != 'unknown':
                self.regime_performance[regime]['effectiveness_scores'].append(effectiveness)
                self.regime_performance[regime]['timestamps'].append(
                    datetime.datetime.now().isoformat()
                )
            
        except Exception as e:
            self.logger.warning(f"Effectiveness analysis failed: {e}")

    async def _update_smartinfobus_comprehensive(self, simulation_results: Dict[str, Any]):
        """Update SmartInfoBus with simulation results"""
        try:
            thesis = f"Applied {len(simulation_results.get('perturbations_applied', []))} perturbations in {self.mode} mode"
            
            # Update simulation data
            self.smart_bus.set('opponent_simulation', {
                'mode': self.mode,
                'intensity': self.intensity,
                'adaptive_intensity': self.adaptive_intensity,
                'simulation_stats': self.simulation_stats.copy(),
                'perturbations_applied': len(simulation_results.get('perturbations_applied', [])),
                'effectiveness_score': self.simulation_stats.get('effectiveness_score', 0.0),
                'context_adjustments': simulation_results.get('context_adjustments', {}),
                'market_context': {
                    'regime': self.market_regime,
                    'volatility_regime': self.volatility_regime,
                    'session': self.market_session
                }
            }, module='OpponentSimulator', thesis=thesis)
            
            # Add simulated market data if perturbations were applied
            perturbations = simulation_results.get('perturbations_applied', [])
            if perturbations:
                simulated_prices = {}
                for p in perturbations:
                    simulated_prices[p['instrument']] = p['simulated_price']
                
                # Add simulated prices to SmartInfoBus
                self.smart_bus.set('simulated_prices', simulated_prices, 
                                 module='OpponentSimulator', thesis=thesis)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "smartinfobus_update")
            self.logger.warning(f"SmartInfoBus update failed: {error_context}")

    async def _handle_processing_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle processing errors with intelligent recovery"""
        self.error_count += 1
        error_context = self.error_pinpointer.analyze_error(error, "OpponentSimulator")
        
        # Circuit breaker logic
        if self.error_count >= self.circuit_breaker_threshold:
            self.is_disabled = True
            self.logger.error(format_operator_message(
                icon="🚨",
                message="OpponentSimulator disabled due to repeated errors",
                error_count=self.error_count,
                threshold=self.circuit_breaker_threshold
            ))
        
        return {
            'perturbations_applied': [],
            'error': str(error_context),
            'status': 'error'
        }

    def _generate_disabled_response(self) -> Dict[str, Any]:
        """Generate response when module is disabled"""
        return {
            'perturbations_applied': [],
            'status': 'disabled',
            'reason': 'circuit_breaker_triggered'
        }

    # ================== PUBLIC INTERFACE METHODS ==================

    def apply_legacy(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy interface for backward compatibility"""
        
        try:
            # Convert legacy data format to perturbations
            market_data = {'prices': {}, 'volatility': 0.01, 'regime': 'unknown'}
            
            # Extract prices from data_dict
            for instrument, timeframes in data_dict.items():
                for timeframe, df in timeframes.items():
                    if hasattr(df, 'columns') and 'close' in df.columns and len(df) > 0:
                        market_data['prices'][f"{instrument}_{timeframe}"] = df['close'].iloc[-1]
            
            # Apply simulation (sync version for legacy)
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                simulation_results = loop.run_until_complete(
                    self._apply_context_aware_simulation(market_data)
                )
            except RuntimeError:
                # No event loop running, create new one
                simulation_results = asyncio.run(
                    self._apply_context_aware_simulation(market_data)
                )
            
            # Convert back to legacy format
            out_dict = {}
            perturbations = simulation_results.get('perturbations_applied', [])
            
            for instrument, timeframes in data_dict.items():
                out_dict[instrument] = {}
                for timeframe, df in timeframes.items():
                    df_copy = df.copy()
                    
                    # Find matching perturbation
                    perturbation_key = f"{instrument}_{timeframe}"
                    matching_perturbation = next(
                        (p for p in perturbations if p['instrument'] == perturbation_key), 
                        None
                    )
                    
                    if matching_perturbation and hasattr(df_copy, 'columns') and 'close' in df_copy.columns:
                        # Apply perturbation to close prices
                        magnitude = matching_perturbation['magnitude']
                        df_copy['close'] = df_copy['close'] + magnitude
                    
                    out_dict[instrument][timeframe] = df_copy
            
            return out_dict
            
        except Exception as e:
            self.logger.error(f"Legacy apply failed: {e}")
            return data_dict  # Return original on error

    def get_observation_components(self) -> np.ndarray:
        """Return simulation features for observation"""
        
        try:
            mode_idx = float(list(self.SIMULATION_MODES.keys()).index(self.mode))
            effectiveness = self.simulation_stats.get('effectiveness_score', 0.0)
            perturbation_rate = min(1.0, self.simulation_stats.get('perturbations_applied', 0) / 100.0)
            
            return np.array([
                float(self.intensity),
                float(self.adaptive_intensity),
                mode_idx / len(self.SIMULATION_MODES),
                float(effectiveness),
                float(perturbation_rate),
                float(self.context_sensitivity)
            ], dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Observation generation failed: {e}")
            return np.array([1.0, 1.0, 0.0, 0.5, 0.0, 0.8], dtype=np.float32)

    def get_opponent_simulation_report(self) -> str:
        """Generate operator-friendly simulation report"""
        
        # Status indicators
        if self.simulation_stats['effectiveness_score'] > 0.8:
            effectiveness_status = "✅ Excellent"
        elif self.simulation_stats['effectiveness_score'] > 0.6:
            effectiveness_status = "⚡ Good"
        elif self.simulation_stats['effectiveness_score'] > 0.4:
            effectiveness_status = "⚠️ Fair"
        else:
            effectiveness_status = "🚨 Poor"
        
        # Mode description
        mode_description = self.SIMULATION_MODES.get(self.mode, "Unknown mode")
        
        # Recent perturbations
        recent_perturbations = list(self.perturbation_effects)[-5:]
        perturbation_lines = []
        for p in recent_perturbations:
            timestamp = p['timestamp'][:19]
            instrument = p['instrument']
            magnitude = p['magnitude']
            perturbation_lines.append(f"  📊 {timestamp}: {instrument} {magnitude:+.5f}")
        
        # Regime adaptations
        adaptation_lines = []
        for adaptation in list(self.regime_adaptations)[-3:]:
            timestamp = adaptation['timestamp'][:19]
            change = f"{adaptation['from_regime']} → {adaptation['to_regime']}"
            adaptation_lines.append(f"  🔄 {timestamp}: {change}")
        
        return f"""
🎮 OPPONENT SIMULATOR
═══════════════════════════════════════
🎯 Mode: {self.mode.title().replace('_', ' ')} - {mode_description}
📊 Effectiveness: {effectiveness_status} ({self.simulation_stats['effectiveness_score']:.1%})
⚖️ Intensity: Base {self.intensity:.2f} | Adaptive {self.adaptive_intensity:.2f}
🌐 Market Context: {self.market_regime.title()} regime, {self.volatility_regime} volatility

📈 SIMULATION CONFIGURATION
• Context Sensitivity: {self.context_sensitivity:.1%}
• Volatility Scaling: {'✅ Enabled' if self.volatility_scaling else '❌ Disabled'}
• Session Awareness: {'✅ Enabled' if self.session_aware else '❌ Disabled'}
• Max Perturbation: {self.max_perturbation:.1%}
• Noise Decay: {self.noise_decay:.1%}
• Regime Multiplier: {self.regime_multiplier:.1f}x

📊 PERFORMANCE STATISTICS
• Total Simulations: {self.simulation_stats['total_simulations']:,}
• Perturbations Applied: {self.simulation_stats['perturbations_applied']:,}
• Avg Perturbation Size: {self.simulation_stats['avg_perturbation_size']:.5f}
• Regime Adaptations: {self.simulation_stats['regime_adaptations']}
• Current Volatility: {self.current_volatility:.4f}
• Error Count: {self.error_count}
• Status: {'🚨 Disabled' if self.is_disabled else '✅ Healthy'}

🔧 ADAPTIVE PARAMETERS
• Base Intensity: {self.intensity:.2f}
• Adaptive Intensity: {self.adaptive_intensity:.2f}
• Adaptation Rate: {self.adaptation_rate:.1%}
• Current Session: {self.market_session.title()}
• Session Multiplier: {self.session_multipliers.get(self.market_session, 1.0):.1f}x

📜 RECENT PERTURBATIONS
{chr(10).join(perturbation_lines) if perturbation_lines else "  📭 No recent perturbations"}

🔄 REGIME ADAPTATIONS
{chr(10).join(adaptation_lines) if adaptation_lines else "  📭 No recent regime changes"}

💡 SIMULATION MODES AVAILABLE
• Random: Gaussian noise injection
• Adversarial: Counter-trend perturbations
• Trend Follow: Momentum amplification
• Volatility Spike: Volatility clustering
• Liquidity Drain: Reduced liquidity simulation
• News Shock: Event-driven price shocks
• Regime Shift: Market regime transitions

🎯 EFFECTIVENESS METRICS
• Current Score: {self.simulation_stats['effectiveness_score']:.1%}
• Impact Distribution: {len(self.perturbation_effects)} recorded effects
• Adaptation Success: {len(self.regime_adaptations)} regime changes handled
        """

    # ================== STATE MANAGEMENT ==================

    def get_state(self) -> Dict[str, Any]:
        """Get complete state for hot-reload and persistence"""
        return {
            'module_info': {
                'name': 'OpponentSimulator',
                'version': '3.0.0',
                'last_updated': datetime.datetime.now().isoformat()
            },
            'configuration': {
                'mode': self.mode,
                'intensity': self.intensity,
                'context_sensitivity': self.context_sensitivity,
                'volatility_scaling': self.volatility_scaling,
                'session_aware': self.session_aware
            },
            'adaptive_parameters': {
                'adaptive_intensity': self.adaptive_intensity,
                'current_volatility': self.current_volatility
            },
            'market_context': {
                'regime': self.market_regime,
                'volatility_regime': self.volatility_regime,
                'session': self.market_session
            },
            'system_state': {
                'statistics': self.simulation_stats.copy(),
                'error_count': self.error_count,
                'is_disabled': self.is_disabled
            },
            'history': {
                'perturbation_effects': list(self.perturbation_effects)[-20:],
                'regime_adaptations': list(self.regime_adaptations)[-10:]
            }
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set state for hot-reload and persistence"""
        
        try:
            # Load configuration
            config = state.get("configuration", {})
            self.mode = config.get("mode", self.mode)
            self.intensity = float(config.get("intensity", self.intensity))
            self.context_sensitivity = float(config.get("context_sensitivity", self.context_sensitivity))
            self.volatility_scaling = bool(config.get("volatility_scaling", self.volatility_scaling))
            self.session_aware = bool(config.get("session_aware", self.session_aware))
            
            # Load adaptive parameters
            adaptive = state.get("adaptive_parameters", {})
            self.adaptive_intensity = float(adaptive.get("adaptive_intensity", self.intensity))
            self.current_volatility = float(adaptive.get("current_volatility", 0.01))
            
            # Load market context
            context = state.get("market_context", {})
            self.market_regime = context.get("regime", "normal")
            self.volatility_regime = context.get("volatility_regime", "medium")
            self.market_session = context.get("session", "unknown")
            
            # Load system state
            system_state = state.get("system_state", {})
            self.simulation_stats.update(system_state.get("statistics", {}))
            self.error_count = system_state.get("error_count", 0)
            self.is_disabled = system_state.get("is_disabled", False)
            
            # Load history
            history = state.get("history", {})
            perturbation_effects = history.get("perturbation_effects", [])
            regime_adaptations = history.get("regime_adaptations", [])
            
            self.perturbation_effects.clear()
            for effect in perturbation_effects:
                self.perturbation_effects.append(effect)
                
            self.regime_adaptations.clear()
            for adaptation in regime_adaptations:
                self.regime_adaptations.append(adaptation)
            
            self.logger.info(format_operator_message(
                icon="🔄",
                message="OpponentSimulator state restored",
                simulations=self.simulation_stats.get('total_simulations', 0),
                perturbations=len(self.perturbation_effects)
            ))
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "state_restoration")
            self.logger.error(f"State restoration failed: {error_context}")

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status for monitoring"""
        return {
            'module_name': 'OpponentSimulator',
            'status': 'disabled' if self.is_disabled else 'healthy',
            'error_count': self.error_count,
            'circuit_breaker_threshold': self.circuit_breaker_threshold,
            'total_simulations': self.simulation_stats['total_simulations'],
            'perturbations_applied': self.simulation_stats['perturbations_applied'],
            'effectiveness_score': self.simulation_stats['effectiveness_score'],
            'mode': self.mode,
            'adaptive_intensity': self.adaptive_intensity
        }

    # ================== EVOLUTIONARY METHODS ==================

    def mutate(self, std: float = 0.2) -> None:
        """Mutate simulation parameters"""
        
        old_intensity = self.intensity
        old_mode = self.mode
        
        # Mutate intensity
        self.intensity = float(np.clip(
            self.intensity + self.rng.normal(0, std), 
            0.05, 10.0
        ))
        
        # Mutate mode occasionally
        if self.rng.random() < 0.2:
            self.mode = self.rng.choice(list(self.SIMULATION_MODES.keys()))
        
        # Mutate other parameters
        if self.rng.random() < 0.1:
            self.context_sensitivity = np.clip(
                self.context_sensitivity + self.rng.normal(0, 0.1),
                0.0, 1.0
            )
        
        self.logger.info(format_operator_message(
            icon="🧬",
            message="Mutation applied",
            intensity=f"{old_intensity:.2f} → {self.intensity:.2f}",
            mode=f"{old_mode} → {self.mode}" if old_mode != self.mode else "unchanged"
        ))

    def crossover(self, other: "OpponentSimulator") -> "OpponentSimulator":
        """Create offspring through crossover"""
        
        # Select parameters from parents using getattr for type safety
        mode = getattr(self, 'mode', 'random') if self.rng.random() < 0.5 else getattr(other, 'mode', 'random')
        intensity = getattr(self, 'intensity', 1.0) if self.rng.random() < 0.5 else getattr(other, 'intensity', 1.0)
        context_sensitivity = (getattr(self, 'context_sensitivity', 0.8) + getattr(other, 'context_sensitivity', 0.8)) / 2
        
        # Create offspring using kwargs to avoid type checker issues
        offspring = OpponentSimulator(**{
            'mode': mode,
            'intensity': intensity,
            'debug': False,
            'seed': self.rng.randint(0, 1000000)
        })
        
        # Set offspring attributes using setattr to avoid type checker issues
        setattr(offspring, 'context_sensitivity', context_sensitivity)
        setattr(offspring, 'volatility_scaling', getattr(self, 'volatility_scaling', True) if self.rng.random() < 0.5 else getattr(other, 'volatility_scaling', True))
        setattr(offspring, 'session_aware', getattr(self, 'session_aware', True) if self.rng.random() < 0.5 else getattr(other, 'session_aware', True))
        
        self.logger.info(format_operator_message(
            icon="🔬",
            message="Crossover created offspring",
            mode=mode,
            intensity=f"{intensity:.2f}",
            context_sensitivity=f"{context_sensitivity:.1%}"
        ))
        
        return offspring

    # ================== LEGACY COMPATIBILITY ==================

    def step(self, **kwargs) -> None:
        """Legacy step interface for backward compatibility"""
        try:
            # Legacy mode processing
            data_dict = kwargs.get('data_dict', {})
            if data_dict:
                # Apply legacy simulation
                simulated_data = self.apply_legacy(data_dict)
                self.simulation_stats['total_simulations'] += 1
                
        except Exception as e:
            self.logger.error(f"Legacy step processing failed: {e}")