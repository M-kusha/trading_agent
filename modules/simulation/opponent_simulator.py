# ─────────────────────────────────────────────────────────────
# File: modules/simulation/opponent_simulator.py
# Enhanced Opponent Simulator with Modern Architecture + Contract Compliance
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
    "OpponentSimulator",
    description="Intelligent market opponent behavior simulation with context-aware perturbations",
    error_handling=True,
    hot_reload=True,
    timeout_ms=3000,
))
class OpponentSimulator(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    Modern opponent simulator with comprehensive SmartInfoBus integration.
    Simulates market opponent behavior by applying intelligent perturbations
    to market data based on market conditions and regime.

    NOTE ON CONTRACTS:
    If your contracts.py uses different key names, update CONTRACT_REQUIRES / CONTRACT_PROVIDES below.
    """

    # ================== CONTRACT SETTINGS ==================
    # If your registry expects different names, tweak them here (single source of truth)
    CONTRACT_REQUIRES = [
        "historical_prices", "market_context", "market_data", "positions", "prices",
        "regime_data", "session_data", "volatility"
    ]
    # Keys that may be present but legitimately empty without triggering warnings
    ALLOW_EMPTY_KEYS = {"positions"}
    CONTRACT_PROVIDES = [
        "adversarial_scenarios", "market_noise", "market_perturbations",
        "opponent_analysis", "opponent_simulation", "perturbation_history",
        "simulated_prices", "simulation_effects", "simulation_statistics"
    ]

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
        "max_perturbation": 0.05,  # absolute price delta cap
        "noise_decay": 0.95
    }

    # ================== LIFECYCLE ==================
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
        self.mode               = mode if mode in self.SIMULATION_MODES else "random"
        self.intensity          = float(intensity)
        self.adaptation_rate    = float(self.sim_config["adaptation_rate"])
        self.context_sensitivity = float(self.sim_config["context_sensitivity"])
        self.regime_multiplier  = float(self.sim_config["regime_multiplier"])
        self.volatility_scaling = bool(self.sim_config["volatility_scaling"])
        self.session_aware      = bool(self.sim_config["session_aware"])
        self.max_perturbation   = float(self.sim_config["max_perturbation"])
        self.noise_decay        = float(self.sim_config["noise_decay"])

        # Reproducible RNG
        self.rng = np.random.RandomState(seed)

        # Enhanced state tracking
        self.simulation_history: deque = deque(maxlen=100)
        self.perturbation_effects: deque = deque(maxlen=50)
        self.regime_adaptations: deque = deque(maxlen=20)

        # Market context awareness
        self.market_regime = "normal"
        self.volatility_regime = "medium"
        self.market_session = "unknown"
        self.current_volatility = 0.01

        # Simulation statistics
        self.simulation_stats: Dict[str, Any] = {
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
        self.simulation_analytics: Dict[str, List[float]] = defaultdict(list)
        self.regime_performance: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

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

    def _initialize(self):
        """Satisfy BaseModule abstract initializer (no-op: real setup in __init__)."""
        return None

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
            icon="[RELOAD]",
            message="Opponent Simulator reset - all state cleared"
        ))

    # ================== CONTRACT HELPERS ==================
    def _validate_contract_inputs(self, md: Dict[str, Any]) -> None:
        """Warn (don’t crash) if required inputs are missing/empty."""
        missing = []
        for k in self.CONTRACT_REQUIRES:
            if k not in md:
                missing.append(k)
                continue
            v = md.get(k)
            # Allow certain keys (like positions) to be empty without noise
            if k in self.ALLOW_EMPTY_KEYS:
                if v is None:
                    missing.append(k)
                continue
            if v in (None, {}, [], ()):  # treat as missing for other keys
                missing.append(k)
        if missing:
            self.logger.warning(f"[Contract] Missing/empty required inputs: {missing}")

    def _contractize_results(
        self,
        market_data: Dict[str, Any],
        simulation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Map internal results to contract 'provides' keys."""
        perturbations = simulation_results.get('perturbations_applied', [])
        effects       = simulation_results.get('simulated_effects', {})
        eff_metrics   = simulation_results.get('effectiveness_metrics', {})
        ctx_adj       = simulation_results.get('context_adjustments', {})

        # Build simulated prices from perturbations
        simulated_prices: Dict[str, float] = {}
        for p in perturbations:
            instrument = p.get('instrument')
            if instrument is not None:
                simulated_prices[instrument] = float(p.get('simulated_price', 0.0))

        # Market noise summary: per-instrument last magnitude (lightweight)
        market_noise = {p['instrument']: float(p.get('magnitude', 0.0)) for p in perturbations}

        # Adversarial scenarios (only meaningful in adversarial mode)
        adversarial_scenarios = []
        if self.mode == 'adversarial':
            for p in perturbations:
                adversarial_scenarios.append({
                    'instrument': p['instrument'],
                    'counter_trend': True,
                    'trend_direction': p.get('trend_direction', 0.0),
                    'magnitude': p.get('magnitude', 0.0),
                    'timestamp': p['timestamp'],
                })

        # Static analysis block
        # Build operator thesis
        thesis = (
            f"OpponentSimulator: {self.mode} | "
            f"perturbations={len(perturbations)} | "
            f"intensity={self.intensity:.2f} (adaptive={self.adaptive_intensity:.2f})"
        )

        payload = {
            'market_perturbations': perturbations,
            'simulation_effects': effects,
            'simulated_prices': simulated_prices,
            'simulation_statistics': self.simulation_stats.copy(),
            'perturbation_history': list(self.perturbation_effects)[-50:],
            'adversarial_scenarios': adversarial_scenarios,
            'market_noise': market_noise,
            '_thesis': thesis,
            'thesis': thesis,
            'opponent_analysis': {
                'mode': self.mode,
                'effectiveness': eff_metrics,
                'context': {
                    'regime': market_data.get('regime', 'unknown'),
                    'session': market_data.get('session', 'unknown'),
                    'volatility_level': market_data.get('volatility_level', 'medium'),
                },
                'notes': 'Counter-trend analysis' if self.mode == 'adversarial' else 'General simulation analysis',
            },
            # IMPORTANT: publish intensity & adaptive_intensity here
            'opponent_simulation': {
                'mode': self.mode,
                'intensity': float(self.intensity),
                'adaptive_intensity': float(self.adaptive_intensity),
                'context_adjustments': ctx_adj,
                'market_context': {
                    'regime': self.market_regime,
                    'volatility_regime': self.volatility_regime,
                    'session': self.market_session
                }
            },
        }
        return payload

    # ================== MAIN PROCESS ==================
    async def process(self, **inputs) -> Dict[str, Any]:
        """Modern async processing with comprehensive simulation"""
        start_time = time.time()

        try:
            # Circuit breaker check
            if self.is_disabled:
                return self._generate_disabled_response()

            # Get comprehensive market data from SmartInfoBus
            market_data = await self._extract_market_data_from_smart_bus()
            self._validate_contract_inputs(market_data)

            # Update market context awareness
            await self._update_market_context(market_data)

            # Apply intelligent perturbations
            simulation_results = await self._apply_context_aware_simulation(market_data)

            # Update adaptive parameters
            self._update_adaptive_parameters(simulation_results)

            # Analyze simulation effectiveness
            self._analyze_simulation_effectiveness(simulation_results)

            # Update SmartInfoBus with results (publishes intensity too)
            await self._update_smartinfobus_comprehensive(simulation_results)

            # Record performance metrics
            processing_time = (time.time() - start_time) * 1000
            self.performance_tracker.record_metric('OpponentSimulator', 'process_time', processing_time, True)

            # Reset error count on successful processing
            self.error_count = 0

            # Return a contract-shaped dict for immediate consumers
            return self._contractize_results(market_data, simulation_results)

        except Exception as e:
            return await self._handle_processing_error(e, start_time)

    # ================== DATA EXTRACTION ==================
    async def _extract_market_data_from_smart_bus(self) -> Dict[str, Any]:
        """Extract all contract-required inputs from SmartInfoBus (robust defaults)."""
        data: Dict[str, Any] = {}
        try:
            get = self.smart_bus.get

            # Contract "requires"
            data['prices']            = get('prices', 'OpponentSimulator') or {}
            data['market_data']       = get('market_data', 'OpponentSimulator') or {}
            data['market_context']    = get('market_context', 'OpponentSimulator') or {}
            data['regime_data']       = get('regime_data', 'OpponentSimulator') or {}
            data['session_data']      = get('session_data', 'OpponentSimulator') or {}
            data['historical_prices'] = get('historical_prices', 'OpponentSimulator') or {}
            # Positions can be published under 'positions' or 'current_positions'; normalize to a list of dicts
            positions_raw = (
                get('positions', 'OpponentSimulator')
                or get('current_positions', 'OpponentSimulator')
                or []
            )
            if isinstance(positions_raw, dict):
                positions_list = list(positions_raw.values())
            elif isinstance(positions_raw, list):
                positions_list = positions_raw
            else:
                positions_list = []
            data['positions'] = positions_list

            # Prefer dedicated 'volatility' key; fall back to context-derived estimate
            vol_scalar = get('volatility', 'OpponentSimulator')
            if isinstance(vol_scalar, (int, float)) and vol_scalar > 0:
                data['volatility'] = float(vol_scalar)
            else:
                ctx_vol = (data['market_context'] or {}).get('volatility', {})
                data['volatility'] = float(np.mean(list(ctx_vol.values()))) if ctx_vol else 0.01

            # Normalize convenient fields we use elsewhere
            mc = data['market_context'] or {}
            data['regime']           = mc.get('regime') or (data['regime_data'] or {}).get('regime', 'unknown')
            data['session']          = mc.get('session') or (data['session_data'] or {}).get('session', 'unknown')
            data['volatility_level'] = mc.get('volatility_level', 'medium')

            # Track current volatility for reports
            self.current_volatility = float(data['volatility'])

        except Exception as e:
            self.logger.warning(f"Market data extraction failed: {e}")
            data = {
                'prices': {}, 'market_data': {}, 'market_context': {},
                'regime_data': {}, 'session_data': {}, 'historical_prices': {},
                'positions': [], 'volatility': 0.01,
                'regime': 'unknown', 'session': 'unknown', 'volatility_level': 'medium'
            }
            self.current_volatility = 0.01

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
                    icon="[STATS]",
                    message=f"Regime change detected: {old_regime} → {self.market_regime}",
                    adaptation="Simulation parameters updated",
                    session=self.market_session
                ))

        except Exception as e:
            self.logger.warning(f"Market context update failed: {e}")

    # ================== SIMULATION ==================
    async def _apply_context_aware_simulation(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply intelligent context-aware market simulation"""
        results: Dict[str, Any] = {
            'mode': self.mode,
            'perturbations_applied': [],
            'effectiveness_metrics': {},
            'context_adjustments': {}
        }

        try:
            # Calculate context-adjusted intensity
            adjusted_intensity = self._calculate_context_adjusted_intensity(market_data)

            # Apply mode-specific simulation
            perturbations = self._generate_mode_specific_perturbations(market_data, adjusted_intensity)

            # Apply perturbations to calculate effects summary
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
            base_intensity = float(self.adaptive_intensity)

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

            # Apply context sensitivity around configured intensity
            context_factor = 1.0 + (base_intensity - self.intensity) * self.context_sensitivity
            final_intensity = self.intensity * context_factor

            # Apply absolute bounds (scaled to price deltas later)
            return float(np.clip(final_intensity, 0.1, self.max_perturbation * 100))

        except Exception as e:
            self.logger.warning(f"Intensity calculation failed: {e}")
            return float(self.intensity)

    def _generate_mode_specific_perturbations(self, market_data: Dict[str, Any], intensity: float) -> List[Dict[str, Any]]:
        """Generate perturbations based on simulation mode"""
        perturbations: List[Dict[str, Any]] = []

        try:
            prices = market_data.get('prices', {})
            volatility = float(market_data.get('volatility', 0.01))

            for instrument, price in prices.items():
                try:
                    price = float(price)
                except Exception:
                    continue
                if price <= 0:
                    continue

                perturbation: Dict[str, Any] = {
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
                    perturbation['rationale'] = "Volatility spike simulation"

                elif self.mode == "liquidity_drain":
                    # Simulated liquidity reduction effects
                    liquidity_impact = volatility * intensity * 2.0  # Amplified impact
                    magnitude = self.rng.normal(0, liquidity_impact)
                    perturbation['type'] = 'liquidity_impact'
                    perturbation['rationale'] = "Liquidity drainage simulation"

                elif self.mode == "news_shock":
                    # Event-driven shocks
                    if self.rng.random() < 0.1:  # 10% chance of shock
                        shock_magnitude = self.rng.choice([-1, 1]) * volatility * intensity * 3.0
                        magnitude = shock_magnitude
                        perturbation['type'] = 'news_shock'
                        perturbation['shock_magnitude'] = shock_magnitude
                        perturbation['rationale'] = "News shock event simulation"
                    else:
                        magnitude = 0.0
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

                # Apply magnitude limits (absolute price delta cap)
                magnitude = float(np.clip(magnitude, -self.max_perturbation, self.max_perturbation))

                perturbation['magnitude'] = magnitude
                perturbation['simulated_price'] = price + magnitude
                perturbation['relative_change'] = (magnitude / price) if price > 0 else 0.0

                if abs(magnitude) > 1e-10:  # Only add non-zero perturbations
                    perturbations.append(perturbation)

        except Exception as e:
            self.logger.error(f"Perturbation generation failed: {e}")

        return perturbations

    def _estimate_trend_direction(self, instrument: str, market_data: Dict[str, Any]) -> float:
        """Estimate trend direction for an instrument"""
        try:
            # Try to get historical prices
            historical = market_data.get('historical_prices', {}).get(instrument, [])
            if isinstance(historical, (list, tuple)) and len(historical) >= 3:
                recent_prices = historical[-3:]
                base = recent_prices[0]
                if base != 0:
                    trend = (recent_prices[-1] - base) / base
                else:
                    trend = 0.0
                return float(np.tanh(trend * 10))  # Bounded between -1 and 1

            # Fallback to random
            return float(self.rng.choice([-1, 0, 1]) * 0.5)

        except Exception:
            return 0.0

    def _calculate_momentum(self, instrument: str, market_data: Dict[str, Any]) -> float:
        """Calculate momentum for an instrument"""
        try:
            # Try to get historical prices
            historical = market_data.get('historical_prices', {}).get(instrument, [])
            if isinstance(historical, (list, tuple)) and len(historical) >= 5:
                last5 = list(historical[-5:])
                # avoid div by zero
                rets = []
                for i in range(1, len(last5)):
                    denom = last5[i-1] if last5[i-1] != 0 else 1e-12
                    rets.append((last5[i] - last5[i-1]) / denom)
                momentum = float(np.mean(rets)) if rets else 0.0
                return float(np.tanh(momentum * 20))  # Bounded momentum

            # Fallback to random
            return float(self.rng.uniform(-0.5, 0.5))

        except Exception:
            return 0.0

    def _apply_perturbations(self, market_data: Dict[str, Any],
                             perturbations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply perturbations and calculate effects"""
        effects: Dict[str, Any] = {
            'total_perturbations': len(perturbations),
            'instruments_affected': set(),
            'total_impact': 0.0,
            'max_impact': 0.0,
            'impact_distribution': defaultdict(list)
        }

        try:
            for perturbation in perturbations:
                instrument = perturbation['instrument']
                magnitude = float(perturbation['magnitude'])

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
            effects['avg_impact'] = (effects['total_impact'] / len(perturbations)) if perturbations else 0.0

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
                float(self.adaptive_intensity) * self.noise_decay +
                float(self.intensity) * (1 - self.noise_decay)
            )

            # Adapt based on effectiveness
            effectiveness = float(simulation_results.get('effectiveness_metrics', {}).get('score', 0.5))
            if effectiveness < 0.3:      # Poor effectiveness
                self.adaptive_intensity *= 1.1  # Increase intensity
            elif effectiveness > 0.8:    # High effectiveness
                self.adaptive_intensity *= 0.95  # Slightly decrease intensity

            # Apply bounds
            self.adaptive_intensity = float(np.clip(self.adaptive_intensity, 0.1, 10.0))

        except Exception as e:
            self.logger.warning(f"Adaptive parameter update failed: {e}")

    def _analyze_simulation_effectiveness(self, simulation_results: Dict[str, Any]) -> None:
        """Analyze simulation effectiveness"""
        try:
            perturbations = simulation_results.get('perturbations_applied', [])
            effects = simulation_results.get('simulated_effects', {})

            # Calculate effectiveness score
            if perturbations:
                mags = [abs(p.get('magnitude', 0.0)) for p in perturbations]
                impact_variance = float(np.var(mags))
                coverage = len(effects.get('instruments_affected', [])) / max(len(perturbations), 1)
                effectiveness = min(1.0, float(impact_variance * 10 + coverage * 0.5))
            else:
                effectiveness = 0.0
                impact_variance = 0.0
                coverage = 0.0

            # Update effectiveness tracking
            self.simulation_stats["effectiveness_score"] = float(effectiveness)
            simulation_results['effectiveness_metrics'] = {
                'score': effectiveness,
                'impact_variance': impact_variance,
                'coverage': coverage
            }

            # Track regime-specific performance
            regime = self.market_regime
            if regime != 'unknown':
                self.regime_performance[regime]['effectiveness_scores'].append(effectiveness)
                # Store timestamps as float epoch seconds to match List[float] typing
                self.regime_performance[regime]['timestamps'].append(time.time())

        except Exception as e:
            self.logger.warning(f"Effectiveness analysis failed: {e}")

    async def _update_smartinfobus_comprehensive(self, simulation_results: Dict[str, Any]):
        """Publish all contract 'provides' to SmartInfoBus (atomic & thesis-tagged)."""
        try:
            thesis = (
                f"OpponentSimulator: {self.mode} | "
                f"perturbations={len(simulation_results.get('perturbations_applied', []))} | "
                f"intensity={self.intensity:.2f} (adaptive={self.adaptive_intensity:.2f})"
            )

            # Build contract payload now that we also have current context on the instance
            market_data = {
                'regime': self.market_regime,
                'session': self.market_session,
                'volatility_level': self.volatility_regime
            }
            contract_payload = self._contractize_results(market_data, simulation_results)

            # Publish every provided key individually (contract-friendly)
            for key in self.CONTRACT_PROVIDES:
                if key in contract_payload:
                    self.smart_bus.set(key, contract_payload[key], module='OpponentSimulator', thesis=thesis)

            # Also keep a compact top-level status for quick consumers (includes intensity)
            self.smart_bus.set('opponent_simulation', contract_payload['opponent_simulation'],
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
                icon="[ALERT]",
                message="OpponentSimulator disabled due to repeated errors",
                error_count=self.error_count,
                threshold=self.circuit_breaker_threshold
            ))

        thesis = f"OpponentSimulator error: {error_context}"
        return {
            'perturbations_applied': [],
            'error': str(error_context),
            'status': 'error',
            '_thesis': thesis,
            'thesis': thesis
        }

    def _generate_disabled_response(self) -> Dict[str, Any]:
        """Generate response when module is disabled"""
        thesis = "OpponentSimulator disabled: circuit_breaker_triggered"
        return {
            'perturbations_applied': [],
            'status': 'disabled',
            'reason': 'circuit_breaker_triggered',
            '_thesis': thesis,
            'thesis': thesis
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
                        market_data['prices'][f"{instrument}_{timeframe}"] = float(df['close'].iloc[-1])

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
            out_dict: Dict[str, Any] = {}
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
                        magnitude = float(matching_perturbation['magnitude'])
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
            effectiveness = float(self.simulation_stats.get('effectiveness_score', 0.0))
            perturbation_rate = min(1.0, float(self.simulation_stats.get('perturbations_applied', 0)) / 100.0)

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
            effectiveness_status = "[OK] Excellent"
        elif self.simulation_stats['effectiveness_score'] > 0.6:
            effectiveness_status = "[FAST] Good"
        elif self.simulation_stats['effectiveness_score'] > 0.4:
            effectiveness_status = "[WARN] Fair"
        else:
            effectiveness_status = "[ALERT] Poor"

        # Mode description
        mode_description = self.SIMULATION_MODES.get(self.mode, "Unknown mode")

        # Recent perturbations
        recent_perturbations = list(self.perturbation_effects)[-5:]
        perturbation_lines = []
        for p in recent_perturbations:
            timestamp = p['timestamp'][:19]
            instrument = p['instrument']
            magnitude = p['magnitude']
            perturbation_lines.append(f"  [STATS] {timestamp}: {instrument} {magnitude:+.5f}")

        # Regime adaptations
        adaptation_lines = []
        for adaptation in list(self.regime_adaptations)[-3:]:
            timestamp = adaptation['timestamp'][:19]
            change = f"{adaptation['from_regime']} → {adaptation['to_regime']}"
            adaptation_lines.append(f"  [RELOAD] {timestamp}: {change}")

        return f"""
🎮 OPPONENT SIMULATOR
═══════════════════════════════════════
[TARGET] Mode: {self.mode.title().replace('_', ' ')} - {mode_description}
[STATS] Effectiveness: {effectiveness_status} ({self.simulation_stats['effectiveness_score']:.1%})
[BALANCE] Intensity: Base {self.intensity:.2f} | Adaptive {self.adaptive_intensity:.2f}
🌐 Market Context: {self.market_regime.title()} regime, {self.volatility_regime} volatility

[CHART] SIMULATION CONFIGURATION
• Context Sensitivity: {self.context_sensitivity:.1%}
• Volatility Scaling: {'[OK] Enabled' if self.volatility_scaling else '[FAIL] Disabled'}
• Session Awareness: {'[OK] Enabled' if self.session_aware else '[FAIL] Disabled'}
• Max Perturbation: {self.max_perturbation:.1%}
• Noise Decay: {self.noise_decay:.1%}
• Regime Multiplier: {self.regime_multiplier:.1f}x

[STATS] PERFORMANCE STATISTICS
• Total Simulations: {self.simulation_stats['total_simulations']:,}
• Perturbations Applied: {self.simulation_stats['perturbations_applied']:,}
• Avg Perturbation Size: {self.simulation_stats['avg_perturbation_size']:.5f}
• Regime Adaptations: {self.simulation_stats['regime_adaptations']}
• Current Volatility: {self.current_volatility:.4f}
• Error Count: {self.error_count}
• Status: {'[ALERT] Disabled' if self.is_disabled else '[OK] Healthy'}

[TOOL] ADAPTIVE PARAMETERS
• Base Intensity: {self.intensity:.2f}
• Adaptive Intensity: {self.adaptive_intensity:.2f}
• Adaptation Rate: {self.adaptation_rate:.1%}
• Current Session: {self.market_session.title()}
• Session Multiplier: {self.session_multipliers.get(self.market_session, 1.0):.1f}x

📜 RECENT PERTURBATIONS
{chr(10).join(perturbation_lines) if perturbation_lines else "  📭 No recent perturbations"}

[RELOAD] REGIME ADAPTATIONS
{chr(10).join(adaptation_lines) if adaptation_lines else "  📭 No recent regime changes"}

💡 SIMULATION MODES AVAILABLE
• Random: Gaussian noise injection
• Adversarial: Counter-trend perturbations
• Trend Follow: Momentum amplification
• Volatility Spike: Volatility clustering
• Liquidity Drain: Reduced liquidity simulation
• News Shock: Event-driven price shocks
• Regime Shift: Market regime transitions

[TARGET] EFFECTIVENESS METRICS
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
                icon="[RELOAD]",
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
            self.context_sensitivity = float(np.clip(
                self.context_sensitivity + self.rng.normal(0, 0.1),
                0.0, 1.0
            ))

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
            'intensity': float(intensity),
            'debug': False,
            'seed': int(self.rng.randint(0, 1000000))
        })

        # Set offspring attributes using setattr to avoid type checker issues
        setattr(offspring, 'context_sensitivity', float(context_sensitivity))
        setattr(offspring, 'volatility_scaling', bool(getattr(self, 'volatility_scaling', True)) if self.rng.random() < 0.5 else bool(getattr(other, 'volatility_scaling', True)))
        setattr(offspring, 'session_aware', bool(getattr(self, 'session_aware', True)) if self.rng.random() < 0.5 else bool(getattr(other, 'session_aware', True)))

        self.logger.info(format_operator_message(
            icon="🔬",
            message="Crossover created offspring",
            mode=mode,
            intensity=f"{float(intensity):.2f}",
            context_sensitivity=f"{float(context_sensitivity):.1%}"
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
