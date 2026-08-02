
import datetime
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Tuple

import numpy as np

from modules.contracts import module_args
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.core.mixins import SmartInfoBusStateMixin, SmartInfoBusTradingMixin
from modules.core.module_base import BaseModule, module
from modules.monitoring.performance_tracker import PerformanceTracker
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.info_bus import InfoBusManager
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities


@module(**module_args(
    "BiasAuditor",
    description="Advanced psychological bias detection and correction system with real-time trading behavior analysis",
    error_handling=True,
    hot_reload=True,
    timeout_ms=3000,
))
class BiasAuditor(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):

    def _initialize(self):

        self._initialize_trading_state()
        self._initialize_state_management()
        self._initialize_advanced_systems()


        self.history_len = self.config.get('history_len', 100)
        self.correction_threshold = self.config.get('correction_threshold', 3)
        self.adaptation_rate = self.config.get('adaptation_rate', 0.1)
        self.debug = self.config.get('debug', False)


        self.bias_history = deque(maxlen=self.history_len)
        self.bias_corrections = defaultdict(int)
        self.bias_performance = defaultdict(list)
        self.bias_frequencies = defaultdict(int)


        self.bias_categories = {
            'revenge': {
                'description': 'Trading to recover losses aggressively',
                'threshold': -50.0,
                'weight_reduction': 0.3,
                'detection_algorithm': 'revenge_pattern_analysis'
            },
            'fear': {
                'description': 'Avoiding trades due to recent losses',
                'threshold': -100.0,
                'weight_reduction': 0.2,
                'detection_algorithm': 'fear_avoidance_analysis'
            },
            'greed': {
                'description': 'Overconfident trading after wins',
                'threshold': 100.0,
                'weight_reduction': 0.25,
                'detection_algorithm': 'greed_escalation_analysis'
            },
            'fomo': {
                'description': 'Fear of missing out on trends',
                'threshold': 50.0,
                'weight_reduction': 0.35,
                'detection_algorithm': 'fomo_chasing_analysis'
            },
            'anchoring': {
                'description': 'Fixation on previous price levels',
                'threshold': 0.0,
                'weight_reduction': 0.15,
                'detection_algorithm': 'anchoring_fixation_analysis'
            }
        }


        self.session_stats = {
            'total_biases_detected': 0,
            'biases_corrected': 0,
            'correction_effectiveness': 0.0,
            'most_common_bias': 'none',
            'bias_impact_score': 0.0,
            'session_start': datetime.datetime.now().isoformat()
        }


        self.error_count = 0
        self.circuit_breaker_threshold = 5
        self.is_disabled = False


        self.process_call_count = 0


        self._generate_initialization_thesis()

        version = getattr(self.metadata, 'version', '3.0.0') if self.metadata else '3.0.0'
        self.logger.info(format_operator_message(
            icon="🧠",
            message=f"Bias Auditor v{version} initialized",
            categories=len(self.bias_categories),
            history_length=self.history_len,
            correction_threshold=self.correction_threshold
        ))

    def _initialize_advanced_systems(self):
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="BiasAuditor",
            log_path="logs/strategy/bias_auditor.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("BiasAuditor", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()


    def _generate_initialization_thesis(self):
        thesis = f"""
        Bias Auditor v3.0 Initialization Complete:

        System Configuration:
        - Tracking {len(self.bias_categories)} bias categories: {', '.join(self.bias_categories.keys())}
        - History buffer: {self.history_len} records with adaptive learning
        - Correction threshold: {self.correction_threshold} occurrences before intervention
        - Adaptation rate: {self.adaptation_rate:.1%} for continuous improvement

        Detection Algorithms:
        - Revenge trading: Pattern analysis of post-loss position sizing
        - Fear avoidance: Statistical analysis of trading frequency degradation
        - Greed escalation: Win-streak position size inflation detection
        - FOMO chasing: Trend-following behavior identification
        - Anchoring fixation: Price level clustering analysis

        Advanced Features:
        - Real-time bias strength measurement with contextual adjustments
        - Intelligent correction mechanisms with performance feedback
        - Session-wide bias impact scoring and effectiveness tracking
        - Market regime awareness for context-sensitive thresholds

        Expected Outcomes:
        - Reduced psychological trading errors through early detection
        - Improved decision-making quality via bias corrections
        - Enhanced self-awareness of trading behavior patterns
        - Data-driven approach to psychological risk management
        """

        self.smart_bus.set('bias_auditor_initialization', {
            'status': 'initialized',
            'thesis': thesis,
            'timestamp': datetime.datetime.now().isoformat()
        }, module='BiasAuditor', thesis=thesis)

    async def process(self, **inputs) -> Dict[str, Any]:
        start_time = time.time()
        self.process_call_count += 1

        try:

            if self.is_disabled:
                return self._generate_disabled_response()


            trading_data = await self._get_comprehensive_trading_data()


            bias_analysis = await self._analyze_psychological_biases_comprehensive(trading_data)


            corrections = await self._generate_intelligent_corrections(bias_analysis)
            adjustments = self._calculate_dynamic_adjustments(bias_analysis, corrections)


            thesis = await self._generate_comprehensive_thesis(bias_analysis, corrections, adjustments)


            results = {
                'bias_analysis': bias_analysis,
                'bias_corrections': corrections,
                'bias_adjustments': adjustments,
                'bias_report': self._generate_comprehensive_report(bias_analysis),
                'bias_recommendations': self._generate_intelligent_recommendations(bias_analysis),
                'psychological_state': self._generate_psychological_state_summary({
                    'bias_analysis': bias_analysis,
                    'bias_corrections': corrections
                }),
                'session_performance': self.session_stats.copy(),
                'health_metrics': self._get_health_metrics(),

                'bias_auditor_initialization': self._get_bias_auditor_initialization_view(),
                '_thesis': thesis
            }


            await self._update_smartinfobus_comprehensive(results, thesis)


            processing_time = (time.time() - start_time) * 1000
            self.performance_tracker.record_metric('BiasAuditor', 'process_time', processing_time, True)


            self.error_count = 0

            return results

        except Exception as e:
            return await self._handle_processing_error(e, start_time)

    async def _get_comprehensive_trading_data(self) -> Dict[str, Any]:
        try:
            return {
                'recent_trades': self.smart_bus.get('recent_trades', 'BiasAuditor') or [],
                'current_pnl': self.smart_bus.get('current_pnl', 'BiasAuditor') or 0.0,
                'positions': self.smart_bus.get('positions', 'BiasAuditor') or [],
                'risk_metrics': self.smart_bus.get('risk_data', 'BiasAuditor') or {},
                'session_context': self.smart_bus.get('session_context', 'BiasAuditor') or {},
                'market_regime': self.smart_bus.get('market_regime', 'BiasAuditor') or 'unknown',
                'volatility_level': self.smart_bus.get('volatility_level', 'BiasAuditor') or 'medium',
                'trading_session': self.smart_bus.get('trading_session', 'BiasAuditor') or {}
            }
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "BiasAuditor")
            self.logger.warning(f"Data retrieval incomplete: {error_context}")
            return self._get_safe_trading_defaults()

    async def _analyze_psychological_biases_comprehensive(self, trading_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            bias_signals = {}
            context_factors = {}


            for bias_type, config in self.bias_categories.items():
                algorithm = config['detection_algorithm']
                strength, factors = await self._run_bias_detection_algorithm(
                    algorithm, bias_type, trading_data
                )

                if strength > 0.1:
                    bias_signals[bias_type] = strength
                    context_factors[bias_type] = factors


                    await self._record_bias_detection_comprehensive(
                        bias_type, strength, factors, trading_data
                    )


            aggregate_metrics = self._calculate_aggregate_bias_metrics(bias_signals)

            return {
                'individual_biases': bias_signals,
                'context_factors': context_factors,
                'aggregate_metrics': aggregate_metrics,
                'detection_timestamp': datetime.datetime.now().isoformat(),
                'market_context': {
                    'regime': trading_data.get('market_regime', 'unknown'),
                    'volatility': trading_data.get('volatility_level', 'medium')
                }
            }

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "BiasAuditor")
            self.logger.error(f"Bias analysis failed: {error_context}")
            return self._get_safe_bias_defaults()

    async def _run_bias_detection_algorithm(self, algorithm: str, bias_type: str,
                                          trading_data: Dict[str, Any]) -> Tuple[float, List[str]]:
        try:
            if algorithm == 'revenge_pattern_analysis':
                return await self._detect_revenge_bias_advanced(trading_data)
            elif algorithm == 'fear_avoidance_analysis':
                return await self._detect_fear_bias_advanced(trading_data)
            elif algorithm == 'greed_escalation_analysis':
                return await self._detect_greed_bias_advanced(trading_data)
            elif algorithm == 'fomo_chasing_analysis':
                return await self._detect_fomo_bias_advanced(trading_data)
            elif algorithm == 'anchoring_fixation_analysis':
                return await self._detect_anchoring_bias_advanced(trading_data)
            else:
                self.logger.warning(f"Unknown detection algorithm: {algorithm}")
                return 0.0, ['unknown_algorithm']

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, f"bias_detection_{bias_type}")
            self.logger.warning(f"Detection algorithm failed for {bias_type}: {error_context}")
            return 0.0, ['detection_error']

    async def _detect_revenge_bias_advanced(self, trading_data: Dict[str, Any]) -> Tuple[float, List[str]]:
        try:
            recent_trades = trading_data.get('recent_trades', [])
            current_pnl = trading_data.get('current_pnl', 0)
            factors = []

            if not recent_trades or len(recent_trades) < 2:
                return 0.0, factors


            recent_losses = [t for t in recent_trades[-5:] if t.get('pnl', 0) < 0]
            if len(recent_losses) >= 2:

                sizes = [abs(t.get('notional', 0) or t.get('units', 0) or t.get('lots', 0) * 100000) for t in recent_losses]
                if len(sizes) >= 2 and sizes[0] > 0 and sizes[-1] > sizes[0] * 1.5:
                    escalation_factor = min(1.0, (sizes[-1] / sizes[0] - 1.0) * 0.5)
                    factors.append('position_size_escalation')


            if current_pnl < -100:
                last_hour_trades = len([t for t in recent_trades[-10:]
                                      if self._is_recent_trade(t, minutes=60)])
                if last_hour_trades >= 5:
                    frequency_factor = min(1.0, last_hour_trades / 10.0)
                    factors.append('rapid_trading_frequency')


            risk_escalation = self._analyze_risk_escalation_pattern(recent_trades)
            if risk_escalation > 0.3:
                factors.append('risk_escalation')


            strength = self._calculate_composite_bias_strength(factors, 'revenge')

            return strength, factors

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "revenge_detection")
            return 0.0, ['detection_error']

    async def _detect_fear_bias_advanced(self, trading_data: Dict[str, Any]) -> Tuple[float, List[str]]:
        try:
            recent_trades = trading_data.get('recent_trades', [])
            risk_data = trading_data.get('risk_metrics', {})
            factors = []


            drawdown = risk_data.get('current_drawdown', 0)
            if drawdown > 0.05:
                recent_count = len([t for t in recent_trades[-20:]
                                  if self._is_recent_trade(t, hours=24)])
                expected_trades = 10

                if recent_count < expected_trades * 0.5:
                    frequency_reduction = 1.0 - (recent_count / expected_trades)
                    factors.append('trading_frequency_reduction')


            size_reduction = self._analyze_position_size_trends(recent_trades)
            if size_reduction > 0.3:
                factors.append('position_size_reduction')


            premature_exits = self._detect_premature_profit_taking(recent_trades)
            if premature_exits > 0.4:
                factors.append('premature_profit_taking')

            strength = self._calculate_composite_bias_strength(factors, 'fear')
            return strength, factors

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "fear_detection")
            return 0.0, ['detection_error']

    async def _detect_greed_bias_advanced(self, trading_data: Dict[str, Any]) -> Tuple[float, List[str]]:
        try:
            recent_trades = trading_data.get('recent_trades', [])
            positions_raw = trading_data.get('positions', []) or []
            factors = []


            pos_list: List[Dict[str, Any]] = []
            if isinstance(positions_raw, dict):
                for v in positions_raw.values():
                    if isinstance(v, dict):
                        pos_list.append(v)
            elif isinstance(positions_raw, list):
                for v in positions_raw:
                    if isinstance(v, dict):
                        pos_list.append(v)

            def _to_float_safe(x: Any) -> float:
                try:
                    if isinstance(x, (int, float, np.generic)):
                        return float(x)
                    if isinstance(x, str):
                        s = x.strip()
                        return float(s) if s else 0.0
                except Exception:
                    return 0.0
                return 0.0

            def _pos_exposure_value(p: Dict[str, Any]) -> float:

                for key in (
                    'notional', 'notional_eur', 'notional_usd', 'notional_value', 'exposure',
                    'size', 'units', 'quantity', 'qty', 'volume', 'amount'
                ):
                    if key in p:
                        return _to_float_safe(p.get(key))
                units = _to_float_safe(p.get('units', p.get('size', 0)))
                price = _to_float_safe(p.get('price', p.get('entry_price', 0)))
                if units and price:
                    return units * price
                return units


            recent_wins = [t for t in recent_trades[-5:] if t.get('pnl', 0) > 0]
            if len(recent_wins) >= 3:
                total_exposure = float(sum(abs(_pos_exposure_value(p)) for p in pos_list))
                if total_exposure > 2.0:
                    win_streak = len(recent_wins)
                    factors.append('position_size_inflation')


            stop_discipline = self._analyze_stop_loss_discipline(recent_trades)
            if stop_discipline < 0.3:
                factors.append('reduced_stop_discipline')


            timing_confidence = self._analyze_market_timing_confidence(recent_trades)
            if timing_confidence > 0.7:
                factors.append('overconfident_timing')

            strength = self._calculate_composite_bias_strength(factors, 'greed')
            return strength, factors

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "greed_detection")
            return 0.0, ['detection_error']

    async def _detect_fomo_bias_advanced(self, trading_data: Dict[str, Any]) -> Tuple[float, List[str]]:
        try:
            market_regime = trading_data.get('market_regime', 'unknown')
            recent_trades = trading_data.get('recent_trades', [])
            factors = []


            if market_regime in ('trending', 'trending_up', 'trending_down'):
                recent_entries = len(recent_trades)

                if recent_entries > 8:
                    entry_frequency = min(1.0, (recent_entries - 8) / 12.0)
                    factors.append('excessive_trend_chasing')


            late_entries = self._analyze_entry_timing_quality(recent_trades)
            if late_entries > 0.7:
                factors.append('poor_entry_timing')


            strategy_abandonment = self._detect_strategy_abandonment(recent_trades)
            if strategy_abandonment > 0.6:
                factors.append('strategy_abandonment')

            strength = self._calculate_composite_bias_strength(factors, 'fomo')
            return strength, factors

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "fomo_detection")
            return 0.0, ['detection_error']

    async def _detect_anchoring_bias_advanced(self, trading_data: Dict[str, Any]) -> Tuple[float, List[str]]:
        try:
            recent_trades = trading_data.get('recent_trades', [])
            factors = []

            if not recent_trades:
                return 0.0, factors


            price_levels = [t.get('entry_price', 0) for t in recent_trades[-5:]
                          if t.get('entry_price', 0) > 0]

            if len(price_levels) >= 3:
                price_std = np.std(price_levels)
                price_mean = np.mean(price_levels)

                if price_std / price_mean < 0.01:
                    factors.append('price_level_clustering')


            round_number_bias = self._analyze_round_number_bias(recent_trades)
            if round_number_bias > 0.3:
                factors.append('round_number_fixation')


            historical_bias = self._analyze_historical_price_bias(recent_trades)
            if historical_bias > 0.4:
                factors.append('historical_price_reference')

            strength = self._calculate_composite_bias_strength(factors, 'anchoring')
            return strength, factors

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "anchoring_detection")
            return 0.0, ['detection_error']

    def _calculate_composite_bias_strength(self, factors: List[str], bias_type: str) -> float:
        if not factors:
            return 0.0


        factor_strength = len(factors) / 3.0


        bias_weights = {
            'revenge': 1.2,
            'fear': 0.8,
            'greed': 1.0,
            'fomo': 1.0,
            'anchoring': 0.7
        }

        weight = bias_weights.get(bias_type, 1.0)


        correction_count = self.bias_corrections.get(bias_type, 0)
        correction_factor = 1.0 + (correction_count * 0.1)

        final_strength = min(1.0, factor_strength * weight * correction_factor)
        return final_strength

    async def _record_bias_detection_comprehensive(self, bias_type: str, strength: float,
                                                 factors: List[str], trading_data: Dict[str, Any]):
        try:
            bias_record = {
                'type': bias_type,
                'strength': strength,
                'factors': factors,
                'timestamp': datetime.datetime.now().isoformat(),
                'trading_context': {
                    'pnl': trading_data.get('current_pnl', 0),
                    'positions_count': len(trading_data.get('positions', [])),
                    'recent_trades_count': len(trading_data.get('recent_trades', [])),
                    'market_regime': trading_data.get('market_regime', 'unknown'),
                    'volatility': trading_data.get('volatility_level', 'medium')
                },
                'pnl_impact': 0.0,
                'correction_applied': False
            }

            self.bias_history.append(bias_record)
            self.bias_frequencies[bias_type] += 1
            self.session_stats['total_biases_detected'] += 1


            if strength > 0.5:
                self.logger.warning(format_operator_message(
                    icon="[WARN]",
                    message=f"Strong {bias_type.title()} bias detected",
                    strength=f"{strength:.1%}",
                    factors=", ".join(factors),
                    session_total=self.session_stats['total_biases_detected']
                ))

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "bias_recording")
            self.logger.error(f"Failed to record bias detection: {error_context}")

    async def _generate_intelligent_corrections(self, bias_analysis: Dict[str, Any]) -> Dict[str, Any]:
        try:
            corrections = {}
            individual_biases = bias_analysis.get('individual_biases', {})

            for bias_type, strength in individual_biases.items():
                if strength > 0.3:
                    correction = await self._generate_bias_specific_correction(
                        bias_type, strength, bias_analysis
                    )
                    corrections[bias_type] = correction


                    self.bias_corrections[bias_type] += 1
                    self.session_stats['biases_corrected'] += 1

            return {
                'individual_corrections': corrections,
                'correction_timestamp': datetime.datetime.now().isoformat(),
                'total_corrections_applied': len(corrections),
                'session_correction_count': self.session_stats['biases_corrected']
            }

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "correction_generation")
            self.logger.error(f"Correction generation failed: {error_context}")
            return {'individual_corrections': {}, 'error': str(error_context)}

    async def _generate_bias_specific_correction(self, bias_type: str, strength: float,
                                               bias_analysis: Dict[str, Any]) -> Dict[str, Any]:
        try:
            base_config = self.bias_categories[bias_type]
            context_factors = bias_analysis.get('context_factors', {}).get(bias_type, [])

            correction = {
                'bias_type': bias_type,
                'strength': strength,
                'weight_reduction': base_config['weight_reduction'],
                'adaptive_adjustment': self._calculate_adaptive_adjustment(bias_type, strength),
                'context_factors': context_factors,
                'recommended_actions': self._generate_recommended_actions(bias_type, context_factors),
                'confidence_level': self._calculate_correction_confidence(bias_type, strength)
            }

            return correction

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "specific_correction")
            return {'error': str(error_context), 'bias_type': bias_type}

    def _calculate_dynamic_adjustments(self, bias_analysis: Dict[str, Any],
                                     corrections: Dict[str, Any]) -> Dict[str, float]:
        try:
            adjustments = {}
            individual_biases = bias_analysis.get('individual_biases', {})
            individual_corrections = corrections.get('individual_corrections', {})

            for bias_type in self.bias_categories.keys():

                base_adjustment = 1.0


                if bias_type in individual_corrections:
                    correction = individual_corrections[bias_type]
                    weight_reduction = correction.get('weight_reduction', 0.0)
                    adaptive_adjustment = correction.get('adaptive_adjustment', 0.0)

                    total_reduction = weight_reduction + adaptive_adjustment
                    base_adjustment = 1.0 - min(0.8, total_reduction)


                correction_count = self.bias_corrections.get(bias_type, 0)
                if correction_count >= self.correction_threshold:
                    learning_factor = min(0.2, correction_count * 0.05)
                    base_adjustment -= learning_factor

                adjustments[bias_type] = max(0.2, base_adjustment)

            return adjustments

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "adjustment_calculation")
            self.logger.error(f"Adjustment calculation failed: {error_context}")
            return {bias_type: 1.0 for bias_type in self.bias_categories.keys()}

    async def _generate_comprehensive_thesis(self, bias_analysis: Dict[str, Any],
                                           corrections: Dict[str, Any],
                                           adjustments: Dict[str, float]) -> str:
        try:
            individual_biases = bias_analysis.get('individual_biases', {})
            aggregate_metrics = bias_analysis.get('aggregate_metrics', {})
            market_context = bias_analysis.get('market_context', {})

            thesis_parts = []


            if individual_biases:
                strongest_bias = max(individual_biases.items(), key=lambda x: x[1])
                thesis_parts.append(
                    f"PRIMARY BIAS DETECTION: {strongest_bias[0].title()} bias at {strongest_bias[1]:.1%} strength"
                )
            else:
                thesis_parts.append("NO SIGNIFICANT PSYCHOLOGICAL BIASES DETECTED")


            if individual_biases:
                thesis_parts.append("BIAS BREAKDOWN:")
                for bias_type, strength in individual_biases.items():
                    factors = bias_analysis.get('context_factors', {}).get(bias_type, [])
                    thesis_parts.append(
                        f"  • {bias_type.title()}: {strength:.1%} strength "
                        f"({', '.join(factors) if factors else 'pattern detected'})"
                    )


            regime = market_context.get('regime', 'unknown')
            volatility = market_context.get('volatility', 'medium')
            if regime != 'unknown':
                thesis_parts.append(
                    f"MARKET CONTEXT: {regime.title()} regime with {volatility} volatility "
                    f"influences bias detection thresholds and correction strategies"
                )


            corrections_applied = corrections.get('total_corrections_applied', 0)
            if corrections_applied > 0:
                thesis_parts.append(
                    f"CORRECTIONS APPLIED: {corrections_applied} bias corrections implemented "
                    f"with adaptive weight adjustments"
                )


                for bias_type, correction in corrections.get('individual_corrections', {}).items():
                    actions = correction.get('recommended_actions', [])
                    confidence = correction.get('confidence_level', 0)
                    thesis_parts.append(
                        f"  • {bias_type.title()}: {confidence:.1%} confidence correction "
                        f"({actions[0] if actions else 'weight adjustment'})"
                    )


            significant_adjustments = {k: v for k, v in adjustments.items() if v < 0.9}
            if significant_adjustments:
                thesis_parts.append("WEIGHT ADJUSTMENTS:")
                for bias_type, weight in significant_adjustments.items():
                    reduction = (1.0 - weight) * 100
                    thesis_parts.append(f"  • {bias_type.title()}: {reduction:.0f}% weight reduction")


            if self.session_stats['biases_corrected'] > 0:
                effectiveness = self.session_stats.get('correction_effectiveness', 0)
                impact_score = self.session_stats.get('bias_impact_score', 0)
                thesis_parts.append(
                    f"SESSION PERFORMANCE: {self.session_stats['biases_corrected']} corrections applied "
                    f"with €{effectiveness:.2f} average effectiveness and €{impact_score:.2f} total impact"
                )


            risk_assessment = self._assess_future_bias_risk(bias_analysis)
            thesis_parts.append(f"RISK OUTLOOK: {risk_assessment}")

            return " | ".join(thesis_parts)

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "thesis_generation")
            return f"Bias analysis thesis generation failed: {error_context}"

    async def _update_smartinfobus_comprehensive(self, results: Dict[str, Any], thesis: str):
        try:

            self.smart_bus.set('bias_analysis', results['bias_analysis'],
                             module='BiasAuditor', thesis=thesis)


            corrections_thesis = f"Applied {len(results['bias_corrections'].get('individual_corrections', {}))} bias corrections"
            self.smart_bus.set('bias_corrections', results['bias_corrections'],
                             module='BiasAuditor', thesis=corrections_thesis)


            adjustments_thesis = f"Dynamic weight adjustments: {len([a for a in results['bias_adjustments'].values() if a < 1.0])} biases adjusted"
            self.smart_bus.set('bias_adjustments', results['bias_adjustments'],
                             module='BiasAuditor', thesis=adjustments_thesis)


            self.smart_bus.set('bias_report', results['bias_report'],
                             module='BiasAuditor', thesis="Comprehensive bias analysis report generated")


            recommendations_thesis = f"Generated {len(results['bias_recommendations'])} actionable recommendations"
            self.smart_bus.set('bias_recommendations', results['bias_recommendations'],
                             module='BiasAuditor', thesis=recommendations_thesis)


            psychological_state = self._generate_psychological_state_summary(results)
            self.smart_bus.set('psychological_state', psychological_state,
                             module='BiasAuditor', thesis="Current psychological trading state assessment")


            try:
                init_existing = self.smart_bus.get('bias_auditor_initialization', 'BiasAuditor') or {}
                init_heartbeat = {
                    **init_existing,
                    'status': init_existing.get('status', 'initialized'),
                    'last_update': datetime.datetime.now().isoformat()
                }
                self.smart_bus.set('bias_auditor_initialization', init_heartbeat,
                                 module='BiasAuditor', thesis='BiasAuditor initialization heartbeat update')
            except Exception:
                pass

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "smartinfobus_update")
            self.logger.error(f"SmartInfoBus update failed: {error_context}")

    async def _handle_processing_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        self.error_count += 1
        error_context = self.error_pinpointer.analyze_error(error, "BiasAuditor")


        if self.error_count >= self.circuit_breaker_threshold:
            self.is_disabled = True
            self.logger.error(format_operator_message(
                icon="[ALERT]",
                message="Bias Auditor disabled due to repeated errors",
                error_count=self.error_count,
                threshold=self.circuit_breaker_threshold
            ))


        processing_time = (time.time() - start_time) * 1000
        self.performance_tracker.record_metric('BiasAuditor', 'process_time', processing_time, False)

        return {
            'bias_analysis': {'individual_biases': {}, 'error': str(error_context)},
            'bias_corrections': {'individual_corrections': {}, 'error': str(error_context)},
            'bias_adjustments': {bias: 1.0 for bias in self.bias_categories.keys()},
            'bias_report': f"Bias analysis failed: {error_context}",
            'bias_recommendations': ["Investigate bias auditor system errors"],
            'psychological_state': self._generate_psychological_state_summary({
                'bias_analysis': {'individual_biases': {}, 'aggregate_metrics': {'total_bias_score': 0.0}},
                'bias_corrections': {'individual_corrections': {}}
            }),
            'session_performance': self.session_stats.copy(),
            'health_metrics': {'status': 'error', 'error_context': str(error_context)},
            'bias_auditor_initialization': self._get_bias_auditor_initialization_view(),
            '_thesis': f"BiasAuditor error: {error_context}"
        }


    def _is_recent_trade(self, trade: Dict, minutes: int = 60, hours: int = 0) -> bool:
        try:
            trade_ts = trade.get('ts', 0)
            if not trade_ts:
                return True

            total_seconds = (hours * 3600) + (minutes * 60)
            current_time = time.time()

            return (current_time - trade_ts) <= total_seconds
        except Exception:
            return True

    def _analyze_risk_escalation_pattern(self, trades: List[Dict]) -> float:
        try:
            if len(trades) < 3:
                return 0.0


            sizes = []
            for t in trades[-10:]:
                size = abs(t.get('notional', 0) or t.get('units', 0) or t.get('lots', 0) * 100000)
                if size > 0:
                    sizes.append(size)

            if len(sizes) < 3:
                return 0.0


            increases = sum(1 for i in range(1, len(sizes)) if sizes[i] > sizes[i-1] * 1.1)
            escalation_ratio = increases / (len(sizes) - 1)

            return min(1.0, escalation_ratio)
        except Exception:
            return 0.0

    def _analyze_position_size_trends(self, trades: List[Dict]) -> float:
        try:
            if len(trades) < 3:
                return 0.0


            sizes = []
            for t in trades[-10:]:
                size = abs(t.get('notional', 0) or t.get('units', 0) or t.get('lots', 0) * 100000)
                if size > 0:
                    sizes.append(size)

            if len(sizes) < 3:
                return 0.0


            decreases = sum(1 for i in range(1, len(sizes)) if sizes[i] < sizes[i-1] * 0.9)
            reduction_ratio = decreases / (len(sizes) - 1)

            return min(1.0, reduction_ratio)
        except Exception:
            return 0.0

    def _detect_premature_profit_taking(self, trades: List[Dict]) -> float:
        try:
            if len(trades) < 3:
                return 0.0


            profits = [t.get('pnl', 0) for t in trades[-20:] if t.get('pnl', 0) > 0]

            if len(profits) < 2:
                return 0.0

            avg_profit = np.mean(profits)
            if avg_profit <= 0:
                return 0.0


            small_profits = sum(1 for p in profits if p < avg_profit * 0.3)
            small_profit_ratio = small_profits / len(profits)

            return min(1.0, small_profit_ratio)
        except Exception:
            return 0.0

    def _analyze_stop_loss_discipline(self, trades: List[Dict]) -> float:
        try:
            if len(trades) < 3:
                return 0.5


            losses = [abs(t.get('pnl', 0)) for t in trades[-20:] if t.get('pnl', 0) < 0]
            profits = [t.get('pnl', 0) for t in trades[-20:] if t.get('pnl', 0) > 0]

            if not losses or not profits:
                return 0.5

            avg_loss = np.mean(losses)
            avg_profit = np.mean(profits)


            if avg_profit > 0:
                rr_ratio = float(avg_loss / avg_profit)

                discipline = max(0.0, 1.0 - (rr_ratio / 2.0))
                return float(discipline)

            return 0.5
        except Exception:
            return 0.5

    def _analyze_market_timing_confidence(self, trades: List[Dict]) -> float:
        try:
            if len(trades) < 5:
                return 0.0


            winning_streak = 0
            max_streak = 0
            for t in trades[-15:]:
                if t.get('pnl', 0) > 0:
                    winning_streak += 1
                    max_streak = max(max_streak, winning_streak)
                else:
                    winning_streak = 0


            if max_streak >= 5:
                return min(1.0, max_streak / 7.0)
            elif max_streak >= 3:
                return min(0.5, max_streak / 6.0)

            return 0.0
        except Exception:
            return 0.0

    def _analyze_entry_timing_quality(self, trades: List[Dict]) -> float:
        try:
            if len(trades) < 5:
                return 0.0


            recent = trades[-20:]
            closed_trades = [t for t in recent if abs(t.get('pnl', 0)) > 0.001 or
                            t.get('action', '').startswith('close') or
                            t.get('comment', '') == 'close']

            if len(closed_trades) < 3:
                return 0.0


            losing_trades = sum(1 for t in closed_trades if t.get('pnl', 0) < 0)
            loss_ratio = losing_trades / len(closed_trades)


            if loss_ratio > 0.8:
                return min(1.0, (loss_ratio - 0.8) * 5)
            return 0.0
        except Exception:
            return 0.0

    def _detect_strategy_abandonment(self, trades: List[Dict]) -> float:
        try:
            if len(trades) < 8:
                return 0.0


            recent = trades[-15:]
            instruments = [t.get('instrument', t.get('symbol', '')) for t in recent]

            if len(instruments) < 5:
                return 0.0


            unique_instruments = set(instruments)


            if len(unique_instruments) <= 2:
                return 0.0


            switches = sum(1 for i in range(1, len(instruments)) if instruments[i] != instruments[i-1])
            switch_ratio = switches / (len(instruments) - 1)


            if len(unique_instruments) >= 4 and switch_ratio > 0.7:
                return min(1.0, (switch_ratio - 0.7) * 3.33)

            return 0.0
        except Exception:
            return 0.0

    def _analyze_round_number_bias(self, trades: List[Dict]) -> float:
        try:
            if len(trades) < 3:
                return 0.0

            prices = [t.get('price', 0) for t in trades[-15:] if t.get('price', 0) > 0]

            if len(prices) < 3:
                return 0.0


            round_count = 0
            for p in prices:

                for round_level in [10, 50, 100, 500, 1000]:
                    remainder = p % round_level
                    if remainder < round_level * 0.01 or remainder > round_level * 0.99:
                        round_count += 1
                        break

            round_ratio = round_count / len(prices)
            return min(1.0, round_ratio)
        except Exception:
            return 0.0

    def _analyze_historical_price_bias(self, trades: List[Dict]) -> float:
        try:
            if len(trades) < 5:
                return 0.0

            prices = [t.get('price', 0) for t in trades[-20:] if t.get('price', 0) > 0]

            if len(prices) < 5:
                return 0.0


            avg_price = np.mean(prices)
            if avg_price <= 0:
                return 0.0


            clustered = sum(1 for p in prices if abs(p - avg_price) / avg_price < 0.01)
            cluster_ratio = clustered / len(prices)


            return min(1.0, cluster_ratio)
        except Exception:
            return 0.0


    def _calculate_adaptive_adjustment(self, bias_type: str, strength: float) -> float:
        correction_count = self.bias_corrections.get(bias_type, 0)
        base_adjustment = strength * 0.1
        learning_multiplier = 1.0 + (correction_count * self.adaptation_rate)
        return min(0.3, base_adjustment * learning_multiplier)

    def _generate_recommended_actions(self, bias_type: str, factors: List[str]) -> List[str]:
        action_map = {
            'revenge': ['Take 15-minute break', 'Reduce position sizes', 'Review stop-loss levels'],
            'fear': ['Start with smaller positions', 'Focus on high-probability setups', 'Review risk parameters'],
            'greed': ['Implement profit-taking rules', 'Reduce position sizes', 'Increase stop-loss discipline'],
            'fomo': ['Wait for pullbacks', 'Stick to strategy rules', 'Avoid momentum chasing'],
            'anchoring': ['Review price level analysis', 'Focus on current market conditions', 'Update reference points']
        }
        return action_map.get(bias_type, ['Monitor trading behavior'])

    def _calculate_correction_confidence(self, bias_type: str, strength: float) -> float:
        base_confidence = min(0.9, strength * 1.2)
        historical_success = self._get_historical_correction_success(bias_type)
        return (base_confidence + historical_success) / 2.0

    def _get_historical_correction_success(self, bias_type: str) -> float:
        performance_data = self.bias_performance.get(bias_type, [])
        if not performance_data:
            return 0.5

        positive_outcomes = sum(1 for p in performance_data if p > 0)
        return positive_outcomes / len(performance_data)

    def _calculate_aggregate_bias_metrics(self, bias_signals: Dict[str, float]) -> Dict[str, Any]:
        if not bias_signals:
            return {'total_bias_score': 0.0, 'dominant_bias': 'none', 'bias_diversity': 0.0}

        total_score = sum(bias_signals.values())
        dominant_bias = max(bias_signals.items(), key=lambda x: x[1])
        bias_diversity = len(bias_signals) / len(self.bias_categories)

        return {
            'total_bias_score': total_score,
            'dominant_bias': dominant_bias[0],
            'dominant_strength': dominant_bias[1],
            'bias_diversity': bias_diversity,
            'bias_count': len(bias_signals)
        }

    def _assess_future_bias_risk(self, bias_analysis: Dict[str, Any]) -> str:
        individual_biases = bias_analysis.get('individual_biases', {})
        aggregate_metrics = bias_analysis.get('aggregate_metrics', {})

        total_score = aggregate_metrics.get('total_bias_score', 0)

        if total_score > 2.0:
            return "HIGH RISK - Multiple strong biases detected, implement immediate corrections"
        elif total_score > 1.0:
            return "MODERATE RISK - Monitor bias development and apply preventive measures"
        elif total_score > 0.3:
            return "LOW RISK - Minor bias indicators, maintain awareness"
        else:
            return "MINIMAL RISK - Psychological state appears balanced"

    def _generate_psychological_state_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        bias_analysis = results.get('bias_analysis', {})
        corrections = results.get('bias_corrections', {})

        individual_biases = bias_analysis.get('individual_biases', {})
        aggregate_metrics = bias_analysis.get('aggregate_metrics', {})

        return {
            'overall_state': self._classify_psychological_state(aggregate_metrics),
            'dominant_bias': aggregate_metrics.get('dominant_bias', 'none'),
            'bias_strength': aggregate_metrics.get('total_bias_score', 0),
            'corrections_active': len(corrections.get('individual_corrections', {})),
            'risk_level': self._assess_future_bias_risk(bias_analysis),
            'stability_trend': self._assess_psychological_stability(),
            'last_updated': datetime.datetime.now().isoformat()
        }

    def _classify_psychological_state(self, aggregate_metrics: Dict[str, Any]) -> str:
        total_score = aggregate_metrics.get('total_bias_score', 0)
        bias_diversity = aggregate_metrics.get('bias_diversity', 0)

        if total_score < 0.3:
            return 'optimal'
        elif total_score < 1.0 and bias_diversity < 0.5:
            return 'stable'
        elif total_score < 2.0:
            return 'elevated'
        else:
            return 'compromised'

    def _assess_psychological_stability(self) -> str:
        if len(self.bias_history) < 10:
            return 'insufficient_data'

        recent_strength = [b['strength'] for b in list(self.bias_history)[-5:]]
        older_strength = [b['strength'] for b in list(self.bias_history)[-10:-5]]

        if not recent_strength or not older_strength:
            return 'insufficient_data'

        recent_avg = np.mean(recent_strength)
        older_avg = np.mean(older_strength)

        change = recent_avg - older_avg

        if change > 0.2:
            return 'deteriorating'
        elif change < -0.2:
            return 'improving'
        else:
            return 'stable'

    def _generate_comprehensive_report(self, bias_analysis: Dict[str, Any]) -> str:
        individual_biases = bias_analysis.get('individual_biases', {})
        aggregate_metrics = bias_analysis.get('aggregate_metrics', {})
        market_context = bias_analysis.get('market_context', {})

        active_biases_str = ""
        if individual_biases:
            active_biases_str = "\n".join([
                f"  • {bias.title()}: {strength:.1%}"
                for bias, strength in individual_biases.items()
            ])
        else:
            active_biases_str = "  • No significant biases detected"

        adjustments = self.get_bias_adjustments()
        adjustment_str = "\n".join([
            f"  • {bias.title()}: {adj:.1%} weight"
            for bias, adj in adjustments.items() if adj < 1.0
        ])

        if not adjustment_str:
            adjustment_str = "  • No adjustments required"

        recommendations = self._generate_intelligent_recommendations(bias_analysis)
        rec_str = "\n".join([f"  • {rec}" for rec in recommendations])

        return f"""
🧠 COMPREHENSIVE BIAS ANALYSIS REPORT
═══════════════════════════════════════════════════════════════
[STATS] Session Overview:
• Total Biases Detected: {self.session_stats['total_biases_detected']}
• Biases Corrected: {self.session_stats['biases_corrected']}
• Most Common: {self.session_stats['most_common_bias'].title()}
• Overall Bias Score: {aggregate_metrics.get('total_bias_score', 0):.2f}

[TARGET] Current Market Context:
• Regime: {market_context.get('regime', 'Unknown').title()}
• Volatility: {market_context.get('volatility', 'Unknown').title()}

[WARN] Active Psychological Biases:
{active_biases_str}

[TOOL] Applied Weight Adjustments:
{adjustment_str}

[CHART] Performance Impact:
• Correction Effectiveness: €{self.session_stats['correction_effectiveness']:.2f}
• Total Bias Impact: €{self.session_stats['bias_impact_score']:.2f}
• Psychological State: {self._classify_psychological_state(aggregate_metrics).title()}

[TARGET] Intelligent Recommendations:
{rec_str}

[STATS] Session Statistics:
• Session Duration: {self._calculate_session_duration()}
• Detection Rate: {self._calculate_detection_rate():.1%}
• Correction Success Rate: {self._calculate_correction_success_rate():.1%}
"""

    def _generate_intelligent_recommendations(self, bias_analysis: Dict[str, Any]) -> List[str]:
        individual_biases = bias_analysis.get('individual_biases', {})
        aggregate_metrics = bias_analysis.get('aggregate_metrics', {})
        market_context = bias_analysis.get('market_context', {})

        recommendations = []


        for bias_type, strength in individual_biases.items():
            if strength > 0.5:
                recommendations.extend(self._generate_recommended_actions(bias_type, []))


        total_score = aggregate_metrics.get('total_bias_score', 0)
        if total_score > 2.0:
            recommendations.append("URGENT: Take extended break to reset psychological state")
        elif total_score > 1.0:
            recommendations.append("Consider reducing position sizes across all trades")


        regime = market_context.get('regime', 'unknown')
        if regime == 'volatile' and individual_biases:
            recommendations.append("High volatility amplifies bias effects - exercise extra caution")


        if self.session_stats['biases_corrected'] > 10:
            recommendations.append("High correction frequency suggests need for strategy review")


        if not recommendations:
            recommendations.append("Continue current disciplined approach - psychological state optimal")

        return list(set(recommendations))

    def _get_safe_trading_defaults(self) -> Dict[str, Any]:
        return {
            'recent_trades': [],
            'current_pnl': 0.0,
            'positions': [],
            'risk_metrics': {},
            'session_context': {},
            'market_regime': 'unknown',
            'volatility_level': 'medium',
            'trading_session': {}
        }

    def _get_safe_bias_defaults(self) -> Dict[str, Any]:
        return {
            'individual_biases': {},
            'context_factors': {},
            'aggregate_metrics': {'total_bias_score': 0.0, 'dominant_bias': 'none'},
            'detection_timestamp': datetime.datetime.now().isoformat(),
            'market_context': {'regime': 'unknown', 'volatility': 'medium'},
            'error': 'bias_analysis_failed'
        }

    def _generate_disabled_response(self) -> Dict[str, Any]:
        return {
            'bias_analysis': {'individual_biases': {}, 'status': 'disabled'},
            'bias_corrections': {'individual_corrections': {}, 'status': 'disabled'},
            'bias_adjustments': {bias: 1.0 for bias in self.bias_categories.keys()},
            'bias_report': "Bias Auditor is temporarily disabled due to errors",
            'bias_recommendations': ["Restart bias auditor system", "Check error logs for issues"],
            'psychological_state': self._generate_psychological_state_summary({
                'bias_analysis': {'individual_biases': {}, 'aggregate_metrics': {'total_bias_score': 0.0}},
                'bias_corrections': {'individual_corrections': {}}
            }),
            'session_performance': self.session_stats.copy(),
            'health_metrics': {'status': 'disabled', 'reason': 'circuit_breaker_triggered'},
            'bias_auditor_initialization': self._get_bias_auditor_initialization_view(),
            '_thesis': 'BiasAuditor disabled via circuit breaker'
        }

    def _get_bias_auditor_initialization_view(self) -> Dict[str, Any]:
        try:
            init = self.smart_bus.get('bias_auditor_initialization', 'BiasAuditor') or {}
        except Exception:
            init = {}
        return {
            'status': init.get('status', 'initialized' if not self.is_disabled else 'disabled'),
            'timestamp': init.get('timestamp', datetime.datetime.now().isoformat())
        }

    def _calculate_session_duration(self) -> str:
        try:
            start_time = datetime.datetime.fromisoformat(self.session_stats['session_start'])
            duration = datetime.datetime.now() - start_time

            hours = duration.seconds // 3600
            minutes = (duration.seconds % 3600) // 60

            if hours > 0:
                return f"{hours}h {minutes}m"
            else:
                return f"{minutes}m"
        except Exception:
            return "Unknown"

    def _calculate_detection_rate(self) -> float:

        try:
            process_calls = getattr(self, 'process_call_count', 0)
            if process_calls == 0:
                return 0.0
            return (self.session_stats['total_biases_detected'] / process_calls) * 100
        except Exception:

            return min(100.0, self.session_stats['total_biases_detected'] * 10)

    def _calculate_correction_success_rate(self) -> float:
        if self.session_stats['biases_corrected'] == 0:
            return 0.0

        successful_corrections = sum(
            1 for outcomes in self.bias_performance.values()
            for outcome in outcomes if outcome > 0
        )
        total_corrections = sum(len(outcomes) for outcomes in self.bias_performance.values())

        if total_corrections == 0:
            return 0.0

        return (successful_corrections / total_corrections) * 100

    def get_bias_adjustments(self) -> Dict[str, float]:
        adjustments = {}

        try:
            for bias_type, category in self.bias_categories.items():
                correction_count = self.bias_corrections.get(bias_type, 0)

                if correction_count >= self.correction_threshold:

                    base_reduction = category['weight_reduction']
                    adaptive_reduction = min(0.8, correction_count * 0.1)
                    total_reduction = base_reduction + adaptive_reduction

                    adjustments[bias_type] = max(0.2, 1.0 - total_reduction)
                else:
                    adjustments[bias_type] = 1.0

            return adjustments

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "adjustment_calculation")
            self.logger.error(f"Failed to calculate bias adjustments: {error_context}")
            return {bias_type: 1.0 for bias_type in self.bias_categories.keys()}

    def record_bias_outcome(self, bias_type: str, pnl: float) -> None:
        try:

            if not isinstance(bias_type, str) or bias_type not in self.bias_categories:
                self.logger.warning(f"Invalid bias type: {bias_type}")
                return

            if np.isnan(pnl):
                self.logger.warning("NaN PnL in bias outcome, ignoring")
                return


            self.bias_performance[bias_type].append(pnl)


            if pnl < 0:
                self.bias_corrections[bias_type] += 1
                self.session_stats['biases_corrected'] += 1

                self.logger.info(format_operator_message(
                    icon="📚",
                    message=f"Learning from {bias_type} bias",
                    pnl=f"€{pnl:.2f}",
                    total_corrections=self.bias_corrections[bias_type]
                ))


            for record in reversed(self.bias_history):
                if record['type'] == bias_type and record.get('pnl_impact') == 0.0:
                    record['pnl_impact'] = pnl
                    break


            self._update_session_stats()

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "outcome_recording")
            self.logger.error(f"Failed to record bias outcome: {error_context}")

    def _update_session_stats(self) -> None:
        try:

            if self.bias_frequencies:
                most_common = max(self.bias_frequencies.items(), key=lambda x: x[1])
                self.session_stats['most_common_bias'] = most_common[0]


            total_impact = 0.0
            for outcomes in self.bias_performance.values():
                if outcomes:
                    total_impact += sum(outcomes)

            self.session_stats['bias_impact_score'] = total_impact


            if self.session_stats['biases_corrected'] > 0:
                recent_outcomes = []
                for bias_type, outcomes in self.bias_performance.items():
                    if outcomes:
                        recent_outcomes.extend(outcomes[-5:])

                if recent_outcomes:
                    self.session_stats['correction_effectiveness'] = np.mean(recent_outcomes)

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "session_stats_update")
            self.logger.warning(f"Session stats update failed: {error_context}")

    def get_observation_components(self) -> np.ndarray:
        try:
            if not self.bias_history:

                num_biases = len(self.bias_categories)
                defaults = np.full(num_biases * 2, 0.2, dtype=np.float32)
                return defaults


            total_biases = len(self.bias_history)
            frequencies = []
            for bias_type in self.bias_categories.keys():
                frequency = self.bias_frequencies.get(bias_type, 0) / max(1, total_biases)
                frequencies.append(frequency)


            corrections = []
            max_corrections = max(self.bias_corrections.values()) if self.bias_corrections else 1
            for bias_type in self.bias_categories.keys():
                correction_strength = self.bias_corrections.get(bias_type, 0) / max(1, max_corrections)
                corrections.append(correction_strength)


            observation = np.array(frequencies + corrections, dtype=np.float32)


            if np.any(~np.isfinite(observation)):
                self.logger.error(f"Invalid observation values: {observation}")
                observation = np.nan_to_num(observation, nan=0.2)

            return observation

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "observation_generation")
            self.logger.error(f"Observation generation failed: {error_context}")
            num_biases = len(self.bias_categories)
            return np.full(num_biases * 2, 0.2, dtype=np.float32)

    def _get_health_metrics(self) -> Dict[str, Any]:
        return {
            'module_name': 'BiasAuditor',
            'status': 'disabled' if self.is_disabled else 'healthy',
            'error_count': self.error_count,
            'circuit_breaker_threshold': self.circuit_breaker_threshold,
            'bias_detection_rate': self._calculate_detection_rate(),
            'correction_success_rate': self._calculate_correction_success_rate(),
            'total_biases_tracked': len(self.bias_history),
            'session_duration': self._calculate_session_duration(),
            'psychological_state': self._classify_psychological_state(
                self._calculate_aggregate_bias_metrics(
                    {bias: sum(self.bias_performance.get(bias, [0])[-1:])
                     for bias in self.bias_categories.keys()}
                )
            )
        }


    def get_state(self) -> Dict[str, Any]:
        return {
            'module_info': {
                'name': 'BiasAuditor',
                'version': '3.0.0',
                'last_updated': datetime.datetime.now().isoformat()
            },
            'configuration': {
                'history_len': self.history_len,
                'correction_threshold': self.correction_threshold,
                'adaptation_rate': self.adaptation_rate,
                'debug': self.debug
            },
            'bias_tracking': {
                'bias_history': list(self.bias_history),
                'corrections': dict(self.bias_corrections),
                'performance': {k: list(v) for k, v in self.bias_performance.items()},
                'frequencies': dict(self.bias_frequencies)
            },
            'session_data': self.session_stats.copy(),
            'error_state': {
                'error_count': self.error_count,
                'is_disabled': self.is_disabled
            },
            'process_tracking': {
                'process_call_count': self.process_call_count
            },
            'performance_metrics': self._get_health_metrics(),
            'bias_categories': self.bias_categories.copy()
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        try:

            config = state.get("configuration", {})
            self.history_len = int(config.get("history_len", self.history_len))
            self.correction_threshold = int(config.get("correction_threshold", self.correction_threshold))
            self.adaptation_rate = float(config.get("adaptation_rate", self.adaptation_rate))
            self.debug = bool(config.get("debug", self.debug))


            bias_data = state.get("bias_tracking", {})
            self.bias_history = deque(bias_data.get("bias_history", []), maxlen=self.history_len)
            self.bias_corrections = defaultdict(int, bias_data.get("corrections", {}))


            performance_data = bias_data.get("performance", {})
            self.bias_performance = defaultdict(list)
            for k, v in performance_data.items():
                self.bias_performance[k] = list(v)

            self.bias_frequencies = defaultdict(int, bias_data.get("frequencies", {}))


            self.session_stats = state.get("session_data", self.session_stats)


            error_state = state.get("error_state", {})
            self.error_count = error_state.get("error_count", 0)
            self.is_disabled = error_state.get("is_disabled", False)


            process_tracking = state.get("process_tracking", {})
            self.process_call_count = process_tracking.get("process_call_count", 0)


            self.bias_categories.update(state.get("bias_categories", {}))

            self.logger.info(format_operator_message(
                icon="[RELOAD]",
                message="Bias Auditor state restored",
                biases_tracked=len(self.bias_history),
                corrections_applied=sum(self.bias_corrections.values())
            ))

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "state_restoration")
            self.logger.error(f"State restoration failed: {error_context}")

    def get_health_status(self) -> Dict[str, Any]:
        return {
            'module_name': 'BiasAuditor',
            'status': 'disabled' if self.is_disabled else 'healthy',
            'metrics': self._get_health_metrics(),
            'alerts': self._generate_health_alerts(),
            'recommendations': self._generate_health_recommendations()
        }

    def _generate_health_alerts(self) -> List[Dict[str, Any]]:
        alerts = []

        if self.is_disabled:
            alerts.append({
                'severity': 'critical',
                'message': 'BiasAuditor disabled due to errors',
                'action': 'Investigate error logs and restart module'
            })

        if self.error_count > 2:
            alerts.append({
                'severity': 'warning',
                'message': f'High error count: {self.error_count}',
                'action': 'Monitor for recurring issues'
            })

        detection_rate = self._calculate_detection_rate()
        if detection_rate > 50:
            alerts.append({
                'severity': 'warning',
                'message': f'High bias detection rate: {detection_rate:.1f}%',
                'action': 'Review trading strategy and psychological factors'
            })

        return alerts

    def _generate_health_recommendations(self) -> List[str]:
        recommendations = []

        if self.is_disabled:
            recommendations.append("Restart BiasAuditor module after investigating errors")

        if len(self.bias_history) < 10:
            recommendations.append("Insufficient bias data - continue monitoring to build baseline")

        if self.session_stats['biases_corrected'] > 20:
            recommendations.append("High correction frequency - consider strategy review")

        if not recommendations:
            recommendations.append("BiasAuditor operating within normal parameters")

        return recommendations

    async def calculate_confidence(self, action: Dict[str, Any], **inputs) -> float:
        try:
            if not isinstance(action, dict):
                return 0.5


            base_confidence = 0.7


            action_type = action.get('action_type', 'unknown')
            if action_type == 'bias_correction':

                target_bias = action.get('target_bias', '')
                correction_strength = action.get('correction_strength', action.get('bias_strength', 0.0))
                base_confidence = 0.6 + (correction_strength * 0.3)
            elif action_type == 'bias_mitigation':

                base_confidence = 0.6
            elif action_type == 'bias_monitoring':

                base_confidence = 0.8


            historical_success = self._get_average_correction_success()
            historical_confidence = 0.7 + (historical_success * 0.3)


            final_confidence = (base_confidence + historical_confidence) / 2.0


            if self.error_count > 0:
                error_penalty = min(0.3, self.error_count * 0.05)
                final_confidence -= error_penalty

            return max(0.1, min(1.0, final_confidence))

        except Exception as e:
            self.logger.warning(f"Confidence calculation failed: {e}")
            return 0.5

    async def propose_action(self, **context) -> Dict[str, Any]:
        try:

            bias_analysis = context.get('bias_analysis') or self.smart_bus.get('bias_analysis', 'BiasAuditor')

            if not bias_analysis:
                return {
                    'action_type': 'bias_monitoring',
                    'action': 'initialize_monitoring',
                    'confidence': 0.7,
                    'reasoning': 'Starting bias monitoring - no bias data available'
                }

            individual_biases = bias_analysis.get('individual_biases', {})

            if not individual_biases:
                return {
                    'action_type': 'bias_monitoring',
                    'action': 'continue_monitoring',
                    'confidence': 0.8,
                    'reasoning': 'No biases detected - continue monitoring'
                }


            strongest_bias = max(individual_biases.items(), key=lambda x: x[1])
            bias_type, strength = strongest_bias


            if strength < 0.4:
                return {
                    'action_type': 'bias_monitoring',
                    'target_bias': bias_type,
                    'action': 'light_monitoring',
                    'confidence': 0.7,
                    'reasoning': f'Minor {bias_type} bias ({strength:.1%}) - monitoring only'
                }


            action_proposal = {
                'action_type': 'bias_correction',
                'target_bias': bias_type,
                'bias_strength': strength,
                'urgency': self._calculate_action_urgency(strength),
                'recommended_adjustments': self._get_action_adjustments(bias_type, strength),
                'rationale': f"Strong {bias_type} bias ({strength:.1%}) detected requiring immediate correction",
                'confidence': await self.calculate_confidence({'bias_analysis': bias_analysis}),
                'timestamp': time.time()
            }


            market_context = bias_analysis.get('market_context', {})
            if market_context.get('regime') == 'volatile':
                action_proposal['additional_caution'] = "High volatility amplifies bias effects"

            return action_proposal

        except Exception as e:
            self.logger.error(f"Action proposal failed: {e}")
            return {
                'action_type': 'error_recovery',
                'action': 'fallback_monitoring',
                'confidence': 0.3,
                'reasoning': f'Error in bias analysis: {e!s}'
            }

    def _get_average_correction_success(self) -> float:
        try:
            if not self.bias_performance:
                return 0.5

            all_outcomes = []
            for outcomes in self.bias_performance.values():
                all_outcomes.extend(outcomes)

            if not all_outcomes:
                return 0.5

            positive_outcomes = sum(1 for outcome in all_outcomes if outcome > 0)
            return positive_outcomes / len(all_outcomes)

        except Exception:
            return 0.5

    def _calculate_action_urgency(self, strength: float) -> str:
        if strength > 0.8:
            return 'critical'
        elif strength > 0.6:
            return 'high'
        elif strength > 0.4:
            return 'medium'
        else:
            return 'low'

    def _get_action_adjustments(self, bias_type: str, strength: float) -> Dict[str, Any]:
        base_config = self.bias_categories.get(bias_type, {})
        base_reduction = base_config.get('weight_reduction', 0.2)


        scaled_reduction = base_reduction * (1.0 + strength)

        return {
            'position_size_multiplier': 1.0 - min(0.8, scaled_reduction),
            'risk_reduction_factor': min(0.5, scaled_reduction * 0.5),
            'cooldown_period_minutes': int(strength * 30),
            'monitoring_frequency_increase': strength * 2.0
        }
