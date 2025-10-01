"""
🕐 Enhanced Time Horizon Aligner with SmartInfoBus Integration v3.1
Advanced time-based weight scaling for voting committees with market adaptation
"""

import asyncio
import time
from modules.contracts import module_args
import numpy as np
import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import deque, defaultdict

# ═══════════════════════════════════════════════════════════════════
# MODERN SMARTINFOBUS IMPORTS
# ═══════════════════════════════════════════════════════════════════
from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusStateMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.health_monitor import HealthMonitor
from modules.monitoring.performance_tracker import PerformanceTracker
from modules.utils.session_utils import normalize_session_name


@module(**module_args(
    "TimeHorizonAligner",
    description="Advanced time-based weight scaling for voting committees with market adaptation",
    error_handling=True,
    hot_reload=True,
    timeout_ms=3000,
))
class TimeHorizonAligner(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    🕐 PRODUCTION-GRADE Time Horizon Aligner v3.1
    """

    # ------------------------------ INIT ------------------------------
    def _initialize(self):
        # mixins
        self._initialize_trading_state()
        self._initialize_state_management()
        self._initialize_advanced_systems()

        # core horizon config
        default_horizons = [1, 3, 5, 10, 15, 30, 60, 120, 240]
        self.horizons = np.array(self.config.get('horizons', default_horizons), dtype=np.float32)
        self.adaptive_scaling = bool(self.config.get('adaptive_scaling', True))
        self.regime_awareness = bool(self.config.get('regime_awareness', True))
        self.performance_feedback = bool(self.config.get('performance_feedback', True))
        self.debug = bool(self.config.get('debug', False))

        # time & session
        self.clock = 0
        self.session_start = 0
        self.last_alignment_time = datetime.datetime.now()

        # alignment state
        self.current_distances = np.ones_like(self.horizons)
        self.base_distances = np.ones_like(self.horizons)
        self.adaptive_multipliers = np.ones_like(self.horizons)
        self.performance_multipliers = np.ones_like(self.horizons)
        self.cyclical_adjustments = np.ones_like(self.horizons)

        # regime multipliers
        self.regime_multipliers = {
            'trending': np.ones_like(self.horizons),
            'volatile': np.ones_like(self.horizons),
            'ranging': np.ones_like(self.horizons),
            'noise': np.ones_like(self.horizons),
            'breakout': np.ones_like(self.horizons),
            'reversal': np.ones_like(self.horizons),
            'unknown': np.ones_like(self.horizons)
        }

        # session patterns (includes 'overlap'); add 'closed' canonical
        self.session_patterns = {
            'american': np.ones_like(self.horizons),
            'european': np.ones_like(self.horizons),
            'asian': np.ones_like(self.horizons),
            'closed': np.ones_like(self.horizons),
            'rollover': np.ones_like(self.horizons),  # legacy alias
            'weekend': np.ones_like(self.horizons),
            'overlap': np.ones_like(self.horizons),
            'unknown': np.ones_like(self.horizons)
        }

        # market state
        self.current_regime = 'unknown'
        self.current_session = 'unknown'
        self.current_volatility = 0.02
        self.volatility_history = deque(maxlen=50)

        # intelligence
        self.alignment_intelligence = {
            'learning_rate': 0.12,
            'adaptation_threshold': 0.15,
            'performance_window': 20,
            'regime_sensitivity': 0.8,
            'session_memory': 0.85,
            'volatility_adaptation': 0.7,
            'horizon_decay': 0.95,
            'performance_momentum': 0.9,
            'impact_ema_beta': 0.9  # <- for avg_alignment_impact smoothing
        }

        # quality
        self.alignment_quality = {
            'effectiveness': 0.5,
            'consistency': 0.5,
            'adaptability': 0.5,
            'regime_alignment': 0.5,
            'session_optimization': 0.5,
            'performance_correlation': 0.5,
            'overall_quality': 0.5  # <- tracked explicitly
        }

        # tracking
        self.alignment_history = deque(maxlen=200)
        self.adaptation_events = deque(maxlen=100)
        self.performance_history = deque(maxlen=150)
        self.horizon_performance = deque(maxlen=100)

        # volatility adaptation bands
        self.volatility_adaptation = {
            'extreme': {'horizon_bias': 'short', 'multiplier_range': (0.3, 1.8)},
            'high': {'horizon_bias': 'short', 'multiplier_range': (0.5, 1.6)},
            'medium': {'horizon_bias': 'balanced', 'multiplier_range': (0.7, 1.4)},
            'low': {'horizon_bias': 'long', 'multiplier_range': (0.8, 1.3)},
            'very_low': {'horizon_bias': 'long', 'multiplier_range': (0.9, 1.2)}
        }

        # stats
        self.alignment_stats = {
            'total_alignments': 0,
            'significant_adaptations': 0,
            'regime_switches': 0,
            'session_transitions': 0,
            'performance_adjustments': 0,
            'avg_alignment_impact': 0.0,
            'effectiveness_trend': 0.0,
            'adaptation_accuracy': 0.5,
            'session_start_time': datetime.datetime.now().isoformat()
        }

        # analytics surfaces (kept for API parity)
        self.alignment_analytics = {'overall_alignment_quality': 0.5}
        self.regime_analytics = {'regime_alignment_score': 0.5, 'regime_stability_score': 0.5}
        self.performance_analytics = {'recent_alignment_performance': 0.5}

        # errors/circuit
        self.error_count = 0
        self.circuit_breaker_threshold = 5
        self.is_disabled = False

        # thesis & boot log
        self._generate_initialization_thesis()
        version = getattr(self.metadata, 'version', '3.1.0') if self.metadata else '3.1.0'
        self.logger.info(format_operator_message(
            icon="🕐",
            message=f"Time Horizon Aligner v{version} initialized",
            horizons=len(self.horizons),
            adaptive=self.adaptive_scaling,
            regime_aware=self.regime_awareness,
            performance_feedback=self.performance_feedback
        ))

    def _initialize_advanced_systems(self):
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="TimeHorizonAligner",
            log_path="logs/voting/time_horizon_aligner.log",
            max_lines=3000,
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("TimeHorizonAligner", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()
        self.health_monitor = HealthMonitor()

    def _generate_initialization_thesis(self):
        thesis = f"""
        Time Horizon Aligner v3.1 Initialization Complete:
        • {len(self.horizons)} horizons {self.horizons.tolist()}
        • Regime/session aware; perf feedback window {self.alignment_intelligence['performance_window']}
        • Adaptation threshold {self.alignment_intelligence['adaptation_threshold']:.2f}
        """
        self.smart_bus.set('time_horizon_aligner_initialization', {
            'status': 'initialized',
            'thesis': thesis,
            'timestamp': datetime.datetime.now().isoformat(),
            'configuration': {
                'horizons': self.horizons.tolist(),
                'adaptive_scaling': self.adaptive_scaling,
                'regime_awareness': self.regime_awareness,
                'performance_feedback': self.performance_feedback,
                'intelligence_parameters': self.alignment_intelligence
            }
        }, module='TimeHorizonAligner', thesis=thesis)

    def _get_tha_init_view(self) -> Dict[str, Any]:
        try:
            payload = self.smart_bus.get('time_horizon_aligner_initialization', 'TimeHorizonAligner') or {}
            if isinstance(payload, dict) and payload.get('status'):
                return payload
        except Exception:
            pass
        return {
            'status': 'initialized',
            'thesis': 'TimeHorizonAligner initialization heartbeat',
            'timestamp': datetime.datetime.now().isoformat(),
            'configuration': {
                'horizons': self.horizons.tolist(),
                'adaptive_scaling': bool(self.adaptive_scaling),
                'regime_awareness': bool(self.regime_awareness),
                'performance_feedback': bool(self.performance_feedback),
                'intelligence_parameters': dict(self.alignment_intelligence)
            }
        }

    # ----------------------------- PROCESS -----------------------------
    async def process(self, **inputs) -> Dict[str, Any]:
        start_time = time.time()
        try:
            if self.is_disabled:
                return self._generate_disabled_response()

            # FIX #2: Read kernel's decision_id for coordination
            decision_id = self.smart_bus.get('kernel_decision_id', 'TimeHorizonAligner')
            
            self.clock += 1
            current_time = datetime.datetime.now()
            alignment_data = await self._get_comprehensive_alignment_data()
            # Soft time budget and trivial-case fast path (avoid stage timeouts)
            soft_budget_ms = float(self.config.get('soft_time_budget_ms', max(500.0, min(2000.0, getattr(self.metadata, 'timeout_ms', 3000) - 800))))
            raw_w = alignment_data.get('voting_weights') or []
            n_members = (len(raw_w) if isinstance(raw_w, (list, tuple)) else (len(raw_w.keys()) if isinstance(raw_w, dict) else 0))
            if n_members <= 1:
                neutral = np.ones_like(self.horizons, dtype=np.float32)
                neutral = neutral / (float(neutral.sum()) + 1e-12)
                quality_analysis = {
                    'overall_quality': 0.5,
                    'effectiveness': 0.5,
                    'consistency': 0.5,
                    'adaptability': 0.5,
                    'regime_alignment': 0.5,
                }
                results = {
                    'aligned_weights': neutral.astype(float).tolist(),
                    'horizon_distances': (np.ones_like(self.horizons, dtype=np.float32)).tolist(),
                    'horizon_multipliers': (np.ones_like(self.horizons, dtype=np.float32)).tolist(),
                    'regime_adjustments': (np.ones_like(self.horizons, dtype=np.float32)).tolist(),
                    'session_patterns': (np.ones_like(self.horizons, dtype=np.float32)).tolist(),
                    'alignment_quality': quality_analysis,
                    'performance_metrics': self._get_performance_metrics_summary(),
                    'adaptation_status': self._get_adaptation_status(),
                    'health_metrics': self._get_health_metrics(),
                    'horizon_alignment': {
                        'distances': (np.ones_like(self.horizons, dtype=np.float32)).tolist(),
                        'multipliers': (np.ones_like(self.horizons, dtype=np.float32)).tolist(),
                        'regime': 'unknown',
                        'session': 'unknown'
                    },
                    'time_horizon_aligner_initialization': self._get_tha_init_view(),
                    'decision_id': decision_id,  # FIX: Use kernel's decision_id instead of stale bus data
                    'horizon_decision_id': decision_id,  # FIX: Contract-required namespaced decision_id
                    'horizon_alignment_meta': {'timestamp': datetime.datetime.now().isoformat()},  # FIX: Contract-required metadata
                    'tick_ts': alignment_data.get('tick_ts') or datetime.datetime.now().isoformat(),
                    '_thesis': 'Neutral alignment (single/no voter); fast-path applied'
                }
                await self._update_smartinfobus_comprehensive(results, results['_thesis'])
                self.performance_tracker.record_metric('TimeHorizonAligner', 'process_time', (time.time() - start_time) * 1000, True)
                self.error_count = 0
                self.last_alignment_time = current_time
                return results
            await self._update_market_state_comprehensive(alignment_data)

            if self.performance_feedback:
                await self._update_horizon_performance_comprehensive(alignment_data)

            await self._calculate_comprehensive_distance_alignment()
            await self._apply_regime_and_session_adaptations(alignment_data)
            await self._update_cyclical_patterns_comprehensive(alignment_data)

            quality_analysis = await self._calculate_comprehensive_alignment_quality()
            _ = await self._generate_intelligent_alignment_recommendations(quality_analysis)

            # base weights (neutral if missing)
            raw_weights = alignment_data.get('voting_weights') or []
            if not isinstance(raw_weights, (list, tuple, np.ndarray)) or (hasattr(raw_weights, '__len__') and len(raw_weights) == 0):
                safe_weights = np.ones_like(self.horizons, dtype=np.float32)
                safe_weights = safe_weights / (safe_weights.sum() + 1e-12)
            else:
                safe_weights = np.asarray(raw_weights, dtype=np.float32)

            try:
                aligned = await self.apply_alignment(safe_weights)
                aligned_list = aligned.astype(float).tolist()
            except Exception:
                fallback = np.ones_like(self.horizons, dtype=np.float32)
                fallback = fallback / (fallback.sum() + 1e-12)
                aligned_list = fallback.astype(float).tolist()

            results = {
                'aligned_weights': aligned_list,
                '_raw_weights': safe_weights.astype(float).tolist(),  # FIX: Store raw weights for VotingKernel
                'horizon_distances': self.current_distances.tolist(),
                'horizon_multipliers': self._get_combined_multipliers().tolist(),
                'regime_adjustments': self.regime_multipliers.get(
                    self.current_regime, np.ones_like(self.horizons)
                ).tolist(),
                'session_patterns': self._safe_session_vector().tolist(),
                'alignment_quality': quality_analysis,
                'performance_metrics': self._get_performance_metrics_summary(),
                'adaptation_status': self._get_adaptation_status(),
                'health_metrics': self._get_health_metrics(),
                'horizon_alignment': {
                    'distances': self.current_distances.tolist(),
                    'multipliers': self._get_combined_multipliers().tolist(),
                    'regime': self.current_regime,
                    'session': self.current_session
                },
                'time_horizon_aligner_initialization': self._get_tha_init_view(),
                'decision_id': decision_id,  # FIX: Use kernel's decision_id instead of stale bus data
                'horizon_decision_id': decision_id,  # FIX: Contract-required namespaced decision_id
                'horizon_alignment_meta': {'timestamp': datetime.datetime.now().isoformat()},  # FIX: Contract-required metadata
                'tick_ts': alignment_data.get('tick_ts') or datetime.datetime.now().isoformat(),
                '_thesis': ''  # set below
            }

            thesis = await self._generate_comprehensive_alignment_thesis(results, quality_analysis)
            results['_thesis'] = thesis
            await self._update_smartinfobus_comprehensive(results, thesis)

            processing_time = (time.time() - start_time) * 1000
            self.performance_tracker.record_metric('TimeHorizonAligner', 'process_time', processing_time, True)
            self.error_count = 0
            self.last_alignment_time = current_time
            return results

        except Exception as e:
            return await self._handle_processing_error(e, start_time)

    # ----------------------- MARKET/SESSION STATE ----------------------
    def _normalize_session(self, s: Optional[str]) -> str:
        return normalize_session_name(s)

    def _safe_session_vector(self, session: Optional[str] = None) -> np.ndarray:
        key = self._normalize_session(session or self.current_session)
        vec = self.session_patterns.get(key)
        if vec is None:
            vec = self.session_patterns.get('unknown')
        try:
            return np.asarray(vec, dtype=np.float32)
        except Exception:
            return np.ones_like(self.horizons, dtype=np.float32)

    async def _get_comprehensive_alignment_data(self) -> Dict[str, Any]:
        try:
            get = self.smart_bus.get
            return {
                'voting_weights': get('voting_weights', 'TimeHorizonAligner') or [],
                'market_regime': get('market_regime', 'TimeHorizonAligner') or 'unknown',
                'session_type': get('session_type', 'TimeHorizonAligner') or 'unknown',
                'volatility_data': get('volatility_data', 'TimeHorizonAligner') or {},
                'market_context': get('market_context', 'TimeHorizonAligner') or {},
                'time_of_day': get('time_of_day', 'TimeHorizonAligner') or 0,  # minutes since session start
                'performance_feedback': get('performance_feedback', 'TimeHorizonAligner') or {},
                'member_confidences': get('member_confidences', 'TimeHorizonAligner') or [],
                'recent_trades': get('recent_trades', 'TimeHorizonAligner') or [],
                'expert_performance': get('expert_performance', 'TimeHorizonAligner') or {},
                'decision_id': get('decision_id', 'TimeHorizonAligner'),
                'tick_ts': get('tick_ts', 'TimeHorizonAligner')
            }
        except Exception as e:
            err = self.error_pinpointer.analyze_error(e, "TimeHorizonAligner")
            self.logger.warning(f"Alignment data retrieval incomplete: {err}")
            return self._get_safe_alignment_defaults()

    async def _update_market_state_comprehensive(self, alignment_data: Dict[str, Any]):
        try:
            old_regime = self.current_regime
            self.current_regime = alignment_data.get('market_regime', 'unknown')

            if old_regime != self.current_regime and old_regime != 'unknown':
                self.alignment_stats['regime_switches'] += 1
                self.logger.info(format_operator_message(
                    icon="[STATS]", message="Market regime changed",
                    old_regime=old_regime, new_regime=self.current_regime, clock=self.clock,
                    impact="Horizon multipliers will adapt"
                ))
                self.adaptation_events.append({
                    'timestamp': datetime.datetime.now().isoformat(),
                    'type': 'regime_change',
                    'old_value': old_regime,
                    'new_value': self.current_regime,
                    'clock': self.clock
                })

            old_session = self.current_session
            self.current_session = self._normalize_session(alignment_data.get('session_type', 'unknown'))
            if old_session != self.current_session and old_session != 'unknown':
                self.alignment_stats['session_transitions'] += 1
                self.logger.info(format_operator_message(
                    icon="🕐", message="Trading session changed",
                    old_session=old_session, new_session=self.current_session, clock=self.clock
                ))

            volatility_data = alignment_data.get('volatility_data', {})
            current_vol = self._extract_numeric_volatility(volatility_data)
            self.volatility_history.append(current_vol)
            self.current_volatility = current_vol

        except Exception as e:
            err = self.error_pinpointer.analyze_error(e, "market_state_update")
            self.logger.warning(f"Market state update failed: {err}")

    # --------------------------- ALIGNMENT CORE ------------------------
    async def apply_alignment(self, weights: np.ndarray) -> np.ndarray:
        try:
            self.alignment_stats['total_alignments'] += 1
            weights = np.asarray(weights, dtype=np.float32)

            if len(weights) != len(self.horizons):
                weights = await self._handle_dimension_mismatch(weights)

            distance_factors = await self._calculate_distance_factors()
            regime_factors = await self._get_regime_factors()
            session_factors = await self._get_session_factors()
            performance_factors = await self._get_performance_factors()
            volatility_factors = await self._get_volatility_factors()

            combined_factors = await self._combine_alignment_factors(
                distance_factors, regime_factors, session_factors,
                performance_factors, volatility_factors
            )

            aligned_weights = weights * combined_factors
            aligned_weights = np.maximum(aligned_weights, 0.01)
            aligned_weights = aligned_weights / (aligned_weights.sum() + 1e-12)

            impact = float(np.linalg.norm(aligned_weights - weights))
            await self._track_alignment_impact(weights.tolist(), aligned_weights.tolist(), {'impact_score': impact})
            await self._record_alignment_event_comprehensive(weights, aligned_weights, combined_factors, impact)
            return aligned_weights

        except Exception as e:
            err = self.error_pinpointer.analyze_error(e, "alignment_application")
            self.logger.error(f"Horizon alignment failed: {err}")
            return np.asarray(weights, dtype=np.float32)

    async def _handle_dimension_mismatch(self, weights: np.ndarray) -> np.ndarray:
        try:
            self.logger.warning(format_operator_message(
                icon="[TOOL]", message="Dimension mismatch detected",
                weights_dim=len(weights), horizons_dim=len(self.horizons), action="Auto-adjusting"
            ))
            if len(weights) > len(self.horizons):
                adjusted_weights = weights[:len(self.horizons)]
            else:
                missing = len(self.horizons) - len(weights)
                default_weight = 1.0 / max(1, len(self.horizons))
                padding = np.full(missing, default_weight, dtype=np.float32)
                adjusted_weights = np.concatenate([weights, padding])
            return adjusted_weights
        except Exception as e:
            _ = self.error_pinpointer.analyze_error(e, "dimension_mismatch_handling")
            return weights

    async def _calculate_comprehensive_distance_alignment(self):
        try:
            time_distances = 1.0 / (1.0 + np.abs(self.clock - self.horizons))
            vol_adjustment = 1.0 + self.current_volatility * 2.0
            adjusted = time_distances * vol_adjustment
            self.current_distances = adjusted / (adjusted.sum() + 1e-12)
            self.base_distances = self.current_distances.copy()
        except Exception as e:
            _ = self.error_pinpointer.analyze_error(e, "distance_calculation")
            self.current_distances = np.ones_like(self.horizons) / len(self.horizons)

    async def _apply_regime_and_session_adaptations(self, alignment_data: Dict[str, Any]):
        try:
            if not self.regime_awareness:
                return
            await self._update_regime_multipliers_comprehensive(alignment_data)
            await self._update_session_patterns_comprehensive(alignment_data)
        except Exception as e:
            _ = self.error_pinpointer.analyze_error(e, "regime_session_adaptation")

    async def _update_regime_multipliers_comprehensive(self, alignment_data: Dict[str, Any]):
        try:
            regime = self.current_regime
            volatility_level = self._determine_volatility_level()

            if regime == 'trending':
                multipliers = np.array([0.7 if h < 5 else 1.4 if h > 30 else 1.1 for h in self.horizons])
            elif regime == 'volatile':
                multipliers = np.array([1.5 if h < 8 else 0.6 if h > 25 else 1.0 for h in self.horizons])
            elif regime == 'ranging':
                multipliers = np.array([1.2 if 8 <= h <= 20 else 0.8 for h in self.horizons])
            elif regime == 'breakout':
                multipliers = np.array([1.6 if h < 5 else 0.5 if h > 15 else 0.9 for h in self.horizons])
            elif regime == 'reversal':
                multipliers = np.array([0.8 if h < 10 else 1.3 if 10 <= h <= 30 else 0.9 for h in self.horizons])
            elif regime == 'noise':
                multipliers = np.array([1.3 if h < 3 else 0.7 if h > 20 else 1.0 for h in self.horizons])
            else:
                multipliers = np.ones_like(self.horizons)

            vol_bias = self.volatility_adaptation.get(volatility_level, {}).get('horizon_bias', 'balanced')
            if vol_bias == 'short':
                multipliers *= np.array([1.3 if h < 10 else 0.7 if h > 20 else 1.0 for h in self.horizons])
            elif vol_bias == 'long':
                multipliers *= np.array([0.8 if h < 5 else 1.2 if h > 15 else 1.0 for h in self.horizons])

            alpha = float(self.alignment_intelligence['regime_sensitivity'])
            old = self.regime_multipliers.get(regime, np.ones_like(self.horizons))
            self.regime_multipliers[regime] = (alpha * multipliers + (1 - alpha) * old).astype(np.float32)

        except Exception as e:
            _ = self.error_pinpointer.analyze_error(e, "regime_multipliers_update")

    def _determine_volatility_level(self) -> str:
        try:
            v = float(self.current_volatility)
            if v > 0.05: return 'extreme'
            if v > 0.03: return 'high'
            if v > 0.015: return 'medium'
            if v > 0.008: return 'low'
            return 'very_low'
        except Exception:
            return 'medium'

    def _extract_numeric_volatility(self, volatility_data: Any) -> float:
        try:
            if isinstance(volatility_data, (int, float, np.floating)):
                return float(abs(volatility_data))
            if isinstance(volatility_data, (list, tuple, np.ndarray)):
                arr = np.asarray(volatility_data, dtype=np.float64)
                return float(np.nanmean(arr)) if arr.size else 0.02
            if isinstance(volatility_data, dict):
                for key in ('value', 'atr', 'sigma', 'vol'):
                    if key in volatility_data:
                        try: return float(abs(volatility_data[key]))
                        except Exception: pass
                level = volatility_data.get('level')
                if isinstance(level, str):
                    return self._map_level_to_numeric(level)
                numeric_vals = [float(v) for v in volatility_data.values() if isinstance(v, (int, float, np.floating))]
                return float(np.mean(numeric_vals)) if numeric_vals else 0.02
            return 0.02
        except Exception:
            return 0.02

    def _map_level_to_numeric(self, level: str) -> float:
        lvl = (level or '').lower()
        return {
            'very_low': 0.005, 'low': 0.01, 'medium': 0.02, 'high': 0.035, 'extreme': 0.06
        }.get(lvl, 0.02)

    # --------------------------- QUALITY/ANALYTICS ---------------------
    async def _calculate_comprehensive_alignment_quality(self) -> Dict[str, Any]:
        try:
            # effectiveness from recent performance
            if len(self.performance_history) >= 10:
                recent = [p.get('improvement', 0.0) for p in list(self.performance_history)[-10:]]
                self.alignment_quality['effectiveness'] = float(np.mean(recent)) if recent else 0.5

            # consistency from impact variance
            if len(self.alignment_history) >= 5:
                impacts = [float(a.get('impact', {}).get('impact_score', 0.0)) if isinstance(a.get('impact'), dict) else float(a.get('impact', 0.0))
                           for a in list(self.alignment_history)[-10:]]
                var = np.std(impacts) if impacts else 0.0
                var_f = float(var)
                self.alignment_quality['consistency'] = float(max(0.0, min(1.0, 1.0 - var_f)))

            # adaptability (rate of meaningful changes)
            total = max(1, self.alignment_stats.get('total_alignments', 1))
            self.alignment_quality['adaptability'] = min(1.0, self.alignment_stats.get('significant_adaptations', 0) / total)

            # regime alignment similarity
            self.alignment_quality['regime_alignment'] = self._calculate_regime_alignment_score()

            # performance correlation (stub -> tie to effectiveness for now)
            self.alignment_quality['performance_correlation'] = self.alignment_quality['effectiveness']

            # session optimization proxy (mean of session vector)
            self.alignment_quality['session_optimization'] = float(np.mean(self._safe_session_vector()))

            # overall
            vals = list(self.alignment_quality.values())
            self.alignment_quality['overall_quality'] = float(np.mean(vals)) if vals else 0.5

            quality_analysis = {
                **self.alignment_quality,
                'quality_trend': self._determine_quality_trend(),
                'improvement_areas': self._identify_improvement_areas()
            }
            return quality_analysis
        except Exception as e:
            _ = self.error_pinpointer.analyze_error(e, "alignment_quality_calculation")
            return {'overall_quality': 0.5, 'quality_trend': 'unknown'}

    def _calculate_regime_alignment_score(self) -> float:
        try:
            regime = self.current_regime
            expected = self.regime_multipliers.get(regime)
            if expected is None:
                return 0.5
            current = self._get_combined_multipliers()
            similarity = 1.0 - np.mean(np.abs(expected - current)) / 2.0
            return float(max(0.0, min(1.0, float(similarity))))
        except Exception:
            return 0.5

    def _get_combined_multipliers(self) -> np.ndarray:
        try:
            combined = self.adaptive_multipliers.copy()
            if self.current_regime in self.regime_multipliers:
                combined *= self.regime_multipliers[self.current_regime]
            if self.performance_feedback:
                combined *= self.performance_multipliers
            combined *= self._safe_session_vector()
            combined *= self.cyclical_adjustments
            return combined
        except Exception:
            return np.ones_like(self.horizons)

    async def _generate_comprehensive_alignment_thesis(self, results: Dict[str, Any], quality_analysis: Dict[str, Any]) -> str:
        try:
            q = quality_analysis.get('overall_quality', 0.5)
            label = "HIGH" if q > 0.7 else "MODERATE" if q > 0.4 else "LOW"
            avg_impact = self.alignment_stats.get('avg_alignment_impact', 0.0)
            total = self.alignment_stats.get('total_alignments', 0)
            adaptations = self.alignment_stats.get('significant_adaptations', 0)
            return " | ".join([
                f"HORIZON ALIGNMENT: {label} effectiveness ({q:.1%})",
                f"MARKET ADAPTATION: {self.current_regime} regime, {self.current_session} session",
                f"ALIGNMENT IMPACT: {avg_impact:.3f} avg Δw",
                f"SYSTEM PERFORMANCE: {total} alignments, {adaptations} adaptations"
            ])
        except Exception as e:
            _ = self.error_pinpointer.analyze_error(e, "alignment_thesis_generation")
            return "Alignment thesis generation failed"

    async def _update_smartinfobus_comprehensive(self, results: Dict[str, Any], thesis: str):
        try:
            sb = self.smart_bus.set
            if results.get('aligned_weights') is not None:
                sb('aligned_weights', results['aligned_weights'], module='TimeHorizonAligner', thesis=thesis)
            sb('horizon_distances', results['horizon_distances'], module='TimeHorizonAligner',
               thesis=f"Horizon distances: {len(results['horizon_distances'])} time scales analyzed")
            sb('horizon_multipliers', results['horizon_multipliers'], module='TimeHorizonAligner',
               thesis=f"Horizon multipliers: Combined scaling factors for {len(results['horizon_multipliers'])} horizons")
            sb('regime_adjustments', results['regime_adjustments'], module='TimeHorizonAligner',
               thesis=f"Regime adjustments: {self.current_regime} market regime adaptations")
            sb('session_patterns', results['session_patterns'], module='TimeHorizonAligner',
               thesis=f"Session patterns: {self.current_session} session optimizations")
            sb('alignment_quality', results['alignment_quality'], module='TimeHorizonAligner',
               thesis=f"Alignment quality: {results['alignment_quality'].get('overall_quality', 0.5):.1%} effectiveness")
            # Publish under a non-canonical key to avoid colliding with SessionManager
            sb('alignment_metrics', results['performance_metrics'], module='TimeHorizonAligner',
               thesis=f"Alignment metrics: Comprehensive alignment analytics")
            sb('horizon_alignment', results['horizon_alignment'], module='TimeHorizonAligner',
               thesis='Combined horizon alignment bundle (distances, multipliers, regime, session)')

            # FIX #2: Decision coordination (namespaced key to avoid conflict)
            if results.get('decision_id'):
                sb('horizon_decision_id', results['decision_id'], module='TimeHorizonAligner',
                   thesis=f"Horizon decision ID for tick coordination: {results['decision_id']}")
            if results.get('tick_ts'):
                sb('tick_ts', results['tick_ts'], module='TimeHorizonAligner',
                   thesis=f"Tick timestamp: {results['tick_ts']}")
            
            # FIX: Publish namespaced keys for VotingKernel coordination (REAL DATA, NO FALLBACKS)
            # Extract raw weights from the alignment data that was processed
            raw_weights_data = results.get('_raw_weights', results.get('aligned_weights', []))
            sb('horizon_raw_weights', raw_weights_data, module='TimeHorizonAligner',
               thesis=f"Raw weights before horizon alignment: {len(raw_weights_data)} values")
            
            sb('horizon_aligned_weights', results.get('aligned_weights', []), module='TimeHorizonAligner',
               thesis=f"Aligned weights after horizon processing: {len(results.get('aligned_weights', []))} values")
        except Exception as e:
            _ = self.error_pinpointer.analyze_error(e, "smartinfobus_update")
            self.logger.error("SmartInfoBus update failed")

    # ------------------------- LEGACY INTERFACE ------------------------
    def apply(self, weights: np.ndarray) -> np.ndarray:
        try:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    return self._simple_alignment_fallback(weights)
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.apply_alignment(weights))
            finally:
                if loop and not loop.is_running():
                    loop.close()
        except Exception:
            return self._simple_alignment_fallback(weights)

    def _simple_alignment_fallback(self, weights: np.ndarray) -> np.ndarray:
        try:
            weights = np.asarray(weights, dtype=np.float32)
            if len(weights) != len(self.horizons):
                if len(weights) > len(self.horizons):
                    weights = weights[:len(self.horizons)]
                else:
                    missing = len(self.horizons) - len(weights)
                    weights = np.concatenate([weights, np.ones(missing, dtype=np.float32) / max(1, len(self.horizons))])
            distances = 1.0 / (1.0 + np.abs(self.clock - self.horizons))
            distances = distances / (distances.sum() + 1e-12)
            regime_mult = self.regime_multipliers.get(self.current_regime, np.ones_like(self.horizons))
            aligned = weights * distances * regime_mult
            aligned = np.maximum(aligned, 0.01)
            return aligned / (aligned.sum() + 1e-12)
        except Exception:
            return np.asarray(weights, dtype=np.float32)

    def resize(self, new_horizons: List[int]) -> None:
        old = self.horizons.copy()
        self.horizons = np.array(new_horizons, dtype=np.float32)
        self.current_distances = np.ones_like(self.horizons)
        self.base_distances = np.ones_like(self.horizons)
        self.adaptive_multipliers = np.ones_like(self.horizons)
        self.performance_multipliers = np.ones_like(self.horizons)
        self.cyclical_adjustments = np.ones_like(self.horizons)
        for regime in self.regime_multipliers:
            self.regime_multipliers[regime] = np.ones_like(self.horizons)
        for session in self.session_patterns:
            self.session_patterns[session] = np.ones_like(self.horizons)
        self.logger.info(format_operator_message(
            icon="[RELOAD]", message="Time Horizon Aligner resized",
            old_horizons=old.tolist(), new_horizons=self.horizons.tolist()
        ))

    # ------------------------------ RL OBS -----------------------------
    def get_observation_components(self) -> np.ndarray:
        try:
            features = [
                float(self.clock % 1000) / 1000.0,
                float(self.alignment_stats.get('avg_alignment_impact', 0.0)),
                float(np.mean(self.current_distances)),
                float(np.mean(self.adaptive_multipliers)),
                float(np.mean(self.performance_multipliers)),
                float(self.alignment_quality.get('overall_quality', 0.5)),
                float(len(self.alignment_history) / 200.0),
                float(self.current_volatility * 10.0)
            ]
            obs = np.array(features, dtype=np.float32)
            if np.any(~np.isfinite(obs)):
                self.logger.error(f"Invalid alignment observation: {obs}")
                obs = np.nan_to_num(obs, nan=0.5)
            return obs
        except Exception:
            return np.array([0.5, 0.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.2], dtype=np.float32)

    # ------------------------------ HEALTH ----------------------------
    def get_health_metrics(self) -> Dict[str, Any]:
        return {
            'module_name': 'TimeHorizonAligner',
            'status': 'disabled' if self.is_disabled else 'healthy',
            'error_count': self.error_count,
            'circuit_breaker_threshold': self.circuit_breaker_threshold,
            'total_alignments': self.alignment_stats.get('total_alignments', 0),
            'clock': self.clock,
            'current_regime': self.current_regime,
            'current_session': self.current_session,
            'current_volatility': self.current_volatility,
            'alignment_quality': self.alignment_quality.get('overall_quality', 0.5),
            'horizons_count': len(self.horizons),
            'adaptation_count': self.alignment_stats.get('significant_adaptations', 0),
            'regime_switches': self.alignment_stats.get('regime_switches', 0),
            'session_duration': (datetime.datetime.now() -
                                 datetime.datetime.fromisoformat(self.alignment_stats['session_start_time'])).total_seconds() / 3600
        }

    def _get_health_metrics(self) -> Dict[str, Any]:
        # kept for internal compatibility
        return self.get_health_metrics()

    # ----------------------- HELPERS & DEFAULTS -----------------------
    def _get_safe_alignment_defaults(self) -> Dict[str, Any]:
        return {
            'voting_weights': [], 'market_regime': 'unknown', 'session_type': 'unknown',
            'volatility_data': {}, 'market_context': {}, 'time_of_day': 0,
            'performance_feedback': {}, 'member_confidences': [], 'recent_trades': [],
            'expert_performance': {}
        }

    def _get_performance_metrics_summary(self) -> Dict[str, Any]:
        try:
            return {
                'total_alignments': self.alignment_stats.get('total_alignments', 0),
                'significant_adaptations': self.alignment_stats.get('significant_adaptations', 0),
                'avg_alignment_impact': self.alignment_stats.get('avg_alignment_impact', 0.0),
                'regime_switches': self.alignment_stats.get('regime_switches', 0),
                'session_transitions': self.alignment_stats.get('session_transitions', 0),
                'effectiveness_trend': self.alignment_stats.get('effectiveness_trend', 0.0),
                'adaptation_accuracy': self.alignment_stats.get('adaptation_accuracy', 0.5)
            }
        except Exception:
            return {}

    def _get_adaptation_status(self) -> Dict[str, Any]:
        try:
            return {
                'adaptive_scaling': self.adaptive_scaling,
                'regime_awareness': self.regime_awareness,
                'performance_feedback': self.performance_feedback,
                'current_regime': self.current_regime,
                'current_session': self.current_session,
                'volatility_level': self._determine_volatility_level(),
                'last_adaptation': self.adaptation_events[-1] if self.adaptation_events else None,
                'alignment_intelligence': self.alignment_intelligence.copy()
            }
        except Exception:
            return {'status': 'error'}

    # --------------------------- ERROR PATH ---------------------------
    async def _handle_processing_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        self.error_count += 1
        error_context = self.error_pinpointer.analyze_error(error, "TimeHorizonAligner")

        if self.error_count >= self.circuit_breaker_threshold:
            self.is_disabled = True
            self.logger.error(format_operator_message(
                icon="[ALERT]", message="Time Horizon Aligner disabled due to repeated errors",
                error_count=self.error_count, threshold=self.circuit_breaker_threshold
            ))

        processing_time = (time.time() - start_time) * 1000
        self.performance_tracker.record_metric(
            'TimeHorizonAligner', 'process_time', processing_time, success=False, error=str(error_context)
        )

        ones = np.ones_like(self.horizons).tolist()
        return {
            'aligned_weights': None,
            'horizon_distances': ones,
            'horizon_multipliers': ones,
            'regime_adjustments': ones,
            'session_patterns': ones,
            'alignment_quality': {'overall_quality': 0.5, 'error': str(error_context)},
            'performance_metrics': {'error': str(error_context)},
            'adaptation_status': {'status': 'error', 'error_context': str(error_context)},
            'health_metrics': {'status': 'error', 'error_context': str(error_context)},
            'decision_id': 'error',  # FIX: Contract-required
            'horizon_decision_id': 'error',  # FIX: Contract-required namespaced decision_id
            'horizon_alignment_meta': {'status': 'error'},  # FIX: Contract-required metadata
            'horizon_alignment': {
                'distances': ones, 'multipliers': ones,
                'regime': getattr(self, 'current_regime', 'unknown'),
                'session': getattr(self, 'current_session', 'unknown')
            },
            'time_horizon_aligner_initialization': self._get_tha_init_view(),
            '_thesis': f"TimeHorizonAligner encountered an error and returned safe defaults: {error_context}"
        }

    # ------------------------------- RESET ---------------------------
    def reset(self) -> None:
        super().reset()
        self.clock = 0
        self.session_start = 0
        self.current_distances = np.ones_like(self.horizons)
        self.base_distances = np.ones_like(self.horizons)
        self.adaptive_multipliers = np.ones_like(self.horizons)
        self.performance_multipliers = np.ones_like(self.horizons)
        self.cyclical_adjustments = np.ones_like(self.horizons)

        self.current_regime = 'unknown'
        self.current_session = 'unknown'
        self.current_volatility = 0.02

        for regime in self.regime_multipliers:
            self.regime_multipliers[regime] = np.ones_like(self.horizons)
        for session in self.session_patterns:
            self.session_patterns[session] = np.ones_like(self.horizons)

        self.alignment_history.clear()
        self.adaptation_events.clear()
        self.performance_history.clear()
        self.volatility_history.clear()
        self.horizon_performance.clear()

        self.alignment_stats = {
            'total_alignments': 0,
            'significant_adaptations': 0,
            'regime_switches': 0,
            'session_transitions': 0,
            'performance_adjustments': 0,
            'avg_alignment_impact': 0.0,
            'effectiveness_trend': 0.0,
            'adaptation_accuracy': 0.5,
            'session_start_time': datetime.datetime.now().isoformat()
        }

        self.alignment_quality.update({
            'effectiveness': 0.5, 'consistency': 0.5, 'adaptability': 0.5,
            'regime_alignment': 0.5, 'session_optimization': 0.5,
            'performance_correlation': 0.5, 'overall_quality': 0.5
        })

        self.error_count = 0
        self.is_disabled = False

        self.logger.info(format_operator_message(
            icon="[RELOAD]", message="Time Horizon Aligner reset completed",
            status="All alignment state cleared and systems reinitialized"
        ))

    # ------------------------------ BASEMODULE ------------------------
    async def calculate_confidence(self, action: Dict[str, Any], **inputs) -> float:
        try:
            aq = float(self.alignment_analytics.get('overall_alignment_quality', 0.5))
            ra = float(self.regime_analytics.get('regime_alignment_score', 0.5))
            rp = float(self.performance_analytics.get('recent_alignment_performance', 0.5))
            vol_level = self._determine_volatility_level()
            vol_conf = {'very_low': 0.9, 'low': 0.8, 'medium': 0.7, 'high': 0.5, 'extreme': 0.3}.get(vol_level, 0.6)
            conf = aq * 0.4 + ra * 0.3 + rp * 0.2 + vol_conf * 0.1
            return float(max(0.1, min(0.95, conf)))
        except Exception:
            return 0.4

    async def propose_action(self, **inputs) -> Dict[str, Any]:
        try:
            alignment_quality = float(self.alignment_analytics.get('overall_alignment_quality', 0.5))
            regime_stability = float(self.regime_analytics.get('regime_stability_score', 0.5))
            if alignment_quality < 0.4:
                action_type, signal_strength, reasoning = 'realign', 0.8, f"Poor alignment quality ({alignment_quality:.3f})"
            elif regime_stability < 0.3:
                action_type, signal_strength, reasoning = 'adapt', 0.6, f"Low regime stability ({regime_stability:.3f})"
            elif alignment_quality > 0.8 and regime_stability > 0.7:
                action_type, signal_strength, reasoning = 'maintain', 0.4, f"Excellent alignment (q={alignment_quality:.3f}, s={regime_stability:.3f})"
            else:
                action_type, signal_strength, reasoning = 'optimize', 0.5, "Moderate metrics suggest optimization"
            return {
                'action': action_type,
                'signal_strength': signal_strength,
                'reasoning': reasoning,
                'alignment_metrics': {
                    'alignment_quality': alignment_quality,
                    'regime_stability': regime_stability,
                    'volatility_level': self._determine_volatility_level(),
                    'current_horizons': len(self.horizons)
                },
                'confidence': await self.calculate_confidence({}, **inputs)
            }
        except Exception as e:
            self.logger.error(f"Action proposal failed: {e}")
            return {'action': 'abstain', 'signal_strength': 0.0, 'reasoning': f'Alignment error: {str(e)}', 'confidence': 0.1}

    # ---------------------- IMPROVED PREVIOUS STUBS -------------------
    def _generate_disabled_response(self) -> Dict[str, Any]:
        neutral = (np.ones_like(self.horizons) / max(1, len(self.horizons))).tolist()
        return {
            'aligned_weights': neutral,
            'alignment_quality': {'overall_quality': 0.5},
            'disabled': True,
            'reason': 'Module disabled',
            'horizon_alignment': {
                'distances': self.current_distances.tolist(),
                'multipliers': self._get_combined_multipliers().tolist(),
                'regime': getattr(self, 'current_regime', 'unknown'),
                'session': getattr(self, 'current_session', 'unknown')
            },
            'time_horizon_aligner_initialization': self._get_tha_init_view(),
            '_thesis': 'TimeHorizonAligner disabled by circuit breaker; returning neutral horizon alignment bundle'
        }

    async def _update_horizon_performance_comprehensive(self, alignment_data: Dict[str, Any]) -> None:
        """
        Lightweight performance feedback:
        - Uses `performance_feedback.get('horizon_scores', {minutes: score})` if present.
        - Else decays to 1.0 with EMA momentum, nudged by recent trade PnL sign if `holding_time` is available.
        """
        try:
            lr = float(self.alignment_intelligence['learning_rate'])
            momentum = float(self.alignment_intelligence['performance_momentum'])

            pf = alignment_data.get('performance_feedback', {}) or {}
            horizon_scores: Dict[Any, float] = pf.get('horizon_scores', {}) or {}

            vec = np.ones_like(self.horizons, dtype=np.float32)
            if horizon_scores:
                for i, h in enumerate(self.horizons):
                    score = float(horizon_scores.get(int(h), 1.0))
                    vec[i] = np.clip(score, 0.5, 1.5)
            else:
                # infer a weak signal from recent_trades if they carry 'holding_time' and 'pnl'
                trades = alignment_data.get('recent_trades', []) or []
                if trades:
                    # build a simple kernel by proximity of trade holding_time to horizons
                    holds, pnls = [], []
                    for t in trades[-self.alignment_intelligence['performance_window']:]:
                        ht = t.get('holding_time') or t.get('duration') or None
                        pnl = t.get('pnl', 0.0)
                        if isinstance(ht, (int, float)) and np.isfinite(ht):
                            holds.append(float(ht))
                            pnls.append(float(pnl))
                    if holds:
                        holds = np.asarray(holds, dtype=np.float32)
                        pnls = np.asarray(pnls, dtype=np.float32)
                        for i, h in enumerate(self.horizons):
                            w = np.exp(-np.abs(holds - h) / max(1.0, h))  # proximity kernel
                            s = float(np.sum(w * np.sign(pnls)) / (np.sum(w) + 1e-9))
                            vec[i] = np.clip(1.0 + 0.2 * s, 0.8, 1.2)

            # EMA update of performance multipliers
            self.performance_multipliers = (momentum * self.performance_multipliers + (1 - momentum) * vec).astype(np.float32)
            self.alignment_stats['performance_adjustments'] += 1

        except Exception as e:
            self.logger.warning(f"Horizon performance update failed: {e}")

    async def _update_session_patterns_comprehensive(self, alignment_data: Dict[str, Any]) -> None:
        """
        Session shaping:
        - american/overlap → slight short-term tilt
        - asian → slight longer-term tilt
        - european → balanced, mid-term tilt
        - rollover/weekend → neutral to conservative
        """
        try:
            sess = self._normalize_session(alignment_data.get('session_type', self.current_session))
            base = np.ones_like(self.horizons, dtype=np.float32)

            if sess in ('american', 'overlap'):
                pattern = np.array([1.15 if h <= 15 else 0.95 if h >= 60 else 1.0 for h in self.horizons], dtype=np.float32)
            elif sess == 'asian':
                pattern = np.array([0.95 if h <= 10 else 1.10 if h >= 60 else 1.0 for h in self.horizons], dtype=np.float32)
            elif sess == 'european':
                pattern = np.array([1.05 if 10 <= h <= 30 else 0.98 for h in self.horizons], dtype=np.float32)
            elif sess in ('rollover', 'weekend'):
                pattern = base
            else:
                pattern = base

            # smooth update
            alpha = float(self.alignment_intelligence['session_memory'])
            old = self.session_patterns.get(sess, base)
            self.session_patterns[sess] = (alpha * pattern + (1 - alpha) * old).astype(np.float32)

        except Exception as e:
            self.logger.warning(f"Session pattern update failed: {e}")

    async def _update_cyclical_patterns_comprehensive(self, alignment_data: Dict[str, Any]) -> None:
        """
        Cyclical nudges over the trading day:
        - Slight sinus modulation across horizons using time_of_day (minutes).
        - Keeps multipliers close to 1.0 (±5%).
        """
        try:
            tod = float(alignment_data.get('time_of_day', 0.0) or 0.0)  # minutes
            # project hours into [0, 2π]
            phase = 2.0 * np.pi * (tod % (24 * 60)) / (24.0 * 60.0)
            # horizon-specific offsets to avoid lockstep
            offsets = (self.horizons / (self.horizons.max() + 1e-9)) * np.pi
            wave = 1.0 + 0.05 * np.sin(phase + offsets)
            self.cyclical_adjustments = np.asarray(wave, dtype=np.float32)
        except Exception as e:
            self.logger.warning(f"Cyclical pattern update failed: {e}")

    async def _generate_intelligent_alignment_recommendations(self, quality_analysis: Dict[str, Any]) -> Dict[str, Any]:
        try:
            q = float(quality_analysis.get('overall_quality', 0.5))
            if q < 0.4:
                rec, pr = 'increase_short_term_weight', 'high'
            elif q > 0.8:
                rec, pr = 'balance_horizons', 'low'
            else:
                rec, pr = 'maintain_current_alignment', 'medium'
            return {'recommendation': rec, 'priority': pr, 'confidence': q, 'reasoning': f'Overall quality {q:.2f}'}
        except Exception:
            return {'recommendation': 'maintain_current_alignment', 'priority': 'medium', 'confidence': 0.5}

    async def _calculate_distance_factors(self) -> np.ndarray:
        return np.asarray(self.current_distances, dtype=np.float32)

    async def _get_regime_factors(self) -> np.ndarray:
        return np.asarray(self.regime_multipliers.get(self.current_regime, np.ones_like(self.horizons)), dtype=np.float32)

    async def _get_session_factors(self) -> np.ndarray:
        return self._safe_session_vector()

    async def _get_performance_factors(self) -> np.ndarray:
        return np.asarray(self.performance_multipliers, dtype=np.float32)

    async def _get_volatility_factors(self) -> np.ndarray:
        level = self._determine_volatility_level()
        scalar = {'very_low': 0.95, 'low': 0.98, 'medium': 1.00, 'high': 1.10, 'extreme': 1.20}.get(level, 1.00)
        return np.full_like(self.horizons, scalar, dtype=np.float32)

    async def _combine_alignment_factors(
        self,
        distance_factors: np.ndarray,
        regime_factors: np.ndarray,
        session_factors: np.ndarray,
        performance_factors: np.ndarray,
        volatility_factors: np.ndarray
    ) -> np.ndarray:
        combined = np.ones_like(self.horizons, dtype=np.float32)
        for arr in (distance_factors, regime_factors, session_factors, performance_factors, volatility_factors):
            a = np.asarray(arr, dtype=np.float32)
            if a.shape != combined.shape:
                a = np.full_like(combined, float(np.mean(a)))
            combined *= a
        return np.clip(combined, 0.01, None)

    async def _track_alignment_impact(self, original_weights: List[float], aligned_weights: List[float], impact: Dict[str, Any]) -> None:
        try:
            rec = {
                'timestamp': datetime.datetime.now().isoformat(),
                'original_weights': original_weights,
                'aligned_weights': aligned_weights,
                'impact': impact
            }
            self.alignment_history.append(rec)
            # update EMA avg impact
            beta = float(self.alignment_intelligence.get('impact_ema_beta', 0.9))
            prev = float(self.alignment_stats.get('avg_alignment_impact', 0.0))
            new = float(impact.get('impact_score', 0.0))
            self.alignment_stats['avg_alignment_impact'] = beta * prev + (1 - beta) * new
        except Exception as e:
            self.logger.warning(f"Alignment impact tracking failed: {e}")

    async def _record_alignment_event_comprehensive(self, weights: np.ndarray, aligned: np.ndarray, factors: np.ndarray, impact: float) -> None:
        try:
            evt = {
                'timestamp': datetime.datetime.now().isoformat(),
                'type': 'alignment',
                'impact': impact,
                'clock': self.clock,
                'regime': self.current_regime,
                'session': self.current_session
            }
            self.adaptation_events.append(evt)
            if impact >= float(self.alignment_intelligence['adaptation_threshold']):
                self.alignment_stats['significant_adaptations'] += 1
        except Exception as e:
            self.logger.warning(f"Alignment event recording failed: {e}")

    def _determine_quality_trend(self) -> str:
        try:
            # compute slope of overall_quality over last N alignment_history entries
            vals = [float(self.alignment_quality.get('overall_quality', 0.5))]
            # supplement with historical overall qualities if we stored them (approx via impacts)
            if len(self.alignment_history) >= 5:
                # proxy: lower impact variance → improving
                impacts = [float(a.get('impact', {}).get('impact_score', 0.0)) if isinstance(a.get('impact'), dict) else float(a.get('impact', 0.0))
                           for a in list(self.alignment_history)[-10:]]
                slope = np.polyfit(np.arange(len(impacts)), impacts, 1)[0] if len(impacts) > 1 else 0.0
                return 'improving' if slope < -0.005 else 'declining' if slope > 0.005 else 'stable'
            return 'stable'
        except Exception:
            return 'unknown'

    def _identify_improvement_areas(self) -> List[str]:
        areas = []
        if self.alignment_quality.get('regime_alignment', 0.5) < 0.45:
            areas.append("regime_adaptation")
        if self.alignment_quality.get('consistency', 0.5) < 0.45:
            areas.append("stability_control")
        if self.alignment_quality.get('performance_correlation', 0.5) < 0.45:
            areas.append("feedback_signal")
        if not areas:
            areas.append("horizon_balance")
        return areas
