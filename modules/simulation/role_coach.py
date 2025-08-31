# ─────────────────────────────────────────────────────────────
# File: modules/simulation/role_coach.py
# Enhanced Role Coach with Modern Architecture + Contract Compliance
# ─────────────────────────────────────────────────────────────

from modules.contracts import module_args
import numpy as np
import datetime
import time
import copy
from typing import Dict, Any, List, Optional
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
    "RoleCoach",
    description="Intelligent trade discipline coaching with context-aware penalties and performance tracking",
    error_handling=True,
    hot_reload=True,
    timeout_ms=3000,
))
class RoleCoach(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    RoleCoach is contract-aware and SmartInfoBus-integrated.

    CONTRACT (from modules.contracts.CONTRACTS['RoleCoach']):
      requires = ['market_context','pending_orders','positions','recent_trades','regime_data','risk_metrics','session_data','trading_performance']
      provides = ['coaching_penalties','coaching_recommendations','coaching_results','coaching_statistics','compliance_tracking','discipline_assessment','discipline_penalty','performance_scoring','trade_limits']
    """

    # Coaching modes
    COACHING_MODES = {
        "strict": "Strict discipline enforcement",
        "adaptive": "Context-aware discipline",
        "lenient": "Flexible trade management",
        "performance_based": "Performance-driven limits",
        "regime_aware": "Market regime specific rules"
    }

    # Enhanced default configuration
    ENHANCED_DEFAULTS = {
        "max_trades": 2,
        "penalty_multiplier": 1.0,
        "coaching_mode": "adaptive",
        "regime_sensitivity": 0.8,
        "performance_adjustment": True,
        "session_aware": True,
        "volatility_scaling": True,
        "learning_rate": 0.1,
        "penalty_decay": 0.95
    }

    # ================== LIFECYCLE ==================
    def __init__(
        self,
        max_trades: int = 2,
        penalty_multiplier: float = 1.0,
        coaching_mode: str = "adaptive",
        debug: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Initialize mixins
        self._initialize_trading_state()

        # Config (defaults + overrides)
        self.coach_config = copy.deepcopy(self.ENHANCED_DEFAULTS)
        if 'config' in kwargs and isinstance(kwargs['config'], dict):
            self.coach_config.update(kwargs['config'])

        # Core parameters
        self.max_trades          = int(max_trades)
        self.penalty_multiplier  = float(penalty_multiplier)
        self.coaching_mode       = coaching_mode if coaching_mode in self.COACHING_MODES else "adaptive"
        self.regime_sensitivity      = float(self.coach_config["regime_sensitivity"])
        self.performance_adjustment  = bool(self.coach_config["performance_adjustment"])
        self.session_aware           = bool(self.coach_config["session_aware"])
        self.volatility_scaling      = bool(self.coach_config["volatility_scaling"])
        self.learning_rate           = float(self.coach_config["learning_rate"])
        self.penalty_decay           = float(self.coach_config["penalty_decay"])

        # State tracking
        self.discipline_history: deque = deque(maxlen=100)
        self.penalty_history: deque = deque(maxlen=50)
        self.performance_history: deque = deque(maxlen=50)
        self.coaching_sessions: deque = deque(maxlen=20)

        # Market context
        self.market_regime = "normal"
        self.volatility_regime = "medium"
        self.market_session = "unknown"

        # Adaptive parameters
        self.adaptive_max_trades = self.max_trades
        self.adaptive_penalty = self.penalty_multiplier
        self.current_performance_score = 0.5
        self._last_effectiveness: float = 0.0

        # Session-specific base limits
        self.session_limits = {
            "asian": max(1, int(self.max_trades * 0.8)),
            "european": max(1, self.max_trades),
            "american": max(1, int(self.max_trades * 1.2)),
            "rollover": max(1, int(self.max_trades * 0.6))
        }

        # Regime-specific multipliers
        self.regime_multipliers = {
            "trending": 1.2,
            "volatile": 0.8,
            "ranging": 1.0,
            "unknown": 1.0
        }

        # Coaching statistics
        self.coaching_stats: Dict[str, Any] = {
            "total_sessions": 0,
            "penalties_applied": 0,
            "total_penalty_amount": 0.0,
            "discipline_score": 1.0,
            "improvement_rate": 0.0,
            "violations": 0,
            "compliance_rate": 1.0
        }

        # Analytics
        self.coaching_analytics: Dict[str, List[float]] = defaultdict(list)
        self.regime_performance: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self.session_performance: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self.learning_history: deque = deque(maxlen=30)
        self.effectiveness_scores: deque = deque(maxlen=20)

        # Fault tolerance
        self.error_count = 0
        self.circuit_breaker_threshold = 5
        self.is_disabled = False

        # Systems
        self._initialize_advanced_systems()

        self.logger.info(format_operator_message(
            icon="[TARGET]",
            message="RoleCoach initialized (contract compliant)",
            max_trades=self.max_trades,
            penalty_multiplier=f"{self.penalty_multiplier:.2f}",
            coaching_mode=self.coaching_mode,
            regime_sensitivity=f"{self.regime_sensitivity:.1%}",
            performance_adjustment=self.performance_adjustment
        ))

    def _initialize_advanced_systems(self):
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="RoleCoach",
            log_path="logs/simulation/role_coach.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("RoleCoach", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()

    def _initialize(self):
        return None

    # ================== CORE FLOW ==================
    def reset(self) -> None:
        super().reset()
        self.discipline_history.clear()
        self.penalty_history.clear()
        self.performance_history.clear()
        self.coaching_sessions.clear()

        self.market_regime = "normal"
        self.volatility_regime = "medium"
        self.market_session = "unknown"

        self.adaptive_max_trades = self.max_trades
        self.adaptive_penalty = self.penalty_multiplier
        self.current_performance_score = 0.5
        self._last_effectiveness = 0.0

        self.coaching_stats = {
            "total_sessions": 0,
            "penalties_applied": 0,
            "total_penalty_amount": 0.0,
            "discipline_score": 1.0,
            "improvement_rate": 0.0,
            "violations": 0,
            "compliance_rate": 1.0
        }

        self.coaching_analytics.clear()
        self.regime_performance.clear()
        self.session_performance.clear()
        self.learning_history.clear()
        self.effectiveness_scores.clear()

        self.error_count = 0
        self.is_disabled = False

        self.logger.info(format_operator_message(
            icon="[RELOAD]",
            message="RoleCoach reset - state cleared"
        ))

    async def process(self, **inputs) -> Dict[str, Any]:
        start_time = time.time()
        try:
            if self.is_disabled:
                return self._generate_disabled_contract()

            # Pull *required* inputs from the SmartInfoBus per contract
            trading_activity = await self._extract_trading_activity_from_smart_bus()

            # Context update + adaptation
            await self._update_market_context(trading_activity)

            # Coaching session
            session_result = await self._conduct_coaching_session(trading_activity)

            # Adaptation + effectiveness analysis
            self._update_adaptive_parameters(session_result)
            self._analyze_coaching_effectiveness(session_result)

            # Build contract payload & publish each provided key
            payload = self._build_contract_payload(session_result)
            await self._publish_contract_outputs(payload)

            # Metrics
            processing_time = (time.time() - start_time) * 1000
            self.performance_tracker.record_metric('RoleCoach', 'process_time', processing_time, True)
            self.error_count = 0

            return payload

        except Exception as e:
            return await self._handle_processing_error(e, start_time)

    # ================== DATA ACQUISITION (REQUIRES) ==================
    async def _extract_trading_activity_from_smart_bus(self) -> Dict[str, Any]:
        """
        Reads all 'requires' from the InfoBus:
          - market_context, pending_orders, positions, recent_trades, regime_data,
            risk_metrics, session_data, trading_performance
        """
        activity: Dict[str, Any] = {}
        try:
            # REQUIRED feeds
            recent_trades   = self.smart_bus.get('recent_trades', 'RoleCoach') or []
            pending_orders  = self.smart_bus.get('pending_orders', 'RoleCoach') or []
            positions       = self.smart_bus.get('positions', 'RoleCoach') or []
            risk_metrics    = self.smart_bus.get('risk_metrics', 'RoleCoach') or {}
            market_context  = self.smart_bus.get('market_context', 'RoleCoach') or {}
            regime_data     = self.smart_bus.get('regime_data', 'RoleCoach') or {}
            session_data    = self.smart_bus.get('session_data', 'RoleCoach') or {}
            trading_perf    = self.smart_bus.get('trading_performance', 'RoleCoach') or {}

            activity['recent_trades']   = recent_trades
            activity['pending_orders']  = pending_orders
            activity['positions']       = positions
            activity['risk_metrics']    = risk_metrics
            activity['market_context']  = market_context
            activity['regime_data']     = regime_data
            activity['session_data']    = session_data
            activity['trading_performance'] = trading_perf

            # Derived fields
            activity['trade_count']  = len(recent_trades)
            activity['order_count']  = len(pending_orders)
            activity['position_count'] = len(positions)
            activity['trading_intensity'] = activity['trade_count'] + activity['order_count']

            # Balances
            current_balance = risk_metrics.get('balance', risk_metrics.get('equity', 10000))
            activity['current_balance'] = current_balance
            recent_pnl = sum(trade.get('pnl', 0.0) for trade in recent_trades)
            activity['recent_pnl'] = float(recent_pnl)

            # Market/session/volatility
            # Priority: explicit 'session_data' & 'regime_data' override 'market_context'
            activity['regime'] = regime_data.get('market_regime',
                                 market_context.get('regime', 'unknown'))
            activity['session'] = session_data.get('session',
                                  market_context.get('session', 'unknown'))
            activity['volatility_level'] = (
                market_context.get('volatility_level', 'medium')
            )

            # Trade timing
            activity['trade_timing'] = self._analyze_trade_timing(recent_trades)

        except Exception as e:
            self.logger.warning(f"Trading activity extraction failed: {e}")
            activity = {
                'recent_trades': [], 'trade_count': 0,
                'pending_orders': [], 'order_count': 0,
                'positions': [], 'position_count': 0,
                'risk_metrics': {}, 'market_context': {}, 'regime_data': {}, 'session_data': {}, 'trading_performance': {},
                'trading_intensity': 0, 'current_balance': 10000, 'recent_pnl': 0.0,
                'regime': 'unknown', 'session': 'unknown', 'volatility_level': 'medium',
                'trade_timing': {'frequency': 'normal', 'clustering': False}
            }

        return activity

    def _analyze_trade_timing(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            if len(trades) < 2:
                return {'frequency': 'low', 'clustering': False, 'intervals': []}

            timestamps = []
            for trade in trades:
                ts = trade.get('timestamp') or trade.get('time') or ""
                if ts:
                    try:
                        # ISO variants safeguard
                        ts_iso = str(ts).replace('Z', '+00:00')
                        timestamps.append(datetime.datetime.fromisoformat(ts_iso))
                    except Exception:
                        continue
            if len(timestamps) < 2:
                return {'frequency': 'low', 'clustering': False, 'intervals': []}

            timestamps.sort()
            intervals = [(timestamps[i] - timestamps[i-1]).total_seconds()
                         for i in range(1, len(timestamps))]

            avg_interval = np.mean(intervals) if intervals else 3600
            if avg_interval < 300:
                frequency = 'high'
            elif avg_interval < 1800:
                frequency = 'medium'
            else:
                frequency = 'low'

            clustering = any(iv < 60 for iv in intervals)

            return {
                'frequency': frequency,
                'clustering': clustering,
                'intervals': intervals,
                'avg_interval': float(avg_interval),
                'min_interval': float(min(intervals)) if intervals else 0.0,
                'rapid_trades': int(sum(1 for iv in intervals if iv < 300))
            }
        except Exception as e:
            self.logger.warning(f"Trade timing analysis failed: {e}")
            return {'frequency': 'normal', 'clustering': False, 'intervals': []}

    async def _update_market_context(self, trading_activity: Dict[str, Any]) -> None:
        try:
            old_regime = self.market_regime
            self.market_regime = trading_activity.get('regime', 'unknown')
            self.volatility_regime = trading_activity.get('volatility_level', 'medium')
            self.market_session = trading_activity.get('session', 'unknown')

            if self.market_regime != old_regime:
                self._adapt_coaching_for_regime_change(old_regime, self.market_regime)
                self.logger.info(format_operator_message(
                    icon="[STATS]",
                    message=f"Regime change: {old_regime} → {self.market_regime}",
                    coaching_adaptation="Trade limits adjusted",
                    session=self.market_session
                ))
        except Exception as e:
            self.logger.warning(f"Market context update failed: {e}")

    def _adapt_coaching_for_regime_change(self, old_regime: str, new_regime: str) -> None:
        try:
            regime_mult = self.regime_multipliers.get(new_regime, 1.0)
            self.adaptive_max_trades = int(self.max_trades * (1.0 + (regime_mult - 1.0) * self.regime_sensitivity))
            self.adaptive_max_trades = max(1, self.adaptive_max_trades)
        except Exception as e:
            self.logger.warning(f"Regime adaptation failed: {e}")

    # ================== COACHING ENGINE ==================
    async def _conduct_coaching_session(self, trading_activity: Dict[str, Any]) -> Dict[str, Any]:
        session_result = {
            'timestamp': datetime.datetime.now().isoformat(),
            'coaching_mode': self.coaching_mode,
            'trade_analysis': {},
            'discipline_assessment': {},
            'penalties': {},
            'recommendations': [],
            'effective_limits': {},
            'context': {
                'regime': self.market_regime,
                'session': self.market_session,
                'volatility_level': self.volatility_regime
            }
        }
        try:
            effective_limits = self._calculate_effective_trade_limits(trading_activity)
            session_result['effective_limits'] = effective_limits

            trade_analysis = self._analyze_trading_activity(trading_activity, effective_limits)
            session_result['trade_analysis'] = trade_analysis

            discipline_assessment = self._assess_trading_discipline(trade_analysis)
            session_result['discipline_assessment'] = discipline_assessment

            penalties = self._calculate_context_aware_penalties(discipline_assessment, trading_activity)
            session_result['penalties'] = penalties

            recs = self._generate_coaching_recommendations(trade_analysis, discipline_assessment, trading_activity)
            session_result['recommendations'] = recs

            self.coaching_sessions.append(session_result)

            # Stats
            self.coaching_stats['total_sessions'] += 1
            if penalties.get('total_penalty', 0.0) > 0:
                self.coaching_stats['penalties_applied'] += 1
                self.coaching_stats['total_penalty_amount'] += penalties['total_penalty']
            if discipline_assessment.get('violations', 0) > 0:
                self.coaching_stats['violations'] += discipline_assessment['violations']

            total_sessions = max(1, self.coaching_stats['total_sessions'])
            self.coaching_stats['compliance_rate'] = 1.0 - (self.coaching_stats['violations'] / total_sessions)
            self.coaching_stats['discipline_score'] = discipline_assessment.get('overall_score', 1.0)

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "coaching_session")
            self.logger.error(f"Coaching session failed: {error_context}")
            session_result['error'] = str(error_context)

        return session_result

    def _calculate_effective_trade_limits(self, trading_activity: Dict[str, Any]) -> Dict[str, Any]:
        try:
            base_limit = self.adaptive_max_trades

            if self.session_aware:
                session = trading_activity.get('session', 'unknown')
                base_limit = self.session_limits.get(session, base_limit)

            vol_adj = {'low': 1.2, 'medium': 1.0, 'high': 0.8, 'extreme': 0.6}
            if self.volatility_scaling:
                vol_level = trading_activity.get('volatility_level', 'medium')
                base_limit = int(base_limit * vol_adj.get(vol_level, 1.0))

            if self.performance_adjustment:
                cps = self.current_performance_score
                if cps > 0.8:
                    base_limit = int(base_limit * 1.2)
                elif cps < 0.3:
                    base_limit = int(base_limit * 0.7)

            base_limit = max(1, base_limit)

            return {
                'max_trades': base_limit,
                'base_trades': self.max_trades,
                'adaptive_trades': self.adaptive_max_trades,
                'session_adjustment': self.session_limits.get(trading_activity.get('session', 'unknown'), 1),
                'volatility_adjustment': vol_adj.get(trading_activity.get('volatility_level', 'medium'), 1.0) if self.volatility_scaling else 1.0,
                'performance_adjustment': (1.2 if self.current_performance_score > 0.8 else (0.7 if self.current_performance_score < 0.3 else 1.0)) if self.performance_adjustment else 1.0
            }
        except Exception as e:
            self.logger.warning(f"Effective limits calculation failed: {e}")
            return {'max_trades': self.max_trades, 'base_trades': self.max_trades}

    def _analyze_trading_activity(self, trading_activity: Dict[str, Any], effective_limits: Dict[str, Any]) -> Dict[str, Any]:
        try:
            trade_count = int(trading_activity.get('trade_count', 0))
            max_trades = int(effective_limits.get('max_trades', self.max_trades))
            trading_intensity = int(trading_activity.get('trading_intensity', 0))
            trade_timing = trading_activity.get('trade_timing', {})

            over_limit = max(0, trade_count - max_trades)
            compliance = trade_count <= max_trades

            # Intensity level
            if trading_intensity > max_trades * 2:
                intensity_level = 'excessive'
            elif trading_intensity > max_trades * 1.5:
                intensity_level = 'high'
            elif trading_intensity > max_trades:
                intensity_level = 'moderate'
            else:
                intensity_level = 'low'

            timing_issues = []
            if trade_timing.get('clustering', False):
                timing_issues.append('rapid_clustering')
            if trade_timing.get('frequency') == 'high':
                timing_issues.append('high_frequency')
            if trade_timing.get('rapid_trades', 0) > 3:
                timing_issues.append('excessive_rapid_trades')

            regime = trading_activity.get('regime', 'unknown')
            context_appropriateness = 'appropriate'
            if regime == 'volatile' and trade_count > max_trades * 0.8:
                context_appropriateness = 'questionable'
            elif regime == 'ranging' and trade_count > max_trades * 1.2:
                context_appropriateness = 'excessive'

            return {
                'trade_count': trade_count,
                'max_trades': max_trades,
                'over_limit': over_limit,
                'compliance': compliance,
                'trading_intensity': trading_intensity,
                'intensity_level': intensity_level,
                'timing_issues': timing_issues,
                'context_appropriateness': context_appropriateness,
                'trade_timing': trade_timing,
                'recent_pnl': float(trading_activity.get('recent_pnl', 0.0))
            }
        except Exception as e:
            self.logger.warning(f"Trading activity analysis failed: {e}")
            return {'trade_count': 0, 'compliance': True, 'over_limit': 0, 'intensity_level': 'low', 'timing_issues': []}

    def _assess_trading_discipline(self, trade_analysis: Dict[str, Any]) -> Dict[str, Any]:
        try:
            compliance_score = 1.0 if trade_analysis.get('compliance', True) else 0.0

            intensity_level = trade_analysis.get('intensity_level', 'low')
            intensity_scores = {'low': 1.0, 'moderate': 0.8, 'high': 0.5, 'excessive': 0.2}
            intensity_score = float(intensity_scores.get(intensity_level, 0.5))

            timing_issues = trade_analysis.get('timing_issues', [])
            timing_score = max(0.0, 1.0 - 0.2 * len(timing_issues))

            context_appropriateness = trade_analysis.get('context_appropriateness', 'appropriate')
            context_scores = {'appropriate': 1.0, 'questionable': 0.6, 'excessive': 0.2}
            context_score = float(context_scores.get(context_appropriateness, 0.5))

            recent_pnl = float(trade_analysis.get('recent_pnl', 0.0))
            performance_score = float(min(1.0, 0.5 + recent_pnl / 1000.0)) if recent_pnl > 0 else max(0.0, 0.5 + recent_pnl / 1000.0)

            weights = {'compliance': 0.3, 'intensity': 0.2, 'timing': 0.2, 'context': 0.2, 'performance': 0.1}
            overall_score = (
                compliance_score * weights['compliance'] +
                intensity_score * weights['intensity'] +
                timing_score * weights['timing'] +
                context_score * weights['context'] +
                performance_score * weights['performance']
            )

            violations = 0
            if not trade_analysis.get('compliance', True):
                violations += 1
            if intensity_level in ['high', 'excessive']:
                violations += 1
            if len(timing_issues) > 1:
                violations += 1
            if context_appropriateness in ['questionable', 'excessive']:
                violations += 1

            return {
                'compliance_score': float(compliance_score),
                'intensity_score': float(intensity_score),
                'timing_score': float(timing_score),
                'context_score': float(context_score),
                'performance_score': float(performance_score),
                'overall_score': float(overall_score),
                'violations': int(violations),
                'discipline_grade': self._get_discipline_grade(overall_score)
            }
        except Exception as e:
            self.logger.warning(f"Discipline assessment failed: {e}")
            return {'overall_score': 0.5, 'violations': 0, 'discipline_grade': 'C'}

    def _get_discipline_grade(self, score: float) -> str:
        if score >= 0.9: return 'A+'
        if score >= 0.8: return 'A'
        if score >= 0.7: return 'B'
        if score >= 0.6: return 'C'
        if score >= 0.5: return 'D'
        return 'F'

    def _calculate_context_aware_penalties(self, discipline_assessment: Dict[str, Any], trading_activity: Dict[str, Any]) -> Dict[str, Any]:
        try:
            violations = int(discipline_assessment.get('violations', 0))
            overall_score = float(discipline_assessment.get('overall_score', 1.0))

            base_penalty = violations * self.adaptive_penalty
            score_penalty = (0.5 - overall_score) * 2.0 * self.adaptive_penalty if overall_score < 0.5 else 0.0

            mode_mult = {'strict': 1.5, 'adaptive': 1.0, 'lenient': 0.7, 'performance_based': 1.2, 'regime_aware': 1.0}.get(self.coaching_mode, 1.0)
            regime_mult = {'volatile': 1.3, 'trending': 0.9, 'ranging': 1.0, 'unknown': 1.0}.get(trading_activity.get('regime', 'unknown'), 1.0)
            session_mult = {'asian': 1.2, 'rollover': 1.4, 'european': 1.0, 'american': 1.0}.get(trading_activity.get('session', 'unknown'), 1.0)

            total_penalty = (base_penalty + score_penalty) * mode_mult * regime_mult * session_mult

            if len(self.penalty_history) > 0:
                last5 = list(self.penalty_history)[-5:]
                if all(p > 0 for p in last5):
                    total_penalty *= self.penalty_decay

            self.penalty_history.append(total_penalty)

            return {
                'base_penalty': float(base_penalty),
                'score_penalty': float(score_penalty),
                'mode_multiplier': float(mode_mult),
                'regime_multiplier': float(regime_mult),
                'session_multiplier': float(session_mult),
                'total_penalty': float(total_penalty),
                'penalty_rationale': f"Violations={violations}, Score={overall_score:.2f}, Mode={self.coaching_mode}"
            }
        except Exception as e:
            self.logger.warning(f"Penalty calculation failed: {e}")
            return {'total_penalty': 0.0, 'penalty_rationale': 'Calculation failed'}

    def _generate_coaching_recommendations(self, trade_analysis: Dict[str, Any], discipline_assessment: Dict[str, Any], trading_activity: Dict[str, Any]) -> List[str]:
        recs: List[str] = []
        try:
            if not trade_analysis.get('compliance', True):
                over = trade_analysis.get('over_limit', 0)
                recs.append(f"🚫 Trade limit exceeded by {over}. Reduce frequency or wait for higher-quality setups.")

            intensity_level = trade_analysis.get('intensity_level', 'low')
            if intensity_level == 'excessive':
                recs.append("[FAST] Excessive trading intensity. Pause and reassess.")
            elif intensity_level == 'high':
                recs.append("[WARN] High trading intensity. Slow down to improve decision quality.")

            timing_issues = trade_analysis.get('timing_issues', [])
            if 'rapid_clustering' in timing_issues:
                recs.append("⏰ Rapid clustering detected. Allow more time between entries.")
            if 'high_frequency' in timing_issues:
                recs.append("[STATS] High-frequency pattern. Prefer higher-R:R setups.")
            if 'excessive_rapid_trades' in timing_issues:
                recs.append("📉 Too many quick trades. Enforce a cool-off window.")

            regime = trading_activity.get('regime', 'unknown')
            vol_level = trading_activity.get('volatility_level', 'medium')
            tc = trade_analysis.get('trade_count', 0)
            if regime == 'volatile' and tc > 1:
                recs.append("[CRASH] Volatile regime. Lower size and tighten criteria.")
            elif regime == 'ranging' and tc > 2:
                recs.append("↔️ Ranging market. Favor breakout/mean-reversion with strict filters.")
            elif regime == 'trending' and tc == 0:
                recs.append("[CHART] Trend in play. Consider momentum setups with strict discipline.")
            if vol_level == 'extreme':
                recs.append("🌪️ Extreme volatility. Reduce size and widen stops judiciously.")

            pnl = trade_analysis.get('recent_pnl', 0.0)
            if pnl < -100:
                recs.append("🧘 Drawdown detected. Step back and review rules.")
            elif pnl > 100:
                recs.append("✅ Solid recent PnL. Protect gains—keep discipline tight.")

            score = discipline_assessment.get('overall_score', 1.0)
            if score < 0.6:
                recs.append("[TARGET] Focus on quality over quantity.")
            elif score > 0.8:
                recs.append("🎯 Excellent discipline—keep it up.")

            session = trading_activity.get('session', 'unknown')
            if session == 'asian' and tc > 1:
                recs.append("🌏 Asian session liquidity is thinner—be extra selective.")
            elif session == 'rollover':
                recs.append("🔄 Rollover period—avoid entries due to spreads.")

            if len(self.penalty_history) > 3 and all(p > 0 for p in list(self.penalty_history)[-3:]):
                recs.append("📚 Repeated penalties. Review playbook & psychology notes.")

        except Exception as e:
            self.logger.warning(f"Recommendation generation failed: {e}")
            recs.append("[WARN] Unable to generate specific recommendations now.")

        return recs[:5]

    def _update_adaptive_parameters(self, session_result: Dict[str, Any]) -> None:
        try:
            overall = session_result.get('discipline_assessment', {}).get('overall_score', 0.5)
            self.current_performance_score = (
                self.current_performance_score * (1 - self.learning_rate) +
                float(overall) * self.learning_rate
            )

            if len(self.effectiveness_scores) > 5:
                avg_eff = float(np.mean(list(self.effectiveness_scores)[-5:]))
                if avg_eff < 0.4:
                    self.adaptive_penalty *= 1.05
                elif avg_eff > 0.8:
                    self.adaptive_penalty *= 0.98
        except Exception as e:
            self.logger.warning(f"Adaptive parameter update failed: {e}")

    def _analyze_coaching_effectiveness(self, session_result: Dict[str, Any]) -> None:
        try:
            overall = session_result.get('discipline_assessment', {}).get('overall_score', 0.5)
            total_penalty = session_result.get('penalties', {}).get('total_penalty', 0.0)
            trade_analysis = session_result.get('trade_analysis', {})
            ctx_ok = trade_analysis.get('context_appropriateness', 'appropriate')

            if overall > 0.8: eff = 0.9
            elif overall > 0.6: eff = 0.7
            elif total_penalty > 0 and overall < 0.5: eff = 0.6
            else: eff = 0.4

            if ctx_ok == 'appropriate': eff += 0.1
            elif ctx_ok == 'excessive': eff -= 0.1

            eff = float(np.clip(eff, 0.0, 1.0))
            self.effectiveness_scores.append(eff)
            self._last_effectiveness = eff

            if len(self.effectiveness_scores) >= 10:
                recent = list(self.effectiveness_scores)[-5:]
                older  = list(self.effectiveness_scores)[-10:-5]
                self.coaching_stats['improvement_rate'] = float(np.mean(recent) - np.mean(older))
            else:
                self.coaching_stats['improvement_rate'] = 0.0

        except Exception as e:
            self.logger.warning(f"Effectiveness analysis failed: {e}")

    # ================== CONTRACT PAYLOAD & PUBLISHING (PROVIDES) ==================
    def _build_contract_payload(self, session_result: Dict[str, Any]) -> Dict[str, Any]:
        penalties = session_result.get('penalties', {})
        trade_analysis = session_result.get('trade_analysis', {})
        discipline_assessment = session_result.get('discipline_assessment', {})
        recommendations = session_result.get('recommendations', [])
        effective_limits = session_result.get('effective_limits', {})

        # compliance block
        compliance_block = {
            'compliance_rate': float(self.coaching_stats.get('compliance_rate', 1.0)),
            'violations_total': int(self.coaching_stats.get('violations', 0)),
            'last_compliance': bool(trade_analysis.get('compliance', True)),
            'last_over_limit': int(trade_analysis.get('over_limit', 0))
        }

        performance_block = {
            'current_performance_score': float(self.current_performance_score),
            'recent_pnl': float(trade_analysis.get('recent_pnl', 0.0)),
            'effectiveness': float(self._last_effectiveness)
        }

        # A compact summary for human/storage
        coaching_results = {
            'timestamp': session_result.get('timestamp'),
            'coaching_mode': self.coaching_mode,
            'context': session_result.get('context', {}),
            'trade_analysis': trade_analysis,
            'discipline_assessment': discipline_assessment,
            'penalties': penalties,
            'recommendations': recommendations,
            'effective_limits': effective_limits
        }

        # contract-shaped payload
        # Compute a concise operator thesis for explainability
        total_penalty_amt = float(penalties.get('total_penalty', 0.0))
        violations_cnt = int(discipline_assessment.get('violations', 0))
        thesis = f"RoleCoach session: violations={violations_cnt}, penalty={total_penalty_amt:.2f}"

        payload: Dict[str, Any] = {
            'coaching_penalties': penalties,                       # dict
            'coaching_recommendations': recommendations,           # list[str]
            'coaching_results': coaching_results,                  # dict
            'coaching_statistics': self.coaching_stats.copy(),     # dict
            'compliance_tracking': compliance_block,               # dict
            'discipline_assessment': discipline_assessment,        # dict
            'discipline_penalty': {                                # dict (explicit single-key view)
                'penalty_amount': float(penalties.get('total_penalty', 0.0)),
                'rationale': penalties.get('penalty_rationale', '')
            },
            'performance_scoring': performance_block,              # dict
            'trade_limits': effective_limits,                      # dict
            '_thesis': thesis,
            'thesis': thesis
        }
        return payload

    async def _publish_contract_outputs(self, payload: Dict[str, Any]) -> None:
        """
        Publish each 'provides' key individually on the InfoBus.
        """
        try:
            total_penalty = payload.get('discipline_penalty', {}).get('penalty_amount', 0.0)
            violations = payload.get('discipline_assessment', {}).get('violations', 0)
            thesis = f"RoleCoach session: violations={violations}, penalty={total_penalty:.2f}"

            # Publish each provided artifact
            for key in [
                'coaching_penalties', 'coaching_recommendations', 'coaching_results',
                'coaching_statistics', 'compliance_tracking', 'discipline_assessment',
                'discipline_penalty', 'performance_scoring', 'trade_limits'
            ]:
                if key in payload:
                    self.smart_bus.set(key, payload[key], module='RoleCoach', thesis=thesis)

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "smartinfobus_publish")
            self.logger.warning(f"SmartInfoBus publish failed: {error_context}")

    # ================== ERRORS & DISABLED ==================
    async def _handle_processing_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        self.error_count += 1
        error_context = self.error_pinpointer.analyze_error(error, "RoleCoach")

        if self.error_count >= self.circuit_breaker_threshold:
            self.is_disabled = True
            self.logger.error(format_operator_message(
                icon="[ALERT]",
                message="RoleCoach disabled due to repeated errors",
                error_count=self.error_count,
                threshold=self.circuit_breaker_threshold
            ))

        return self._generate_error_contract(str(error_context))

    def _generate_disabled_contract(self) -> Dict[str, Any]:
        thesis = "RoleCoach disabled: circuit_breaker_triggered"
        skeleton = {
            'coaching_penalties': {'total_penalty': 0.0, 'penalty_rationale': 'disabled'},
            'coaching_recommendations': [],
            'coaching_results': {'status': 'disabled', 'reason': 'circuit_breaker_triggered'},
            'coaching_statistics': self.coaching_stats.copy(),
            'compliance_tracking': {'compliance_rate': self.coaching_stats.get('compliance_rate', 1.0), 'violations_total': self.coaching_stats.get('violations', 0), 'last_compliance': True, 'last_over_limit': 0},
            'discipline_assessment': {'overall_score': self.coaching_stats.get('discipline_score', 1.0), 'violations': 0, 'discipline_grade': 'N/A'},
            'discipline_penalty': {'penalty_amount': 0.0, 'rationale': 'disabled'},
            'performance_scoring': {'current_performance_score': self.current_performance_score, 'recent_pnl': 0.0, 'effectiveness': 0.0},
            'trade_limits': {'max_trades': self.adaptive_max_trades, 'base_trades': self.max_trades},
            '_thesis': thesis,
            'thesis': thesis
        }
        return skeleton

    def _generate_error_contract(self, error_msg: str) -> Dict[str, Any]:
        thesis = f"RoleCoach error: {error_msg}"
        skeleton = {
            'coaching_penalties': {'total_penalty': 0.0, 'penalty_rationale': 'error'},
            'coaching_recommendations': ["[WARN] RoleCoach encountered an error."],
            'coaching_results': {'status': 'error', 'error': error_msg},
            'coaching_statistics': self.coaching_stats.copy(),
            'compliance_tracking': {'compliance_rate': self.coaching_stats.get('compliance_rate', 1.0), 'violations_total': self.coaching_stats.get('violations', 0), 'last_compliance': True, 'last_over_limit': 0},
            'discipline_assessment': {'overall_score': self.coaching_stats.get('discipline_score', 1.0), 'violations': 0, 'discipline_grade': 'N/A'},
            'discipline_penalty': {'penalty_amount': 0.0, 'rationale': 'error'},
            'performance_scoring': {'current_performance_score': self.current_performance_score, 'recent_pnl': 0.0, 'effectiveness': 0.0},
            'trade_limits': {'max_trades': self.adaptive_max_trades, 'base_trades': self.max_trades},
            '_thesis': thesis,
            'thesis': thesis
        }
        return skeleton

    # ================== PUBLIC (LEGACY) ==================
    def get_coaching_penalty(self, trade_count: int, context: Optional[Dict[str, Any]] = None) -> float:
        if context is None:
            context = {'regime': 'unknown', 'volatility_level': 'medium', 'session': 'unknown'}
        effective_limits = self._calculate_effective_trade_limits({
            'session': context.get('session', 'unknown'),
            'volatility_level': context.get('volatility_level', 'medium')
        })
        max_trades = int(effective_limits.get('max_trades', self.max_trades))
        over_limit = max(0, int(trade_count) - max_trades)
        return float(over_limit * self.adaptive_penalty)

    def get_observation_components(self) -> np.ndarray:
        try:
            mode_idx = float(list(self.COACHING_MODES.keys()).index(self.coaching_mode))
            recent_penalty = list(self.penalty_history)[-1] if self.penalty_history else 0.0
            return np.array([
                float(self.max_trades),
                float(self.adaptive_max_trades),
                float(self.penalty_multiplier),
                float(self.adaptive_penalty),
                mode_idx / len(self.COACHING_MODES),
                float(self.current_performance_score),
                float(self.coaching_stats.get('discipline_score', 1.0)),
                float(min(1.0, float(recent_penalty) / 10.0))
            ], dtype=np.float32)
        except Exception as e:
            self.logger.error(f"Observation generation failed: {e}")
            return np.array([2.0, 2.0, 1.0, 1.0, 0.0, 0.5, 1.0, 0.0], dtype=np.float32)

    def get_role_coaching_report(self) -> str:
        discipline_score = self.coaching_stats.get('discipline_score', 1.0)
        if discipline_score > 0.8:
            discipline_status = "[OK] Excellent"
        elif discipline_score > 0.6:
            discipline_status = "[FAST] Good"
        elif discipline_score > 0.4:
            discipline_status = "[WARN] Needs Improvement"
        else:
            discipline_status = "[ALERT] Poor"

        session_lines = []
        for session in list(self.coaching_sessions)[-3:]:
            ts = session.get('timestamp', '')[:19]
            penalties = session.get('penalties', {})
            penalty = penalties.get('total_penalty', 0.0)
            violations = session.get('discipline_assessment', {}).get('violations', 0)
            if penalty > 0:
                emoji = "[ALERT]" if violations > 2 else "[WARN]"
                session_lines.append(f"  {emoji} {ts}: {violations} violations, penalty {penalty:.2f}")
            else:
                session_lines.append(f"  [OK] {ts}: No violations, good discipline")

        latest_session = list(self.coaching_sessions)[-1] if self.coaching_sessions else {}
        recs = latest_session.get('recommendations', [])
        rec_lines = [f"  • {r}" for r in recs[:3]]

        return f"""
[TARGET] ROLE COACH
═══════════════════════════════════════
[TROPHY] Discipline Status: {discipline_status} ({discipline_score:.1%})
[STATS] Coaching Mode: {self.coaching_mode.title().replace('_', ' ')}
[TARGET] Trade Limits: Base {self.max_trades} | Adaptive {self.adaptive_max_trades}
[BALANCE] Penalty Scale: Base {self.penalty_multiplier:.1f} | Adaptive {self.adaptive_penalty:.1f}
[TOOL] Status: {'[ALERT] Disabled' if self.is_disabled else '[OK] Healthy'}

[CHART] COACHING CONFIGURATION
• Regime Sensitivity: {self.regime_sensitivity:.1%}
• Performance Adjustment: {'[OK] Enabled' if self.performance_adjustment else '[FAIL] Disabled'}
• Session Awareness: {'[OK] Enabled' if self.session_aware else '[FAIL] Disabled'}
• Volatility Scaling: {'[OK] Enabled' if self.volatility_scaling else '[FAIL] Disabled'}
• Learning Rate: {self.learning_rate:.1%}
• Penalty Decay: {self.penalty_decay:.1%}

[STATS] PERFORMANCE STATISTICS
• Total Sessions: {self.coaching_stats['total_sessions']:,}
• Penalties Applied: {self.coaching_stats['penalties_applied']:,}
• Total Penalty Amount: {self.coaching_stats['total_penalty_amount']:.2f}
• Compliance Rate: {self.coaching_stats['compliance_rate']:.1%}
• Violations: {self.coaching_stats['violations']}
• Improvement Rate: {self.coaching_stats['improvement_rate']:+.1%}
• Error Count: {self.error_count}

[TOOL] ADAPTIVE PARAMETERS
• Current Performance Score: {self.current_performance_score:.1%}
• Market Regime: {self.market_regime.title()}
• Volatility Level: {self.volatility_regime.title()}
• Market Session: {self.market_session.title()}

📜 RECENT COACHING SESSIONS
{chr(10).join(session_lines) if session_lines else "  📭 No recent coaching sessions"}

💡 LATEST RECOMMENDATIONS
{chr(10).join(rec_lines) if rec_lines else "  📭 No current recommendations"}
        """

    # ================== STATE MGMT ==================
    def get_state(self) -> Dict[str, Any]:
        return {
            'module_info': {
                'name': 'RoleCoach',
                'version': '3.0.0',
                'last_updated': datetime.datetime.now().isoformat()
            },
            'configuration': {
                'max_trades': self.max_trades,
                'penalty_multiplier': self.penalty_multiplier,
                'coaching_mode': self.coaching_mode,
                'regime_sensitivity': self.regime_sensitivity,
                'performance_adjustment': self.performance_adjustment,
                'session_aware': self.session_aware,
                'volatility_scaling': self.volatility_scaling
            },
            'adaptive_parameters': {
                'adaptive_max_trades': self.adaptive_max_trades,
                'adaptive_penalty': self.adaptive_penalty,
                'current_performance_score': self.current_performance_score
            },
            'market_context': {
                'regime': self.market_regime,
                'volatility_regime': self.volatility_regime,
                'session': self.market_session
            },
            'system_state': {
                'statistics': self.coaching_stats.copy(),
                'error_count': self.error_count,
                'is_disabled': self.is_disabled
            },
            'history': {
                'penalty_history': list(self.penalty_history)[-20:],
                'effectiveness_scores': list(self.effectiveness_scores)[-10:],
                'learning_history': list(self.learning_history)[-10:]
            }
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        try:
            config = state.get("configuration", {})
            self.max_trades = int(config.get("max_trades", self.max_trades))
            self.penalty_multiplier = float(config.get("penalty_multiplier", self.penalty_multiplier))
            self.coaching_mode = config.get("coaching_mode", self.coaching_mode)
            self.regime_sensitivity = float(config.get("regime_sensitivity", self.regime_sensitivity))
            self.performance_adjustment = bool(config.get("performance_adjustment", self.performance_adjustment))
            self.session_aware = bool(config.get("session_aware", self.session_aware))
            self.volatility_scaling = bool(config.get("volatility_scaling", self.volatility_scaling))

            adaptive = state.get("adaptive_parameters", {})
            self.adaptive_max_trades = int(adaptive.get("adaptive_max_trades", self.max_trades))
            self.adaptive_penalty = float(adaptive.get("adaptive_penalty", self.penalty_multiplier))
            self.current_performance_score = float(adaptive.get("current_performance_score", 0.5))

            context = state.get("market_context", {})
            self.market_regime = context.get("regime", "normal")
            self.volatility_regime = context.get("volatility_regime", "medium")
            self.market_session = context.get("session", "unknown")

            system_state = state.get("system_state", {})
            self.coaching_stats.update(system_state.get("statistics", {}))
            self.error_count = system_state.get("error_count", 0)
            self.is_disabled = system_state.get("is_disabled", False)

            history = state.get("history", {})
            self.penalty_history.clear()
            for p in history.get("penalty_history", []):
                self.penalty_history.append(p)
            self.effectiveness_scores.clear()
            for s in history.get("effectiveness_scores", []):
                self.effectiveness_scores.append(s)
            self.learning_history.clear()
            for h in history.get("learning_history", []):
                self.learning_history.append(h)

            self.logger.info(format_operator_message(
                icon="[RELOAD]",
                message="RoleCoach state restored",
                sessions=self.coaching_stats.get('total_sessions', 0),
                penalties=len(self.penalty_history),
                discipline_score=f"{self.coaching_stats.get('discipline_score', 1.0):.1%}"
            ))
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "state_restoration")
            self.logger.error(f"State restoration failed: {error_context}")

    def get_health_status(self) -> Dict[str, Any]:
        return {
            'module_name': 'RoleCoach',
            'status': 'disabled' if self.is_disabled else 'healthy',
            'error_count': self.error_count,
            'circuit_breaker_threshold': self.circuit_breaker_threshold,
            'total_sessions': self.coaching_stats['total_sessions'],
            'penalties_applied': self.coaching_stats['penalties_applied'],
            'discipline_score': self.coaching_stats['discipline_score'],
            'compliance_rate': self.coaching_stats['compliance_rate'],
            'coaching_mode': self.coaching_mode,
            'adaptive_max_trades': self.adaptive_max_trades
        }

    # ================== LEGACY STEP ==================
    def step(self, **kwargs) -> float:
        try:
            trades = kwargs.get('trades', [])
            trade_count = len(trades)
            over_limit = max(0, trade_count - self.max_trades)
            penalty = float(over_limit * self.penalty_multiplier)

            self.coaching_stats['total_sessions'] += 1
            if penalty > 0:
                self.coaching_stats['penalties_applied'] += 1
                self.coaching_stats['total_penalty_amount'] += penalty

            if over_limit > 0:
                self.logger.warning(format_operator_message(
                    icon="[TARGET]",
                    message="Trade discipline violation",
                    trades=f"{trade_count}/{self.max_trades}",
                    penalty=f"{penalty:.2f}",
                    over_limit=over_limit
                ))
            else:
                self.logger.info(format_operator_message(
                    icon="[OK]",
                    message="Trade discipline maintained",
                    trades=f"{trade_count}/{self.max_trades}",
                    penalty="none"
                ))
            return penalty
        except Exception as e:
            self.logger.error(f"Legacy step processing failed: {e}")
            return 0.0
