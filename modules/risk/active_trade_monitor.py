"""
Enhanced Active Trade Monitor with SmartInfoBus Integration
Monitors position duration and provides intelligent alerts with context awareness

Contract Guarantees:
- Always returns: position_duration_risk, duration_alerts, position_tracking, and _thesis
- Writes only its owned keys on the SmartInfoBus (single-writer style)
- All timestamps ISO-8601, numpy scalars cast to Python types
- Background health monitor (daemon) + circuit breaker for repeated errors
"""

from __future__ import annotations

import datetime
import threading
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional, Union, Tuple, Iterable
from collections import deque, defaultdict

from modules.contracts import module_args
import numpy as np

from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusRiskMixin, SmartInfoBusStateMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.performance_tracker import PerformanceTracker


# ─────────────────────────────────────────────────────────────
# Typed configuration (lint-safe) + dict bridge for BaseModule
# ─────────────────────────────────────────────────────────────

@dataclass
class ActiveTradeMonitorConfig:
    enabled: bool = True
    max_duration: int = 200               # hard stop for duration-based ageing
    critical_duration: int = 150          # escalated risk after this
    warning_duration: int = 50            # early attention threshold
    # monitoring
    health_interval_sec: float = 5.0      # daemon heartbeat
    breaker_error_window: int = 10        # how many recent cycles we inspect
    breaker_open_threshold: int = 4       # open breaker if ≥ N failures in window
    breaker_cooldown_sec: float = 20.0    # auto-reset to half-open after cooldown
    # velocity tuning
    rapid_velocity: int = 5               # steps per cycle flagged as rapid
    fast_velocity: int = 3
    history_maxlen: int = 200
    # severity score weights
    w_alert_critical: float = 1.0
    w_alert_warning: float = 0.6
    w_alert_info: float = 0.3
    w_concentration_warning: float = 0.5
    w_concentration_critical: float = 1.0
    # context multipliers
    vol_mult: Dict[str, float] = field(default_factory=lambda: {
        'low': 0.8, 'medium': 1.0, 'high': 1.3, 'extreme': 1.6
    })
    regime_mult: Dict[str, float] = field(default_factory=lambda: {
        'volatile_market': 1.5, 'trending_market': 0.8, 'ranging_market': 1.0
    })
    # namespacing for health/status (avoid collisions)
    status_key: str = "active_trade_monitor_status"
    health_key: str = "active_trade_monitor_health"


# ─────────────────────────────────────────────────────────────
# Module
# ─────────────────────────────────────────────────────────────

@module(**module_args(
    "ActiveTradeMonitor",
    description="Enhanced active trade monitor with intelligent risk assessment",
    error_handling=True,
    hot_reload=True,
    timeout_ms=120,
))
class ActiveTradeMonitor(BaseModule, SmartInfoBusRiskMixin, SmartInfoBusStateMixin):
    """
    Enhanced Active Trade Monitor with SmartInfoBus Integration

    Monitors position durations with intelligent context-aware thresholds,
    velocity analysis, and progressive risk assessment.
    """

    # ── lifecycle ────────────────────────────────────────────

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        # Typed config for internal logic; keep dict for BaseModule
        cfg_dict = (config or {}).copy()
        self._cfg = ActiveTradeMonitorConfig(**{**asdict(ActiveTradeMonitorConfig()), **cfg_dict})
        self.config = cfg_dict  # BaseModule may expect a dict-like

        self._fully_initialized = False
        self._breaker_state = "CLOSED"       # CLOSED | OPEN | HALF_OPEN
        self._last_failure_ts: float = 0.0
        self._recent_failures: deque[bool] = deque(maxlen=self._cfg.breaker_error_window)

        # Initialize advanced systems (logger, bus, perf, etc.)
        self._initialize_advanced_systems()

        # Parent init (may call _initialize)
        super().__init__()

        # Derived shortcuts from config
        self.enabled: bool = bool(self._cfg.enabled)
        self.max_duration: int = int(self._cfg.max_duration)
        self.warning_duration: int = int(self._cfg.warning_duration)
        self.critical_duration: int = int(self._cfg.critical_duration)

        # State
        self._fully_initialized = True
        self._lock = threading.RLock()
        self.position_durations: Dict[str, int] = {}
        self.position_first_seen: Dict[str, str] = {}
        self.position_velocity: Dict[str, int] = {}  # integer steps/cycle
        self.duration_history: deque = deque(maxlen=self._cfg.history_maxlen)

        self.risk_score: float = 0.0
        self.severity_level: str = "normal"
        self.alert_count: int = 0
        self.step_count: int = 0

        self.closure_analytics: Dict[str, int] = {'normal': 0, 'timeout': 0, 'emergency': 0}
        self.regime_performance: Dict[str, Dict[str, Any]] = defaultdict(lambda: {'durations': [], 'closures': 0})

        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_stop = threading.Event()
        self._start_monitoring()

        self.logger.info(format_operator_message(
            icon="[SEARCH]",
            message="ActiveTradeMonitor initialized",
            max_duration=f"{self.max_duration} steps",
            warning_threshold=f"{self.warning_duration} steps",
            enabled=self.enabled
        ))

    def _initialize_advanced_systems(self):
        """Initialize advanced monitoring and error handling systems"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="ActiveTradeMonitor",
            log_path="logs/risk/active_trade_monitor.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("ActiveTradeMonitor", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()

    # ── main loop ────────────────────────────────────────────

    async def process(self, **kwargs) -> Dict[str, Any]:
        """
        Enhanced position duration monitoring with comprehensive analysis.

        Returns a payload that ALWAYS contains:
          - position_duration_risk
          - duration_alerts
          - position_tracking
          - _thesis
        """
        start = time.time()
        try:
            if not self.enabled:
                payload = self._generate_disabled_response()
                self._write_bus_from_payload(payload, thesis=payload["_thesis"])
                return payload

            # Circuit breaker: short-circuit when OPEN (fallback but still return contract)
            if self._breaker_state == "OPEN":
                thesis = ("Circuit breaker OPEN due to repeated errors. "
                          "Skipping monitoring cycle; system will auto-attempt reset.")
                payload = self._fallback_payload(thesis=thesis)
                self._write_bus_from_payload(payload, thesis=thesis)
                return payload

            self.step_count += 1

            # Extract context (safe defaults)
            market_context = self._safe_get_bus_dict('market_context')
            raw_positions = self._safe_get_bus_any('positions', default=[])
            positions_dict = self._normalize_positions(raw_positions)

            # Process monitoring
            monitoring_results = await self._monitor_positions_comprehensive(positions_dict, market_context)

            # Thesis + metrics
            thesis = await self._generate_monitoring_thesis(monitoring_results, market_context)
            risk_metrics = self._calculate_comprehensive_risk_metrics(monitoring_results)

            # Build contract-conform payload
            payload = self._format_provides_output(
                monitoring_results=monitoring_results,
                risk_metrics=risk_metrics,
                thesis=thesis
            )

            # Update bus (single-writer keys only)
            self._write_bus_from_payload(payload, thesis=thesis)

            # Record performance
            elapsed_ms = int((time.time() - start) * 1000)
            self.performance_tracker.record_metric('ActiveTradeMonitor', 'monitoring_cycle', elapsed_ms, True)

            # Bookend: healthy cycle
            self._recent_failures.append(False)
            return payload

        except Exception as e:
            # Failure path: update breaker and return error-safe payload
            self._recent_failures.append(True)
            self._last_failure_ts = time.time()
            self._update_breaker_state_on_failure()

            error_context = self.error_pinpointer.analyze_error(e, "ActiveTradeMonitor.process")
            self.logger.error(f"Position monitoring failed: {error_context}")
            elapsed_ms = int((time.time() - start) * 1000)
            self.performance_tracker.record_metric('ActiveTradeMonitor', 'monitoring_cycle', elapsed_ms, False)

            payload = self._generate_error_response(str(error_context))
            # Try to publish what we can (never violate single-writer)
            try:
                self._write_bus_from_payload(payload, thesis=payload["_thesis"])
            except Exception:
                pass
            return payload

    # ── helpers: bus + contract ─────────────────────────────

    def _write_bus_from_payload(self, payload: Dict[str, Any], thesis: str) -> None:
        """Write only owned keys to the bus; keep single-writer discipline."""
        try:
            self.smart_bus.set('position_duration_risk', payload['position_duration_risk'],
                               module='ActiveTradeMonitor', thesis=thesis)
            self.smart_bus.set('duration_alerts', payload['duration_alerts'],
                               module='ActiveTradeMonitor', thesis=f"Duration alerts updated")
            self.smart_bus.set('position_tracking', payload['position_tracking'],
                               module='ActiveTradeMonitor', thesis="Position tracking metrics updated")
        except Exception as e:
            err = self.error_pinpointer.analyze_error(e, "bus_write")
            self.logger.error(f"SmartInfoBus update failed: {err}")

    def _format_provides_output(
        self,
        monitoring_results: Dict[str, Any],
        risk_metrics: Dict[str, Any],
        thesis: str
    ) -> Dict[str, Any]:
        """Strictly format the provides payload to always include required keys + _thesis."""
        alerts = monitoring_results.get('alerts') or {}
        alerts = {
            'critical': list(alerts.get('critical', [])),
            'warning': list(alerts.get('warning', [])),
            'info': list(alerts.get('info', [])),
        }
        stats = monitoring_results.get('duration_statistics') or {}

        # Cast to python types for serialization safety
        def _py(v):  # small caster for numpy types
            if isinstance(v, (np.generic,)):
                return v.item()
            return v

        payload = {
            'position_duration_risk': {
                'risk_score': float(_py(self.risk_score)),
                'severity_level': str(self.severity_level),
                'monitoring_results': {
                    'positions_tracked': int(_py(monitoring_results.get('positions_tracked', 0))),
                    'duration_statistics': {k: _py(v) for k, v in stats.items()},
                    'closure_info': monitoring_results.get('closure_info', {}),
                    'processing_time_ms': int(_py(monitoring_results.get('processing_time_ms', 0))),
                    'market_context': monitoring_results.get('market_context', {}),
                    'alerts': alerts
                },
                'risk_metrics': {k: _py(v) for k, v in risk_metrics.items()},
                'timestamp': datetime.datetime.now().isoformat()
            },
            'duration_alerts': alerts,
            'position_tracking': {
                'durations': {k: int(_py(v)) for k, v in self.position_durations.items()},
                'velocities': {k: int(_py(v)) for k, v in self.position_velocity.items()},
                'statistics': {k: _py(v) for k, v in stats.items()}
            },
            '_thesis': thesis
        }
        return payload

    # ── monitoring core ─────────────────────────────────────

    async def _monitor_positions_comprehensive(
        self,
        positions: Dict[str, Dict[str, Any]],
        market_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Comprehensive position monitoring with intelligent analysis"""
        start_time = time.time()

        alerts = {'critical': [], 'warning': [], 'info': []}
        current_ids: set[str] = set()

        with self._lock:
            for pid, position in positions.items():
                try:
                    symbol = position.get('symbol') or position.get('instrument') or 'UNKNOWN'
                    current_ids.add(pid)

                    # Calculate duration + velocity
                    duration_info = self._calculate_enhanced_duration(position, pid)
                    self.position_durations[pid] = duration_info['duration']
                    self.position_velocity[pid] = duration_info['velocity']

                    # First-seen timestamp
                    if pid not in self.position_first_seen:
                        self.position_first_seen[pid] = datetime.datetime.now().isoformat()

                    # Assess severity with context
                    severity_info = self._assess_position_severity_enhanced(
                        pid, duration_info, position, market_context
                    )

                    level = severity_info['level']
                    if level in ('critical', 'warning', 'info'):
                        alerts[level].append({
                            'position_id': pid,
                            'symbol': symbol,
                            'duration': duration_info['duration'],
                            'velocity': duration_info['velocity'],
                            'severity': severity_info,
                            'position_info': self._safe_position_snapshot(position),
                        })

                except Exception as e:
                    error_context = self.error_pinpointer.analyze_error(e, "position_processing")
                    self.logger.warning(f"Position processing failed for {pid}: {error_context}")

            # Handle closures
            closure_info = self._process_position_closures(current_ids, market_context)

            processing_time = int((time.time() - start_time) * 1000)

            return {
                'alerts': alerts,
                'positions_tracked': len(current_ids),
                'duration_statistics': self._calculate_duration_statistics(),
                'closure_info': closure_info,
                'processing_time_ms': processing_time,
                'market_context': market_context
            }

    def _calculate_enhanced_duration(self, position: Dict[str, Any], pid: str) -> Dict[str, Any]:
        """Calculate enhanced duration metrics with velocity analysis (per position id)"""
        try:
            # Priority 1: explicit duration/bars_held
            raw_duration = position.get('duration', position.get('bars_held', None))
            if raw_duration is not None:
                base = max(0, int(raw_duration))
            else:
                # Priority 2: derive from step_idx - entry_step
                entry_step = int(position.get('entry_step', 0) or 0)
                current_step = int(self.smart_bus.get('step_idx', 'ActiveTradeMonitor') or self.step_count)
                base = max(0, current_step - entry_step) if entry_step > 0 else 0

            prev = int(self.position_durations.get(pid, 0))
            velocity = max(0, base - prev)  # monotonic duration; clamp negatives to 0

            # History for analytics
            self.duration_history.append({
                'position_id': pid,
                'duration': base,
                'velocity': velocity,
                'timestamp': datetime.datetime.now().isoformat()
            })

            return {
                'duration': base,
                'velocity': velocity,
                'acceleration': velocity - int(self.position_velocity.get(pid, 0)),
                'trend': 'increasing' if velocity > 0 else ('stable' if velocity == 0 else 'decreasing')
            }

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "duration_calculation")
            self.logger.warning(f"Duration calculation failed for {pid}: {error_context}")
            return {'duration': 0, 'velocity': 0, 'acceleration': 0, 'trend': 'unknown'}

    def _assess_position_severity_enhanced(
        self,
        pid: str,
        duration_info: Dict[str, Any],
        position: Dict[str, Any],
        market_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhanced severity assessment with context awareness and velocity analysis"""
        try:
            duration = int(duration_info['duration'])
            velocity = int(duration_info['velocity'])

            # Get context-adjusted thresholds
            regime = str(market_context.get('regime', 'ranging'))
            volatility = str(market_context.get('volatility_level', 'medium'))
            thresholds = self._get_context_adjusted_thresholds(regime, volatility)

            # Base severity
            if duration >= thresholds['critical']:
                base_level = 'critical'
            elif duration >= thresholds['warning']:
                base_level = 'warning'
            elif duration >= thresholds['info']:
                base_level = 'info'
            else:
                base_level = 'normal'

            context_factors: List[str] = []

            # Velocity escalators/relaxers
            if velocity >= self._cfg.rapid_velocity and base_level == 'info':
                base_level = 'warning'
                context_factors.append('rapid_duration_increase')

            # Profit tolerance
            pnl = float(position.get('unrealised_pnl', position.get('pnl', 0.0)) or 0.0)
            if pnl > 0 and base_level == 'warning' and duration < thresholds['critical']:
                base_level = 'info'
                context_factors.append('profitable_position_tolerance')

            # High volatility tolerance
            if volatility in ('high', 'extreme') and base_level == 'warning' and duration < int(thresholds['critical'] * 0.9):
                base_level = 'info'
                context_factors.append('high_volatility_tolerance')

            return {
                'level': base_level,
                'threshold_used': int(thresholds['info'] if base_level == 'normal' else thresholds[base_level]),
                'context_factors': context_factors,
                'position_pnl': pnl,
                'regime_factor': regime,
                'volatility_factor': volatility
            }

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "severity_assessment")
            self.logger.warning(f"Severity assessment failed for {pid}: {error_context}")
            return {'level': 'unknown', 'threshold_used': 0, 'context_factors': ['assessment_error']}

    def _get_context_adjusted_thresholds(self, regime: str, volatility: str) -> Dict[str, int]:
        """Compute thresholds with regime/volatility multipliers"""
        base = {'info': self.warning_duration, 'warning': self.critical_duration, 'critical': self.max_duration}

        # Map regime to key expected in config
        regime_key = f"{regime}_market" if not regime.endswith("_market") else regime
        reg_mult = float(self._cfg.regime_mult.get(regime_key, self._cfg.regime_mult['ranging_market']))
        vol_mult = float(self._cfg.vol_mult.get(volatility, 1.0))
        final_mult = reg_mult * vol_mult

        return {lvl: int(val * final_mult) for lvl, val in base.items()}

    def _process_position_closures(self, current_ids: set[str], market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Process closures and update analytics"""
        closed_ids = set(self.position_durations.keys()) - current_ids
        closure_info = {'closed_count': len(closed_ids), 'closure_details': []}

        for pid in closed_ids:
            duration = int(self.position_durations.get(pid, 0))

            if duration >= self.max_duration:
                ctype = 'timeout'
            elif duration >= self.critical_duration:
                ctype = 'emergency'
            else:
                ctype = 'normal'

            self.closure_analytics[ctype] += 1

            regime = str(market_context.get('regime', 'unknown'))
            self.regime_performance[regime]['durations'].append(duration)
            self.regime_performance[regime]['closures'] += 1

            closure_info['closure_details'].append({
                'position_id': pid,
                'duration': duration,
                'type': ctype,
                'first_seen': self.position_first_seen.get(pid)
            })

            # cleanup
            self.position_durations.pop(pid, None)
            self.position_first_seen.pop(pid, None)
            self.position_velocity.pop(pid, None)

            if ctype in ('timeout', 'emergency'):
                self.logger.warning(format_operator_message(
                    icon="⏰",
                    message=f"Position closed - {ctype}",
                    position_id=pid,
                    duration=f"{duration} steps",
                    regime=regime
                ))

        return closure_info

    def _calculate_duration_statistics(self) -> Dict[str, Any]:
        """Aggregate duration stats (safe and typed)"""
        try:
            if not self.position_durations:
                return {'active_positions': 0, 'avg_duration': 0, 'max_duration': 0, 'min_duration': 0,
                        'std_duration': 0, 'median_duration': 0, 'positions_over_warning': 0,
                        'positions_over_critical': 0}

            durations = list(int(v) for v in self.position_durations.values())
            arr = np.array(durations, dtype=np.int32)

            return {
                'active_positions': int(arr.size),
                'avg_duration': float(np.mean(arr)),
                'max_duration': int(np.max(arr)),
                'min_duration': int(np.min(arr)),
                'std_duration': float(np.std(arr)),
                'median_duration': float(np.median(arr)),
                'positions_over_warning': int(np.sum(arr >= self.warning_duration)),
                'positions_over_critical': int(np.sum(arr >= self.critical_duration)),
            }

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "statistics_calculation")
            self.logger.warning(f"Statistics calculation failed: {error_context}")
            return {'active_positions': 0, 'avg_duration': 0, 'max_duration': 0, 'min_duration': 0,
                    'std_duration': 0, 'median_duration': 0, 'positions_over_warning': 0,
                    'positions_over_critical': 0}

    def _calculate_comprehensive_risk_metrics(self, monitoring_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute risk score from alerts, concentration and velocity distribution."""
        try:
            alerts = monitoring_results['alerts']
            stats = monitoring_results['duration_statistics']

            # Alert risk
            alert_risk = (
                len(alerts['critical']) * self._cfg.w_alert_critical +
                len(alerts['warning']) * self._cfg.w_alert_warning +
                len(alerts['info']) * self._cfg.w_alert_info
            ) / max(stats.get('active_positions', 1), 1)

            # Concentration risk
            concentration_risk = 0.0
            ap = stats.get('active_positions', 0)
            if ap > 0:
                over_warn = stats.get('positions_over_warning', 0) / ap
                over_crit = stats.get('positions_over_critical', 0) / ap
                concentration_risk = (over_warn * self._cfg.w_concentration_warning +
                                      over_crit * self._cfg.w_concentration_critical)

            # Velocity risk
            vel_vals = list(int(v) for v in self.position_velocity.values())
            rapid = sum(1 for v in vel_vals if v > self._cfg.fast_velocity)
            velocity_risk = (rapid / max(len(vel_vals), 1)) if vel_vals else 0.0

            # Combined risk score (bounded)
            self.risk_score = float(np.clip(alert_risk + concentration_risk + velocity_risk, 0.0, 1.0))

            # Severity
            if self.risk_score > 0.7 or len(alerts['critical']) > 0:
                self.severity_level = 'critical'
            elif self.risk_score > 0.4 or len(alerts['warning']) > 0:
                self.severity_level = 'warning'
            elif self.risk_score > 0.1 or len(alerts['info']) > 0:
                self.severity_level = 'elevated'
            else:
                self.severity_level = 'normal'

            return {
                'risk_score': self.risk_score,
                'severity_level': self.severity_level,
                'alert_risk': float(alert_risk),
                'concentration_risk': float(concentration_risk),
                'velocity_risk': float(velocity_risk),
                'total_alerts': int(len(alerts['critical']) + len(alerts['warning']) + len(alerts['info']))
            }

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "risk_metrics")
            self.logger.error(f"Risk metrics calculation failed: {error_context}")
            self.risk_score = 0.5
            self.severity_level = 'unknown'
            return {'risk_score': 0.5, 'severity_level': 'unknown',
                    'alert_risk': 0.0, 'concentration_risk': 0.0, 'velocity_risk': 0.0,
                    'total_alerts': 0}

    async def _generate_monitoring_thesis(self, monitoring_results: Dict[str, Any],
                                          market_context: Dict[str, Any]) -> str:
        """Generate plain-English thesis explaining monitoring decisions."""
        try:
            stats = monitoring_results.get('duration_statistics', {})
            alerts = monitoring_results.get('alerts', {'critical': [], 'warning': [], 'info': []})
            regime = str(market_context.get('regime', 'ranging'))
            volatility = str(market_context.get('volatility_level', 'medium'))

            parts: List[str] = []

            ap = int(stats.get('active_positions', 0))
            if ap > 0:
                parts.append(f"Monitoring {ap} active positions (avg {stats.get('avg_duration', 0):.1f} steps).")
                if int(stats.get('positions_over_critical', 0)) > 0:
                    parts.append(f"CRITICAL: {int(stats['positions_over_critical'])} exceed {self.critical_duration} steps.")
                elif int(stats.get('positions_over_warning', 0)) > 0:
                    parts.append(f"WARNING: {int(stats['positions_over_warning'])} near duration limits.")
                else:
                    parts.append("Durations within acceptable ranges.")
            else:
                parts.append("No active positions.")

            adj = self._get_context_adjusted_thresholds(regime, volatility)
            parts.append(f"Context: regime={regime}, vol={volatility}, warning_threshold={adj['warning']} steps.")

            total_alerts = len(alerts['critical']) + len(alerts['warning']) + len(alerts['info'])
            if total_alerts > 0:
                parts.append(f"Alerts: {len(alerts['critical'])} critical, {len(alerts['warning'])} warning, {len(alerts['info'])} info.")

            parts.append(f"Overall duration risk: {self.severity_level.upper()} (score {self.risk_score:.2f}).")

            total_closures = sum(self.closure_analytics.values())
            if total_closures > 0 and self.closure_analytics['timeout'] > 0:
                timeout_rate = self.closure_analytics['timeout'] / total_closures
                parts.append(f"Historical timeout rate {timeout_rate:.1%}. Consider tightening management.")

            return " ".join(parts)

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "thesis_generation")
            return f"Thesis generation failed: {error_context}"

    # ── recommendations & actions ────────────────────────────

    def _generate_recommendations(self, monitoring_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations (not bus-published; used by propose_action)."""
        recs: List[str] = []
        try:
            alerts = monitoring_results.get('alerts', {'critical': [], 'warning': [], 'info': []})
            stats = monitoring_results.get('duration_statistics', {})

            if alerts['critical']:
                recs.append("IMMEDIATE: Close or reduce positions exceeding maximum duration.")
                recs.append("Consider emergency risk reduction measures.")

            if alerts['warning']:
                recs.append("Review positions approaching duration limits.")
                recs.append("Tighten stops or realize partial profits.")

            if int(stats.get('active_positions', 0)) > 5:
                recs.append("High concentration: consider reducing open positions.")

            rapid_positions = [pid for pid, v in self.position_velocity.items() if v > self._cfg.fast_velocity]
            if rapid_positions:
                recs.append(f"Monitor rapidly aging positions: {', '.join(list(map(str, rapid_positions))[:5])}")

            if self.closure_analytics['timeout'] > self.closure_analytics['normal']:
                recs.append("Timeouts > normal closures: revisit duration management rules.")

            if not recs:
                recs.append("Duration monitoring optimal—maintain current strategy.")

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "recommendations")
            recs.append(f"Recommendation generation failed: {error_context}")

        return recs

    async def calculate_confidence(self, action: Dict[str, Any], **inputs) -> float:
        """Confidence in proposed action, bounded [0.1, 1.0]."""
        try:
            base = 0.7
            n = len(self.position_durations)
            if n > 0:
                base += min(0.3, (n / 10.0) * 0.3)

            if self.risk_score < 0.3:
                base += 0.2
            elif self.risk_score > 0.7:
                base += 0.1

            base += {'normal': 0.1, 'warning': 0.0, 'critical': -0.1, 'emergency': -0.2}.get(self.severity_level, 0.0)
            return float(np.clip(base, 0.1, 1.0))
        except Exception as e:
            self.logger.warning(f"Calculate confidence failed: {e}")
            return 0.5

    async def propose_action(self, **inputs) -> Dict[str, Any]:
        """Propose risk actions; supports positions as dict or list."""
        try:
            raw_positions = inputs.get('positions')
            if raw_positions is None:
                raw_positions = self._safe_get_bus_any('positions', default=[])
            positions = self._normalize_positions(raw_positions)

            # snapshot severity
            duration_risks: Dict[str, Any] = {}
            recommendations: List[Dict[str, Any]] = []

            for pid, _ in positions.items():
                if pid in self.position_durations:
                    dur = int(self.position_durations[pid])
                    if dur > self.critical_duration:
                        risk_level = 'critical'
                        recommendations.append({
                            'position_id': pid,
                            'action': 'close_position',
                            'reason': f'Duration {dur} exceeds critical {self.critical_duration}',
                            'urgency': 'high'
                        })
                    elif dur > self.warning_duration:
                        risk_level = 'warning'
                        recommendations.append({
                            'position_id': pid,
                            'action': 'review_position',
                            'reason': f'Duration {dur} exceeds warning {self.warning_duration}',
                            'urgency': 'medium'
                        })
                    else:
                        risk_level = 'normal'

                    duration_risks[pid] = {
                        'duration': dur,
                        'risk_level': risk_level,
                        'threshold_ratio': float(dur / max(1, self.max_duration))
                    }

            if self.severity_level == 'emergency':
                overall = 'reduce_exposure'
            elif self.severity_level == 'critical':
                overall = 'close_risky_positions'
            elif self.severity_level == 'warning':
                overall = 'increase_monitoring'
            else:
                overall = 'monitor'

            return {
                'action_type': 'duration_risk_management',
                'overall_action': overall,
                'severity_level': self.severity_level,
                'risk_score': self.risk_score,
                'position_risks': duration_risks,
                'recommendations': recommendations,
                'confidence': await self.calculate_confidence({}, **inputs),
                'timestamp': datetime.datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Propose action failed: {e}")
            return {
                'action_type': 'duration_risk_management',
                'overall_action': 'monitor',
                'error': str(e),
                'confidence': 0.1
            }

    # ── health & lifecycle ───────────────────────────────────

    def _start_monitoring(self):
        """Background daemon that posts health + manages circuit breaker resets."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return

        def _loop():
            self.logger.info("[MONITOR] ActiveTradeMonitor health monitor started.")
            while not self._monitor_stop.is_set():
                try:
                    # breaker auto-reset
                    self._maybe_reset_breaker()

                    # health snapshot
                    health = self.get_health_metrics()
                    status = {
                        'initialized': True,
                        'enabled': self.enabled,
                        'max_duration': self.max_duration,
                        'warning_duration': self.warning_duration,
                        'positions_tracked': int(health.get('positions_tracked', 0)),
                        'breaker_state': self._breaker_state,
                        'ts': datetime.datetime.now().isoformat()
                    }
                    # Namespaced keys to avoid collisions with other modules
                    self.smart_bus.set(self._cfg.status_key, status, module='ActiveTradeMonitor',
                                       thesis="ActiveTradeMonitor status heartbeat")
                    self.smart_bus.set(self._cfg.health_key, health, module='ActiveTradeMonitor',
                                       thesis="ActiveTradeMonitor health metrics")

                except Exception as e:
                    self.logger.warning(f"[MONITOR] health update failed: {e}")

                time.sleep(max(0.5, float(self._cfg.health_interval_sec)))

        self._monitor_thread = threading.Thread(target=_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self):
        """Stop background monitor."""
        self._monitor_stop.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

    def _maybe_reset_breaker(self):
        """Reset OPEN breaker to HALF_OPEN after cooldown; close after a successful cycle."""
        if self._breaker_state == "OPEN":
            if (time.time() - self._last_failure_ts) >= self._cfg.breaker_cooldown_sec:
                self._breaker_state = "HALF_OPEN"
        # If HALF_OPEN and recent failures are not growing, close it optimistically
        if self._breaker_state == "HALF_OPEN":
            # If last N entries show mostly success (<=1 failure), close
            if list(self._recent_failures).count(True) <= 1:
                self._breaker_state = "CLOSED"

    def _update_breaker_state_on_failure(self):
        """Open breaker when repeated failures in the sliding window exceed threshold."""
        if list(self._recent_failures).count(True) >= self._cfg.breaker_open_threshold:
            self._breaker_state = "OPEN"

    def _initialize(self):
        """Initialize module-specific state (called by BaseModule.__init__)."""
        if not getattr(self, '_fully_initialized', False):
            return
        self.logger.info("[RELOAD] ActiveTradeMonitor async initialization")
        # Initial status post (namespaced)
        try:
            self.smart_bus.set(
                self._cfg.status_key,
                {
                    'initialized': True,
                    'enabled': self.enabled,
                    'max_duration': self.max_duration,
                    'warning_duration': self.warning_duration,
                    'positions_tracked': 0,
                    'breaker_state': self._breaker_state,
                    'ts': datetime.datetime.now().isoformat()
                },
                module='ActiveTradeMonitor',
                thesis="Trade monitor initialization status for system awareness"
            )
        except Exception as e:
            self.logger.warning(f"Initial status bus write failed: {e}")

    # ── state, health, utils ─────────────────────────────────

    def get_state(self) -> Dict[str, Any]:
        """Get complete module state for hot-reload (serialization-safe)."""
        with self._lock:
            return {
                'position_durations': {k: int(v) for k, v in self.position_durations.items()},
                'position_first_seen': dict(self.position_first_seen),
                'position_velocity': {k: int(v) for k, v in self.position_velocity.items()},
                'risk_score': float(self.risk_score),
                'severity_level': str(self.severity_level),
                'closure_analytics': dict(self.closure_analytics),
                'step_count': int(self.step_count),
                'config': dict(self.config),
                'breaker_state': self._breaker_state
            }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set module state for hot-reload (defensive)."""
        with self._lock:
            self.position_durations = {k: int(v) for k, v in state.get('position_durations', {}).items()}
            self.position_first_seen = dict(state.get('position_first_seen', {}))
            self.position_velocity = {k: int(v) for k, v in state.get('position_velocity', {}).items()}
            self.risk_score = float(state.get('risk_score', 0.0))
            self.severity_level = str(state.get('severity_level', 'normal'))
            self.closure_analytics = dict(state.get('closure_analytics', {'normal': 0, 'timeout': 0, 'emergency': 0}))
            self.step_count = int(state.get('step_count', 0))
            self.config.update(dict(state.get('config', {})))
            self._breaker_state = str(state.get('breaker_state', 'CLOSED'))

    def get_health_metrics(self) -> Dict[str, Any]:
        """Health metrics for monitoring (serialization-safe)."""
        with self._lock:
            total_closures = max(1, sum(self.closure_analytics.values()))
            avg_dur = np.mean(list(self.position_durations.values())) if self.position_durations else 0.0
            return {
                'positions_tracked': int(len(self.position_durations)),
                'risk_score': float(self.risk_score),
                'severity_level': str(self.severity_level),
                'timeout_rate': float(self.closure_analytics['timeout'] / total_closures),
                'avg_position_duration': float(avg_dur),
                'enabled': bool(self.enabled)
            }

    # ── error + fallback payloads ────────────────────────────

    def _generate_disabled_response(self) -> Dict[str, Any]:
        """Return payload when module is disabled (contract-conform)."""
        thesis = "Active Trade Monitor is disabled"
        return {
            'position_duration_risk': {
                'risk_score': 0.0,
                'severity_level': 'disabled',
                'monitoring_results': {
                    'positions_tracked': 0,
                    'duration_statistics': {'active_positions': 0, 'avg_duration': 0, 'max_duration': 0,
                                            'min_duration': 0, 'std_duration': 0, 'median_duration': 0,
                                            'positions_over_warning': 0, 'positions_over_critical': 0},
                    'closure_info': {'closed_count': 0, 'closure_details': []},
                    'processing_time_ms': 0,
                    'market_context': {},
                    'alerts': {'critical': [], 'warning': [], 'info': []}
                },
                'risk_metrics': {'risk_score': 0.0, 'severity_level': 'disabled'},
                'timestamp': datetime.datetime.now().isoformat()
            },
            'duration_alerts': {'critical': [], 'warning': [], 'info': []},
            'position_tracking': {'durations': {}, 'velocities': {}, 'statistics': {}},
            '_thesis': thesis
        }

    def _fallback_payload(self, thesis: str) -> Dict[str, Any]:
        """Return a safe payload used when breaker is OPEN."""
        return {
            'position_duration_risk': {
                'risk_score': float(self.risk_score),
                'severity_level': str(self.severity_level),
                'monitoring_results': {
                    'positions_tracked': int(len(self.position_durations)),
                    'duration_statistics': self._calculate_duration_statistics(),
                    'closure_info': {'closed_count': 0, 'closure_details': []},
                    'processing_time_ms': 0,
                    'market_context': {},
                    'alerts': {'critical': [], 'warning': [], 'info': []}
                },
                'risk_metrics': {'risk_score': float(self.risk_score), 'severity_level': str(self.severity_level)},
                'timestamp': datetime.datetime.now().isoformat()
            },
            'duration_alerts': {'critical': [], 'warning': [], 'info': []},
            'position_tracking': {
                'durations': {k: int(v) for k, v in self.position_durations.items()},
                'velocities': {k: int(v) for k, v in self.position_velocity.items()},
                'statistics': self._calculate_duration_statistics()
            },
            '_thesis': thesis
        }

    def _generate_error_response(self, error_context: str) -> Dict[str, Any]:
        """Return payload on processing failure (contract-conform)."""
        thesis = f"Position monitoring failed: {error_context}"
        return {
            'position_duration_risk': {
                'risk_score': 0.5,
                'severity_level': 'error',
                'monitoring_results': {
                    'positions_tracked': 0,
                    'duration_statistics': {'active_positions': 0, 'avg_duration': 0, 'max_duration': 0,
                                            'min_duration': 0, 'std_duration': 0, 'median_duration': 0,
                                            'positions_over_warning': 0, 'positions_over_critical': 0},
                    'closure_info': {'closed_count': 0, 'closure_details': []},
                    'processing_time_ms': 0,
                    'market_context': {},
                    'alerts': {'critical': [], 'warning': [], 'info': []}
                },
                'risk_metrics': {'risk_score': 0.5, 'severity_level': 'error'},
                'timestamp': datetime.datetime.now().isoformat()
            },
            'duration_alerts': {'critical': [], 'warning': [], 'info': []},
            'position_tracking': {'durations': {}, 'velocities': {}, 'statistics': {}},
            '_thesis': thesis
        }

    # ── utils: inputs & normalization ────────────────────────

    def _safe_get_bus_any(self, key: str, default: Any = None) -> Any:
        try:
            v = self.smart_bus.get(key, 'ActiveTradeMonitor')
            return v if v is not None else default
        except Exception:
            return default

    def _safe_get_bus_dict(self, key: str) -> Dict[str, Any]:
        v = self._safe_get_bus_any(key, default={})
        return v if isinstance(v, dict) else {}

    def _normalize_positions(self, raw: Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        """
        Accepts:
            - list of position dicts (must contain an id if available or we synthesize one)
            - dict mapping id -> position dict
        Returns dict: id -> position dict (with at least 'symbol'/'instrument' if available)
        """
        positions: Dict[str, Dict[str, Any]] = {}
        if isinstance(raw, dict):
            for k, v in raw.items():
                if isinstance(v, dict):
                    positions[str(k)] = v
        elif isinstance(raw, list):
            for i, p in enumerate(raw):
                if not isinstance(p, dict):
                    continue
                pid = str(p.get('id') or p.get('ticket') or p.get('order_id') or p.get('symbol') or f"pos_{i}")
                positions[pid] = p
        else:
            # unknown shape; return empty
            pass
        return positions

    @staticmethod
    def _safe_position_snapshot(position: Dict[str, Any]) -> Dict[str, Any]:
        """Small snapshot to avoid dumping the whole position structure to alerts."""
        keys = ('symbol', 'instrument', 'volume', 'side', 'unrealised_pnl', 'pnl', 'entry_step', 'duration', 'bars_held')
        snap = {k: position.get(k) for k in keys if k in position}
        # cast numpy
        for k, v in list(snap.items()):
            if isinstance(v, np.generic):
                snap[k] = v.item()
        return snap
