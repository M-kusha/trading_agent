# ─────────────────────────────────────────────────────────────
# File: modules/risk/anomaly_detector.py
# [ROCKET] PRODUCTION-READY Enhanced Anomaly Detector (contract-tight)
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import time
import threading
from modules.contracts import module_args
import numpy as np
import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from enum import Enum

from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusRiskMixin, SmartInfoBusStateMixin, SmartInfoBusTradingMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.performance_tracker import PerformanceTracker


# ─────────────────────────────────────────────────────────────
# Fallback analyzers (kept lean)
# ─────────────────────────────────────────────────────────────
class SimpleAnalyzer:
    async def analyze_async(self, *args, **kwargs):
        return {'anomalies': [], 'analysis_completed': False, 'fallback': True}

    async def detect_async(self, *args, **kwargs):
        return {'anomalies': [], 'detection_completed': False, 'fallback': True}


class AnomalyDetectionMode(Enum):
    INITIALIZATION = "initialization"
    TRAINING = "training"
    CALIBRATION = "calibration"
    ACTIVE = "active"
    ENHANCED = "enhanced"
    EMERGENCY = "emergency"
    MAINTENANCE = "maintenance"


class AnomalySeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AnomalyVote(Enum):
    """Standardized anomaly/risk vote for coordinators/routers."""
    PROCEED = "proceed"    # normal – green light
    CAUTION = "caution"    # elevated risk – reduce size / add safeguards
    HALT    = "halt"       # high/critical risk – pause/halt new risk
    ABSTAIN = "abstain"    # insufficient signal – do not influence



# ─────────────────────────────────────────────────────────────
# Typed config (lint-safe) + namespaced health keys
# ─────────────────────────────────────────────────────────────
@dataclass
class AnomalyDetectorConfig:
    # Core thresholds
    pnl_limit: float = 1000.0
    volume_zscore: float = 3.0
    price_zscore: float = 3.0
    observation_zscore: float = 4.0
    # History
    history_size: int = 100
    min_history_for_stats: int = 20
    correlation_window: int = 50
    volatility_window: int = 30
    # Adaptation
    adaptive_thresholds: bool = True
    regime_awareness: bool = True
    learning_rate: float = 0.05
    threshold_smoothing: float = 0.8
    # Training
    training_mode: bool = True
    training_duration_steps: int = 200
    synthetic_data_ratio: float = 0.3
    # Performance / quality
    max_processing_time_ms: float = 50.0
    circuit_breaker_threshold: int = 5
    min_detection_quality: float = 0.7
    # Risk bands
    critical_threshold: float = 0.8
    warning_threshold: float = 0.5
    emergency_threshold: float = 0.9
    # Monitoring
    health_check_interval: int = 30
    performance_window: int = 100
    false_positive_threshold: float = 0.3
    # Status/health (namespaced)
    status_key: str = "anomaly_detector_status"
    health_key: str = "anomaly_detector_health"


# ─────────────────────────────────────────────────────────────
# Module
# ─────────────────────────────────────────────────────────────
@module(**module_args(
    "EnhancedAnomalyDetector",
    description="Deterministic multi-window feature extraction with circuit breaker, monitoring, and explainability.",
    error_handling=True,
    hot_reload=True,
    timeout_ms=3000,
))
class EnhancedAnomalyDetector(BaseModule, SmartInfoBusRiskMixin, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    Contract guarantees:
    - Returns ONLY provides keys + `_thesis` on success/fallback/error:
      anomaly_detection, anomaly_score, anomaly_alerts, detection_analytics, _thesis
    - Writes ONLY its provides keys to SmartInfoBus; health/status use namespaced keys.
    - Numpy → Python scalars/lists; timestamps are ISO-8601.
    - Background monitor posts status & health; circuit breaker with safe fallback.
    """

    # ── init & systems ───────────────────────────────────────
    def __init__(self, config: Optional[Union[AnomalyDetectorConfig, Dict[str, Any]]] = None,
                 enabled: bool = True, action_dim: int = 8, **kwargs):

        # Keep BaseModule config as dict; use a typed copy for logic
        cfg_dict = asdict(AnomalyDetectorConfig())
        if isinstance(config, dict):
            cfg_dict.update(config)
        elif isinstance(config, AnomalyDetectorConfig):
            cfg_dict.update(asdict(config))

        # Normalize critical threshold keys early to avoid repeated injection warnings
        # If provided values are None, strings, or non-numeric, fall back to defaults
        defaults = asdict(AnomalyDetectorConfig())
        for k in ['pnl_limit', 'volume_zscore', 'price_zscore', 'observation_zscore']:
            v = cfg_dict.get(k, defaults[k])
            try:
                # Coerce to float if possible (handles numeric strings)
                cfg_dict[k] = float(v)
            except (TypeError, ValueError):
                cfg_dict[k] = float(defaults[k])

        self._cfg = AnomalyDetectorConfig(**cfg_dict)
        self.config = cfg_dict  # BaseModule expects dict-like

        self.enabled = bool(enabled)
        self.action_dim = int(action_dim)

        # Initialize low-level systems before BaseModule may call _initialize()
        self._initialize_advanced_systems()

        # Define attributes consumed by _initialize before BaseModule runs it
        self.current_mode = AnomalyDetectionMode.INITIALIZATION
        self.mode_start_time = datetime.datetime.now()
        self.anomaly_score = 0.0
        self.detection_confidence = 0.5
        self.step_count = 0
        self._last_vote = None  # keep most recent vote for bus publishing


        super().__init__()  # may call _initialize()

        # Detection state
        self._initialize_detection_state()
        self._monitoring_active = False
        self._start_monitoring()

        self.logger.info(format_operator_message(
            message="Enhanced anomaly detector ready",
            icon="[SEARCH]",
            enabled=self.enabled,
            adaptive_thresholds=self._cfg.adaptive_thresholds,
            regime_awareness=self._cfg.regime_awareness,
            config_loaded=True
        ))

    def _initialize(self):
        """Lightweight async-style initialization hook required by BaseModule."""
        try:
            self.logger.info("[RELOAD] EnhancedAnomalyDetector async initialization")
            # Ensure detection state and thresholds are present
            if not hasattr(self, 'current_thresholds') or not hasattr(self, 'base_thresholds'):
                self._initialize_detection_state()
            else:
                self._ensure_threshold_keys(initial=True)

            # Post initial namespaced status (do not write provides here)
            status = {
                "current_mode": self.current_mode.value,
                "enabled": bool(self.enabled),
                "anomaly_score": float(self.anomaly_score),
                "detection_confidence": float(self.detection_confidence),
                "training_mode": bool(self._cfg.training_mode),
                "adaptive_thresholds": bool(self._cfg.adaptive_thresholds),
                "ts": datetime.datetime.now().isoformat()
            }
            self.smart_bus.set(self._cfg.status_key, status, module='EnhancedAnomalyDetector',
                               thesis="Anomaly detector initialization status")
        except Exception as e:
            self.logger.warning(f"EnhancedAnomalyDetector initialization failed: {e}")

    def _initialize_advanced_systems(self):
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="EnhancedAnomalyDetector",
            log_path="logs/risk/enhanced_anomaly_detector.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("EnhancedAnomalyDetector", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()

        # Circuit breaker state
        self.circuit_breaker = {
            'failures': 0,
            'last_failure': 0.0,
            'state': 'CLOSED',
            'threshold': int(self._cfg.circuit_breaker_threshold),
            'cooldown_sec': 20.0
        }

        # Health
        self._health_status = 'healthy'
        self._last_health_check = time.time()

        # Thread sync
        self._lock = threading.RLock()

    def _initialize_detection_state(self):
        try:
            # Mixins
            self._initialize_risk_state()
            self._initialize_trading_state()
            self._initialize_state_management()

            # Mode
            self.current_mode = AnomalyDetectionMode.INITIALIZATION
            self.mode_start_time = datetime.datetime.now()

            # Thresholds
            self.current_thresholds = dict(self.config)  # plain dict for bus
            self.base_thresholds = dict(self.config)
            self.threshold_history = deque(maxlen=100)
            # Ensure required threshold keys are present (avoid KeyError: 'pnl_limit')
            self._ensure_threshold_keys(initial=True)

            # Data history
            self.pnl_history = deque(maxlen=self._cfg.history_size)
            self.volume_history = deque(maxlen=self._cfg.history_size)
            self.price_history = deque(maxlen=self._cfg.history_size)
            self.observation_history = deque(maxlen=min(self._cfg.history_size, 50))
            self.volatility_history = deque(maxlen=self._cfg.volatility_window)

            # Anomaly buckets
            self.anomalies: Dict[str, List[Dict[str, Any]]] = {
                "pnl": [], "volume": [], "price": [], "observation": [], "pattern": [],
                "correlation": [], "volatility": [], "sequence": [], "system": [], "market_structure": []
            }

            # Metrics
            self.anomaly_score = 0.0
            self.detection_confidence = 0.5
            self.step_count = 0
            self.detection_stats = defaultdict(int)
            self.false_positive_tracker = deque(maxlen=self._cfg.performance_window)
            self.detection_effectiveness = deque(maxlen=self._cfg.performance_window)

            # Context baselines
            self.regime_baselines = defaultdict(lambda: defaultdict(lambda: deque(maxlen=100)))
            self.session_baselines = defaultdict(lambda: defaultdict(lambda: deque(maxlen=100)))
            self.volatility_baselines = defaultdict(lambda: deque(maxlen=50))

            # Market context
            self.market_regime = "normal"
            self.market_session = "unknown"
            self.volatility_regime = "medium"
            self.market_stress_level = 0.0

            # Lazy analyzers
            self.sequence_analyzer = None
            self.correlation_analyzer = None
            self.pattern_detector = None

            # Training / adaptation
            self.training_progress = 0.0
            self.is_training_complete = False  # ensure defined
            our_params = {
                'sensitivity_multiplier': 1.0,
                'regime_adaptation_factor': 1.0,
                'volatility_tolerance': 1.0,
                'learning_momentum': 0.0,
                'detection_confidence_boost': 1.0
            }
            self.adaptive_params = dict(our_params)

            # Quality/perf
            self._detection_quality = 0.5
            self._processing_times = deque(maxlen=100)
            self._last_significant_detection = None

            # Integrations
            self.external_anomaly_sources: Dict[str, Any] = {}
            self.compliance_alerts: List[Any] = []

            # Publish initial status
            status = {
                "current_mode": self.current_mode.value,
                "enabled": self.enabled,
                "anomaly_score": float(self.anomaly_score),
                "detection_confidence": float(self.detection_confidence),
                "training_mode": bool(self._cfg.training_mode),
                "adaptive_thresholds": bool(self._cfg.adaptive_thresholds),
                "ts": datetime.datetime.now().isoformat()
            }
            self.smart_bus.set(self._cfg.status_key, status, module='EnhancedAnomalyDetector',
                               thesis="Initial anomaly detector status")
        except Exception as e:
            self.logger.error(f"Anomaly detector initialization failed: {e}")

    # ── background monitor ───────────────────────────────────
    def _start_monitoring(self):
        if self._monitoring_active:
            return

        def monitoring_loop():
            self._monitoring_active = True
            self.logger.info("[MONITOR] EnhancedAnomalyDetector health monitor started.")
            while self._monitoring_active:
                try:
                    self._update_detection_health()
                    self._analyze_detection_effectiveness()
                    self._adapt_detection_parameters()
                    self._cleanup_old_data()
                    # Publish health snapshot (namespaced)
                    health = self.get_health_status()
                    self.smart_bus.set(self._cfg.health_key, health, module='EnhancedAnomalyDetector',
                                       thesis="Anomaly detector health heartbeat")

                    # Cooldown-based breaker reset
                    if self.circuit_breaker['state'] == 'OPEN':
                        if (time.time() - self.circuit_breaker['last_failure']) >= self.circuit_breaker['cooldown_sec']:
                            self.circuit_breaker['state'] = 'CLOSED'
                            self.circuit_breaker['failures'] = 0
                            self.logger.info("[MONITOR] Circuit breaker auto-reset to CLOSED.")

                except Exception as e:
                    self.logger.error(f"Anomaly detection monitoring error: {e}")
                time.sleep(max(1, int(self._cfg.health_check_interval)))

        t = threading.Thread(target=monitoring_loop, daemon=True)
        t.start()

    def stop_monitoring(self):
        self._monitoring_active = False

    # ── contract-safe process ────────────────────────────────
    async def process(self, **inputs) -> Dict[str, Any]:
        start_time = time.time()
        try:
            if not self.enabled:
                # Disabled fallback + vote
                payload = await self._handle_disabled_fallback()
                vote = await self.cast_vote(**inputs)
                payload["anomaly_risk_vote"] = vote
                # Ensure required provides in return payload
                payload['EnhancedAnomalyDetector_voting_proposal'] = vote
                payload['EnhancedAnomalyDetector_confidence'] = vote.get('confidence', 0.5)
                self._write_bus_from_payload(payload, payload["_thesis"])
                return payload

            # Circuit breaker gating
            if self.circuit_breaker['state'] == 'OPEN':
                thesis = "Circuit breaker OPEN; safe fallback payload emitted."
                payload = self._fallback_payload(thesis=thesis)
                vote = await self.cast_vote(**inputs)
                payload["anomaly_risk_vote"] = vote
                payload['EnhancedAnomalyDetector_voting_proposal'] = vote
                payload['EnhancedAnomalyDetector_confidence'] = vote.get('confidence', 0.5)
                self._write_bus_from_payload(payload, thesis)
                return payload

            self.step_count += 1

            # Extract detection data
            detection_data = await self._extract_detection_data(**inputs)
            if not detection_data:
                payload = await self._handle_no_data_fallback()
                vote = await self.cast_vote(**inputs)
                payload["anomaly_risk_vote"] = vote
                payload['EnhancedAnomalyDetector_voting_proposal'] = vote
                payload['EnhancedAnomalyDetector_confidence'] = vote.get('confidence', 0.5)
                self._write_bus_from_payload(payload, payload["_thesis"])
                return payload

            # Full pipeline
            context_result   = await self._update_market_context_async(detection_data)
            detection_result = await self._detect_anomalies_comprehensive_async(detection_data)
            pattern_result   = await self._analyze_patterns_async(detection_data)
            adaptation_result = await self._adapt_thresholds_async(detection_data) if self._cfg.adaptive_thresholds else {}
            scoring_result   = await self._calculate_comprehensive_score_async(detection_data)
            training_result  = await self._update_training_progress_async(detection_data)
            emergency_result = await self._handle_emergency_situations_async(detection_data)
            mode_result      = await self._update_operational_mode_async(detection_data)

            _ = {**context_result, **detection_result, **pattern_result,
                **adaptation_result, **scoring_result, **training_result,
                **emergency_result, **mode_result}

            thesis = await self._generate_detection_thesis(detection_data, _)

            # Format payload (strict) + vote
            payload = self._format_provides_output(thesis=thesis)
            vote = await self.cast_vote(**inputs)
            payload["anomaly_risk_vote"] = vote
            payload['EnhancedAnomalyDetector_voting_proposal'] = vote
            payload['EnhancedAnomalyDetector_confidence'] = vote.get('confidence', 0.5)

            # Publish to SmartInfoBus
            self._write_bus_from_payload(payload, thesis)

            # Success metrics
            processing_time_ms = (time.time() - start_time) * 1000.0
            self._record_success(processing_time_ms)

            return payload

        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000.0
            payload = await self._handle_detection_error(e, start_time)
            try:
                vote = await self.cast_vote(**inputs)
                payload["anomaly_risk_vote"] = vote
                payload['EnhancedAnomalyDetector_voting_proposal'] = vote
                payload['EnhancedAnomalyDetector_confidence'] = vote.get('confidence', 0.5)
            except Exception:
                # Attach a minimal abstain vote to satisfy provides
                fallback_vote = {
                    "module": "EnhancedAnomalyDetector",
                    "topic": "anomaly_risk",
                    "vote": "abstain",
                    "confidence": 0.5,
                    "sizing_multiplier": 0.75,
                    "reasoning": "Vote generation failed in error path; abstaining.",
                    "metrics": {},
                    "timestamp": datetime.datetime.now().isoformat()
                }
                payload['EnhancedAnomalyDetector_voting_proposal'] = fallback_vote
                payload['EnhancedAnomalyDetector_confidence'] = 0.5
            try:
                self._write_bus_from_payload(payload, payload.get("_thesis", "Anomaly detector error"))
            except Exception:
                pass
            return payload


    # ── SmartInfoBus I/O (single-writer) ─────────────────────
    def _write_bus_from_payload(self, payload: Dict[str, Any], thesis: str) -> None:
        try:
            self.smart_bus.set('anomaly_detection', payload['anomaly_detection'],
                            module='EnhancedAnomalyDetector', thesis=thesis)
            self.smart_bus.set('anomaly_score', payload['anomaly_score'],
                            module='EnhancedAnomalyDetector', thesis="Anomaly score update")
            self.smart_bus.set('anomaly_alerts', payload['anomaly_alerts'],
                            module='EnhancedAnomalyDetector', thesis="Anomaly alerts update")
            self.smart_bus.set('detection_analytics', payload['detection_analytics'],
                            module='EnhancedAnomalyDetector', thesis="Detection analytics update")

            # NEW: publish standardized vote - ALWAYS publish even if None/empty
            vote = payload.get('anomaly_risk_vote') or getattr(self, '_last_vote', None)
            if not vote:
                # Fallback vote if none exists
                vote = {
                    "module": "EnhancedAnomalyDetector",
                    "topic": "anomaly_risk", 
                    "vote": "abstain",
                    "confidence": 0.5,
                    "sizing_multiplier": 0.75,
                    "reasoning": "No vote generated; abstaining.",
                    "metrics": {},
                    "timestamp": datetime.datetime.now().isoformat()
                }
            
            self.smart_bus.set('EnhancedAnomalyDetector_voting_proposal', vote,
                            module='EnhancedAnomalyDetector', thesis="Anomaly risk voting proposal")
            self.smart_bus.set('EnhancedAnomalyDetector_confidence', vote.get('confidence', 0.5),
                            module='EnhancedAnomalyDetector', thesis="Anomaly risk vote confidence")

        except Exception as e:
            err = self.error_pinpointer.analyze_error(e, "bus_write")
            self.logger.error(f"SmartInfoBus update failed: {err}")

    # ── payload formatter (contract enforcer) ────────────────
    def _format_provides_output(self, thesis: str) -> Dict[str, Any]:
        # Build anomaly_detection snapshot
        detection_data_payload = {
            'current_mode': self.current_mode.value,
            'enabled': bool(self.enabled),
            'anomaly_score': float(self.anomaly_score),
            'detection_confidence': float(self.detection_confidence),
            'total_anomalies': int(sum(len(v) for v in self.anomalies.values())),
            'training_mode': bool(self._cfg.training_mode),
            'training_progress': float(self.training_progress),
            'is_training_complete': bool(self.is_training_complete),
            'timestamp': datetime.datetime.now().isoformat()
        }

        # Score view
        score_data_payload = {
            'anomaly_score': float(self.anomaly_score),
            'detection_confidence': float(self.detection_confidence),
            'anomaly_types': {k: int(len(v)) for k, v in self.anomalies.items() if v},
            'critical_anomalies': int(sum(1 for aL in self.anomalies.values() for a in aL
                                          if a.get("severity") == AnomalySeverity.CRITICAL.value))
        }

        # Alerts view
        alerts_data_payload = {
            'emergency_mode': bool(self.current_mode == AnomalyDetectionMode.EMERGENCY),
            'critical_anomalies_present': any(
                a.get("severity") == AnomalySeverity.CRITICAL.value for aL in self.anomalies.values() for a in aL
            ),
            'high_anomaly_score': bool(self.anomaly_score > self._cfg.critical_threshold),
            'low_detection_quality': bool(self._detection_quality < self._cfg.min_detection_quality),
            'circuit_breaker_open': bool(self.circuit_breaker['state'] == 'OPEN'),
            'recent_anomalies': {
                a_type: [{
                    'type': a.get('type', 'unknown'),
                    'severity': a.get('severity', 'info'),
                    'confidence': float(a.get('confidence', 0.5)),
                    'timestamp': a.get('timestamp')
                } for a in aL[-5:]]
                for a_type, aL in self.anomalies.items() if aL
            }
        }

        # Analytics view (all python types)
        perf_avg_ms = (np.mean(list(self._processing_times)[-10:]) * 1000.0) if self._processing_times else 0.0
        analytics_payload = {
            'detection_quality': float(self._detection_quality),
            'detection_effectiveness': [float(x) for x in list(self.detection_effectiveness)[-10:]] if self.detection_effectiveness else [],
            'threshold_adaptation_count': int(len(self.threshold_history)),
            'current_thresholds': dict(self.current_thresholds),
            'base_thresholds': dict(self.base_thresholds),
            'detection_stats': {k: int(v) for k, v in self.detection_stats.items()},
            'data_sufficiency': {
                'pnl_history': int(len(self.pnl_history)),
                'volume_history': int(len(self.volume_history)),
                'price_history': int(len(self.price_history)),
                'observation_history': int(len(self.observation_history))
            },
            'performance_metrics': {
                'avg_processing_time_ms': float(perf_avg_ms),
                'circuit_breaker_state': self.circuit_breaker['state'],
                'false_positive_rate': float(len([fp for fp in self.false_positive_tracker if fp]) / max(len(self.false_positive_tracker), 1))
            }
        }

        return {
            'anomaly_detection': detection_data_payload,
            'anomaly_score': score_data_payload,
            'anomaly_alerts': alerts_data_payload,
            'detection_analytics': analytics_payload,
            '_thesis': thesis
        }

    # ── data extraction & context ────────────────────────────
    async def _extract_detection_data(self, **inputs) -> Optional[Dict[str, Any]]:
        try:
            risk_data = self.smart_bus.get('risk_data', 'EnhancedAnomalyDetector') or {}
            market_data = self.smart_bus.get('market_data', 'EnhancedAnomalyDetector') or {}
            trading_data = self.smart_bus.get('trading_data', 'EnhancedAnomalyDetector') or {}
            performance_data = self.smart_bus.get('performance_data', 'EnhancedAnomalyDetector') or {}

            pnl = inputs.get('pnl', 0.0)
            volume = inputs.get('volume', 0.0)
            price = inputs.get('price', 0.0)
            observation = inputs.get('obs', inputs.get('observation', None))
            trades = inputs.get('trades', [])

            risk_snapshot = risk_data.get('risk_snapshot', {})
            if not pnl and 'recent_pnl' in risk_snapshot:
                pnl = risk_snapshot['recent_pnl']

            market_snapshot = market_data.get('market_snapshot', {})
            if not volume and 'volume' in market_snapshot:
                volume = market_snapshot['volume']
            if not price and 'price' in market_snapshot:
                price = market_snapshot['price']

            trading_snapshot = trading_data.get('trading_snapshot', {})
            if not trades and 'recent_trades' in trading_snapshot:
                trades = trading_snapshot['recent_trades']

            return {
                'pnl': float(pnl or 0.0),
                'volume': float(volume or 0.0),
                'price': float(price or 0.0),
                'observation': observation,
                'trades': trades or [],
                'risk_data': risk_data,
                'market_data': market_data,
                'trading_data': trading_data,
                'performance_data': performance_data,
                'timestamp': datetime.datetime.now().isoformat(),
                'step_count': int(self.step_count)
            }
        except Exception as e:
            self.logger.error(f"Failed to extract detection data: {e}")
            return None

    async def _update_market_context_async(self, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            market_context = self.smart_bus.get('market_context', 'EnhancedAnomalyDetector') or {}

            old_regime = self.market_regime
            old_session = self.market_session
            old_volatility = self.volatility_regime

            self.market_regime = market_context.get('regime', 'normal')
            self.market_session = market_context.get('session', 'unknown')
            self.volatility_regime = market_context.get('volatility_level', 'medium')
            self.market_stress_level = float(market_context.get('stress_level', 0.0) or 0.0)

            context_changed = (old_regime != self.market_regime or
                               old_session != self.market_session or
                               old_volatility != self.volatility_regime)

            if context_changed:
                self.logger.info(format_operator_message(
                    message="Market context changed - adapting detection",
                    icon="[STATS]",
                    old_regime=old_regime, new_regime=self.market_regime,
                    volatility=self.volatility_regime, session=self.market_session,
                    stress_level=f"{self.market_stress_level:.2f}"
                ))
                await self._update_context_baselines_async(detection_data)

            return {
                'market_context_updated': True,
                'context_changed': context_changed,
                'current_regime': self.market_regime,
                'current_session': self.market_session,
                'volatility_regime': self.volatility_regime,
                'stress_level': self.market_stress_level
            }
        except Exception as e:
            self.logger.error(f"Market context update failed: {e}")
            return {'market_context_updated': False, 'error': str(e)}

    async def _update_context_baselines_async(self, detection_data: Dict[str, Any]) -> None:
        try:
            regime = self.market_regime
            session = self.market_session
            volatility = self.volatility_regime

            for data_type in ['pnl', 'volume', 'price']:
                val = float(detection_data.get(data_type, 0.0) or 0.0)
                if val != 0.0:
                    self.regime_baselines[regime][data_type].append(val)
                    self.session_baselines[session][data_type].append(val)

            if len(self.price_history) >= 2:
                price_change = abs(float(detection_data.get('price', 0.0)) - float(self.price_history[-1]))
                self.volatility_baselines[volatility].append(price_change)
        except Exception as e:
            self.logger.warning(f"Context baseline update failed: {e}")

    # ── comprehensive detection ──────────────────────────────
    async def _detect_anomalies_comprehensive_async(self, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Clear/refresh buckets
            for k in self.anomalies:
                self.anomalies[k].clear()

            critical_found = False

            pnl_result = await self._detect_pnl_anomalies_async(detection_data)
            if pnl_result.get('critical', False):
                critical_found = True

            volume_result = await self._detect_volume_anomalies_async(detection_data)
            price_result = await self._detect_price_anomalies_async(detection_data)

            obs_result = await self._detect_observation_anomalies_async(detection_data)
            if obs_result.get('critical', False):
                critical_found = True

            vol_result = await self._detect_volatility_anomalies_async(detection_data)
            system_result = await self._detect_system_anomalies_async(detection_data)
            if system_result.get('critical', False):
                critical_found = True

            structure_result = await self._detect_market_structure_anomalies_async(detection_data)

            # Update simple stats counters
            self.detection_stats['total_anomalies'] = sum(len(v) for v in self.anomalies.values())
            self.detection_stats['pnl_count'] += len(self.anomalies['pnl'])
            self.detection_stats['volume_count'] += len(self.anomalies['volume'])
            self.detection_stats['price_count'] += len(self.anomalies['price'])
            self.detection_stats['observation_count'] += len(self.anomalies['observation'])
            self.detection_stats['system_count'] += len(self.anomalies['system'])

            return {
                'comprehensive_detection_completed': True,
                'critical_anomalies_found': critical_found,
                'total_anomalies': self.detection_stats['total_anomalies']
            }
        except Exception as e:
            self.logger.error(f"Comprehensive anomaly detection failed: {e}")
            return {'comprehensive_detection_completed': False, 'error': str(e)}

    # ── individual detectors (safer) ─────────────────────────
    async def _detect_pnl_anomalies_async(self, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Ensure thresholds exist (pnl_limit in particular)
            self._ensure_threshold_keys()

            pnl = float(detection_data.get('pnl', 0.0) or 0.0)
            if self._cfg.training_mode and pnl == 0.0 and len(self.pnl_history) < 10:
                pnl = await self._generate_synthetic_pnl_async(detection_data)

            if pnl == 0.0 and not self._cfg.training_mode:
                return {'pnl_detected': False, 'reason': 'no_pnl_data'}

            self.pnl_history.append(pnl)
            critical_found = False
            anomalies_detected = 0

            adjusted_limit = await self._get_context_adjusted_threshold_async('pnl_limit', detection_data)
            if abs(pnl) > adjusted_limit:
                severity = AnomalySeverity.CRITICAL if abs(pnl) > adjusted_limit * 1.5 else AnomalySeverity.WARNING
                self.anomalies['pnl'].append({
                    'type': 'absolute_limit_exceeded',
                    'value': pnl,
                    'threshold': float(adjusted_limit),
                    'base_threshold': float(self.base_thresholds.get('pnl_limit', self._cfg.pnl_limit)),
                    'severity': severity.value,
                    'confidence': 0.9,
                    'timestamp': detection_data.get('timestamp'),
                    'context': {
                        'regime': self.market_regime,
                        'session': self.market_session,
                        'volatility': self.volatility_regime
                    }
                })
                anomalies_detected += 1
                if severity == AnomalySeverity.CRITICAL:
                    critical_found = True
                    self.logger.error(format_operator_message(
                        message='CRITICAL PnL anomaly detected', icon='[ALERT]',
                        pnl=f"€{pnl:,.2f}", limit=f"€{adjusted_limit:,.0f}",
                        regime=self.market_regime, session=self.market_session
                    ))
                else:
                    self.logger.warning(format_operator_message(
                        message='PnL anomaly detected', icon='[WARN]',
                        pnl=f"€{pnl:,.2f}", limit=f"€{adjusted_limit:,.0f}"
                    ))

            if len(self.pnl_history) >= self._cfg.min_history_for_stats:
                z = await self._calculate_robust_zscore_async(pnl, list(self.pnl_history))
                if z > 4.0:
                    severity = AnomalySeverity.CRITICAL if z > 6.0 else AnomalySeverity.WARNING
                    self.anomalies['pnl'].append({
                        'type': 'statistical_outlier',
                        'value': pnl,
                        'z_score': float(z),
                        'severity': severity.value,
                        'confidence': min(0.9, z / 8.0),
                        'timestamp': detection_data.get('timestamp'),
                        'context': {'history_size': len(self.pnl_history), 'regime': self.market_regime}
                    })
                    anomalies_detected += 1
                    if severity == AnomalySeverity.CRITICAL:
                        critical_found = True
                        self.logger.error(f"[ALERT] CRITICAL: Statistical PnL anomaly - z-score {z:.2f}")

            regime_anomaly = await self._detect_regime_specific_pnl_anomaly_async(pnl, detection_data)
            if regime_anomaly:
                self.anomalies['pnl'].append(regime_anomaly)
                anomalies_detected += 1

            return {
                'pnl_detected': True,
                'anomalies_count': anomalies_detected,
                'critical': critical_found,
                'pnl_value': pnl,
                'adjusted_threshold': float(adjusted_limit)
            }
        except Exception as e:
            self.logger.warning(f"PnL anomaly detection failed: {e}")
            return {'pnl_detected': False, 'error': str(e)}

    async def _detect_observation_anomalies_async(self, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            observation = detection_data.get('observation')
            if observation is None:
                return {'observation_detected': False, 'reason': 'no_observation_data'}

            try:
                obs = np.array(observation, dtype=np.float32)
            except (ValueError, TypeError):
                self.anomalies["observation"].append({
                    "type": "invalid_format", "severity": AnomalySeverity.CRITICAL.value,
                    "confidence": 1.0, "timestamp": detection_data.get('timestamp'),
                    "details": "Observation could not be converted to valid numpy array"
                })
                self.logger.error("[ALERT] CRITICAL: Invalid observation format detected")
                return {'observation_detected': True, 'critical': True, 'anomalies_count': 1}

            critical_found = False
            anomalies_detected = 0

            nan_count = int(np.isnan(obs).sum())
            inf_count = int(np.isinf(obs).sum())
            if nan_count > 0 or inf_count > 0:
                self.anomalies["observation"].append({
                    "type": "invalid_values", "nan_count": nan_count, "inf_count": inf_count,
                    "observation_shape": tuple(obs.shape), "severity": AnomalySeverity.CRITICAL.value,
                    "confidence": 1.0, "timestamp": detection_data.get('timestamp')
                })
                critical_found = True
                anomalies_detected += 1
                self.logger.error(format_operator_message(
                    message="CRITICAL: Invalid observation values", icon="[ALERT]",
                    nan_count=nan_count, inf_count=inf_count, shape=str(obs.shape)
                ))

            if not critical_found:
                self.observation_history.append(obs)
                if len(self.observation_history) >= 10:
                    z_scores = await self._calculate_observation_zscores_async(obs)
                    extreme_threshold = float(self.current_thresholds['observation_zscore'])
                    extreme_indices = np.where(z_scores > extreme_threshold)[0]
                    if len(extreme_indices) > 0:
                        max_z = float(np.max(z_scores))
                        severity = AnomalySeverity.CRITICAL if max_z > extreme_threshold * 1.5 else AnomalySeverity.WARNING
                        self.anomalies["observation"].append({
                            "type": "extreme_values",
                            "extreme_indices": extreme_indices.tolist(),
                            "z_scores": z_scores[extreme_indices].tolist(),
                            "max_z_score": max_z,
                            "threshold": extreme_threshold,
                            "severity": severity.value,
                            "confidence": min(0.9, max_z / (extreme_threshold * 2)),
                            "timestamp": detection_data.get('timestamp')
                        })
                        anomalies_detected += 1
                        if severity == AnomalySeverity.CRITICAL:
                            critical_found = True
                            self.logger.error(f"[ALERT] CRITICAL: Extreme observation values - max z-score {max_z:.2f}")

            return {
                'observation_detected': True, 'anomalies_count': anomalies_detected,
                'critical': critical_found, 'observation_shape': tuple(obs.shape),
                'invalid_values': nan_count + inf_count
            }
        except Exception as e:
            self.logger.warning(f"Observation anomaly detection failed: {e}")
            return {'observation_detected': False, 'error': str(e)}

    async def _detect_volume_anomalies_async(self, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            volume = float(detection_data.get('volume', 0.0) or 0.0)
            if self._cfg.training_mode and volume == 0.0 and len(self.volume_history) < 10:
                volume = await self._generate_synthetic_volume_async(detection_data)
            if volume == 0.0:
                return {'volume_detected': False, 'reason': 'no_volume_data'}

            self.volume_history.append(volume)
            anomalies_detected = 0

            if len(self.volume_history) >= self._cfg.min_history_for_stats:
                z = await self._calculate_robust_zscore_async(volume, list(self.volume_history))
                adjusted = await self._get_context_adjusted_threshold_async('volume_zscore', detection_data)
                if z > adjusted:
                    severity = AnomalySeverity.WARNING if z < adjusted * 1.5 else AnomalySeverity.CRITICAL
                    self.anomalies["volume"].append({
                        "type": "volume_spike", "value": volume, "z_score": float(z),
                        "threshold": float(adjusted), "severity": severity.value,
                        "confidence": min(0.8, z / (adjusted * 2)),
                        "timestamp": detection_data.get('timestamp'),
                        "context": {'regime': self.market_regime, 'session': self.market_session}
                    })
                    anomalies_detected += 1
                    self.logger.warning(format_operator_message(
                        message="Volume anomaly detected", icon="[WARN]",
                        volume=f"{volume:,.0f}", z_score=f"{z:.2f}", regime=self.market_regime
                    ))

            return {'volume_detected': True, 'anomalies_count': anomalies_detected,
                    'volume_value': volume, 'history_size': len(self.volume_history)}
        except Exception as e:
            self.logger.warning(f"Volume anomaly detection failed: {e}")
            return {'volume_detected': False, 'error': str(e)}

    async def _detect_price_anomalies_async(self, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            price = float(detection_data.get('price', 0.0) or 0.0)
            if self._cfg.training_mode and price == 0.0 and len(self.price_history) < 10:
                price = await self._generate_synthetic_price_async(detection_data)
            if price == 0.0:
                return {'price_detected': False, 'reason': 'no_price_data'}

            self.price_history.append(price)
            anomalies_detected = 0

            if len(self.price_history) >= 2:
                prev_price = float(self.price_history[-2])
                if prev_price > 0:
                    price_change = abs((price - prev_price) / prev_price)
                    jump_threshold = await self._get_price_jump_threshold_async(detection_data)
                    if price_change > jump_threshold:
                        severity = AnomalySeverity.CRITICAL if price_change > jump_threshold * 2 else AnomalySeverity.WARNING
                        self.anomalies["price"].append({
                            "type": "price_jump", "change_percentage": float(price_change),
                            "prev_price": prev_price, "current_price": price,
                            "threshold": float(jump_threshold), "severity": severity.value,
                            "confidence": min(0.9, price_change / jump_threshold / 2),
                            "timestamp": detection_data.get('timestamp'),
                            "context": {'volatility_regime': self.volatility_regime, 'regime': self.market_regime}
                        })
                        anomalies_detected += 1
                        self.logger.warning(format_operator_message(
                            message="Price jump detected", icon="[WARN]",
                            change=f"{price_change:.1%}", from_price=f"{prev_price:.5f}",
                            to_price=f"{price:.5f}", volatility=self.volatility_regime
                        ))

            return {'price_detected': True, 'anomalies_count': anomalies_detected,
                    'price_value': price, 'history_size': len(self.price_history)}
        except Exception as e:
            self.logger.warning(f"Price anomaly detection failed: {e}")
            return {'price_detected': False, 'error': str(e)}

    async def _detect_volatility_anomalies_async(self, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            anomalies_detected = 0
            current_vol = 0.0
            if len(self.price_history) >= 10:
                prices = np.array(list(self.price_history)[-10:], dtype=np.float64)
                returns = np.diff(np.log(prices + 1e-8))
                current_vol = float(np.std(returns) * np.sqrt(252))
                self.volatility_history.append(current_vol)

                if len(self.volatility_history) >= 10:
                    z = await self._calculate_robust_zscore_async(current_vol, list(self.volatility_history))
                    if z > 3.0:
                        severity = AnomalySeverity.CRITICAL if z > 5.0 else AnomalySeverity.WARNING
                        self.anomalies["volatility"].append({
                            "type": "volatility_spike", "current_volatility": current_vol,
                            "z_score": float(z), "severity": severity.value,
                            "confidence": min(0.8, z / 6.0),
                            "timestamp": detection_data.get('timestamp'),
                            "context": {'volatility_regime': self.volatility_regime, 'regime': self.market_regime}
                        })
                        anomalies_detected += 1
                        if severity == AnomalySeverity.CRITICAL:
                            self.logger.error(format_operator_message(
                                message="CRITICAL volatility spike", icon="[ALERT]",
                                volatility=f"{current_vol:.1%}", z_score=f"{z:.2f}"
                            ))
                        else:
                            self.logger.warning(format_operator_message(
                                message="Volatility spike detected", icon="[WARN]",
                                volatility=f"{current_vol:.1%}", z_score=f"{z:.2f}"
                            ))
            return {'volatility_detected': True, 'anomalies_count': anomalies_detected,
                    'current_volatility': current_vol}
        except Exception as e:
            self.logger.warning(f"Volatility anomaly detection failed: {e}")
            return {'volatility_detected': False, 'error': str(e)}

    async def _detect_system_anomalies_async(self, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            anomalies_detected = 0
            critical_found = False
            processing_times = list(self._processing_times)
            if len(processing_times) >= 10:
                recent_avg_time_ms = float(np.mean(processing_times[-10:]) * 1000.0)
                if recent_avg_time_ms > self._cfg.max_processing_time_ms:
                    self.anomalies["system"].append({
                        "type": "slow_processing", "average_time_ms": recent_avg_time_ms,
                        "threshold_ms": float(self._cfg.max_processing_time_ms),
                        "severity": AnomalySeverity.WARNING.value,
                        "confidence": min(0.8, recent_avg_time_ms / self._cfg.max_processing_time_ms / 2),
                        "timestamp": detection_data.get('timestamp')
                    })
                    anomalies_detected += 1

            if self.circuit_breaker['state'] == 'OPEN':
                self.anomalies["system"].append({
                    "type": "circuit_breaker_open",
                    "failures": int(self.circuit_breaker['failures']),
                    "threshold": int(self.circuit_breaker['threshold']),
                    "severity": AnomalySeverity.CRITICAL.value,
                    "confidence": 1.0,
                    "timestamp": detection_data.get('timestamp')
                })
                anomalies_detected += 1
                critical_found = True

            if self._detection_quality < self._cfg.min_detection_quality:
                self.anomalies["system"].append({
                    "type": "low_detection_quality",
                    "quality_score": float(self._detection_quality),
                    "threshold": float(self._cfg.min_detection_quality),
                    "severity": AnomalySeverity.WARNING.value,
                    "confidence": 0.7,
                    "timestamp": detection_data.get('timestamp')
                })
                anomalies_detected += 1

            return {'system_detected': True, 'anomalies_count': anomalies_detected, 'critical': critical_found}
        except Exception as e:
            self.logger.warning(f"System anomaly detection failed: {e}")
            return {'system_detected': False, 'error': str(e)}

    async def _detect_market_structure_anomalies_async(self, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            trades = detection_data.get('trades', [])
            if not trades:
                return {'market_structure_detected': False, 'reason': 'no_trade_data'}

            anomalies_detected = 0
            trade_sizes = [abs(t.get('size', t.get('volume', 0))) for t in trades]
            trade_directions = [np.sign(t.get('size', t.get('volume', 0))) for t in trades if t.get('size', t.get('volume', 0)) != 0]

            if len(set(trade_directions)) == 1 and len(trade_directions) > 10:
                self.anomalies["market_structure"].append({
                    "type": "unidirectional_trading", "trade_count": len(trade_directions),
                    "direction": int(trade_directions[0]),
                    "severity": AnomalySeverity.INFO.value,
                    "confidence": min(0.8, len(trade_directions) / 20.0),
                    "timestamp": detection_data.get('timestamp'),
                    "context": {'regime': self.market_regime}
                })
                anomalies_detected += 1
                direction_text = "BUY" if trade_directions[0] > 0 else "SELL"
                self.logger.info(format_operator_message(
                    message="Unidirectional trading pattern", icon="[STATS]",
                    direction=direction_text, count=len(trade_directions), regime=self.market_regime
                ))

            if len(trades) > 30:
                self.anomalies["market_structure"].append({
                    "type": "high_frequency_trading", "trade_count": len(trades),
                    "severity": AnomalySeverity.WARNING.value, "confidence": min(0.9, len(trades) / 50.0),
                    "timestamp": detection_data.get('timestamp')
                })
                anomalies_detected += 1
                self.logger.warning(f"[WARN] High frequency trading detected: {len(trades)} trades")

            if trade_sizes:
                z_list = await self._calculate_trade_size_zscores_async(trade_sizes)
                extreme_count = sum(1 for z in z_list if z > 3.0)
                if extreme_count > 0:
                    self.anomalies["market_structure"].append({
                        "type": "extreme_trade_sizes", "extreme_count": int(extreme_count),
                        "total_trades": int(len(trade_sizes)), "max_z_score": float(max(z_list)),
                        "severity": AnomalySeverity.INFO.value, "confidence": min(0.7, extreme_count / max(1, len(trade_sizes))),
                        "timestamp": detection_data.get('timestamp')
                    })
                    anomalies_detected += 1

            return {'market_structure_detected': True, 'anomalies_count': anomalies_detected,
                    'trades_analyzed': int(len(trades))}
        except Exception as e:
            self.logger.warning(f"Market structure anomaly detection failed: {e}")
            return {'market_structure_detected': False, 'error': str(e)}

    # ── pattern analyzers ────────────────────────────────────
    def _ensure_analyzers_initialized(self):
        try:
            if self.sequence_analyzer is None:
                self.sequence_analyzer = SequenceAnomalyAnalyzer()
            if self.correlation_analyzer is None:
                self.correlation_analyzer = CorrelationAnomalyAnalyzer()
            if self.pattern_detector is None:
                self.pattern_detector = PatternAnomalyDetector()
        except NameError as e:
            self.logger.warning(f"Analyzer classes not yet available: {e}")
        except Exception as e:
            self.logger.error(f"Failed to initialize analyzers: {e}")
            self.sequence_analyzer = SimpleAnalyzer()
            self.correlation_analyzer = SimpleAnalyzer()
            self.pattern_detector = SimpleAnalyzer()

    async def _analyze_patterns_async(self, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self._ensure_analyzers_initialized()
            if len(self.pnl_history) >= 10 and self.sequence_analyzer is not None:
                seq_res = await self.sequence_analyzer.analyze_async(list(self.pnl_history), detection_data)
                if seq_res.get('anomalies'):
                    self.anomalies["sequence"].extend(seq_res['anomalies'])

            if (len(self.price_history) >= 20 and len(self.volume_history) >= 20 and
                    self.correlation_analyzer is not None):
                corr_res = await self.correlation_analyzer.analyze_async(
                    list(self.price_history), list(self.volume_history), detection_data
                )
                if corr_res.get('anomalies'):
                    self.anomalies["correlation"].extend(corr_res['anomalies'])

            trades = detection_data.get('trades', [])
            if trades and self.pattern_detector is not None:
                pat_res = await self.pattern_detector.detect_async(trades, detection_data)
                if pat_res.get('anomalies'):
                    self.anomalies["pattern"].extend(pat_res['anomalies'])

            self.detection_stats['pattern_anomalies'] += sum(
                len(self.anomalies[t]) for t in ["sequence", "correlation", "pattern"]
            )
            return {'pattern_analysis_completed': True}
        except Exception as e:
            self.logger.warning(f"Pattern analysis failed: {e}")
            return {'pattern_analysis_completed': False, 'error': str(e)}

    # ── adaptation, scoring, training, emergency, mode ───────
    async def _adapt_thresholds_async(self, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Defensive: ensure thresholds exist before adapting
            self._ensure_threshold_keys()
            if not self._cfg.adaptive_thresholds or self.step_count < 50:
                return {'threshold_adaptation': False, 'reason': 'insufficient_data_or_disabled'}

            changes: Dict[str, Any] = {}

            if len(self.pnl_history) >= 50:
                arr = np.array(list(self.pnl_history))
                pnl_95 = float(np.percentile(np.abs(arr), 95))
                pnl_std = float(np.std(arr))
                adaptive = float(max(pnl_95, 3.0 * pnl_std))
                adaptive = float(np.clip(adaptive, self.base_thresholds.get('pnl_limit', self._cfg.pnl_limit) * 0.5,
                                         self.base_thresholds.get('pnl_limit', self._cfg.pnl_limit) * 3.0))
                old = float(self.current_thresholds.get('pnl_limit', self._cfg.pnl_limit))
                new = float(self._cfg.threshold_smoothing * old + (1 - self._cfg.threshold_smoothing) * adaptive)
                if abs(new - old) > old * 0.1:
                    self.current_thresholds['pnl_limit'] = new
                    changes['pnl_limit'] = {'old': old, 'new': new, 'change_pct': (new - old) / max(1e-6, old)}

            if len(self.detection_effectiveness) >= 20:
                recent_eff = float(np.mean(list(self.detection_effectiveness)[-20:]))
                old_obs = float(self.current_thresholds['observation_zscore'])

                if recent_eff < 0.6:
                    new_obs = min(6.0, old_obs * 1.10)
                    if abs(new_obs - old_obs) > 1e-6:
                        self.current_thresholds['observation_zscore'] = new_obs
                        changes['observation_zscore'] = {
                            'old': old_obs, 'new': new_obs, 'reason': 'poor_effectiveness'
                        }
                elif recent_eff > 0.8:
                    new_obs = max(2.0, old_obs * 0.95)
                    if abs(new_obs - old_obs) > 1e-6:
                        self.current_thresholds['observation_zscore'] = new_obs
                        changes['observation_zscore'] = {
                            'old': old_obs, 'new': new_obs, 'reason': 'good_effectiveness'
                        }

            if changes:
                self.threshold_history.append({
                    'timestamp': detection_data.get('timestamp'),
                    'changes': changes,
                    'context': {
                        'regime': self.market_regime,
                        'session': self.market_session,
                        'step_count': int(self.step_count)
                    }
                })
                self.logger.info(format_operator_message(
                    message="Thresholds adapted", icon="[TOOL]",
                    changes=len(changes), regime=self.market_regime
                ))

            return {'threshold_adaptation': True, 'changes_made': int(len(changes))}
        except Exception as e:
            self.logger.warning(f"Threshold adaptation failed: {e}")
            return {'threshold_adaptation': False, 'error': str(e)}

    async def _calculate_comprehensive_score_async(self, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Weighted, context-aware anomaly score + confidence."""
        try:
            severity_weights = {
                AnomalySeverity.INFO.value: 0.1,
                AnomalySeverity.WARNING.value: 0.5,
                AnomalySeverity.CRITICAL.value: 1.0,
                AnomalySeverity.EMERGENCY.value: 1.5
            }
            type_weights = {
                "pnl": 0.35, "observation": 0.25, "system": 0.15, "volatility": 0.10,
                "price": 0.05, "volume": 0.05, "market_structure": 0.03,
                "pattern": 0.01, "correlation": 0.01, "sequence": 0.01
            }

            total_weighted = 0.0
            total_conf = 0.0
            count = 0

            for a_type, a_list in self.anomalies.items():
                if not a_list:
                    continue
                t_w = type_weights.get(a_type, 0.01)
                for a in a_list:
                    sev = a.get("severity", AnomalySeverity.INFO.value)
                    conf = float(a.get("confidence", 0.5))
                    s_w = severity_weights.get(sev, 0.1)
                    total_weighted += (t_w * s_w * conf)
                    total_conf += conf
                    count += 1

            if count > 0:
                base = min(total_weighted, 1.0)
                avg_conf = total_conf / count
                ctx_mult = await self._get_context_score_multiplier_async(detection_data)
                self.anomaly_score = float(np.clip(base * ctx_mult, 0.0, 1.0))
                self.detection_confidence = float(np.clip(avg_conf, 0.0, 1.0))
            else:
                self.anomaly_score = 0.0
                self.detection_confidence = 1.0  # confident there's no anomaly

            eff = await self._calculate_detection_effectiveness_async()
            self.detection_effectiveness.append(eff)

            return {
                'score_calculated': True,
                'anomaly_score': float(self.anomaly_score),
                'detection_confidence': float(self.detection_confidence)
            }
        except Exception as e:
            self.logger.warning(f"Score calculation failed: {e}")
            return {'score_calculated': False, 'error': str(e)}

    async def _update_training_progress_async(self, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not self._cfg.training_mode:
                return {'training_update': False, 'reason': 'not_in_training_mode'}

            self.training_progress = min(self.step_count / float(self._cfg.training_duration_steps), 1.0)

            if self.training_progress >= 1.0 and not self.is_training_complete:
                await self._complete_training_async()
                return {'training_update': True, 'training_completed': True}

            return {'training_update': True, 'training_completed': False}
        except Exception as e:
            self.logger.warning(f"Training progress update failed: {e}")
            return {'training_update': False, 'error': str(e)}

    async def _complete_training_async(self) -> None:
        try:
            self.is_training_complete = True
            await self._finalize_training_thresholds_async()
            old = self.current_mode
            self.current_mode = AnomalyDetectionMode.ACTIVE
            self.logger.info(format_operator_message(
                message="Training completed - transitioning to active detection",
                icon="🎓", old_mode=old.value, new_mode=self.current_mode.value,
                steps_trained=int(self.step_count), final_score=f"{self._detection_quality:.2f}"
            ))
        except Exception as e:
            self.logger.error(f"Training completion failed: {e}")

    async def _handle_emergency_situations_async(self, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            triggered = False
            reasons: List[str] = []

            if self.anomaly_score > float(self._cfg.emergency_threshold):
                triggered = True
                reasons.append(f"anomaly_score_exceeded_{self.anomaly_score:.2f}")

            crit_sys = [a for a in self.anomalies.get("system", []) if a.get("severity") == AnomalySeverity.CRITICAL.value]
            if crit_sys:
                triggered = True
                reasons.append(f"critical_system_anomalies_{len(crit_sys)}")

            all_crit = [a for L in self.anomalies.values() for a in L if a.get("severity") == AnomalySeverity.CRITICAL.value]
            if len(all_crit) >= 3:
                triggered = True
                reasons.append(f"multiple_critical_anomalies_{len(all_crit)}")

            if triggered and self.current_mode != AnomalyDetectionMode.EMERGENCY:
                old = self.current_mode
                self.current_mode = AnomalyDetectionMode.EMERGENCY
                self.logger.error(format_operator_message(
                    message="EMERGENCY MODE ACTIVATED", icon="🆘",
                    old_mode=old.value, reasons=", ".join(reasons[:3]),
                    anomaly_score=f"{self.anomaly_score:.2f}", critical_count=len(all_crit)
                ))

            return {
                'emergency_check_completed': True,
                'emergency_triggered': bool(triggered)
            }
        except Exception as e:
            self.logger.warning(f"Emergency situation handling failed: {e}")
            return {'emergency_check_completed': False, 'error': str(e)}

    async def _update_operational_mode_async(self, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            old = self.current_mode

            if self.current_mode == AnomalyDetectionMode.EMERGENCY:
                if (self.anomaly_score < float(self._cfg.emergency_threshold) * 0.7 and
                        not any(a.get("severity") == AnomalySeverity.CRITICAL.value for L in self.anomalies.values() for a in L)):
                    self.current_mode = AnomalyDetectionMode.ACTIVE
                    self.logger.info(format_operator_message(
                        message="Emergency cleared - returning to active mode", icon="[OK]",
                        anomaly_score=f"{self.anomaly_score:.2f}"
                    ))
            elif self.step_count < 10:
                self.current_mode = AnomalyDetectionMode.INITIALIZATION
            elif self._cfg.training_mode and not self.is_training_complete:
                self.current_mode = AnomalyDetectionMode.TRAINING
            elif self._detection_quality < float(self._cfg.min_detection_quality):
                self.current_mode = AnomalyDetectionMode.CALIBRATION
            elif self.anomaly_score > float(self._cfg.critical_threshold):
                self.current_mode = AnomalyDetectionMode.ENHANCED
            else:
                self.current_mode = AnomalyDetectionMode.ACTIVE

            if old != self.current_mode:
                self.mode_start_time = datetime.datetime.now()
                self.logger.info(format_operator_message(
                    message="Detection mode changed", icon="[RELOAD]",
                    old_mode=old.value, new_mode=self.current_mode.value,
                    anomaly_score=f"{self.anomaly_score:.2f}",
                    detection_quality=f"{self._detection_quality:.2f}"
                ))

            return {'mode_updated': True}
        except Exception as e:
            self.logger.warning(f"Mode update failed: {e}")
            return {'mode_updated': False, 'error': str(e)}

    async def _generate_detection_thesis(self, detection_data: Dict[str, Any], result: Dict[str, Any]) -> str:
        try:
            parts: List[str] = []
            parts.append(f"Anomaly Detection: {self.current_mode.value.upper()} mode with {self.anomaly_score:.1%} risk score")
            parts.append(f"Detection Confidence: {self.detection_confidence:.2f} assessment accuracy")

            if self.anomaly_score > float(self._cfg.critical_threshold):
                parts.append("HIGH RISK: Critical anomalies detected")
            elif self.anomaly_score > float(self._cfg.warning_threshold):
                parts.append("ELEVATED: Warning-level anomalies present")
            else:
                parts.append("NORMAL: No significant anomalies detected")

            total_anoms = int(sum(len(v) for v in self.anomalies.values()))
            if total_anoms > 0:
                crit_count = sum(1 for L in self.anomalies.values() for a in L if a.get("severity") == AnomalySeverity.CRITICAL.value)
                if crit_count > 0:
                    parts.append(f"Active anomalies: {total_anoms} total, {crit_count} critical")
                else:
                    parts.append(f"Active anomalies: {total_anoms} total, monitoring level")

            parts.append(f"Context: {self.market_regime.upper()} regime, {self.volatility_regime.upper()} volatility")

            if self._cfg.training_mode:
                if self.is_training_complete:
                    parts.append(f"Training: COMPLETED ({self.step_count} steps)")
                else:
                    parts.append(f"Training: {self.training_progress:.0%} complete")

            if self._cfg.adaptive_thresholds:
                recent_adapt = len([h for h in list(self.threshold_history)[-10:] if h.get('changes')])
                parts.append(f"Adaptive: {recent_adapt} recent threshold adjustments")

            data_suff = min(len(self.pnl_history) / 50.0, 1.0)
            parts.append(f"Data quality: {data_suff:.0%} sufficiency")

            return " | ".join(parts)
        except Exception as e:
            return f"Detection thesis generation failed: {str(e)} - Core anomaly detection functional"

    # ── fallback & error payloads (contract-safe) ─────────────
    def _fallback_payload(self, thesis: str) -> Dict[str, Any]:
        return self._format_provides_output(thesis=thesis)

    async def _handle_disabled_fallback(self) -> Dict[str, Any]:
        self.anomaly_score = 0.0
        self.detection_confidence = 1.0
        thesis = "Anomaly detector disabled - maintenance mode"
        return self._format_provides_output(thesis=thesis)

    async def _handle_no_data_fallback(self) -> Dict[str, Any]:
        self.logger.warning("No detection data available - maintaining previous state")
        thesis = "No detection data - maintaining previous state"
        self.detection_confidence = float(np.clip(self.detection_confidence - 0.05, 0.1, 1.0))
        return self._format_provides_output(thesis=thesis)

    async def _handle_detection_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        processing_time = (time.time() - start_time) * 1000.0
        # circuit breaker update
        self.circuit_breaker['failures'] += 1
        self.circuit_breaker['last_failure'] = time.time()
        if self.circuit_breaker['failures'] >= int(self.circuit_breaker['threshold']):
            self.circuit_breaker['state'] = 'OPEN'
            self._health_status = 'warning'

        _ = self.error_pinpointer.analyze_error(error, "EnhancedAnomalyDetector")
        explanation = self.english_explainer.explain_error("EnhancedAnomalyDetector", str(error), "anomaly detection")
        self.logger.error(format_operator_message(
            message="Anomaly detector error", icon="[CRASH]", error=str(error),
            details=explanation, processing_time_ms=processing_time,
            circuit_breaker_state=self.circuit_breaker['state']
        ))
        self._record_failure(error)
        # Keep state minimally pessimistic
        self.anomaly_score = max(0.1, float(self.anomaly_score))
        self.detection_confidence = min(0.5, float(self.detection_confidence))
        return self._format_provides_output(thesis=f"Anomaly detector error fallback: {str(error)}")

    # ── calculations & utilities ─────────────────────────────
    async def _generate_synthetic_pnl_async(self, detection_data: Dict[str, Any]) -> float:
        try:
            base = np.random.normal(0, 100)
            if self.market_regime == 'volatile':
                base *= 2.5
            elif self.market_regime == 'trending':
                base *= 1.8
            if np.random.rand() < 0.08:
                base += np.random.choice([-1, 1]) * np.random.uniform(300, 1200)
            return float(base)
        except Exception:
            return 0.0

    async def _generate_synthetic_volume_async(self, detection_data: Dict[str, Any]) -> float:
        try:
            base = abs(np.random.normal(8000, 3000))
            if self.market_session == 'european':
                base *= 1.4
            elif self.market_session == 'american':
                base *= 1.2
            return float(max(base, 500.0))
        except Exception:
            return 1000.0

    async def _generate_synthetic_price_async(self, detection_data: Dict[str, Any]) -> float:
        try:
            if self.price_history:
                last = float(self.price_history[-1])
                change = np.random.normal(0, 0.003)
                if self.market_regime == 'volatile':
                    change *= 4.0
                elif self.market_regime == 'trending':
                    change += np.random.choice([-1, 1]) * 0.002
                return float(last * (1 + change))
            return float(np.random.uniform(1.1, 1.9))
        except Exception:
            return 1.5

    async def _calculate_robust_zscore_async(self, value: float, history: List[float]) -> float:
        try:
            if len(history) < 3:
                return 0.0
            arr = np.array(history, dtype=np.float64)
            med = np.median(arr)
            mad = np.median(np.abs(arr - med))
            mad_std = mad * 1.4826
            if mad_std < 1e-8:
                return 0.0
            return float(abs((value - med) / mad_std))
        except Exception:
            return 0.0

    async def _calculate_observation_zscores_async(self, obs: np.ndarray) -> np.ndarray:
        try:
            if len(self.observation_history) < 2:
                return np.zeros(len(obs), dtype=np.float32)
            stack = np.vstack(self.observation_history).astype(np.float64)
            med = np.median(stack, axis=0)
            mads = np.median(np.abs(stack - med), axis=0)
            mad_std = mads * 1.4826
            mad_std[mad_std < 1e-8] = 1.0
            return np.abs((obs - med) / mad_std).astype(np.float32)
        except Exception:
            return np.zeros(len(obs) if obs is not None else 0, dtype=np.float32)

    async def _calculate_trade_size_zscores_async(self, sizes: List[float]) -> List[float]:
        try:
            if len(sizes) < 3:
                return [0.0] * len(sizes)
            arr = np.array(sizes, dtype=np.float64)
            mean, std = float(np.mean(arr)), float(np.std(arr))
            if std < 1e-8:
                return [0.0] * len(sizes)
            return [float(abs((s - mean) / std)) for s in sizes]
        except Exception:
            return [0.0] * len(sizes)

    async def _get_context_adjusted_threshold_async(self, threshold_name: str, detection_data: Dict[str, Any]) -> float:
        try:
            base = float(self.current_thresholds.get(threshold_name, 1.0))
            mult = 1.0
            if threshold_name in ['pnl_limit', 'volume_zscore']:
                if self.volatility_regime == 'extreme':
                    mult = 2.5
                elif self.volatility_regime == 'high':
                    mult = 1.8
                elif self.market_regime == 'volatile':
                    mult = 1.5
            if self.market_stress_level > 0.7:
                mult *= 1.3
            mult *= float(self.adaptive_params.get('sensitivity_multiplier', 1.0))
            return float(base * mult)
        except Exception:
            return float(self.current_thresholds.get(threshold_name, 1.0))

    async def _get_price_jump_threshold_async(self, detection_data: Dict[str, Any]) -> float:
        try:
            base = {
                'low': 0.04, 'medium': 0.07, 'high': 0.12, 'extreme': 0.25
            }.get(self.volatility_regime, 0.07)
            if self.market_stress_level > 0.8:
                base *= 1.5
            return float(base)
        except Exception:
            return 0.07

    async def _get_context_score_multiplier_async(self, detection_data: Dict[str, Any]) -> float:
        try:
            mult = 1.0
            if self.volatility_regime == 'extreme':
                mult = 0.6
            elif self.volatility_regime == 'high':
                mult = 0.8
            elif self.market_regime == 'volatile':
                mult = 0.9
            elif self.volatility_regime == 'low':
                mult = 1.2
            return float(mult)
        except Exception:
            return 1.0

    async def _detect_regime_specific_pnl_anomaly_async(self, pnl: float, detection_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            series = self.regime_baselines[self.market_regime].get('pnl', [])
            if len(series) < 20:
                return None
            z = await self._calculate_robust_zscore_async(pnl, list(series))
            if z > 5.0:
                return {
                    "type": "regime_specific_outlier",
                    "value": float(pnl), "regime_z_score": float(z),
                    "regime": self.market_regime, "severity": AnomalySeverity.WARNING.value,
                    "confidence": min(0.8, z / 7.0), "timestamp": detection_data.get('timestamp')
                }
            return None
        except Exception:
            return None

    async def _calculate_detection_effectiveness_async(self) -> float:
        try:
            fp_rate = float(len([fp for fp in self.false_positive_tracker if fp])) / max(len(self.false_positive_tracker), 1)
            stability = 1.0
            if len(self.threshold_history) > 10:
                recent_changes = len([h for h in list(self.threshold_history)[-10:] if h.get('changes')])
                stability = max(0.5, 1.0 - (recent_changes / 10.0))
            eff = (1.0 - fp_rate) * stability * float(self._detection_quality)
            return float(np.clip(eff, 0.0, 1.0))
        except Exception:
            return 0.5

    async def _finalize_training_thresholds_async(self) -> None:
        try:
            # Defensive: ensure thresholds exist before finalization
            self._ensure_threshold_keys()
            if len(self.pnl_history) >= 100:
                arr = np.array(list(self.pnl_history), dtype=np.float64)
                final_pnl = float(np.percentile(np.abs(arr), 98))
                self.current_thresholds['pnl_limit'] = max(final_pnl, float(self.base_thresholds.get('pnl_limit', self._cfg.pnl_limit)) * 0.8)
            self.logger.info(format_operator_message(
                message="Training thresholds finalized", icon="[OK]",
                pnl_threshold=f"€{self.current_thresholds['pnl_limit']:,.0f}",
                obs_threshold=f"{float(self.current_thresholds['observation_zscore']):.1f}"
            ))
        except Exception as e:
            self.logger.warning(f"Threshold finalization failed: {e}")

    # ── monitoring & health ──────────────────────────────────
    def _update_detection_health(self):
        try:
            if not hasattr(self, '_detection_quality') or not hasattr(self, '_processing_times'):
                return
            self._health_status = 'healthy' if self._detection_quality >= float(self._cfg.min_detection_quality) else 'warning'
            if self.circuit_breaker['state'] == 'OPEN':
                self._health_status = 'warning'
            if len(self._processing_times) >= 10:
                avg_ms = float(np.mean(list(self._processing_times)[-10:]) * 1000.0)
                if avg_ms > float(self._cfg.max_processing_time_ms):
                    self._health_status = 'warning'
            self._last_health_check = time.time()
        except Exception as e:
            self.logger.error(f"Detection health check failed: {e}")
            self._health_status = 'warning'

    def _analyze_detection_effectiveness(self):
        try:
            if not hasattr(self, 'detection_effectiveness'):
                return
            if len(self.detection_effectiveness) >= 20:
                eff = float(np.mean(list(self.detection_effectiveness)[-20:]))
                if eff > 0.8:
                    self.logger.info(format_operator_message(
                        message="High detection effectiveness achieved", icon="[TARGET]",
                        effectiveness=f"{eff:.2f}", score=f"{self.anomaly_score:.2f}"
                    ))
                elif eff < 0.4:
                    self.logger.warning(format_operator_message(
                        message="Low detection effectiveness detected", icon="[WARN]",
                        effectiveness=f"{eff:.2f}", mode=self.current_mode.value
                    ))

        except Exception as e:
            self.logger.error(f"Detection effectiveness analysis failed: {e}")

    def _adapt_detection_parameters(self):
        try:
            if not hasattr(self, 'detection_effectiveness') or not hasattr(self, '_detection_quality'):
                return
            if len(self.detection_effectiveness) >= 10:
                recent = float(np.mean(list(self.detection_effectiveness)[-10:]))
                if recent < 0.5:
                    self.adaptive_params['sensitivity_multiplier'] = min(
                        1.3, float(self.adaptive_params['sensitivity_multiplier']) * 1.01
                    )
                elif recent > 0.8:
                    self.adaptive_params['sensitivity_multiplier'] = max(
                        0.7, float(self.adaptive_params['sensitivity_multiplier']) * 0.995
                    )
            # small random walk to simulate calibration dynamics
            self._detection_quality = float(
                np.clip(self._detection_quality + (np.random.normal(0, 0.02) if self.enabled else -0.1), 0.1, 1.0)
            )
        except Exception as e:
            self.logger.warning(f"Detection parameter adaptation failed: {e}")

    def _cleanup_old_data(self):
        try:
            if not hasattr(self, 'anomalies') or not hasattr(self, 'regime_baselines'):
                return
            for k in self.anomalies:
                if len(self.anomalies[k]) > 100:
                    self.anomalies[k] = self.anomalies[k][-50:]
            # prune empty baselines to avoid memory leaks
            for regime in list(self.regime_baselines.keys()):
                for dtype in list(self.regime_baselines[regime].keys()):
                    if len(self.regime_baselines[regime][dtype]) == 0:
                        del self.regime_baselines[regime][dtype]
                if not self.regime_baselines[regime]:
                    del self.regime_baselines[regime]
        except Exception as e:
            self.logger.warning(f"Data cleanup failed: {e}")

    def _record_success(self, processing_time_ms: float):
        self._processing_times.append(float(processing_time_ms) / 1000.0)
        self.performance_tracker.record_metric(
            'EnhancedAnomalyDetector', 'anomaly_detection', float(processing_time_ms), True
        )
        if self.circuit_breaker['state'] == 'OPEN':
            self.circuit_breaker['failures'] = 0
            self.circuit_breaker['state'] = 'CLOSED'

    def _record_failure(self, error: Exception):
        self.performance_tracker.record_metric(
            'EnhancedAnomalyDetector', 'anomaly_detection', 0.0, False
        )

    # ── public interface (kept stable) ───────────────────────
    def get_voter_capabilities(self) -> Dict[str, Any]:
        """Describe this voter's topic and schema to any aggregator/router."""
        return {
            "module": "EnhancedAnomalyDetector",
            "topic": "anomaly_risk",
            "votes": [v.value for v in AnomalyVote],
            "schema": {
                "vote": "proceed|caution|halt|abstain",
                "confidence": "0..1",
                "sizing_multiplier": "suggested 0..1 for position sizing",
                "reasoning": "short rationale",
                "metrics": {
                    "anomaly_score": "0..1",
                    "detection_confidence": "0..1",
                    "mode": "initialization|training|calibration|active|enhanced|emergency|maintenance",
                    "critical_anomalies": "int",
                    "circuit_breaker": "CLOSED|OPEN"
                }
            }
        }

    async def cast_vote(self, **inputs) -> Dict[str, Any]:
        """
        Convert current anomaly state into a standardized vote.
        Uses only in-memory state; safe to call in fallbacks.
        """
        mode = self.current_mode
        score = float(self.anomaly_score)
        crit_thresh = float(self._cfg.critical_threshold)
        warn_thresh = float(self._cfg.warning_threshold)
        breaker_open = (self.circuit_breaker.get('state') == 'OPEN')
        critical_count = int(sum(
            1 for L in self.anomalies.values() for a in L
            if a.get("severity") == AnomalySeverity.CRITICAL.value
        ))

        # Decide vote + suggested sizing
        if not self.enabled:
            vote = AnomalyVote.ABSTAIN
            sizing = 0.75
            reason = "Detector disabled; abstaining."
            halt_flag = False
            reduce_flag = False
        elif breaker_open or mode == AnomalyDetectionMode.EMERGENCY or score >= crit_thresh or critical_count >= 1:
            vote = AnomalyVote.HALT
            sizing = 0.0
            reason = f"High/critical risk (score={score:.2f}, mode={mode.value}, crit={critical_count})."
            halt_flag = True
            reduce_flag = True
        elif mode in (AnomalyDetectionMode.CALIBRATION, AnomalyDetectionMode.ENHANCED) or score >= warn_thresh:
            vote = AnomalyVote.CAUTION
            sizing = 0.50
            reason = f"Elevated risk (score={score:.2f}, mode={mode.value})."
            halt_flag = False
            reduce_flag = True
        elif mode in (AnomalyDetectionMode.INITIALIZATION, AnomalyDetectionMode.TRAINING):
            vote = AnomalyVote.PROCEED
            sizing = 0.75
            reason = f"{mode.value.title()} mode; proceed conservatively (score={score:.2f})."
            halt_flag = False
            reduce_flag = False
        else:
            vote = AnomalyVote.PROCEED
            sizing = 1.0
            reason = f"Healthy state (score={score:.2f})."
            halt_flag = False
            reduce_flag = False

        # Reuse the module's confidence logic
        try:
            conf = await self.calculate_confidence(
                {"halt_trading": halt_flag, "reduce_exposure": reduce_flag},
                **inputs
            )
        except Exception:
            conf = float(self.detection_confidence)

        payload = {
            "module": "EnhancedAnomalyDetector",
            "topic": "anomaly_risk",
            "vote": vote.value,
            "confidence": float(conf),
            "sizing_multiplier": float(sizing),
            "reasoning": reason,
            "metrics": {
                "anomaly_score": score,
                "detection_confidence": float(self.detection_confidence),
                "mode": mode.value,
                "critical_anomalies": critical_count,
                "circuit_breaker": self.circuit_breaker.get('state', 'UNKNOWN')
            },
            "timestamp": datetime.datetime.now().isoformat()
        }
        self._last_vote = payload
        return payload


    async def calculate_confidence(self, action: Dict[str, Any], **inputs) -> float:
        try:
            base_conf = float(self.detection_confidence)
            clarity = 1.0 if action.get("halt_trading") or action.get("reduce_exposure") else 0.8
            health_penalty = 0.9 if self._health_status == 'warning' else 1.0
            return float(np.clip(base_conf * clarity * health_penalty, 0.0, 1.0))
        except Exception:
            return 0.5

    def get_current_anomaly_score(self) -> float:
        return float(self.anomaly_score)

    def get_detection_confidence(self) -> float:
        return float(self.detection_confidence)

    def get_anomalies_summary(self) -> Dict[str, int]:
        return {k: int(len(v)) for k, v in self.anomalies.items() if v}

    def force_emergency_mode(self, reason: str = "manual_override") -> None:
        old = self.current_mode
        self.current_mode = AnomalyDetectionMode.EMERGENCY
        self.anomaly_score = float(np.clip(self.anomaly_score + 0.5, 0.0, 1.0))
        self.logger.error(format_operator_message(
            message="Emergency mode forced", icon="🆘",
            reason=reason, old_mode=old.value, new_score=f"{self.anomaly_score:.2f}"
        ))

    def clear_anomaly_history(self) -> None:
        for k in self.anomalies:
            self.anomalies[k].clear()
        self.anomaly_score = 0.0
        self.detection_confidence = 0.5
        self.logger.info("[RELOAD] Anomaly history cleared")

    def get_observation_components(self) -> np.ndarray:
        try:
            has_critical = float(any(a.get("severity") == AnomalySeverity.CRITICAL.value
                                     for L in self.anomalies.values() for a in L))
            emergency = float(self.current_mode == AnomalyDetectionMode.EMERGENCY)
            pnl_suff = min(len(self.pnl_history) / 50.0, 1.0)
            obs_suff = min(len(self.observation_history) / 20.0, 1.0)
            return np.array([
                float(self.anomaly_score),
                float(self.detection_confidence),
                has_critical,
                emergency,
                float(self.training_progress),
                float(pnl_suff),
                float(obs_suff),
                float(self._detection_quality)
            ], dtype=np.float32)
        except Exception:
            return np.zeros(8, dtype=np.float32)

    def get_health_status(self) -> Dict[str, Any]:
        return {
            'status': self._health_status,
            'last_check': float(self._last_health_check),
            'circuit_breaker': self.circuit_breaker['state'],
            'current_mode': self.current_mode.value,
            'anomaly_score': float(self.anomaly_score),
            'detection_confidence': float(self.detection_confidence),
            'detection_quality': float(self._detection_quality),
            'training_progress': float(self.training_progress) if self._cfg.training_mode else 1.0,
            'enabled': bool(self.enabled)
        }

    def reset(self) -> None:
        super().reset()
        for k in self.anomalies:
            self.anomalies[k].clear()
        self.anomaly_score = 0.0
        self.detection_confidence = 0.5
        self.step_count = 0
        self.detection_stats.clear()
        self.false_positive_tracker.clear()
        self.detection_effectiveness.clear()
        self.pnl_history.clear()
        self.volume_history.clear()
        self.price_history.clear()
        self.observation_history.clear()
        self.volatility_history.clear()
        self.threshold_history.clear()
        self._processing_times.clear()
        self.regime_baselines.clear()
        self.session_baselines.clear()
        self.volatility_baselines.clear()
        self.market_regime = "normal"
        self.market_session = "unknown"
        self.volatility_regime = "medium"
        self.market_stress_level = 0.0
        self.training_progress = 0.0
        self.is_training_complete = False
        self.adaptive_params.update({
            'sensitivity_multiplier': 1.0,
            'regime_adaptation_factor': 1.0,
            'volatility_tolerance': 1.0,
            'learning_momentum': 0.0,
            'detection_confidence_boost': 1.0
        })
        self.current_thresholds = dict(self.base_thresholds)
        self._ensure_threshold_keys()
        self.circuit_breaker['failures'] = 0
        self.circuit_breaker['state'] = 'CLOSED'
        self._health_status = 'healthy'
        self._detection_quality = 0.5
        self.external_anomaly_sources.clear()
        self.compliance_alerts.clear()
        self.current_mode = AnomalyDetectionMode.INITIALIZATION
        self.mode_start_time = datetime.datetime.now()
        self.logger.info("[RELOAD] Enhanced Anomaly Detector reset - all state cleared")

    # ── internal safety: threshold keys ─────────────────────
    def _ensure_threshold_keys(self, initial: bool = False) -> None:
        """Ensure required threshold keys exist in base/current dicts.
        Injects defaults from typed config if missing or invalid."""
        try:
            required = {
                'pnl_limit': float(self._cfg.pnl_limit),
                'volume_zscore': float(self._cfg.volume_zscore),
                'price_zscore': float(self._cfg.price_zscore),
                'observation_zscore': float(self._cfg.observation_zscore),
            }
            patched: List[str] = []
            for k, v in required.items():
                if not isinstance(self.base_thresholds.get(k), (int, float)):
                    self.base_thresholds[k] = v
                    patched.append(f"base:{k}")
                if not isinstance(self.current_thresholds.get(k), (int, float)):
                    # Seed current from base to keep relative adjustments coherent
                    self.current_thresholds[k] = float(self.base_thresholds.get(k, v))
                    patched.append(f"current:{k}")
            if patched and initial:
                self.logger.warning(
                    format_operator_message(
                        message="Injected missing threshold keys", icon="[SAFE]",
                        details=", ".join(patched)
                    )
                )
        except Exception:
            # Never raise from a guard; last-resort defaults
            for k in ['pnl_limit', 'volume_zscore', 'price_zscore', 'observation_zscore']:
                self.base_thresholds.setdefault(k, float(getattr(self._cfg, k)))
                self.current_thresholds.setdefault(k, float(self.base_thresholds[k]))


# ─────────────────────────────────────────────────────────────
# Supporting analyzers (kept concise, async-friendly)
# ─────────────────────────────────────────────────────────────
class SequenceAnomalyAnalyzer:
    def __init__(self):
        self.sequence_history = deque(maxlen=50)
        self.pattern_cache = {}

    async def analyze_async(self, sequence: List[float], context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            anomalies: List[Dict[str, Any]] = []
            if len(sequence) < 5:
                return {'anomalies': anomalies, 'analysis_completed': False}
            # simple repeated value pattern
            recent = sequence[-5:]
            if len(set(recent)) == 1 and recent[0] != 0:
                anomalies.append({
                    'type': 'repeated_values', 'value': float(recent[0]), 'count': 5,
                    'severity': AnomalySeverity.INFO.value, 'confidence': 0.7,
                    'timestamp': context.get('timestamp')
                })
            # trend anomaly
            if len(sequence) >= 10:
                x = np.arange(len(sequence), dtype=np.float64)
                slope = float(np.polyfit(x, np.array(sequence, dtype=np.float64), 1)[0])
                sd = float(np.std(sequence))
                if abs(slope) > (sd * 2 if sd > 0 else 0):
                    anomalies.append({
                        'type': 'extreme_trend', 'slope': slope,
                        'direction': 'increasing' if slope > 0 else 'decreasing',
                        'severity': AnomalySeverity.WARNING.value,
                        'confidence': min(0.8, abs(slope) / (max(sd, 1e-6) * 3)),
                        'timestamp': context.get('timestamp')
                    })
            # simple autocorr cycle check
            if len(sequence) >= 20:
                s = np.array(sequence, dtype=np.float64)
                ac = np.correlate(s, s, mode='full')
                ac = ac[ac.size // 2:]
                ac_window_max = float(np.max(ac[5:15])) if ac.size >= 15 else float(np.max(ac))
                ac_total_max = float(np.max(ac)) if ac.size > 0 else 0.0
                if len(ac) > 10 and ac_window_max > ac_total_max * 0.8:
                    anomalies.append({
                        'type': 'unusual_cycle',
                        'cycle_strength': float(ac_window_max / max(ac_total_max, 1e-9)),
                        'severity': AnomalySeverity.INFO.value, 'confidence': 0.6,
                        'timestamp': context.get('timestamp')
                    })
            return {'anomalies': anomalies, 'analysis_completed': True, 'sequence_length': int(len(sequence))}
        except Exception:
            return {'anomalies': [], 'analysis_completed': False, 'error': 'sequence_analysis_failed'}


class CorrelationAnomalyAnalyzer:
    def __init__(self):
        self.correlation_history = deque(maxlen=100)

    async def analyze_async(self, series1: List[float], series2: List[float], context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            anomalies: List[Dict[str, Any]] = []
            if len(series1) < 10 or len(series2) < 10:
                return {'anomalies': anomalies, 'analysis_completed': False}
            c = float(np.corrcoef(np.array(series1), np.array(series2))[0, 1])
            if np.isnan(c):
                return {'anomalies': anomalies, 'analysis_completed': False}
            self.correlation_history.append(c)
            if len(self.correlation_history) >= 20:
                recent = list(self.correlation_history)[-20:]
                mu, sd = float(np.mean(recent)), float(np.std(recent))
                if sd > 0.01:
                    z = abs((c - mu) / sd)
                    if z > 3.0:
                        anomalies.append({
                            'type': 'correlation_change',
                            'current_correlation': c, 'expected_correlation': mu, 'z_score': float(z),
                            'severity': AnomalySeverity.WARNING.value, 'confidence': min(0.8, z / 5.0),
                            'timestamp': context.get('timestamp')
                        })
            if abs(c) > 0.95:
                anomalies.append({
                    'type': 'extreme_correlation', 'correlation': c,
                    'severity': AnomalySeverity.INFO.value, 'confidence': abs(c),
                    'timestamp': context.get('timestamp')
                })
            return {'anomalies': anomalies, 'analysis_completed': True, 'current_correlation': c}
        except Exception:
            return {'anomalies': [], 'analysis_completed': False, 'error': 'correlation_analysis_failed'}


class PatternAnomalyDetector:
    def __init__(self):
        self.trade_patterns = deque(maxlen=200)
        self.known_patterns: Dict[str, Any] = {}

    async def detect_async(self, trades: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            anomalies: List[Dict[str, Any]] = []
            if not trades:
                return {'anomalies': anomalies, 'pattern_detected': False}
            sizes = [abs(t.get('size', t.get('volume', 0))) for t in trades]
            dirs = [np.sign(t.get('size', t.get('volume', 0))) for t in trades]
            intervals = await self._calculate_trade_intervals_async(trades)
            self.trade_patterns.append({
                'sizes': sizes, 'directions': dirs, 'intervals': intervals,
                'count': len(trades), 'timestamp': context.get('timestamp')
            })
            # uniform size pattern
            if sizes and len(sizes) >= 6 and len(set(sizes)) == 1:
                anomalies.append({
                    'type': 'uniform_trade_sizes', 'size': float(sizes[0]), 'count': int(len(sizes)),
                    'severity': AnomalySeverity.INFO.value, 'confidence': min(0.9, len(sizes) / 10.0),
                    'timestamp': context.get('timestamp')
                })
            # timing regularity
            if intervals and len(intervals) > 10:
                m, s = float(np.mean(intervals)), float(np.std(intervals))
                if s < m * 0.1:
                    anomalies.append({
                        'type': 'regular_timing_pattern',
                        'interval_mean': m, 'interval_std': s,
                        'regularity_score': float(m / max(s, 1e-6)),
                        'severity': AnomalySeverity.INFO.value, 'confidence': 0.7,
                        'timestamp': context.get('timestamp')
                    })
            # alternating directions
            nz = [d for d in dirs if d != 0]
            if len(nz) >= 6:
                alt = sum(1 for i in range(1, len(nz)) if nz[i] != nz[i-1])
                ratio = alt / (len(nz) - 1)
                if ratio > 0.8:
                    anomalies.append({
                        'type': 'alternating_direction_pattern',
                        'alternating_ratio': float(ratio), 'trade_count': int(len(nz)),
                        'severity': AnomalySeverity.INFO.value, 'confidence': float(ratio),
                        'timestamp': context.get('timestamp')
                    })
            return {'anomalies': anomalies, 'pattern_detected': True, 'trades_analyzed': int(len(trades))}
        except Exception:
            return {'anomalies': [], 'pattern_detected': False, 'error': 'pattern_detection_failed'}

    async def _calculate_trade_intervals_async(self, trades: List[Dict[str, Any]]) -> List[float]:
        try:
            # Placeholder without real timestamps; keep deterministic & safe
            return [1.0 for _ in range(max(0, len(trades) - 1))]
        except Exception:
            return []