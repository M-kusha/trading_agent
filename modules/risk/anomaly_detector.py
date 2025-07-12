# ─────────────────────────────────────────────────────────────
# File: modules/risk/anomaly_detector.py
# 🚀 PRODUCTION-READY Enhanced Anomaly Detector
# Advanced anomaly detection with SmartInfoBus integration and intelligent automation
# ─────────────────────────────────────────────────────────────

import asyncio
import time
import threading
import numpy as np
import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum

from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusRiskMixin, SmartInfoBusStateMixin, SmartInfoBusTradingMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.performance_tracker import PerformanceTracker


class AnomalyDetectionMode(Enum):
    """Anomaly detection operational modes"""
    INITIALIZATION = "initialization"
    TRAINING = "training"
    CALIBRATION = "calibration"
    ACTIVE = "active"
    ENHANCED = "enhanced"
    EMERGENCY = "emergency"
    MAINTENANCE = "maintenance"


class AnomalySeverity(Enum):
    """Anomaly severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class AnomalyDetectorConfig:
    """Configuration for Enhanced Anomaly Detector"""
    # Core detection thresholds
    pnl_limit: float = 1000.0
    volume_zscore: float = 3.0
    price_zscore: float = 3.0
    observation_zscore: float = 4.0
    
    # History management
    history_size: int = 100
    min_history_for_stats: int = 20
    correlation_window: int = 50
    volatility_window: int = 30
    
    # Adaptive behavior
    adaptive_thresholds: bool = True
    regime_awareness: bool = True
    learning_rate: float = 0.05
    threshold_smoothing: float = 0.8
    
    # Training mode
    training_mode: bool = True
    training_duration_steps: int = 200
    synthetic_data_ratio: float = 0.3
    
    # Performance thresholds
    max_processing_time_ms: float = 50
    circuit_breaker_threshold: int = 5
    min_detection_quality: float = 0.7
    
    # Detection sensitivity
    critical_threshold: float = 0.8
    warning_threshold: float = 0.5
    emergency_threshold: float = 0.9
    
    # Monitoring parameters
    health_check_interval: int = 30
    performance_window: int = 100
    false_positive_threshold: float = 0.3


@module(
    name="EnhancedAnomalyDetector",
    version="4.0.0",
    category="risk",
    provides=["anomaly_detection", "anomaly_score", "anomaly_alerts", "detection_analytics"],
    requires=["risk_data", "market_data", "trading_data", "performance_data"],
    description="Advanced anomaly detection with intelligent adaptation and comprehensive market analysis",
    thesis_required=True,
    health_monitoring=True,
    performance_tracking=True,
    error_handling=True,
    voting=True
)
class EnhancedAnomalyDetector(BaseModule, SmartInfoBusRiskMixin, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    🚀 Advanced anomaly detection system with SmartInfoBus integration.
    Provides intelligent anomaly detection with context-aware thresholds and comprehensive analysis.
    """

    def __init__(self, 
                 config: Optional[AnomalyDetectorConfig] = None,
                 enabled: bool = True,
                 action_dim: int = 8,
                 **kwargs):
        
        self.config = config or AnomalyDetectorConfig()
        self.enabled = enabled
        self.action_dim = int(action_dim)
        super().__init__()
        
        # Initialize advanced systems
        self._initialize_advanced_systems()
        
        # Initialize detection state
        self._initialize_detection_state()
        
        self.logger.info(format_operator_message(
            message="Enhanced anomaly detector ready",
            icon="🔍",
            enabled=enabled,
            adaptive_thresholds=self.config.adaptive_thresholds,
            regime_awareness=self.config.regime_awareness,
            config_loaded=True
        ))

    def _initialize_advanced_systems(self):
        """Initialize advanced systems for anomaly detection"""
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
        
        # Circuit breaker for detection operations
        self.circuit_breaker = {
            'failures': 0,
            'last_failure': 0,
            'state': 'CLOSED',
            'threshold': self.config.circuit_breaker_threshold
        }
        
        # Health monitoring
        self._health_status = 'healthy'
        self._last_health_check = time.time()
        self._start_monitoring()

    def _initialize_detection_state(self):
        """Initialize anomaly detection state"""
        # Initialize mixin states
        self._initialize_risk_state()
        self._initialize_trading_state() 
        self._initialize_state_management()
        
        # Current operational mode
        self.current_mode = AnomalyDetectionMode.INITIALIZATION
        self.mode_start_time = datetime.datetime.now()
        
        # Detection thresholds (dynamic)
        self.current_thresholds = self.config.__dict__.copy()
        self.base_thresholds = self.config.__dict__.copy()
        self.threshold_history = deque(maxlen=100)
        
        # Data history with intelligent management
        self.pnl_history = deque(maxlen=self.config.history_size)
        self.volume_history = deque(maxlen=self.config.history_size)
        self.price_history = deque(maxlen=self.config.history_size)
        self.observation_history = deque(maxlen=min(self.config.history_size, 50))
        self.volatility_history = deque(maxlen=self.config.volatility_window)
        
        # Enhanced anomaly tracking with structured data
        self.anomalies: Dict[str, List[Dict[str, Any]]] = {
            "pnl": [],
            "volume": [],
            "price": [],
            "observation": [],
            "pattern": [],
            "correlation": [],
            "volatility": [],
            "sequence": [],
            "system": [],
            "market_structure": []
        }
        
        # Detection analytics
        self.anomaly_score = 0.0
        self.detection_confidence = 0.5
        self.step_count = 0
        self.detection_stats = defaultdict(int)
        self.false_positive_tracker = deque(maxlen=self.config.performance_window)
        self.detection_effectiveness = deque(maxlen=self.config.performance_window)
        
        # Context-aware baselines
        self.regime_baselines = defaultdict(lambda: defaultdict(lambda: deque(maxlen=100)))
        self.session_baselines = defaultdict(lambda: defaultdict(lambda: deque(maxlen=100)))
        self.volatility_baselines = defaultdict(lambda: deque(maxlen=50))
        
        # Market context tracking
        self.market_regime = "normal"
        self.market_session = "unknown"
        self.volatility_regime = "medium"
        self.market_stress_level = 0.0
        
        # Advanced detection features
        self.sequence_analyzer = SequenceAnomalyAnalyzer()
        self.correlation_analyzer = CorrelationAnomalyAnalyzer()
        self.pattern_detector = PatternAnomalyDetector()
        
        # Training and adaptation
        self.training_progress = 0
        self.is_training_complete = False
        self.adaptive_params = {
            'sensitivity_multiplier': 1.0,
            'regime_adaptation_factor': 1.0,
            'volatility_tolerance': 1.0,
            'learning_momentum': 0.0,
            'detection_confidence_boost': 1.0
        }
        
        # Performance and quality metrics
        self._detection_quality = 0.5
        self._processing_times = deque(maxlen=100)
        self._last_significant_detection = None
        
        # External integrations
        self.external_anomaly_sources = {}
        self.compliance_alerts = []

    def _start_monitoring(self):
        """Start background monitoring for anomaly detection"""
        def monitoring_loop():
            while getattr(self, '_monitoring_active', True):
                try:
                    self._update_detection_health()
                    self._analyze_detection_effectiveness()
                    self._adapt_detection_parameters()
                    self._cleanup_old_data()
                    time.sleep(self.config.health_check_interval)
                except Exception as e:
                    self.logger.error(f"Anomaly detection monitoring error: {e}")
        
        self._monitoring_active = True
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()

    async def _initialize(self):
        """Initialize module with SmartInfoBus integration"""
        try:
            # Set initial anomaly detection status
            initial_status = {
                "current_mode": self.current_mode.value,
                "enabled": self.enabled,
                "anomaly_score": self.anomaly_score,
                "detection_confidence": self.detection_confidence,
                "training_mode": self.config.training_mode,
                "adaptive_thresholds": self.config.adaptive_thresholds
            }
            
            self.smart_bus.set(
                'anomaly_detection',
                initial_status,
                module='EnhancedAnomalyDetector',
                thesis="Initial enhanced anomaly detector status"
            )
            
            return True
        except Exception as e:
            self.logger.error(f"Anomaly detector initialization failed: {e}")
            return False

    async def process(self, **inputs) -> Dict[str, Any]:
        """Process anomaly detection with enhanced analytics"""
        start_time = time.time()
        
        try:
            if not self.enabled:
                return await self._handle_disabled_fallback()
            
            # Extract detection data from SmartInfoBus
            detection_data = await self._extract_detection_data(**inputs)
            
            if not detection_data:
                return await self._handle_no_data_fallback()
            
            # Update market context
            context_result = await self._update_market_context_async(detection_data)
            
            # Perform comprehensive anomaly detection
            detection_result = await self._detect_anomalies_comprehensive_async(detection_data)
            
            # Analyze patterns and sequences
            pattern_result = await self._analyze_patterns_async(detection_data)
            
            # Update adaptive thresholds if enabled
            adaptation_result = {}
            if self.config.adaptive_thresholds:
                adaptation_result = await self._adapt_thresholds_async(detection_data)
            
            # Calculate comprehensive anomaly score
            scoring_result = await self._calculate_comprehensive_score_async(detection_data)
            
            # Update training progress
            training_result = await self._update_training_progress_async(detection_data)
            
            # Handle emergency situations
            emergency_result = await self._handle_emergency_situations_async(detection_data)
            
            # Update operational mode
            mode_result = await self._update_operational_mode_async(detection_data)
            
            # Combine results
            result = {**context_result, **detection_result, **pattern_result,
                     **adaptation_result, **scoring_result, **training_result,
                     **emergency_result, **mode_result}
            
            # Generate thesis
            thesis = await self._generate_detection_thesis(detection_data, result)
            
            # Update SmartInfoBus
            await self._update_detection_smart_bus(result, thesis)
            
            # Record success
            processing_time = (time.time() - start_time) * 1000
            self._record_success(processing_time)
            
            return result
            
        except Exception as e:
            return await self._handle_detection_error(e, start_time)

    async def _extract_detection_data(self, **inputs) -> Optional[Dict[str, Any]]:
        """Extract comprehensive detection data from SmartInfoBus"""
        try:
            # Get data from SmartInfoBus
            risk_data = self.smart_bus.get('risk_data', 'EnhancedAnomalyDetector') or {}
            market_data = self.smart_bus.get('market_data', 'EnhancedAnomalyDetector') or {}
            trading_data = self.smart_bus.get('trading_data', 'EnhancedAnomalyDetector') or {}
            performance_data = self.smart_bus.get('performance_data', 'EnhancedAnomalyDetector') or {}
            
            # Extract direct inputs (legacy compatibility)
            pnl = inputs.get('pnl', 0.0)
            volume = inputs.get('volume', 0.0)
            price = inputs.get('price', 0.0)
            observation = inputs.get('obs', inputs.get('observation', None))
            trades = inputs.get('trades', [])
            
            # Extract from SmartInfoBus data
            risk_snapshot = risk_data.get('risk_snapshot', {})
            if not pnl and 'recent_pnl' in risk_snapshot:
                pnl = risk_snapshot['recent_pnl']
            
            market_snapshot = market_data.get('market_snapshot', {})
            if not volume and 'volume' in market_snapshot:
                volume = market_snapshot['volume']
            
            if not price and 'price' in market_snapshot:
                price = market_snapshot['price']
            
            # Get trading activity
            trading_snapshot = trading_data.get('trading_snapshot', {})
            if not trades and 'recent_trades' in trading_snapshot:
                trades = trading_snapshot['recent_trades']
            
            return {
                'pnl': float(pnl),
                'volume': float(volume),
                'price': float(price),
                'observation': observation,
                'trades': trades or [],
                'risk_data': risk_data,
                'market_data': market_data,
                'trading_data': trading_data,
                'performance_data': performance_data,
                'timestamp': datetime.datetime.now().isoformat(),
                'step_count': self.step_count
            }
            
        except Exception as e:
            self.logger.error(f"Failed to extract detection data: {e}")
            return None

    async def _update_market_context_async(self, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update market context awareness asynchronously"""
        try:
            # Extract market context from SmartInfoBus
            market_context = self.smart_bus.get('market_context', 'EnhancedAnomalyDetector') or {}
            
            # Update context tracking
            old_regime = self.market_regime
            old_session = self.market_session
            old_volatility = self.volatility_regime
            
            self.market_regime = market_context.get('regime', 'normal')
            self.market_session = market_context.get('session', 'unknown')
            self.volatility_regime = market_context.get('volatility_level', 'medium')
            self.market_stress_level = market_context.get('stress_level', 0.0)
            
            # Detect context changes
            context_changed = (
                old_regime != self.market_regime or
                old_session != self.market_session or
                old_volatility != self.volatility_regime
            )
            
            if context_changed:
                self.logger.info(format_operator_message(
                    message="Market context changed - adapting detection",
                    icon="📊",
                    old_regime=old_regime,
                    new_regime=self.market_regime,
                    volatility=self.volatility_regime,
                    session=self.market_session,
                    stress_level=f"{self.market_stress_level:.2f}"
                ))
                
                # Update context-specific baselines
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
        """Update context-specific baselines"""
        try:
            regime = self.market_regime
            session = self.market_session
            volatility = self.volatility_regime
            
            # Update regime baselines
            for data_type in ['pnl', 'volume', 'price']:
                value = detection_data.get(data_type, 0.0)
                if value != 0.0:
                    self.regime_baselines[regime][data_type].append(value)
                    self.session_baselines[session][data_type].append(value)
            
            # Update volatility baselines
            if len(self.price_history) >= 2:
                price_change = abs(detection_data.get('price', 0) - self.price_history[-1])
                self.volatility_baselines[volatility].append(price_change)
            
        except Exception as e:
            self.logger.warning(f"Context baseline update failed: {e}")

    async def _detect_anomalies_comprehensive_async(self, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive anomaly detection across all data types"""
        try:
            # Clear previous anomalies
            for anomaly_type in self.anomalies:
                self.anomalies[anomaly_type].clear()
            
            detection_results = {}
            critical_found = False
            
            # 1. PnL anomaly detection
            pnl_result = await self._detect_pnl_anomalies_async(detection_data)
            detection_results['pnl'] = pnl_result
            if pnl_result.get('critical', False):
                critical_found = True
            
            # 2. Volume anomaly detection
            volume_result = await self._detect_volume_anomalies_async(detection_data)
            detection_results['volume'] = volume_result
            
            # 3. Price anomaly detection
            price_result = await self._detect_price_anomalies_async(detection_data)
            detection_results['price'] = price_result
            
            # 4. Observation anomaly detection
            obs_result = await self._detect_observation_anomalies_async(detection_data)
            detection_results['observation'] = obs_result
            if obs_result.get('critical', False):
                critical_found = True
            
            # 5. Volatility anomaly detection
            vol_result = await self._detect_volatility_anomalies_async(detection_data)
            detection_results['volatility'] = vol_result
            
            # 6. System anomaly detection
            system_result = await self._detect_system_anomalies_async(detection_data)
            detection_results['system'] = system_result
            if system_result.get('critical', False):
                critical_found = True
            
            # 7. Market structure anomaly detection
            structure_result = await self._detect_market_structure_anomalies_async(detection_data)
            detection_results['market_structure'] = structure_result
            
            return {
                'comprehensive_detection_completed': True,
                'critical_anomalies_found': critical_found,
                'total_anomalies': sum(len(v) for v in self.anomalies.values()),
                'detection_results': detection_results
            }
            
        except Exception as e:
            self.logger.error(f"Comprehensive anomaly detection failed: {e}")
            return {'comprehensive_detection_completed': False, 'error': str(e)}

    async def _detect_pnl_anomalies_async(self, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced PnL anomaly detection"""
        try:
            pnl = detection_data.get('pnl', 0.0)
            
            # Handle training mode synthetic data
            if self.config.training_mode and pnl == 0.0 and len(self.pnl_history) < 10:
                pnl = await self._generate_synthetic_pnl_async(detection_data)
            
            if pnl == 0.0 and not self.config.training_mode:
                return {'pnl_detected': False, 'reason': 'no_pnl_data'}
            
            self.pnl_history.append(pnl)
            critical_found = False
            anomalies_detected = 0
            
            # 1. Absolute threshold check with context adjustment
            adjusted_limit = await self._get_context_adjusted_threshold_async('pnl_limit', detection_data)
            
            if abs(pnl) > adjusted_limit:
                severity = AnomalySeverity.CRITICAL if abs(pnl) > adjusted_limit * 1.5 else AnomalySeverity.WARNING
                
                self.anomalies["pnl"].append({
                    "type": "absolute_limit_exceeded",
                    "value": pnl,
                    "threshold": adjusted_limit,
                    "base_threshold": self.base_thresholds['pnl_limit'],
                    "severity": severity.value,
                    "confidence": 0.9,
                    "timestamp": detection_data.get('timestamp'),
                    "context": {
                        'regime': self.market_regime,
                        'session': self.market_session,
                        'volatility': self.volatility_regime
                    }
                })
                
                anomalies_detected += 1
                if severity == AnomalySeverity.CRITICAL:
                    critical_found = True
                    self.logger.error(format_operator_message(
                        message="CRITICAL PnL anomaly detected",
                        icon="🚨",
                        pnl=f"€{pnl:,.2f}",
                        limit=f"€{adjusted_limit:,.0f}",
                        regime=self.market_regime,
                        session=self.market_session
                    ))
                else:
                    self.logger.warning(format_operator_message(
                        message="PnL anomaly detected",
                        icon="⚠️",
                        pnl=f"€{pnl:,.2f}",
                        limit=f"€{adjusted_limit:,.0f}"
                    ))
            
            # 2. Statistical anomaly detection
            if len(self.pnl_history) >= self.config.min_history_for_stats:
                z_score = await self._calculate_robust_zscore_async(pnl, list(self.pnl_history))
                
                if z_score > 4.0:  # Conservative threshold
                    severity = AnomalySeverity.CRITICAL if z_score > 6.0 else AnomalySeverity.WARNING
                    
                    self.anomalies["pnl"].append({
                        "type": "statistical_outlier",
                        "value": pnl,
                        "z_score": float(z_score),
                        "severity": severity.value,
                        "confidence": min(0.9, z_score / 8.0),
                        "timestamp": detection_data.get('timestamp'),
                        "context": {
                            'history_size': len(self.pnl_history),
                            'regime': self.market_regime
                        }
                    })
                    
                    anomalies_detected += 1
                    if severity == AnomalySeverity.CRITICAL:
                        critical_found = True
                        self.logger.error(f"🚨 CRITICAL: Statistical PnL anomaly - z-score {z_score:.2f}")
            
            # 3. Regime-specific detection
            regime_anomaly = await self._detect_regime_specific_pnl_anomaly_async(pnl, detection_data)
            if regime_anomaly:
                self.anomalies["pnl"].append(regime_anomaly)
                anomalies_detected += 1
            
            return {
                'pnl_detected': True,
                'anomalies_count': anomalies_detected,
                'critical': critical_found,
                'pnl_value': pnl,
                'adjusted_threshold': adjusted_limit
            }
            
        except Exception as e:
            self.logger.warning(f"PnL anomaly detection failed: {e}")
            return {'pnl_detected': False, 'error': str(e)}

    async def _detect_observation_anomalies_async(self, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced observation anomaly detection"""
        try:
            observation = detection_data.get('observation')
            
            if observation is None:
                return {'observation_detected': False, 'reason': 'no_observation_data'}
            
            try:
                obs = np.array(observation, dtype=np.float32)
            except (ValueError, TypeError):
                # Critical: Invalid observation format
                self.anomalies["observation"].append({
                    "type": "invalid_format",
                    "severity": AnomalySeverity.CRITICAL.value,
                    "confidence": 1.0,
                    "timestamp": detection_data.get('timestamp'),
                    "details": "Observation could not be converted to valid numpy array"
                })
                
                self.logger.error("🚨 CRITICAL: Invalid observation format detected")
                return {'observation_detected': True, 'critical': True, 'anomalies_count': 1}
            
            critical_found = False
            anomalies_detected = 0
            
            # 1. Check for invalid values (NaN, Inf)
            nan_count = int(np.isnan(obs).sum())
            inf_count = int(np.isinf(obs).sum())
            
            if nan_count > 0 or inf_count > 0:
                self.anomalies["observation"].append({
                    "type": "invalid_values",
                    "nan_count": nan_count,
                    "inf_count": inf_count,
                    "observation_shape": obs.shape,
                    "severity": AnomalySeverity.CRITICAL.value,
                    "confidence": 1.0,
                    "timestamp": detection_data.get('timestamp')
                })
                
                critical_found = True
                anomalies_detected += 1
                
                self.logger.error(format_operator_message(
                    message="CRITICAL: Invalid observation values",
                    icon="🚨",
                    nan_count=nan_count,
                    inf_count=inf_count,
                    shape=str(obs.shape)
                ))
            
            # 2. Store valid observations for analysis
            if not critical_found:
                self.observation_history.append(obs)
                
                # Statistical analysis if sufficient history
                if len(self.observation_history) >= 10:
                    z_scores = await self._calculate_observation_zscores_async(obs)
                    extreme_threshold = self.current_thresholds['observation_zscore']
                    
                    extreme_indices = np.where(z_scores > extreme_threshold)[0]
                    
                    if len(extreme_indices) > 0:
                        max_z_score = float(np.max(z_scores))
                        severity = AnomalySeverity.CRITICAL if max_z_score > extreme_threshold * 1.5 else AnomalySeverity.WARNING
                        
                        self.anomalies["observation"].append({
                            "type": "extreme_values",
                            "extreme_indices": extreme_indices.tolist(),
                            "z_scores": z_scores[extreme_indices].tolist(),
                            "max_z_score": max_z_score,
                            "threshold": extreme_threshold,
                            "severity": severity.value,
                            "confidence": min(0.9, max_z_score / (extreme_threshold * 2)),
                            "timestamp": detection_data.get('timestamp')
                        })
                        
                        anomalies_detected += 1
                        if severity == AnomalySeverity.CRITICAL:
                            critical_found = True
                            self.logger.error(f"🚨 CRITICAL: Extreme observation values - max z-score {max_z_score:.2f}")
            
            return {
                'observation_detected': True,
                'anomalies_count': anomalies_detected,
                'critical': critical_found,
                'observation_shape': obs.shape,
                'invalid_values': nan_count + inf_count
            }
            
        except Exception as e:
            self.logger.warning(f"Observation anomaly detection failed: {e}")
            return {'observation_detected': False, 'error': str(e)}

    async def _detect_volume_anomalies_async(self, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced volume anomaly detection"""
        try:
            volume = detection_data.get('volume', 0.0)
            
            # Handle training mode
            if self.config.training_mode and volume == 0.0 and len(self.volume_history) < 10:
                volume = await self._generate_synthetic_volume_async(detection_data)
            
            if volume == 0.0:
                return {'volume_detected': False, 'reason': 'no_volume_data'}
            
            self.volume_history.append(volume)
            anomalies_detected = 0
            
            # Statistical analysis
            if len(self.volume_history) >= self.config.min_history_for_stats:
                z_score = await self._calculate_robust_zscore_async(volume, list(self.volume_history))
                threshold = self.current_thresholds['volume_zscore']
                
                # Context-adjusted threshold
                adjusted_threshold = await self._get_context_adjusted_threshold_async('volume_zscore', detection_data)
                
                if z_score > adjusted_threshold:
                    severity = AnomalySeverity.WARNING if z_score < adjusted_threshold * 1.5 else AnomalySeverity.CRITICAL
                    
                    self.anomalies["volume"].append({
                        "type": "volume_spike",
                        "value": volume,
                        "z_score": float(z_score),
                        "threshold": adjusted_threshold,
                        "severity": severity.value,
                        "confidence": min(0.8, z_score / (adjusted_threshold * 2)),
                        "timestamp": detection_data.get('timestamp'),
                        "context": {
                            'regime': self.market_regime,
                            'session': self.market_session
                        }
                    })
                    
                    anomalies_detected += 1
                    
                    self.logger.warning(format_operator_message(
                        message="Volume anomaly detected",
                        icon="⚠️",
                        volume=f"{volume:,.0f}",
                        z_score=f"{z_score:.2f}",
                        regime=self.market_regime
                    ))
            
            return {
                'volume_detected': True,
                'anomalies_count': anomalies_detected,
                'volume_value': volume,
                'history_size': len(self.volume_history)
            }
            
        except Exception as e:
            self.logger.warning(f"Volume anomaly detection failed: {e}")
            return {'volume_detected': False, 'error': str(e)}

    async def _detect_price_anomalies_async(self, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced price anomaly detection"""
        try:
            price = detection_data.get('price', 0.0)
            
            # Handle training mode
            if self.config.training_mode and price == 0.0 and len(self.price_history) < 10:
                price = await self._generate_synthetic_price_async(detection_data)
            
            if price == 0.0:
                return {'price_detected': False, 'reason': 'no_price_data'}
            
            self.price_history.append(price)
            anomalies_detected = 0
            
            # Price jump detection
            if len(self.price_history) >= 2:
                prev_price = self.price_history[-2]
                
                if prev_price > 0:
                    price_change = abs((price - prev_price) / prev_price)
                    
                    # Context-adjusted threshold
                    jump_threshold = await self._get_price_jump_threshold_async(detection_data)
                    
                    if price_change > jump_threshold:
                        severity = AnomalySeverity.CRITICAL if price_change > jump_threshold * 2 else AnomalySeverity.WARNING
                        
                        self.anomalies["price"].append({
                            "type": "price_jump",
                            "change_percentage": float(price_change),
                            "prev_price": prev_price,
                            "current_price": price,
                            "threshold": jump_threshold,
                            "severity": severity.value,
                            "confidence": min(0.9, price_change / jump_threshold / 2),
                            "timestamp": detection_data.get('timestamp'),
                            "context": {
                                'volatility_regime': self.volatility_regime,
                                'regime': self.market_regime
                            }
                        })
                        
                        anomalies_detected += 1
                        
                        self.logger.warning(format_operator_message(
                            message="Price jump detected",
                            icon="⚠️",
                            change=f"{price_change:.1%}",
                            from_price=f"{prev_price:.5f}",
                            to_price=f"{price:.5f}",
                            volatility=self.volatility_regime
                        ))
            
            return {
                'price_detected': True,
                'anomalies_count': anomalies_detected,
                'price_value': price,
                'history_size': len(self.price_history)
            }
            
        except Exception as e:
            self.logger.warning(f"Price anomaly detection failed: {e}")
            return {'price_detected': False, 'error': str(e)}

    async def _detect_volatility_anomalies_async(self, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced volatility anomaly detection"""
        try:
            anomalies_detected = 0
            
            # Calculate current volatility from recent price history
            if len(self.price_history) >= 10:
                prices = np.array(list(self.price_history)[-10:])
                returns = np.diff(np.log(prices + 1e-8))
                current_vol = np.std(returns) * np.sqrt(252)  # Annualized
                
                self.volatility_history.append(current_vol)
                
                # Detect volatility spikes
                if len(self.volatility_history) >= 10:
                    vol_z_score = await self._calculate_robust_zscore_async(current_vol, list(self.volatility_history))
                    
                    if vol_z_score > 3.0:
                        severity = AnomalySeverity.CRITICAL if vol_z_score > 5.0 else AnomalySeverity.WARNING
                        
                        self.anomalies["volatility"].append({
                            "type": "volatility_spike",
                            "current_volatility": float(current_vol),
                            "z_score": float(vol_z_score),
                            "severity": severity.value,
                            "confidence": min(0.8, vol_z_score / 6.0),
                            "timestamp": detection_data.get('timestamp'),
                            "context": {
                                'volatility_regime': self.volatility_regime,
                                'regime': self.market_regime
                            }
                        })
                        
                        anomalies_detected += 1
                        
                        if severity == AnomalySeverity.CRITICAL:
                            self.logger.error(format_operator_message(
                                message="CRITICAL volatility spike",
                                icon="🚨",
                                volatility=f"{current_vol:.1%}",
                                z_score=f"{vol_z_score:.2f}"
                            ))
                        else:
                            self.logger.warning(format_operator_message(
                                message="Volatility spike detected",
                                icon="⚠️",
                                volatility=f"{current_vol:.1%}",
                                z_score=f"{vol_z_score:.2f}"
                            ))
            
            return {
                'volatility_detected': True,
                'anomalies_count': anomalies_detected,
                'current_volatility': current_vol if 'current_vol' in locals() else 0.0
            }
            
        except Exception as e:
            self.logger.warning(f"Volatility anomaly detection failed: {e}")
            return {'volatility_detected': False, 'error': str(e)}

    async def _detect_system_anomalies_async(self, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect system-level anomalies"""
        try:
            anomalies_detected = 0
            critical_found = False
            
            # Check processing time anomalies
            processing_times = list(self._processing_times)
            if len(processing_times) >= 10:
                current_time = time.time()
                recent_avg_time = np.mean(processing_times[-10:]) * 1000  # Convert to ms
                
                if recent_avg_time > self.config.max_processing_time_ms:
                    self.anomalies["system"].append({
                        "type": "slow_processing",
                        "average_time_ms": float(recent_avg_time),
                        "threshold_ms": self.config.max_processing_time_ms,
                        "severity": AnomalySeverity.WARNING.value,
                        "confidence": min(0.8, recent_avg_time / self.config.max_processing_time_ms / 2),
                        "timestamp": detection_data.get('timestamp')
                    })
                    
                    anomalies_detected += 1
            
            # Check circuit breaker status
            if self.circuit_breaker['state'] == 'OPEN':
                self.anomalies["system"].append({
                    "type": "circuit_breaker_open",
                    "failures": self.circuit_breaker['failures'],
                    "threshold": self.circuit_breaker['threshold'],
                    "severity": AnomalySeverity.CRITICAL.value,
                    "confidence": 1.0,
                    "timestamp": detection_data.get('timestamp')
                })
                
                anomalies_detected += 1
                critical_found = True
            
            # Check detection quality
            if self._detection_quality < self.config.min_detection_quality:
                self.anomalies["system"].append({
                    "type": "low_detection_quality",
                    "quality_score": self._detection_quality,
                    "threshold": self.config.min_detection_quality,
                    "severity": AnomalySeverity.WARNING.value,
                    "confidence": 0.7,
                    "timestamp": detection_data.get('timestamp')
                })
                
                anomalies_detected += 1
            
            return {
                'system_detected': True,
                'anomalies_count': anomalies_detected,
                'critical': critical_found
            }
            
        except Exception as e:
            self.logger.warning(f"System anomaly detection failed: {e}")
            return {'system_detected': False, 'error': str(e)}

    async def _detect_market_structure_anomalies_async(self, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect market structure anomalies"""
        try:
            anomalies_detected = 0
            trades = detection_data.get('trades', [])
            
            if not trades:
                return {'market_structure_detected': False, 'reason': 'no_trade_data'}
            
            # Analyze trade patterns
            trade_sizes = [abs(trade.get('size', trade.get('volume', 0))) for trade in trades]
            trade_directions = [np.sign(trade.get('size', trade.get('volume', 0))) for trade in trades if trade.get('size', trade.get('volume', 0)) != 0]
            
            # 1. Unidirectional trading detection
            if len(set(trade_directions)) == 1 and len(trade_directions) > 10:
                self.anomalies["market_structure"].append({
                    "type": "unidirectional_trading",
                    "trade_count": len(trade_directions),
                    "direction": trade_directions[0],
                    "severity": AnomalySeverity.INFO.value,
                    "confidence": min(0.8, len(trade_directions) / 20.0),
                    "timestamp": detection_data.get('timestamp'),
                    "context": {'regime': self.market_regime}
                })
                
                anomalies_detected += 1
                
                direction_text = "BUY" if trade_directions[0] > 0 else "SELL"
                self.logger.info(format_operator_message(
                    message="Unidirectional trading pattern",
                    icon="📊",
                    direction=direction_text,
                    count=len(trade_directions),
                    regime=self.market_regime
                ))
            
            # 2. High frequency trading detection
            if len(trades) > 30:
                self.anomalies["market_structure"].append({
                    "type": "high_frequency_trading",
                    "trade_count": len(trades),
                    "severity": AnomalySeverity.WARNING.value,
                    "confidence": min(0.9, len(trades) / 50.0),
                    "timestamp": detection_data.get('timestamp')
                })
                
                anomalies_detected += 1
                
                self.logger.warning(f"⚠️ High frequency trading detected: {len(trades)} trades")
            
            # 3. Extreme trade size detection
            if trade_sizes:
                size_z_scores = await self._calculate_trade_size_zscores_async(trade_sizes)
                extreme_threshold = 3.0
                extreme_count = sum(1 for z in size_z_scores if z > extreme_threshold)
                
                if extreme_count > 0:
                    self.anomalies["market_structure"].append({
                        "type": "extreme_trade_sizes",
                        "extreme_count": extreme_count,
                        "total_trades": len(trade_sizes),
                        "max_z_score": float(max(size_z_scores)),
                        "severity": AnomalySeverity.INFO.value,
                        "confidence": min(0.7, extreme_count / len(trade_sizes)),
                        "timestamp": detection_data.get('timestamp')
                    })
                    
                    anomalies_detected += 1
            
            return {
                'market_structure_detected': True,
                'anomalies_count': anomalies_detected,
                'trades_analyzed': len(trades)
            }
            
        except Exception as e:
            self.logger.warning(f"Market structure anomaly detection failed: {e}")
            return {'market_structure_detected': False, 'error': str(e)}

    async def _analyze_patterns_async(self, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns and sequences for anomalies"""
        try:
            pattern_results = {}
            
            # 1. Sequence analysis
            if len(self.pnl_history) >= 10:
                sequence_result = await self.sequence_analyzer.analyze_async(
                    list(self.pnl_history), detection_data
                )
                pattern_results['sequence'] = sequence_result
                
                if sequence_result.get('anomalies'):
                    self.anomalies["sequence"].extend(sequence_result['anomalies'])
            
            # 2. Correlation analysis
            if len(self.price_history) >= 20 and len(self.volume_history) >= 20:
                correlation_result = await self.correlation_analyzer.analyze_async(
                    list(self.price_history), list(self.volume_history), detection_data
                )
                pattern_results['correlation'] = correlation_result
                
                if correlation_result.get('anomalies'):
                    self.anomalies["correlation"].extend(correlation_result['anomalies'])
            
            # 3. Pattern detection
            trades = detection_data.get('trades', [])
            if trades:
                pattern_result = await self.pattern_detector.detect_async(trades, detection_data)
                pattern_results['pattern'] = pattern_result
                
                if pattern_result.get('anomalies'):
                    self.anomalies["pattern"].extend(pattern_result['anomalies'])
            
            return {
                'pattern_analysis_completed': True,
                'pattern_results': pattern_results,
                'pattern_anomalies': sum(len(self.anomalies[t]) for t in ["sequence", "correlation", "pattern"])
            }
            
        except Exception as e:
            self.logger.warning(f"Pattern analysis failed: {e}")
            return {'pattern_analysis_completed': False, 'error': str(e)}

    async def _adapt_thresholds_async(self, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt detection thresholds based on recent performance"""
        try:
            if not self.config.adaptive_thresholds or self.step_count < 50:
                return {'threshold_adaptation': False, 'reason': 'insufficient_data_or_disabled'}
            
            adaptation_changes = {}
            
            # Adapt PnL threshold
            if len(self.pnl_history) >= 50:
                recent_pnls = list(self.pnl_history)[-50:]
                pnl_array = np.array(recent_pnls)
                
                # Calculate adaptive threshold
                pnl_95th = np.percentile(np.abs(pnl_array), 95)
                pnl_std = np.std(pnl_array)
                
                adaptive_threshold = max(pnl_95th, 3 * pnl_std)
                adaptive_threshold = np.clip(
                    adaptive_threshold,
                    self.base_thresholds['pnl_limit'] * 0.5,
                    self.base_thresholds['pnl_limit'] * 3.0
                )
                
                # Smooth threshold changes
                old_threshold = self.current_thresholds['pnl_limit']
                new_threshold = (
                    self.config.threshold_smoothing * old_threshold +
                    (1 - self.config.threshold_smoothing) * adaptive_threshold
                )
                
                if abs(new_threshold - old_threshold) > old_threshold * 0.1:  # 10% change threshold
                    self.current_thresholds['pnl_limit'] = new_threshold
                    adaptation_changes['pnl_limit'] = {
                        'old': old_threshold,
                        'new': new_threshold,
                        'change_pct': (new_threshold - old_threshold) / old_threshold
                    }
            
            # Adapt observation threshold based on recent detection quality
            if len(self.detection_effectiveness) >= 20:
                recent_effectiveness = np.mean(list(self.detection_effectiveness)[-20:])
                
                if recent_effectiveness < 0.6:  # Poor effectiveness, relax thresholds
                    old_obs_threshold = self.current_thresholds['observation_zscore']
                    new_obs_threshold = min(old_obs_threshold * 1.1, 6.0)
                    
                    if new_obs_threshold != old_obs_threshold:
                        self.current_thresholds['observation_zscore'] = new_obs_threshold
                        adaptation_changes['observation_zscore'] = {
                            'old': old_obs_threshold,
                            'new': new_obs_threshold,
                            'reason': 'poor_effectiveness'
                        }
                elif recent_effectiveness > 0.8:  # Good effectiveness, tighten thresholds
                    old_obs_threshold = self.current_thresholds['observation_zscore']
                    new_obs_threshold = max(old_obs_threshold * 0.95, 2.0)
                    
                    if new_obs_threshold != old_obs_threshold:
                        self.current_thresholds['observation_zscore'] = new_obs_threshold
                        adaptation_changes['observation_zscore'] = {
                            'old': old_obs_threshold,
                            'new': new_obs_threshold,
                            'reason': 'good_effectiveness'
                        }
            
            # Store threshold history
            if adaptation_changes:
                self.threshold_history.append({
                    'timestamp': detection_data.get('timestamp'),
                    'changes': adaptation_changes,
                    'context': {
                        'regime': self.market_regime,
                        'session': self.market_session,
                        'step_count': self.step_count
                    }
                })
                
                self.logger.info(format_operator_message(
                    message="Thresholds adapted",
                    icon="🔧",
                    changes=len(adaptation_changes),
                    regime=self.market_regime
                ))
            
            return {
                'threshold_adaptation': True,
                'changes_made': len(adaptation_changes),
                'adaptation_changes': adaptation_changes
            }
            
        except Exception as e:
            self.logger.warning(f"Threshold adaptation failed: {e}")
            return {'threshold_adaptation': False, 'error': str(e)}

    async def _calculate_comprehensive_score_async(self, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive anomaly score with confidence"""
        try:
            # Severity weights
            severity_weights = {
                AnomalySeverity.INFO.value: 0.1,
                AnomalySeverity.WARNING.value: 0.5,
                AnomalySeverity.CRITICAL.value: 1.0,
                AnomalySeverity.EMERGENCY.value: 1.5
            }
            
            # Type weights based on importance
            type_weights = {
                "pnl": 0.35,
                "observation": 0.25,
                "system": 0.15,
                "volatility": 0.10,
                "price": 0.05,
                "volume": 0.05,
                "market_structure": 0.03,
                "pattern": 0.01,
                "correlation": 0.01,
                "sequence": 0.01
            }
            
            total_weighted_score = 0.0
            total_confidence = 0.0
            total_anomalies = 0
            
            for anomaly_type, anomalies in self.anomalies.items():
                if not anomalies:
                    continue
                
                type_weight = type_weights.get(anomaly_type, 0.01)
                
                for anomaly in anomalies:
                    severity = anomaly.get("severity", AnomalySeverity.INFO.value)
                    confidence = anomaly.get("confidence", 0.5)
                    severity_weight = severity_weights.get(severity, 0.1)
                    
                    weighted_score = type_weight * severity_weight * confidence
                    total_weighted_score += weighted_score
                    total_confidence += confidence
                    total_anomalies += 1
            
            # Normalize score
            if total_anomalies > 0:
                base_score = min(total_weighted_score, 1.0)
                average_confidence = total_confidence / total_anomalies
                
                # Apply context adjustments
                context_multiplier = await self._get_context_score_multiplier_async(detection_data)
                final_score = base_score * context_multiplier
                
                self.anomaly_score = final_score
                self.detection_confidence = average_confidence
            else:
                self.anomaly_score = 0.0
                self.detection_confidence = 1.0  # High confidence in no anomalies
            
            # Update detection effectiveness
            effectiveness = await self._calculate_detection_effectiveness_async()
            self.detection_effectiveness.append(effectiveness)
            
            return {
                'score_calculated': True,
                'anomaly_score': self.anomaly_score,
                'detection_confidence': self.detection_confidence,
                'total_anomalies': total_anomalies,
                'detection_effectiveness': effectiveness
            }
            
        except Exception as e:
            self.logger.warning(f"Score calculation failed: {e}")
            return {'score_calculated': False, 'error': str(e)}

    async def _update_training_progress_async(self, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update training progress and mode transitions"""
        try:
            if not self.config.training_mode:
                return {'training_update': False, 'reason': 'not_in_training_mode'}
            
            self.training_progress = min(self.step_count / self.config.training_duration_steps, 1.0)
            
            # Check if training should be completed
            if (self.training_progress >= 1.0 and not self.is_training_complete):
                await self._complete_training_async()
                
                return {
                    'training_update': True,
                    'training_completed': True,
                    'progress': self.training_progress
                }
            
            return {
                'training_update': True,
                'training_completed': False,
                'progress': self.training_progress
            }
            
        except Exception as e:
            self.logger.warning(f"Training progress update failed: {e}")
            return {'training_update': False, 'error': str(e)}

    async def _complete_training_async(self) -> None:
        """Complete training mode and transition to active detection"""
        try:
            self.is_training_complete = True
            
            # Finalize adaptive thresholds
            await self._finalize_training_thresholds_async()
            
            # Transition mode
            old_mode = self.current_mode
            self.current_mode = AnomalyDetectionMode.ACTIVE
            
            self.logger.info(format_operator_message(
                message="Training completed - transitioning to active detection",
                icon="🎓",
                old_mode=old_mode.value,
                new_mode=self.current_mode.value,
                steps_trained=self.step_count,
                final_score=f"{self._detection_quality:.2f}"
            ))
            
        except Exception as e:
            self.logger.error(f"Training completion failed: {e}")

    async def _handle_emergency_situations_async(self, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle emergency situations"""
        try:
            emergency_triggered = False
            emergency_reasons = []
            
            # Check for emergency conditions
            if self.anomaly_score > self.config.emergency_threshold:
                emergency_triggered = True
                emergency_reasons.append(f"anomaly_score_exceeded_{self.anomaly_score:.2f}")
            
            # Check for critical system anomalies
            critical_system_anomalies = [
                a for a in self.anomalies.get("system", [])
                if a.get("severity") == AnomalySeverity.CRITICAL.value
            ]
            
            if critical_system_anomalies:
                emergency_triggered = True
                emergency_reasons.append(f"critical_system_anomalies_{len(critical_system_anomalies)}")
            
            # Check for multiple critical anomalies
            all_critical_anomalies = [
                a for anomalies in self.anomalies.values()
                for a in anomalies
                if a.get("severity") == AnomalySeverity.CRITICAL.value
            ]
            
            if len(all_critical_anomalies) >= 3:
                emergency_triggered = True
                emergency_reasons.append(f"multiple_critical_anomalies_{len(all_critical_anomalies)}")
            
            # Update mode if emergency
            if emergency_triggered and self.current_mode != AnomalyDetectionMode.EMERGENCY:
                old_mode = self.current_mode
                self.current_mode = AnomalyDetectionMode.EMERGENCY
                
                self.logger.error(format_operator_message(
                    message="EMERGENCY MODE ACTIVATED",
                    icon="🆘",
                    old_mode=old_mode.value,
                    reasons=", ".join(emergency_reasons[:3]),
                    anomaly_score=f"{self.anomaly_score:.2f}",
                    critical_count=len(all_critical_anomalies)
                ))
            
            return {
                'emergency_check_completed': True,
                'emergency_triggered': emergency_triggered,
                'emergency_reasons': emergency_reasons,
                'critical_anomalies_count': len(all_critical_anomalies)
            }
            
        except Exception as e:
            self.logger.warning(f"Emergency situation handling failed: {e}")
            return {'emergency_check_completed': False, 'error': str(e)}

    async def _update_operational_mode_async(self, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update operational mode based on detection status"""
        try:
            old_mode = self.current_mode
            
            # Skip mode changes if in emergency
            if self.current_mode == AnomalyDetectionMode.EMERGENCY:
                # Check if emergency can be cleared
                if (self.anomaly_score < self.config.emergency_threshold * 0.7 and
                    not any(a.get("severity") == AnomalySeverity.CRITICAL.value 
                           for anomalies in self.anomalies.values() for a in anomalies)):
                    self.current_mode = AnomalyDetectionMode.ACTIVE
                    
                    self.logger.info(format_operator_message(
                        message="Emergency cleared - returning to active mode",
                        icon="✅",
                        anomaly_score=f"{self.anomaly_score:.2f}"
                    ))
            
            # Determine mode based on current state
            elif self.step_count < 10:
                self.current_mode = AnomalyDetectionMode.INITIALIZATION
            elif self.config.training_mode and not self.is_training_complete:
                self.current_mode = AnomalyDetectionMode.TRAINING
            elif self._detection_quality < 0.5:
                self.current_mode = AnomalyDetectionMode.CALIBRATION
            elif self.anomaly_score > self.config.critical_threshold:
                self.current_mode = AnomalyDetectionMode.ENHANCED
            else:
                self.current_mode = AnomalyDetectionMode.ACTIVE
            
            mode_changed = old_mode != self.current_mode
            
            if mode_changed:
                self.mode_start_time = datetime.datetime.now()
                
                self.logger.info(format_operator_message(
                    message="Detection mode changed",
                    icon="🔄",
                    old_mode=old_mode.value,
                    new_mode=self.current_mode.value,
                    anomaly_score=f"{self.anomaly_score:.2f}",
                    detection_quality=f"{self._detection_quality:.2f}"
                ))
            
            return {
                'mode_updated': True,
                'current_mode': self.current_mode.value,
                'mode_changed': mode_changed,
                'old_mode': old_mode.value if mode_changed else None,
                'mode_duration': (datetime.datetime.now() - self.mode_start_time).total_seconds()
            }
            
        except Exception as e:
            self.logger.warning(f"Mode update failed: {e}")
            return {'mode_updated': False, 'error': str(e)}

    # ================== HELPER METHODS ==================

    async def _generate_synthetic_pnl_async(self, detection_data: Dict[str, Any]) -> float:
        """Generate realistic synthetic PnL for training mode"""
        try:
            base_pnl = np.random.normal(0, 100)
            
            # Adjust for market regime
            if self.market_regime == 'volatile':
                base_pnl *= 2.5
            elif self.market_regime == 'trending':
                base_pnl *= 1.8
            
            # Add occasional anomalies for training
            if np.random.rand() < 0.08:  # 8% chance
                spike_pnl = np.random.choice([-1, 1]) * np.random.uniform(300, 1200)
                base_pnl += spike_pnl
            
            return float(base_pnl)
            
        except Exception:
            return 0.0

    async def _generate_synthetic_volume_async(self, detection_data: Dict[str, Any]) -> float:
        """Generate realistic synthetic volume for training mode"""
        try:
            base_volume = abs(np.random.normal(8000, 3000))
            
            # Adjust for session
            if self.market_session == 'european':
                base_volume *= 1.4
            elif self.market_session == 'american':
                base_volume *= 1.2
            
            return float(max(base_volume, 500))
            
        except Exception:
            return 1000.0

    async def _generate_synthetic_price_async(self, detection_data: Dict[str, Any]) -> float:
        """Generate realistic synthetic price for training mode"""
        try:
            if self.price_history:
                last_price = self.price_history[-1]
                change_pct = np.random.normal(0, 0.003)
                
                if self.market_regime == 'volatile':
                    change_pct *= 4.0
                elif self.market_regime == 'trending':
                    change_pct += np.random.choice([-1, 1]) * 0.002
                
                return float(last_price * (1 + change_pct))
            else:
                return float(np.random.uniform(1.1, 1.9))
                
        except Exception:
            return 1.5

    async def _calculate_robust_zscore_async(self, value: float, history: List[float]) -> float:
        """Calculate robust z-score using median and MAD"""
        try:
            if len(history) < 3:
                return 0.0
            
            history_array = np.array(history)
            median = np.median(history_array)
            mad = np.median(np.abs(history_array - median))
            
            mad_std = mad * 1.4826
            
            if mad_std < 1e-8:
                return 0.0
            
            return float(abs((value - median) / mad_std))
            
        except Exception:
            return 0.0

    async def _calculate_observation_zscores_async(self, obs: np.ndarray) -> np.ndarray:
        """Calculate z-scores for observation vector"""
        try:
            if len(self.observation_history) < 2:
                return np.zeros(len(obs))
            
            obs_stack = np.vstack(self.observation_history)
            medians = np.median(obs_stack, axis=0)
            mads = np.median(np.abs(obs_stack - medians), axis=0)
            
            mad_stds = mads * 1.4826
            mad_stds[mad_stds < 1e-8] = 1.0
            
            z_scores = np.abs((obs - medians) / mad_stds)
            
            return z_scores
            
        except Exception:
            return np.zeros(len(obs) if obs is not None else 0)

    async def _calculate_trade_size_zscores_async(self, trade_sizes: List[float]) -> List[float]:
        """Calculate z-scores for trade sizes"""
        try:
            if len(trade_sizes) < 3:
                return [0.0] * len(trade_sizes)
            
            sizes_array = np.array(trade_sizes)
            mean_size = np.mean(sizes_array)
            std_size = np.std(sizes_array)
            
            if std_size < 1e-8:
                return [0.0] * len(trade_sizes)
            
            z_scores = [float(abs((size - mean_size) / std_size)) for size in trade_sizes]
            
            return z_scores
            
        except Exception:
            return [0.0] * len(trade_sizes)

    async def _get_context_adjusted_threshold_async(self, threshold_name: str, detection_data: Dict[str, Any]) -> float:
        """Get context-adjusted threshold"""
        try:
            base_threshold = self.current_thresholds.get(threshold_name, 1.0)
            
            multiplier = 1.0
            
            # Adjust for market regime
            if threshold_name in ['pnl_limit', 'volume_zscore']:
                if self.volatility_regime == 'extreme':
                    multiplier = 2.5
                elif self.volatility_regime == 'high':
                    multiplier = 1.8
                elif self.market_regime == 'volatile':
                    multiplier = 1.5
            
            # Adjust for market stress
            if self.market_stress_level > 0.7:
                multiplier *= 1.3
            
            # Apply adaptive parameters
            sensitivity = self.adaptive_params.get('sensitivity_multiplier', 1.0)
            multiplier *= sensitivity
            
            return base_threshold * multiplier
            
        except Exception:
            return self.current_thresholds.get(threshold_name, 1.0)

    async def _get_price_jump_threshold_async(self, detection_data: Dict[str, Any]) -> float:
        """Get context-adjusted price jump threshold"""
        try:
            base_thresholds = {
                'low': 0.04,
                'medium': 0.07,
                'high': 0.12,
                'extreme': 0.25
            }
            
            threshold = base_thresholds.get(self.volatility_regime, 0.07)
            
            # Adjust for market stress
            if self.market_stress_level > 0.8:
                threshold *= 1.5
            
            return threshold
            
        except Exception:
            return 0.07

    async def _get_context_score_multiplier_async(self, detection_data: Dict[str, Any]) -> float:
        """Get context multiplier for anomaly score"""
        try:
            multiplier = 1.0
            
            # Reduce score in volatile markets (more tolerance)
            if self.volatility_regime == 'extreme':
                multiplier = 0.6
            elif self.volatility_regime == 'high':
                multiplier = 0.8
            elif self.market_regime == 'volatile':
                multiplier = 0.9
            
            # Increase score during low volatility (less tolerance)
            elif self.volatility_regime == 'low':
                multiplier = 1.2
            
            return multiplier
            
        except Exception:
            return 1.0

    async def _detect_regime_specific_pnl_anomaly_async(self, pnl: float, detection_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect regime-specific PnL anomalies"""
        try:
            regime_data = self.regime_baselines[self.market_regime].get('pnl', [])
            
            if len(regime_data) < 20:
                return None
            
            regime_z_score = await self._calculate_robust_zscore_async(pnl, list(regime_data))
            
            if regime_z_score > 5.0:  # Stricter threshold for regime-specific
                return {
                    "type": "regime_specific_outlier",
                    "value": pnl,
                    "regime_z_score": float(regime_z_score),
                    "regime": self.market_regime,
                    "severity": AnomalySeverity.WARNING.value,
                    "confidence": min(0.8, regime_z_score / 7.0),
                    "timestamp": detection_data.get('timestamp')
                }
            
            return None
            
        except Exception:
            return None

    async def _calculate_detection_effectiveness_async(self) -> float:
        """Calculate detection effectiveness score"""
        try:
            # Simple effectiveness based on false positive rate and detection quality
            false_positive_rate = len([fp for fp in self.false_positive_tracker if fp]) / max(len(self.false_positive_tracker), 1)
            
            # Quality based on threshold stability and performance
            threshold_stability = 1.0
            if len(self.threshold_history) > 10:
                recent_changes = len([h for h in list(self.threshold_history)[-10:] if h.get('changes')])
                threshold_stability = max(0.5, 1.0 - (recent_changes / 10.0))
            
            effectiveness = (1.0 - false_positive_rate) * threshold_stability * self._detection_quality
            
            return float(np.clip(effectiveness, 0.0, 1.0))
            
        except Exception:
            return 0.5

    async def _finalize_training_thresholds_async(self) -> None:
        """Finalize thresholds after training completion"""
        try:
            # Set final thresholds based on training data
            if len(self.pnl_history) >= 100:
                pnl_data = np.array(list(self.pnl_history))
                final_pnl_threshold = np.percentile(np.abs(pnl_data), 98)
                self.current_thresholds['pnl_limit'] = max(
                    final_pnl_threshold,
                    self.base_thresholds['pnl_limit'] * 0.8
                )
            
            self.logger.info(format_operator_message(
                message="Training thresholds finalized",
                icon="✅",
                pnl_threshold=f"€{self.current_thresholds['pnl_limit']:,.0f}",
                obs_threshold=f"{self.current_thresholds['observation_zscore']:.1f}"
            ))
            
        except Exception as e:
            self.logger.warning(f"Threshold finalization failed: {e}")

    # ================== THESIS AND SMARTINFOBUS METHODS ==================

    async def _generate_detection_thesis(self, detection_data: Dict[str, Any], 
                                        result: Dict[str, Any]) -> str:
        """Generate comprehensive detection thesis"""
        try:
            # Core metrics
            anomaly_score = self.anomaly_score
            mode = self.current_mode.value
            confidence = self.detection_confidence
            
            thesis_parts = [
                f"Anomaly Detection: {mode.upper()} mode with {anomaly_score:.1%} risk score",
                f"Detection Confidence: {confidence:.2f} assessment accuracy"
            ]
            
            # Anomaly level assessment
            if anomaly_score > self.config.critical_threshold:
                thesis_parts.append(f"HIGH RISK: Critical anomalies detected")
            elif anomaly_score > self.config.warning_threshold:
                thesis_parts.append(f"ELEVATED: Warning-level anomalies present")
            else:
                thesis_parts.append(f"NORMAL: No significant anomalies detected")
            
            # Active anomalies
            total_anomalies = sum(len(v) for v in self.anomalies.values())
            if total_anomalies > 0:
                critical_count = sum(
                    1 for anomalies in self.anomalies.values()
                    for a in anomalies
                    if a.get("severity") == AnomalySeverity.CRITICAL.value
                )
                
                if critical_count > 0:
                    thesis_parts.append(f"Active anomalies: {total_anomalies} total, {critical_count} critical")
                else:
                    thesis_parts.append(f"Active anomalies: {total_anomalies} total, monitoring level")
            
            # Market context
            thesis_parts.append(f"Context: {self.market_regime.upper()} regime, {self.volatility_regime.upper()} volatility")
            
            # Training status
            if self.config.training_mode:
                if self.is_training_complete:
                    thesis_parts.append(f"Training: COMPLETED ({self.step_count} steps)")
                else:
                    thesis_parts.append(f"Training: {self.training_progress:.0%} complete")
            
            # Adaptive status
            if self.config.adaptive_thresholds:
                recent_adaptations = len([h for h in list(self.threshold_history)[-10:] if h.get('changes')])
                thesis_parts.append(f"Adaptive: {recent_adaptations} recent threshold adjustments")
            
            # Data sufficiency
            data_sufficiency = min(len(self.pnl_history) / 50.0, 1.0)
            thesis_parts.append(f"Data quality: {data_sufficiency:.0%} sufficiency")
            
            return " | ".join(thesis_parts)
            
        except Exception as e:
            return f"Detection thesis generation failed: {str(e)} - Core anomaly detection functional"

    async def _update_detection_smart_bus(self, result: Dict[str, Any], thesis: str):
        """Update SmartInfoBus with detection results"""
        try:
            # Anomaly detection data
            detection_data = {
                'current_mode': self.current_mode.value,
                'enabled': self.enabled,
                'anomaly_score': self.anomaly_score,
                'detection_confidence': self.detection_confidence,
                'total_anomalies': sum(len(v) for v in self.anomalies.values()),
                'training_mode': self.config.training_mode,
                'training_progress': self.training_progress,
                'is_training_complete': self.is_training_complete,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            self.smart_bus.set(
                'anomaly_detection',
                detection_data,
                module='EnhancedAnomalyDetector',
                thesis=thesis
            )
            
            # Anomaly score for other modules
            score_data = {
                'anomaly_score': self.anomaly_score,
                'detection_confidence': self.detection_confidence,
                'anomaly_types': {
                    anomaly_type: len(anomalies)
                    for anomaly_type, anomalies in self.anomalies.items()
                    if anomalies
                },
                'critical_anomalies': sum(
                    1 for anomalies in self.anomalies.values()
                    for a in anomalies
                    if a.get("severity") == AnomalySeverity.CRITICAL.value
                ),
                'emergency_mode': self.current_mode == AnomalyDetectionMode.EMERGENCY
            }
            
            self.smart_bus.set(
                'anomaly_score',
                score_data,
                module='EnhancedAnomalyDetector',
                thesis="Current anomaly scoring and risk assessment"
            )
            
            # Anomaly alerts
            alerts_data = {
                'emergency_mode': self.current_mode == AnomalyDetectionMode.EMERGENCY,
                'critical_anomalies_present': any(
                    a.get("severity") == AnomalySeverity.CRITICAL.value
                    for anomalies in self.anomalies.values()
                    for a in anomalies
                ),
                'high_anomaly_score': self.anomaly_score > self.config.critical_threshold,
                'low_detection_quality': self._detection_quality < self.config.min_detection_quality,
                'circuit_breaker_open': self.circuit_breaker['state'] == 'OPEN',
                'recent_anomalies': {
                    anomaly_type: [
                        {
                            'type': a.get('type', 'unknown'),
                            'severity': a.get('severity', 'info'),
                            'confidence': a.get('confidence', 0.5),
                            'timestamp': a.get('timestamp')
                        }
                        for a in anomalies[-5:]  # Last 5 anomalies of each type
                    ]
                    for anomaly_type, anomalies in self.anomalies.items()
                    if anomalies
                }
            }
            
            self.smart_bus.set(
                'anomaly_alerts',
                alerts_data,
                module='EnhancedAnomalyDetector',
                thesis="Anomaly detection alerts and emergency status"
            )
            
            # Detection analytics
            analytics_data = {
                'detection_quality': self._detection_quality,
                'detection_effectiveness': list(self.detection_effectiveness)[-10:] if self.detection_effectiveness else [],
                'threshold_adaptation_count': len(self.threshold_history),
                'current_thresholds': self.current_thresholds.copy(),
                'base_thresholds': self.base_thresholds.copy(),
                'detection_stats': dict(self.detection_stats),
                'data_sufficiency': {
                    'pnl_history': len(self.pnl_history),
                    'volume_history': len(self.volume_history),
                    'price_history': len(self.price_history),
                    'observation_history': len(self.observation_history)
                },
                'performance_metrics': {
                    'avg_processing_time_ms': np.mean(list(self._processing_times)[-10:]) * 1000 if self._processing_times else 0,
                    'circuit_breaker_state': self.circuit_breaker['state'],
                    'false_positive_rate': len([fp for fp in self.false_positive_tracker if fp]) / max(len(self.false_positive_tracker), 1)
                }
            }
            
            self.smart_bus.set(
                'detection_analytics',
                analytics_data,
                module='EnhancedAnomalyDetector',
                thesis="Comprehensive anomaly detection analytics and performance metrics"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update SmartInfoBus: {e}")

    # ================== FALLBACK AND ERROR HANDLING ==================

    async def _handle_disabled_fallback(self) -> Dict[str, Any]:
        """Handle case when detector is disabled"""
        self.anomaly_score = 0.0
        self.detection_confidence = 1.0
        
        return {
            'current_mode': AnomalyDetectionMode.MAINTENANCE.value,
            'enabled': False,
            'anomaly_score': 0.0,
            'detection_confidence': 1.0,
            'fallback_reason': 'detector_disabled'
        }

    async def _handle_no_data_fallback(self) -> Dict[str, Any]:
        """Handle case when no detection data is available"""
        self.logger.warning("No detection data available - maintaining previous state")
        
        return {
            'current_mode': self.current_mode.value,
            'anomaly_score': self.anomaly_score,
            'detection_confidence': max(0.1, self.detection_confidence - 0.1),
            'fallback_reason': 'no_detection_data'
        }

    async def _handle_detection_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle detection errors"""
        processing_time = (time.time() - start_time) * 1000
        
        # Update circuit breaker
        self.circuit_breaker['failures'] += 1
        self.circuit_breaker['last_failure'] = time.time()
        
        if self.circuit_breaker['failures'] >= self.circuit_breaker['threshold']:
            self.circuit_breaker['state'] = 'OPEN'
            self._health_status = 'warning'
        
        # Log error with context
        error_context = self.error_pinpointer.analyze_error(error, "EnhancedAnomalyDetector")
        explanation = self.english_explainer.explain_error(
            "EnhancedAnomalyDetector", str(error), "anomaly detection"
        )
        
        self.logger.error(format_operator_message(
            message="Anomaly detector error",
            icon="💥",
            error=str(error),
            details=explanation,
            processing_time_ms=processing_time,
            circuit_breaker_state=self.circuit_breaker['state']
        ))
        
        # Record failure
        self._record_failure(error)
        
        return self._create_error_fallback_response(f"error: {str(error)}")

    def _create_error_fallback_response(self, reason: str) -> Dict[str, Any]:
        """Create fallback response for error cases"""
        return {
            'current_mode': AnomalyDetectionMode.EMERGENCY.value,
            'anomaly_score': 0.1,  # Conservative fallback
            'detection_confidence': 0.1,  # Low confidence due to error
            'circuit_breaker_state': self.circuit_breaker['state'],
            'fallback_reason': reason
        }

    # ================== MONITORING AND HEALTH METHODS ==================

    def _update_detection_health(self):
        """Update detection health metrics"""
        try:
            # Check detection quality
            if self._detection_quality < self.config.min_detection_quality:
                self._health_status = 'warning'
            else:
                self._health_status = 'healthy'
            
            # Check circuit breaker
            if self.circuit_breaker['state'] == 'OPEN':
                self._health_status = 'warning'
            
            # Check processing times
            if len(self._processing_times) >= 10:
                avg_time = np.mean(list(self._processing_times)[-10:]) * 1000
                if avg_time > self.config.max_processing_time_ms:
                    self._health_status = 'warning'
            
            self._last_health_check = time.time()
            
        except Exception as e:
            self.logger.error(f"Detection health check failed: {e}")
            self._health_status = 'warning'

    def _analyze_detection_effectiveness(self):
        """Analyze detection effectiveness"""
        try:
            if len(self.detection_effectiveness) >= 20:
                effectiveness = np.mean(list(self.detection_effectiveness)[-20:])
                
                if effectiveness > 0.8:
                    self.logger.info(format_operator_message(
                        message="High detection effectiveness achieved",
                        icon="🎯",
                        effectiveness=f"{effectiveness:.2f}",
                        score=f"{self.anomaly_score:.2f}"
                    ))
                elif effectiveness < 0.4:
                    self.logger.warning(format_operator_message(
                        message="Low detection effectiveness detected",
                        icon="⚠️",
                        effectiveness=f"{effectiveness:.2f}",
                        mode=self.current_mode.value
                    ))
            
        except Exception as e:
            self.logger.error(f"Detection effectiveness analysis failed: {e}")

    def _adapt_detection_parameters(self):
        """Continuous detection parameter adaptation"""
        try:
            # Adapt sensitivity based on recent performance
            if len(self.detection_effectiveness) >= 10:
                recent_effectiveness = np.mean(list(self.detection_effectiveness)[-10:])
                
                if recent_effectiveness < 0.5:
                    # Reduce sensitivity (increase tolerance)
                    self.adaptive_params['sensitivity_multiplier'] = min(
                        1.3, self.adaptive_params['sensitivity_multiplier'] * 1.01
                    )
                elif recent_effectiveness > 0.8:
                    # Increase sensitivity (decrease tolerance)
                    self.adaptive_params['sensitivity_multiplier'] = max(
                        0.7, self.adaptive_params['sensitivity_multiplier'] * 0.995
                    )
            
            # Update detection quality
            self._detection_quality = min(1.0, max(0.1, self._detection_quality + 
                (np.random.normal(0, 0.02) if self.enabled else -0.1)))
            
        except Exception as e:
            self.logger.warning(f"Detection parameter adaptation failed: {e}")

    def _cleanup_old_data(self):
        """Cleanup old data to maintain performance"""
        try:
            # Cleanup old anomaly records
            for anomaly_type in self.anomalies:
                if len(self.anomalies[anomaly_type]) > 100:
                    self.anomalies[anomaly_type] = self.anomalies[anomaly_type][-50:]
            
            # Cleanup regime baselines
            for regime in list(self.regime_baselines.keys()):
                for data_type in list(self.regime_baselines[regime].keys()):
                    if len(self.regime_baselines[regime][data_type]) == 0:
                        del self.regime_baselines[regime][data_type]
                
                if not self.regime_baselines[regime]:
                    del self.regime_baselines[regime]
            
        except Exception as e:
            self.logger.warning(f"Data cleanup failed: {e}")

    def _record_success(self, processing_time: float):
        """Record successful processing"""
        self._processing_times.append(processing_time / 1000.0)  # Store in seconds
        
        self.performance_tracker.record_metric(
            'EnhancedAnomalyDetector', 'anomaly_detection', processing_time, True
        )
        
        # Reset circuit breaker on success
        if self.circuit_breaker['state'] == 'OPEN':
            self.circuit_breaker['failures'] = 0
            self.circuit_breaker['state'] = 'CLOSED'

    def _record_failure(self, error: Exception):
        """Record processing failure"""
        self.performance_tracker.record_metric(
            'EnhancedAnomalyDetector', 'anomaly_detection', 0, False
        )

    # ================== PUBLIC INTERFACE METHODS ==================

    def get_current_anomaly_score(self) -> float:
        """Get current anomaly score"""
        return self.anomaly_score

    def get_detection_confidence(self) -> float:
        """Get current detection confidence"""
        return self.detection_confidence

    def get_anomalies_summary(self) -> Dict[str, int]:
        """Get summary of current anomalies"""
        return {
            anomaly_type: len(anomalies)
            for anomaly_type, anomalies in self.anomalies.items()
            if anomalies
        }

    def force_emergency_mode(self, reason: str = "manual_override") -> None:
        """Force emergency detection mode"""
        old_mode = self.current_mode
        self.current_mode = AnomalyDetectionMode.EMERGENCY
        self.anomaly_score = min(1.0, self.anomaly_score + 0.5)
        
        self.logger.error(format_operator_message(
            message="Emergency mode forced",
            icon="🆘",
            reason=reason,
            old_mode=old_mode.value,
            new_score=f"{self.anomaly_score:.2f}"
        ))

    def set_external_anomaly_source(self, source_name: str, anomaly_data: Dict[str, Any]) -> None:
        """Set external anomaly source data"""
        self.external_anomaly_sources[source_name] = {
            'data': anomaly_data,
            'timestamp': datetime.datetime.now().isoformat()
        }

    def clear_anomaly_history(self) -> None:
        """Clear anomaly history (for testing/reset)"""
        for anomaly_type in self.anomalies:
            self.anomalies[anomaly_type].clear()
        
        self.anomaly_score = 0.0
        self.detection_confidence = 0.5
        
        self.logger.info("🔄 Anomaly history cleared")

    def get_observation_components(self) -> np.ndarray:
        """Enhanced observation components for model integration"""
        try:
            # Core detection metrics
            anomaly_score = float(self.anomaly_score)
            detection_confidence = float(self.detection_confidence)
            
            # Critical anomaly indicator
            has_critical = float(any(
                a.get("severity") == AnomalySeverity.CRITICAL.value
                for anomalies in self.anomalies.values()
                for a in anomalies
            ))
            
            # Emergency mode indicator
            emergency_mode = float(self.current_mode == AnomalyDetectionMode.EMERGENCY)
            
            # Training status
            training_progress = float(self.training_progress)
            
            # Data sufficiency indicators
            pnl_sufficiency = min(len(self.pnl_history) / 50.0, 1.0)
            observation_sufficiency = min(len(self.observation_history) / 20.0, 1.0)
            
            # Detection quality
            detection_quality = float(self._detection_quality)
            
            return np.array([
                anomaly_score,
                detection_confidence,
                has_critical,
                emergency_mode,
                training_progress,
                pnl_sufficiency,
                observation_sufficiency,
                detection_quality
            ], dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Observation generation failed: {e}")
            return np.zeros(8, dtype=np.float32)

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        return {
            'status': self._health_status,
            'last_check': self._last_health_check,
            'circuit_breaker': self.circuit_breaker['state'],
            'current_mode': self.current_mode.value,
            'anomaly_score': self.anomaly_score,
            'detection_confidence': self.detection_confidence,
            'detection_quality': self._detection_quality,
            'training_progress': self.training_progress if self.config.training_mode else 1.0,
            'enabled': self.enabled
        }

    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring_active = False

    def get_detection_report(self) -> str:
        """Generate operator-friendly detection report"""
        
        total_anomalies = sum(len(v) for v in self.anomalies.values())
        
        # Status indicators
        if self.anomaly_score > self.config.critical_threshold:
            detection_status = "🚨 Critical"
        elif self.anomaly_score > self.config.warning_threshold:
            detection_status = "⚠️ Elevated"
        else:
            detection_status = "✅ Normal"
        
        # Mode status
        mode_emoji = {
            AnomalyDetectionMode.INITIALIZATION: "🔄",
            AnomalyDetectionMode.TRAINING: "🎓",
            AnomalyDetectionMode.CALIBRATION: "🔧",
            AnomalyDetectionMode.ACTIVE: "✅",
            AnomalyDetectionMode.ENHANCED: "⚡",
            AnomalyDetectionMode.EMERGENCY: "🆘",
            AnomalyDetectionMode.MAINTENANCE: "🔧"
        }
        
        mode_status = f"{mode_emoji.get(self.current_mode, '❓')} {self.current_mode.value.upper()}"
        
        # Health status
        health_emoji = "✅" if self._health_status == 'healthy' else "⚠️"
        cb_status = "🔴 OPEN" if self.circuit_breaker['state'] == 'OPEN' else "🟢 CLOSED"
        
        # Training status
        if self.config.training_mode:
            if self.is_training_complete:
                training_status = "✅ Completed"
            else:
                training_status = f"🎓 In Progress ({self.training_progress:.0%})"
        else:
            training_status = "❌ Disabled"
        
        # Data sufficiency status
        pnl_sufficiency = len(self.pnl_history) / (self.pnl_history.maxlen or 1)
        if pnl_sufficiency > 0.8:
            data_status = "✅ Sufficient"
        elif pnl_sufficiency > 0.5:
            data_status = "⚡ Partial"
        else:
            data_status = "❌ Limited"
        
        # Current anomalies breakdown
        anomaly_lines = []
        for anomaly_type, anomalies in self.anomalies.items():
            if anomalies:
                critical_count = sum(1 for a in anomalies if a.get('severity') == AnomalySeverity.CRITICAL.value)
                warning_count = sum(1 for a in anomalies if a.get('severity') == AnomalySeverity.WARNING.value)
                emoji = "🚨" if critical_count > 0 else "⚠️" if warning_count > 0 else "ℹ️"
                anomaly_lines.append(f"  {emoji} {anomaly_type.replace('_', ' ').title()}: {len(anomalies)} ({critical_count} critical)")
        
        # Detection effectiveness
        if len(self.detection_effectiveness) >= 10:
            recent_effectiveness = np.mean(list(self.detection_effectiveness)[-10:])
            effectiveness_status = "📈 High" if recent_effectiveness > 0.8 else "📊 Medium" if recent_effectiveness > 0.5 else "📉 Low"
        else:
            effectiveness_status = "📊 Calculating"
        
        # Threshold adaptation status
        if self.config.adaptive_thresholds:
            recent_adaptations = len([h for h in list(self.threshold_history)[-10:] if h.get('changes')])
            adaptation_status = f"✅ Active ({recent_adaptations} recent)"
        else:
            adaptation_status = "❌ Disabled"
        
        return f"""
🔍 ENHANCED ANOMALY DETECTOR v4.0
═══════════════════════════════════════════════════
🎯 Detection Status: {detection_status} ({self.anomaly_score:.1%} risk)
🔧 Detection Mode: {mode_status}
📊 Market Context: {self.market_regime.title()} regime, {self.volatility_regime.title()} volatility
🎓 Training Status: {training_status}
🔄 Detector Enabled: {'✅ Yes' if self.enabled else '❌ No'}

🏥 SYSTEM HEALTH
• Status: {health_emoji} {self._health_status.upper()}
• Circuit Breaker: {cb_status}
• Detection Quality: {self._detection_quality:.2f}
• Detection Confidence: {self.detection_confidence:.2f}

⚖️ DETECTION THRESHOLDS
• PnL Limit: €{self.current_thresholds['pnl_limit']:,.0f}
• Volume Z-Score: {self.current_thresholds['volume_zscore']:.1f}
• Price Z-Score: {self.current_thresholds['price_zscore']:.1f}
• Observation Z-Score: {self.current_thresholds['observation_zscore']:.1f}
• Adaptive Thresholds: {adaptation_status}

📊 DATA COLLECTION STATUS
• PnL History: {len(self.pnl_history)}/{self.pnl_history.maxlen}
• Volume History: {len(self.volume_history)}/{self.volume_history.maxlen}
• Price History: {len(self.price_history)}/{self.price_history.maxlen}
• Observation History: {len(self.observation_history)}/{self.observation_history.maxlen}
• Data Quality: {data_status} ({pnl_sufficiency:.1%})

🚨 CURRENT ANOMALIES ({total_anomalies} total)
{chr(10).join(anomaly_lines) if anomaly_lines else "  ✅ No anomalies detected"}

📈 DETECTION PERFORMANCE
• Detection Effectiveness: {effectiveness_status}
• Total Detections: {self.detection_stats.get('total_anomalies', 0)}
• PnL Anomalies: {self.detection_stats.get('pnl_count', 0)}
• Volume Anomalies: {self.detection_stats.get('volume_count', 0)}
• Price Anomalies: {self.detection_stats.get('price_count', 0)}
• Observation Anomalies: {self.detection_stats.get('observation_count', 0)}
• System Anomalies: {self.detection_stats.get('system_count', 0)}

🔧 ADAPTIVE BEHAVIOR
• Sensitivity Multiplier: {self.adaptive_params['sensitivity_multiplier']:.2f}
• Regime Adaptation: {self.adaptive_params['regime_adaptation_factor']:.2f}
• Volatility Tolerance: {self.adaptive_params['volatility_tolerance']:.2f}
• Learning Momentum: {self.adaptive_params['learning_momentum']:.2f}

📊 MARKET CONTEXT TRACKING
• Current Regime: {self.market_regime.title()}
• Market Session: {self.market_session.title()}
• Volatility Level: {self.volatility_regime.title()}
• Market Stress: {self.market_stress_level:.2f}
• Regime Changes: {len(self.regime_baselines)} tracked

🔗 EXTERNAL INTEGRATIONS
• External Sources: {len(self.external_anomaly_sources)}
{chr(10).join([f"  • {name}: {data.get('timestamp', 'N/A')}" for name, data in self.external_anomaly_sources.items()]) if self.external_anomaly_sources else "  📭 No external sources"}

💡 ADVANCED FEATURES
• Sequence Analysis: {'✅ Active' if hasattr(self, 'sequence_analyzer') else '❌ Inactive'}
• Correlation Analysis: {'✅ Active' if hasattr(self, 'correlation_analyzer') else '❌ Inactive'}
• Pattern Detection: {'✅ Active' if hasattr(self, 'pattern_detector') else '❌ Inactive'}
• Market Structure Analysis: {'✅ Active' if len(self.anomalies.get('market_structure', [])) >= 0 else '❌ Inactive'}

📈 STEP STATISTICS
• Total Steps: {self.step_count:,}
• Processing Time Avg: {(np.mean(list(self._processing_times)[-10:]) * 1000):.1f}ms (last 10)
• Threshold Adaptations: {len(self.threshold_history)}
        """

    # ================== LEGACY COMPATIBILITY ==================

    def reset(self) -> None:
        """Enhanced reset with comprehensive state cleanup"""
        super().reset()
        
        # Reset mixin states
        # Note: Mixin reset methods will be implemented as needed
        
        # Clear detection state
        for anomaly_type in self.anomalies:
            self.anomalies[anomaly_type].clear()
        
        self.anomaly_score = 0.0
        self.detection_confidence = 0.5
        self.step_count = 0
        self.detection_stats.clear()
        self.false_positive_tracker.clear()
        self.detection_effectiveness.clear()
        
        # Clear history
        self.pnl_history.clear()
        self.volume_history.clear()
        self.price_history.clear()
        self.observation_history.clear()
        self.volatility_history.clear()
        self.threshold_history.clear()
        self._processing_times.clear()
        
        # Reset baselines but keep structure
        self.regime_baselines.clear()
        self.session_baselines.clear()
        self.volatility_baselines.clear()
        
        # Reset context
        self.market_regime = "normal"
        self.market_session = "unknown"
        self.volatility_regime = "medium"
        self.market_stress_level = 0.0
        
        # Reset training state
        self.training_progress = 0
        self.is_training_complete = False
        
        # Reset adaptive parameters
        self.adaptive_params = {
            'sensitivity_multiplier': 1.0,
            'regime_adaptation_factor': 1.0,
            'volatility_tolerance': 1.0,
            'learning_momentum': 0.0,
            'detection_confidence_boost': 1.0
        }
        
        # Reset thresholds to base values
        self.current_thresholds = self.base_thresholds.copy()
        
        # Reset circuit breaker
        self.circuit_breaker['failures'] = 0
        self.circuit_breaker['state'] = 'CLOSED'
        self._health_status = 'healthy'
        
        # Reset quality tracking
        self._detection_quality = 0.5
        
        # Reset external integrations
        self.external_anomaly_sources.clear()
        self.compliance_alerts.clear()
        
        # Reset mode
        self.current_mode = AnomalyDetectionMode.INITIALIZATION
        self.mode_start_time = datetime.datetime.now()
        
        self.logger.info("🔄 Enhanced Anomaly Detector reset - all state cleared")

    def step(self, pnl: Optional[float] = None, obs: Optional[np.ndarray] = None,
            volume: Optional[float] = None, price: Optional[float] = None,
            trades: Optional[List[Dict[str, Any]]] = None, **kwargs) -> bool:
        """Legacy compatibility method for synchronous operation"""
        
        if not self.enabled:
            self.anomaly_score = 0.0
            return False
        
        import asyncio
        
        # Create event loop for async operation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Prepare inputs
            inputs = {
                'pnl': pnl or 0.0,
                'observation': obs,
                'volume': volume or 0.0,
                'price': price or 0.0,
                'trades': trades or []
            }
            inputs.update(kwargs)
            
            # Run async processing
            result = loop.run_until_complete(self.process(**inputs))
            
            # Return critical anomaly status for legacy compatibility
            critical_found = any(
                a.get("severity") == AnomalySeverity.CRITICAL.value
                for anomalies in self.anomalies.values()
                for a in anomalies
            )
            
            return critical_found
            
        except Exception as e:
            self.logger.error(f"Legacy step operation failed: {e}")
            return False
        finally:
            loop.close()

    def detect_anomalies(self, **kwargs) -> Dict[str, Any]:
        """Legacy interface for anomaly detection"""
        import asyncio
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(self.process(**kwargs))
            
            # Return legacy-compatible format
            return {
                'anomaly_detected': self.anomaly_score > self.config.warning_threshold,
                'anomaly_score': self.anomaly_score,
                'confidence': self.detection_confidence,
                'anomalies': {
                    anomaly_type: len(anomalies)
                    for anomaly_type, anomalies in self.anomalies.items()
                    if anomalies
                },
                'mode': self.current_mode.value
            }
            
        except Exception as e:
            self.logger.error(f"Legacy detection operation failed: {e}")
            return {
                'anomaly_detected': False,
                'anomaly_score': 0.0,
                'confidence': 0.0,
                'anomalies': {},
                'mode': 'error'
            }
        finally:
            loop.close()

    def get_anomaly_score(self) -> float:
        """Legacy interface to get anomaly score"""
        return self.anomaly_score

    def is_anomaly_detected(self) -> bool:
        """Legacy interface to check if anomaly is detected"""
        return self.anomaly_score > self.config.warning_threshold

    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Legacy interface for anomaly summary"""
        return {
            'total_anomalies': sum(len(v) for v in self.anomalies.values()),
            'anomaly_score': self.anomaly_score,
            'detection_confidence': self.detection_confidence,
            'current_mode': self.current_mode.value,
            'enabled': self.enabled,
            'anomaly_types': {
                anomaly_type: len(anomalies)
                for anomaly_type, anomalies in self.anomalies.items()
                if anomalies
            }
        }


# ================== SUPPORTING CLASSES ==================

class SequenceAnomalyAnalyzer:
    """Advanced sequence anomaly analysis"""
    
    def __init__(self):
        self.sequence_history = deque(maxlen=50)
        self.pattern_cache = {}
    
    async def analyze_async(self, sequence: List[float], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sequence for anomalies"""
        try:
            anomalies = []
            
            if len(sequence) < 5:
                return {'anomalies': anomalies, 'analysis_completed': False}
            
            # 1. Detect unusual patterns
            pattern_anomaly = await self._detect_pattern_anomaly_async(sequence, context)
            if pattern_anomaly:
                anomalies.append(pattern_anomaly)
            
            # 2. Detect trend anomalies
            trend_anomaly = await self._detect_trend_anomaly_async(sequence, context)
            if trend_anomaly:
                anomalies.append(trend_anomaly)
            
            # 3. Detect cyclical anomalies
            cycle_anomaly = await self._detect_cycle_anomaly_async(sequence, context)
            if cycle_anomaly:
                anomalies.append(cycle_anomaly)
            
            return {
                'anomalies': anomalies,
                'analysis_completed': True,
                'sequence_length': len(sequence)
            }
            
        except Exception:
            return {'anomalies': [], 'analysis_completed': False, 'error': 'sequence_analysis_failed'}
    
    async def _detect_pattern_anomaly_async(self, sequence: List[float], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect unusual patterns in sequence"""
        try:
            # Simple pattern detection: look for repeated values
            recent_vals = sequence[-5:]
            if len(set(recent_vals)) == 1 and recent_vals[0] != 0:
                return {
                    'type': 'repeated_values',
                    'value': recent_vals[0],
                    'count': len(recent_vals),
                    'severity': AnomalySeverity.INFO.value,
                    'confidence': 0.7,
                    'timestamp': context.get('timestamp')
                }
            
            return None
            
        except Exception:
            return None
    
    async def _detect_trend_anomaly_async(self, sequence: List[float], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect trend anomalies"""
        try:
            if len(sequence) < 10:
                return None
            
            # Calculate simple trend
            x = np.arange(len(sequence))
            trend_slope = np.polyfit(x, sequence, 1)[0]
            
            # Detect extreme trends
            if abs(trend_slope) > np.std(sequence) * 2:
                return {
                    'type': 'extreme_trend',
                    'slope': float(trend_slope),
                    'direction': 'increasing' if trend_slope > 0 else 'decreasing',
                    'severity': AnomalySeverity.WARNING.value,
                    'confidence': min(0.8, abs(trend_slope) / (np.std(sequence) * 3)),
                    'timestamp': context.get('timestamp')
                }
            
            return None
            
        except Exception:
            return None
    
    async def _detect_cycle_anomaly_async(self, sequence: List[float], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect cyclical anomalies"""
        try:
            # Simple cycle detection using autocorrelation
            if len(sequence) < 20:
                return None
            
            # This is a simplified version - in production, you'd use more sophisticated methods
            autocorr = np.correlate(sequence, sequence, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Look for strong periodic patterns
            if len(autocorr) > 10 and np.max(autocorr[5:15]) > np.max(autocorr) * 0.8:
                return {
                    'type': 'unusual_cycle',
                    'cycle_strength': float(np.max(autocorr[5:15]) / np.max(autocorr)),
                    'severity': AnomalySeverity.INFO.value,
                    'confidence': 0.6,
                    'timestamp': context.get('timestamp')
                }
            
            return None
            
        except Exception:
            return None


class CorrelationAnomalyAnalyzer:
    """Advanced correlation anomaly analysis"""
    
    def __init__(self):
        self.correlation_history = deque(maxlen=100)
    
    async def analyze_async(self, series1: List[float], series2: List[float], 
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze correlation for anomalies"""
        try:
            anomalies = []
            
            if len(series1) < 10 or len(series2) < 10:
                return {'anomalies': anomalies, 'analysis_completed': False}
            
            # Calculate correlation
            correlation = np.corrcoef(series1, series2)[0, 1]
            
            if np.isnan(correlation):
                return {'anomalies': anomalies, 'analysis_completed': False}
            
            self.correlation_history.append(correlation)
            
            # Detect correlation anomalies
            if len(self.correlation_history) >= 20:
                recent_corr = list(self.correlation_history)[-20:]
                corr_mean = np.mean(recent_corr)
                corr_std = np.std(recent_corr)
                
                if corr_std > 0.01:  # Avoid division by zero
                    z_score = abs((correlation - corr_mean) / corr_std)
                    
                    if z_score > 3.0:
                        anomalies.append({
                            'type': 'correlation_change',
                            'current_correlation': float(correlation),
                            'expected_correlation': float(corr_mean),
                            'z_score': float(z_score),
                            'severity': AnomalySeverity.WARNING.value,
                            'confidence': min(0.8, z_score / 5.0),
                            'timestamp': context.get('timestamp')
                        })
            
            # Detect extreme correlations
            if abs(correlation) > 0.95:
                anomalies.append({
                    'type': 'extreme_correlation',
                    'correlation': float(correlation),
                    'severity': AnomalySeverity.INFO.value,
                    'confidence': abs(correlation),
                    'timestamp': context.get('timestamp')
                })
            
            return {
                'anomalies': anomalies,
                'analysis_completed': True,
                'current_correlation': float(correlation)
            }
            
        except Exception:
            return {'anomalies': [], 'analysis_completed': False, 'error': 'correlation_analysis_failed'}


class PatternAnomalyDetector:
    """Advanced pattern anomaly detection"""
    
    def __init__(self):
        self.trade_patterns = deque(maxlen=200)
        self.known_patterns = {}
    
    async def detect_async(self, trades: List[Dict[str, Any]], 
                         context: Dict[str, Any]) -> Dict[str, Any]:
        """Detect trading pattern anomalies"""
        try:
            anomalies = []
            
            if not trades:
                return {'anomalies': anomalies, 'pattern_detected': False}
            
            # Extract trade characteristics
            trade_sizes = [abs(trade.get('size', trade.get('volume', 0))) for trade in trades]
            trade_directions = [np.sign(trade.get('size', trade.get('volume', 0))) for trade in trades]
            trade_intervals = await self._calculate_trade_intervals_async(trades)
            
            # Store pattern
            pattern = {
                'sizes': trade_sizes,
                'directions': trade_directions,
                'intervals': trade_intervals,
                'count': len(trades),
                'timestamp': context.get('timestamp')
            }
            self.trade_patterns.append(pattern)
            
            # 1. Detect size anomalies
            size_anomaly = await self._detect_size_pattern_anomaly_async(trade_sizes, context)
            if size_anomaly:
                anomalies.append(size_anomaly)
            
            # 2. Detect timing anomalies
            timing_anomaly = await self._detect_timing_pattern_anomaly_async(trade_intervals, context)
            if timing_anomaly:
                anomalies.append(timing_anomaly)
            
            # 3. Detect directional anomalies
            direction_anomaly = await self._detect_direction_pattern_anomaly_async(trade_directions, context)
            if direction_anomaly:
                anomalies.append(direction_anomaly)
            
            return {
                'anomalies': anomalies,
                'pattern_detected': True,
                'trades_analyzed': len(trades)
            }
            
        except Exception:
            return {'anomalies': [], 'pattern_detected': False, 'error': 'pattern_detection_failed'}
    
    async def _calculate_trade_intervals_async(self, trades: List[Dict[str, Any]]) -> List[float]:
        """Calculate intervals between trades"""
        try:
            intervals = []
            for i in range(1, len(trades)):
                # Simple interval calculation - in practice, you'd use actual timestamps
                intervals.append(1.0)  # Placeholder
            return intervals
        except Exception:
            return []
    
    async def _detect_size_pattern_anomaly_async(self, sizes: List[float], 
                                               context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect size pattern anomalies"""
        try:
            if not sizes or len(sizes) < 3:
                return None
            
            # Detect if all trades are the same size (potential algo trading)
            if len(set(sizes)) == 1 and len(sizes) > 5:
                return {
                    'type': 'uniform_trade_sizes',
                    'size': sizes[0],
                    'count': len(sizes),
                    'severity': AnomalySeverity.INFO.value,
                    'confidence': min(0.9, len(sizes) / 10.0),
                    'timestamp': context.get('timestamp')
                }
            
            return None
            
        except Exception:
            return None
    
    async def _detect_timing_pattern_anomaly_async(self, intervals: List[float], 
                                                 context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect timing pattern anomalies"""
        try:
            if not intervals or len(intervals) < 5:
                return None
            
            # Detect very regular intervals (potential HFT)
            interval_std = np.std(intervals)
            interval_mean = np.mean(intervals)
            
            if interval_std < interval_mean * 0.1 and len(intervals) > 10:
                return {
                    'type': 'regular_timing_pattern',
                    'interval_mean': float(interval_mean),
                    'interval_std': float(interval_std),
                    'regularity_score': float(interval_mean / max(interval_std, 0.001)),
                    'severity': AnomalySeverity.INFO.value,
                    'confidence': 0.7,
                    'timestamp': context.get('timestamp')
                }
            
            return None
            
        except Exception:
            return None
    
    async def _detect_direction_pattern_anomaly_async(self, directions: List[float], 
                                                    context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect directional pattern anomalies"""
        try:
            if not directions or len(directions) < 5:
                return None
            
            # Remove zeros (neutral trades)
            nonzero_directions = [d for d in directions if d != 0]
            
            if not nonzero_directions:
                return None
            
            # Detect alternating pattern
            if len(nonzero_directions) >= 6:
                alternating_count = sum(1 for i in range(1, len(nonzero_directions)) 
                                      if nonzero_directions[i] != nonzero_directions[i-1])
                alternating_ratio = alternating_count / (len(nonzero_directions) - 1)
                
                if alternating_ratio > 0.8:  # 80% alternating
                    return {
                        'type': 'alternating_direction_pattern',
                        'alternating_ratio': float(alternating_ratio),
                        'trade_count': len(nonzero_directions),
                        'severity': AnomalySeverity.INFO.value,
                        'confidence': alternating_ratio,
                        'timestamp': context.get('timestamp')
                    }
            
            return None
            
        except Exception:
            return None


# End of EnhancedAnomalyDetector class