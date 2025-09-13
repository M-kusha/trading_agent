"""
Unified Voting System - Base Classes
Provides shared base functionality for all voting components
"""

from __future__ import annotations
import asyncio
import time
import numpy as np
import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import deque, defaultdict
from abc import ABC, abstractmethod

from modules.core.module_base import BaseModule
from modules.core.mixins import (
    SmartInfoBusTradingMixin, 
    SmartInfoBusVotingMixin, 
    SmartInfoBusStateMixin
)
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.health_monitor import HealthMonitor
from modules.monitoring.performance_tracker import PerformanceTracker

from .types import (
    MarketContext, VotingProposal, ConsensusResult, QualityMetrics,
    IntelligenceParameters, CircuitBreakerState, MarketRegime, 
    TradingSession, VolatilityLevel, AlertSeverity, DebugSnapshot
)


class VotingComponentBase(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusVotingMixin, SmartInfoBusStateMixin):
    """
    Base class for all voting system components
    Provides shared initialization, error handling, market context, and telemetry
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize base voting component"""
        super().__init__(config or {})
        
    def _initialize(self):
        """Base initialization for all voting components"""
        # Initialize mixins
        self._initialize_trading_state()
        self._initialize_voting_state()
        self._initialize_state_management()
        
        # Initialize systems
        self._initialize_core_systems()
        
        # Initialize component state
        self._initialize_component_state()
        
        # Initialize debug system if enabled
        if self.config.get('debug', {}).get('enabled', False):
            self._initialize_debug_system()
            
        # Component-specific initialization
        self._initialize_specific()
        
    def _initialize_core_systems(self):
        """Initialize core system components"""
        self.smart_bus = InfoBusManager.get_instance()
        
        # Logging with component-specific name
        self.logger = RotatingLogger(
            name=self._get_component_name(),
            log_path=f"logs/voting/{self._get_component_name().lower()}.log",
            max_lines=10000 if self._is_debug_enabled() else 5000,
            operator_mode=True,
            plain_english=True,
            info_bus_aware=True
        )
        
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler(self._get_component_name(), self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()
        self.health_monitor = HealthMonitor(auto_start=True)
        
    def _initialize_component_state(self):
        """Initialize shared component state"""
        # Market context
        self.market_context = MarketContext()
        self.market_history = deque(maxlen=100)
        
        # Intelligence parameters
        self.intelligence = IntelligenceParameters(**self.config.get('intelligence', {}))
        
        # Quality metrics
        self.quality_metrics = QualityMetrics()
        
        # Circuit breaker
        self.circuit_breaker = {
            'state': CircuitBreakerState.CLOSED,
            'failure_count': 0,
            'threshold': int(self.config.get('circuit_breaker_threshold', 5)),
            'reset_time': int(self.config.get('circuit_breaker_reset_time', 300)),
            'last_failure': 0
        }
        
        # Performance tracking
        self.component_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'avg_processing_time': 0.0,
            'session_start': datetime.datetime.now().isoformat()
        }
        
        # Debug snapshots
        self.debug_snapshots = deque(maxlen=1000 if self._is_debug_enabled() else 100)
        
    def _initialize_debug_system(self):
        """Initialize debug system if enabled"""
        debug_config = self.config.get('debug', {})
        self.debug_enabled = debug_config.get('enabled', False)
        self.debug_level = debug_config.get('level', 'INFO').upper()
        self.trace_enabled = self.debug_level == 'TRACE'
        
        if self.trace_enabled:
            self.logger.info(format_operator_message(
                icon="🔍",
                message=f"TRACE debugging enabled for {self._get_component_name()}",
                detail="Full execution tracing active"
            ))
    
    @abstractmethod
    def _initialize_specific(self):
        """Component-specific initialization - to be implemented by subclasses"""
        pass
    
    @abstractmethod
    def _get_component_name(self) -> str:
        """Get component name for logging - to be implemented by subclasses"""
        pass
    
    # -------------------- Market Context Management --------------------
    
    async def _update_market_context(self) -> MarketContext:
        """Update and return current market context"""
        try:
            # Fetch market data from bus
            market_data = await self._get_market_data_from_bus()
            
            # Update context
            old_context = self.market_context
            self.market_context = self._parse_market_context(market_data)
            
            # Track significant changes
            if old_context.regime != self.market_context.regime:
                await self._handle_regime_change(old_context.regime, self.market_context.regime)
                
            if old_context.session != self.market_context.session:
                await self._handle_session_change(old_context.session, self.market_context.session)
                
            if old_context.emergency_mode != self.market_context.emergency_mode:
                await self._handle_emergency_mode_change(self.market_context.emergency_mode)
            
            # Record history
            self.market_history.append(self.market_context.to_dict())
            
            if self.trace_enabled:
                self._trace("market_context_update", {
                    'old': old_context.to_dict(),
                    'new': self.market_context.to_dict(),
                    'changes': self._detect_context_changes(old_context, self.market_context)
                })
            
            return self.market_context
            
        except Exception as e:
            self.logger.error(f"Market context update failed: {e}")
            return self.market_context
    
    async def _get_market_data_from_bus(self) -> Dict[str, Any]:
        """Fetch market data from SmartInfoBus"""
        component = self._get_component_name()
        get = self.smart_bus.get
        
        return {
            'market_regime': get('market_regime', component) or 'unknown',
            'session_type': get('session_type', component) or 'unknown',
            'volatility_data': get('volatility_data', component) or {},
            'risk_score': get('risk_score', component) or 0.0,
            'market_open': get('market_open', component, default=True),
            'emergency_mode': get('emergency_mode', component, default=False),
            'market_context': get('market_context', component) or {},
            'recent_trades': get('recent_trades', component) or [],
            'portfolio_state': get('portfolio_state', component) or {}
        }
    
    def _parse_market_context(self, market_data: Dict[str, Any]) -> MarketContext:
        """Parse market data into MarketContext"""
        volatility_value = self._extract_volatility_value(market_data.get('volatility_data', {}))
        
        return MarketContext(
            regime=self._parse_regime(market_data.get('market_regime')),
            session=self._parse_session(market_data.get('session_type')),
            volatility_level=self._determine_volatility_level(volatility_value),
            volatility_value=volatility_value,
            risk_score=float(market_data.get('risk_score', 0.0)),
            emergency_mode=bool(market_data.get('emergency_mode', False)),
            market_open=bool(market_data.get('market_open', True))
        )
    
    def _parse_regime(self, regime_str: Any) -> MarketRegime:
        """Parse regime string to enum"""
        try:
            if isinstance(regime_str, MarketRegime):
                return regime_str
            return MarketRegime(str(regime_str).lower())
        except (ValueError, AttributeError):
            return MarketRegime.UNKNOWN
    
    def _parse_session(self, session_str: Any) -> TradingSession:
        """Parse session string to enum"""
        try:
            if isinstance(session_str, TradingSession):
                return session_str
            session = str(session_str).lower()
            # Handle common aliases
            session_map = {
                'us': 'american', 'ny': 'american', 'new_york': 'american',
                'eu': 'european', 'london': 'european',
                'asia': 'asian', 'tokyo': 'asian', 'apac': 'asian'
            }
            session = session_map.get(session, session)
            return TradingSession(session)
        except (ValueError, AttributeError):
            return TradingSession.UNKNOWN
    
    def _extract_volatility_value(self, volatility_data: Any) -> float:
        """Extract numeric volatility value from various formats"""
        try:
            # Direct numeric value
            if isinstance(volatility_data, (int, float, np.floating)):
                return float(abs(volatility_data))
            
            # Dictionary with various keys
            if isinstance(volatility_data, dict):
                for key in ('value', 'atr', 'sigma', 'vol', 'volatility'):
                    if key in volatility_data:
                        val = volatility_data[key]
                        if isinstance(val, (int, float, np.floating)):
                            return float(abs(val))
                
                # Try averaging all numeric values
                numeric_vals = [
                    float(v) for v in volatility_data.values() 
                    if isinstance(v, (int, float, np.floating))
                ]
                if numeric_vals:
                    return float(np.mean(numeric_vals))
            
            # Array/list - take mean
            if isinstance(volatility_data, (list, tuple, np.ndarray)):
                arr = np.asarray(volatility_data, dtype=np.float64)
                return float(np.nanmean(arr)) if arr.size else 0.02
                
        except Exception:
            pass
        
        return 0.02  # Default volatility
    
    def _determine_volatility_level(self, volatility_value: float) -> VolatilityLevel:
        """Determine volatility level from numeric value"""
        if volatility_value > 0.05:
            return VolatilityLevel.EXTREME
        elif volatility_value > 0.03:
            return VolatilityLevel.HIGH
        elif volatility_value > 0.015:
            return VolatilityLevel.MEDIUM
        elif volatility_value > 0.008:
            return VolatilityLevel.LOW
        else:
            return VolatilityLevel.VERY_LOW
    
    def _detect_context_changes(self, old: MarketContext, new: MarketContext) -> Dict[str, Any]:
        """Detect what changed between contexts"""
        changes = {}
        if old.regime != new.regime:
            changes['regime'] = {'old': old.regime.value, 'new': new.regime.value}
        if old.session != new.session:
            changes['session'] = {'old': old.session.value, 'new': new.session.value}
        if old.emergency_mode != new.emergency_mode:
            changes['emergency_mode'] = {'old': old.emergency_mode, 'new': new.emergency_mode}
        if abs(old.volatility_value - new.volatility_value) > 0.005:
            changes['volatility'] = {'old': old.volatility_value, 'new': new.volatility_value}
        if abs(old.risk_score - new.risk_score) > 0.1:
            changes['risk_score'] = {'old': old.risk_score, 'new': new.risk_score}
        return changes
    
    async def _handle_regime_change(self, old_regime: MarketRegime, new_regime: MarketRegime):
        """Handle market regime change"""
        self.logger.info(format_operator_message(
            icon="📊",
            message="Market regime changed",
            old_regime=old_regime.value,
            new_regime=new_regime.value,
            component=self._get_component_name()
        ))
        
        if self.trace_enabled:
            self._trace("regime_change", {
                'old': old_regime.value,
                'new': new_regime.value,
                'adaptations': await self._get_regime_adaptations(new_regime)
            })
    
    async def _handle_session_change(self, old_session: TradingSession, new_session: TradingSession):
        """Handle trading session change"""
        self.logger.info(format_operator_message(
            icon="🕐",
            message="Trading session changed",
            old_session=old_session.value,
            new_session=new_session.value,
            component=self._get_component_name()
        ))
    
    async def _handle_emergency_mode_change(self, emergency_active: bool):
        """Handle emergency mode change"""
        if emergency_active:
            self.logger.warning(format_operator_message(
                icon="⚠️",
                message="EMERGENCY MODE ACTIVATED",
                component=self._get_component_name(),
                action="Risk parameters adjusted"
            ))
        else:
            self.logger.info(format_operator_message(
                icon="✅",
                message="Emergency mode deactivated",
                component=self._get_component_name()
            ))
    
    async def _get_regime_adaptations(self, regime: MarketRegime) -> Dict[str, Any]:
        """Get regime-specific adaptations"""
        # Override in subclasses for specific adaptations
        return {'regime': regime.value}
    
    # -------------------- Circuit Breaker --------------------
    
    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker allows operation"""
        cb = self.circuit_breaker
        now = time.time()
        
        if cb['state'] == CircuitBreakerState.OPEN:
            if now - cb['last_failure'] > cb['reset_time']:
                cb['state'] = CircuitBreakerState.HALF_OPEN
                cb['failure_count'] = 0
                self.logger.info(f"Circuit breaker moved to HALF_OPEN for {self._get_component_name()}")
                return True
            return False
            
        return True
    
    def _trip_circuit_breaker(self):
        """Trip the circuit breaker"""
        self.circuit_breaker['failure_count'] += 1
        self.circuit_breaker['last_failure'] = time.time()
        
        if self.circuit_breaker['failure_count'] >= self.circuit_breaker['threshold']:
            self.circuit_breaker['state'] = CircuitBreakerState.OPEN
            self.logger.error(format_operator_message(
                icon="🔴",
                message="CIRCUIT BREAKER TRIPPED",
                component=self._get_component_name(),
                failures=self.circuit_breaker['failure_count']
            ))
    
    def _reset_circuit_breaker(self):
        """Reset circuit breaker after successful operation"""
        if self.circuit_breaker['state'] == CircuitBreakerState.HALF_OPEN:
            self.circuit_breaker['state'] = CircuitBreakerState.CLOSED
            self.circuit_breaker['failure_count'] = 0
            self.logger.info(f"Circuit breaker reset for {self._get_component_name()}")
    
    # -------------------- Debug & Tracing --------------------
    
    def _is_debug_enabled(self) -> bool:
        """Check if debug is enabled"""
        return getattr(self, 'debug_enabled', False)
    
    def _trace(self, stage: str, data: Dict[str, Any], rationale: str = ""):
        """Add trace snapshot if tracing is enabled"""
        if not getattr(self, 'trace_enabled', False):
            return
            
        snapshot = DebugSnapshot(
            stage=stage,
            timestamp=datetime.datetime.now().isoformat(),
            inputs=data.get('inputs', {}),
            calculations=data.get('calculations', {}),
            outputs=data.get('outputs', {}),
            rationale=rationale or data.get('rationale', ''),
            performance_ms=data.get('performance_ms', 0.0)
        )
        
        self.debug_snapshots.append(snapshot)
        
        # Log trace
        self.logger.debug(format_operator_message(
            icon="🔍",
            message=f"TRACE: {stage}",
            component=self._get_component_name(),
            snapshot=snapshot.to_dict()
        ))
    
    def _get_debug_state(self) -> Dict[str, Any]:
        """Get current debug state"""
        return {
            'component': self._get_component_name(),
            'debug_enabled': self._is_debug_enabled(),
            'trace_enabled': getattr(self, 'trace_enabled', False),
            'snapshots_count': len(self.debug_snapshots),
            'recent_snapshots': [s.to_dict() for s in list(self.debug_snapshots)[-10:]],
            'market_context': self.market_context.to_dict(),
            'circuit_breaker': {
                'state': self.circuit_breaker['state'].value,
                'failures': self.circuit_breaker['failure_count']
            },
            'quality_metrics': self.quality_metrics.to_dict(),
            'component_stats': self.component_stats
        }
    
    # -------------------- Performance Tracking --------------------
    
    async def _track_operation(self, operation_name: str, success: bool, duration_ms: float):
        """Track component operation performance"""
        self.component_stats['total_operations'] += 1
        
        if success:
            self.component_stats['successful_operations'] += 1
        else:
            self.component_stats['failed_operations'] += 1
        
        # Update average processing time
        n = self.component_stats['total_operations']
        old_avg = self.component_stats['avg_processing_time']
        self.component_stats['avg_processing_time'] = (old_avg * (n - 1) + duration_ms) / n
        
        # Track in performance tracker
        self.performance_tracker.record_metric(
            self._get_component_name(),
            operation_name,
            duration_ms,
            success
        )
        
        if self.trace_enabled:
            self._trace(f"operation_{operation_name}", {
                'performance_ms': duration_ms,
                'success': success,
                'stats': self.component_stats
            })
    
    # -------------------- Health Monitoring --------------------
    
    def _get_health_metrics(self) -> Dict[str, Any]:
        """Get component health metrics"""
        total_ops = self.component_stats['total_operations']
        success_rate = (
            self.component_stats['successful_operations'] / total_ops 
            if total_ops > 0 else 0.0
        )
        
        return {
            'component': self._get_component_name(),
            'status': 'healthy' if self._check_circuit_breaker() else 'degraded',
            'circuit_breaker_state': self.circuit_breaker['state'].value,
            'total_operations': total_ops,
            'success_rate': success_rate,
            'avg_processing_time_ms': self.component_stats['avg_processing_time'],
            'quality_score': self.quality_metrics.overall_effectiveness,
            'market_context': {
                'regime': self.market_context.regime.value,
                'session': self.market_context.session.value,
                'emergency_mode': self.market_context.emergency_mode
            },
            'uptime_hours': (
                datetime.datetime.now() - 
                datetime.datetime.fromisoformat(self.component_stats['session_start'])
            ).total_seconds() / 3600
        }
    
    # -------------------- Error Handling --------------------
    
    async def _handle_component_error(self, error: Exception, operation: str) -> Dict[str, Any]:
        """Handle component error with circuit breaker"""
        self._trip_circuit_breaker()
        
        error_context = self.error_pinpointer.analyze_error(error, self._get_component_name())
        
        self.logger.error(format_operator_message(
            icon="❌",
            message=f"Component error in {operation}",
            component=self._get_component_name(),
            error=str(error_context)
        ))
        
        if self.trace_enabled:
            self._trace(f"error_{operation}", {
                'error': str(error),
                'context': str(error_context),
                'circuit_breaker': self.circuit_breaker
            })
        
        return {
            'success': False,
            'error': str(error_context),
            'component': self._get_component_name(),
            'operation': operation
        }
    
    # -------------------- State Management --------------------
    
    def get_state(self) -> Dict[str, Any]:
        """Get component state for persistence"""
        base_state = super().get_state()
        
        return {
            **base_state,
            'component_specific': {
                'market_context': self.market_context.to_dict(),
                'quality_metrics': self.quality_metrics.to_dict(),
                'intelligence': self.intelligence.to_dict(),
                'circuit_breaker': {
                    'state': self.circuit_breaker['state'].value,
                    'failure_count': self.circuit_breaker['failure_count']
                },
                'component_stats': self.component_stats,
                'debug_enabled': self._is_debug_enabled()
            }
        }
    
    def set_state(self, state: Dict[str, Any]):
        """Restore component state"""
        super().set_state(state)
        
        if 'component_specific' in state:
            specific = state['component_specific']
            
            if 'market_context' in specific:
                self.market_context = MarketContext.from_dict(specific['market_context'])
            
            if 'quality_metrics' in specific:
                for key, value in specific['quality_metrics'].items():
                    setattr(self.quality_metrics, key, value)
            
            if 'intelligence' in specific:
                for key, value in specific['intelligence'].items():
                    setattr(self.intelligence, key, value)
            
            if 'circuit_breaker' in specific:
                cb = specific['circuit_breaker']
                self.circuit_breaker['state'] = CircuitBreakerState(cb.get('state', 'CLOSED'))
                self.circuit_breaker['failure_count'] = cb.get('failure_count', 0)
            
            if 'component_stats' in specific:
                self.component_stats.update(specific['component_stats'])
    
    def reset(self):
        """Reset component state"""
        super().reset()
        
        # Reset market context
        self.market_context = MarketContext()
        self.market_history.clear()
        
        # Reset quality metrics
        self.quality_metrics = QualityMetrics()
        
        # Reset circuit breaker
        self.circuit_breaker['state'] = CircuitBreakerState.CLOSED
        self.circuit_breaker['failure_count'] = 0
        
        # Reset stats
        self.component_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'avg_processing_time': 0.0,
            'session_start': datetime.datetime.now().isoformat()
        }
        
        # Clear debug snapshots
        self.debug_snapshots.clear()
        
        self.logger.info(format_operator_message(
            icon="🔄",
            message=f"{self._get_component_name()} reset completed"
        ))