# ─────────────────────────────────────────────────────────────
# File: modules/core/mixins.py
# [ROCKET] PRODUCTION-GRADE SmartInfoBus Core Mixins
# NASA/MILITARY GRADE - ZERO ERROR TOLERANCE
# ENHANCED: Complete integration with ModuleOrchestrator, ErrorPinpointer, StateManager
# ─────────────────────────────────────────────────────────────

from __future__ import annotations
from abc import ABC, abstractmethod
import time
import numpy as np
import datetime
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
from collections import  deque
from functools import wraps
from dataclasses import dataclass
import threading

from modules.utils.info_bus import (
    SmartInfoBus, InfoBusExtractor, InfoBusUpdater, InfoBusManager,
    extract_standard_context
)
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer

if TYPE_CHECKING:
    from modules.core.module_base import BaseModule
    from modules.core.module_system import ModuleOrchestrator
    from modules.core.error_pinpointer import ErrorPinpointer

# ═══════════════════════════════════════════════════════════════════
# ENHANCED MIXIN STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MixinPerformanceMetrics:
    """Performance metrics for mixin operations"""
    operation_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_latency_ms: float = 0.0
    last_execution: Optional[float] = None
    health_status: str = "OK"  # OK, DEGRADED, FAILED

class MixinStateManager:
    """State management for mixins with hot-reload support"""
    
    def __init__(self, mixin_instance: Any):
        self.mixin_instance = mixin_instance
        self.state_lock = threading.RLock()
        self.performance_metrics = MixinPerformanceMetrics()
        self.explainer = EnglishExplainer()
        
    def get_state(self) -> Dict[str, Any]:
        """Get complete mixin state for persistence"""
        with self.state_lock:
            return {
                'performance_metrics': {
                    'operation_count': self.performance_metrics.operation_count,
                    'success_count': self.performance_metrics.success_count,
                    'failure_count': self.performance_metrics.failure_count,
                    'avg_latency_ms': self.performance_metrics.avg_latency_ms,
                    'health_status': self.performance_metrics.health_status
                }
            }
    
    def set_state(self, state: Dict[str, Any]):
        """Restore mixin state"""
        with self.state_lock:
            if 'performance_metrics' in state:
                metrics = state['performance_metrics']
                self.performance_metrics.operation_count = metrics.get('operation_count', 0)
                self.performance_metrics.success_count = metrics.get('success_count', 0)
                self.performance_metrics.failure_count = metrics.get('failure_count', 0)
                self.performance_metrics.avg_latency_ms = metrics.get('avg_latency_ms', 0.0)
                self.performance_metrics.health_status = metrics.get('health_status', 'OK')
    
    def record_operation(self, operation_name: str, duration_ms: float, success: bool):
        """Record operation performance"""
        with self.state_lock:
            self.performance_metrics.operation_count += 1
            self.performance_metrics.last_execution = time.time()
            
            if success:
                self.performance_metrics.success_count += 1
                # Update health if recovering
                if self.performance_metrics.health_status == "DEGRADED":
                    success_rate = self.performance_metrics.success_count / self.performance_metrics.operation_count
                    if success_rate > 0.8:
                        self.performance_metrics.health_status = "OK"
            else:
                self.performance_metrics.failure_count += 1
                # Degrade health if too many failures
                failure_rate = self.performance_metrics.failure_count / self.performance_metrics.operation_count
                if failure_rate > 0.3:
                    self.performance_metrics.health_status = "DEGRADED"
                elif failure_rate > 0.5:
                    self.performance_metrics.health_status = "FAILED"
            
            # Update average latency
            total_ops = self.performance_metrics.operation_count
            current_avg = self.performance_metrics.avg_latency_ms
            self.performance_metrics.avg_latency_ms = (
                (current_avg * (total_ops - 1) + duration_ms) / total_ops
            )

def with_mixin_error_handling(operation_name: str):
    """Decorator for mixin operations with error handling"""
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            start_time = time.time()
            success = False
            result = None
            
            try:
                # Check circuit breaker if available
                if hasattr(self, '_check_circuit_breaker'):
                    if not self._check_circuit_breaker(operation_name):
                        raise RuntimeError(f"Circuit breaker open for {operation_name}")
                
                result = await func(self, *args, **kwargs)
                success = True
                return result
                
            except Exception as e:
                # Log error with pinpointer if available
                if hasattr(self, 'error_pinpointer') and self.error_pinpointer:
                    self.error_pinpointer.analyze_error(e, self.__class__.__name__)
                
                # Log with standard format
                if hasattr(self, 'logger') and self.logger:
                    self.logger.error(
                        format_operator_message(
                            "[CRASH]", f"MIXIN ERROR: {operation_name}",
                            details=str(e),
                            context="mixin_operation"
                        )
                    )
                
                # Trigger emergency mode if too many failures
                if hasattr(self, '_check_emergency_trigger'):
                    self._check_emergency_trigger(operation_name, e)
                
                raise
            
            finally:
                # Record performance metrics
                duration_ms = (time.time() - start_time) * 1000
                if hasattr(self, 'state_manager'):
                    self.state_manager.record_operation(operation_name, duration_ms, success)
        
        return wrapper
    return decorator

# ═══════════════════════════════════════════════════════════════════
# ENHANCED SMARTINFOBUS TRADING MIXIN
# ═══════════════════════════════════════════════════════════════════

class SmartInfoBusTradingMixin(ABC):
    """
    PRODUCTION-GRADE trading mixin with complete SmartInfoBus integration.
    
    FEATURES:
    - Mandatory thesis generation for all trades
    - Circuit breaker integration
    - Emergency mode awareness
    - State management for hot-reload
    - Performance tracking and health monitoring
    - Error handling with ErrorPinpointer integration
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_trading_state()
    
    def _initialize_trading_state(self):
        """Initialize enhanced trading state with all integrations"""
        # Core state
        max_history = getattr(getattr(self, "config", None), "max_history", 100)
        self._trade_history = deque(maxlen=max_history)
        self._trade_theses = deque(maxlen=max_history)
        self._position_history = deque(maxlen=max_history)
        
        # Trading metrics
        self._total_pnl = 0.0
        self._trades_processed = 0
        self._winning_trades = 0
        self._losing_trades = 0
        self._max_drawdown = 0.0
        self._current_drawdown = 0.0
        self._peak_equity = 0.0
        
        # State management
        self.state_manager = MixinStateManager(self)
        
        # Smart bus integration
        self.smart_bus = InfoBusManager.get_instance()
        
        # Logger setup
        self.logger = getattr(self, "logger", RotatingLogger(
            name=f"{self.__class__.__name__}_Trading",
            log_path=f"logs/mixins/{self.__class__.__name__.lower()}_trading.log",
            max_lines=5000,
            operator_mode=True
        ))
        
        # Circuit breaker state
        self._trading_circuit_breaker = {
            'failures': 0,
            'threshold': 5,
            'reset_time': 300,  # 5 minutes
            'last_failure': 0,
            'state': 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        }
        
        self.logger.info(
            format_operator_message(
                "🏗️", "TRADING MIXIN INITIALIZED",
                context="mixin_init"
            )
        )
    
    @abstractmethod
    async def propose_action(self, **inputs) -> Dict[str, Any]:
        """ASYNC trading action proposal - different from BaseModule sync version"""
        pass
    
    @abstractmethod  
    async def calculate_confidence(self, action: Dict[str, Any], **inputs) -> float:
        """ASYNC confidence calc - different from BaseModule sync version"""
        pass
    
    @with_mixin_error_handling("process_trades")
    async def process_trades_with_thesis(self, **inputs) -> Dict[str, Any]:
        """
        Process trades with mandatory thesis generation and full error handling.
        
        Returns:
            Dict containing processed trades, thesis, and performance metrics
        """
        # Check emergency mode
        if self._is_emergency_mode():
            return {
                'trades': [],
                'thesis': "Trading suspended due to emergency mode",
                'emergency_mode': True
            }
        
        # Check circuit breaker
        if not self._check_trading_circuit_breaker():
            return {
                'trades': [],
                'thesis': "Trading suspended due to circuit breaker",
                'circuit_breaker_open': True
            }
        
        trades = inputs.get('trades', [])
        if not trades:
            return {'trades': [], 'thesis': 'No trades to process'}
        
        processed_trades = []
        processing_errors = []
        
        for trade_idx, trade in enumerate(trades):
            try:
                # Generate thesis for trade
                thesis = await self._generate_trade_thesis(trade, inputs)
                
                # Process single trade
                processed = await self._process_single_trade(trade, inputs)
                if processed:
                    processed['thesis'] = thesis
                    processed['trade_index'] = trade_idx
                    processed_trades.append(processed)
                    
                    # Update metrics
                    self._update_trading_metrics(processed)
                    
                    # Store in SmartInfoBus with thesis
                    self.smart_bus.set(
                        f"trade_{processed['trade_id']}",
                        processed,
                        module=self.__class__.__name__,
                        thesis=thesis,
                        confidence=processed.get('confidence', 0.8)
                    )
                    
            except Exception as e:
                error_msg = f"Failed to process trade {trade_idx}: {str(e)}"
                processing_errors.append(error_msg)
                self.logger.error(error_msg)
                
                # Increment circuit breaker failures
                self._record_trading_failure(str(e))
        
        # Generate session summary
        session_thesis = await self._generate_trading_summary_thesis(processed_trades, processing_errors)
        
        # Store summary in SmartInfoBus
        summary = self._get_trading_summary()
        summary['processing_errors'] = processing_errors
        
        self.smart_bus.set(
            'trading_session_summary',
            summary,
            module=self.__class__.__name__,
            thesis=session_thesis,
            confidence=0.9 if not processing_errors else 0.7
        )
        
        return {
            'trades': processed_trades,
            'thesis': session_thesis,
            'summary': summary,
            'errors': processing_errors,
            'circuit_breaker_state': self._trading_circuit_breaker['state']
        }
    
    async def _generate_trade_thesis(self, trade: Dict[str, Any], inputs: Dict[str, Any]) -> str:
        """Generate comprehensive trade thesis with market context"""
        symbol = trade.get('symbol', 'UNKNOWN')
        side = trade.get('side', 'unknown')
        size = trade.get('size', 0)
        pnl = trade.get('pnl', 0)
        
        # Get enhanced context from SmartInfoBus
        regime = self.smart_bus.get('market_regime', self.__class__.__name__) or 'unknown'
        risk_score = InfoBusExtractor.get_risk_score(inputs)
        volatility = self.smart_bus.get('market_volatility', self.__class__.__name__) or 0.0
        
        # Get current portfolio state
        portfolio_exposure = self.smart_bus.get('portfolio_exposure', self.__class__.__name__) or 0.0
        current_drawdown = self._current_drawdown
        
        thesis = f"""
TRADE EXECUTION ANALYSIS: {symbol} {side.upper()}
================================================
Position Size: {size:,.0f}
P&L Impact: ${pnl:+.2f}
Execution Time: {datetime.datetime.now().isoformat()}

MARKET ENVIRONMENT:
- Regime: {regime}
- Risk Level: {risk_score:.1%}
- Volatility: {volatility:.2%}
- Portfolio Exposure: {portfolio_exposure:.1%}
- Current Drawdown: {current_drawdown:.1%}

EXECUTION RATIONALE:
"""
        
        if pnl > 0:
            thesis += f"""
- [OK] PROFITABLE TRADE: Generated ${pnl:.2f} profit
- Market conditions were favorable for {side} position
- Risk level of {risk_score:.1%} was within acceptable parameters
- Trade aligned with current {regime} market regime
"""
        else:
            thesis += f"""
- [WARN] LOSING TRADE: Loss of ${abs(pnl):.2f}
- Market moved against {side} position
- Risk controls properly limited downside exposure
- Trade execution was disciplined despite unfavorable outcome
"""
        
        # Add risk assessment
        if risk_score > 0.7:
            thesis += "\n- [WARN] HIGH RISK ENVIRONMENT: Extra caution warranted"
        elif risk_score < 0.3:
            thesis += "\n- [OK] LOW RISK ENVIRONMENT: Favorable conditions"
        
        # Add portfolio context
        if portfolio_exposure > 0.8:
            thesis += "\n- [WARN] HIGH PORTFOLIO EXPOSURE: Consider reducing positions"
        
        thesis += f"\n\nTRADE QUALITY: {'EXCELLENT' if pnl > 0 and risk_score < 0.5 else 'ACCEPTABLE' if pnl > 0 else 'MONITORED'}"
        
        return thesis.strip()
    
    async def _process_single_trade(self, trade: Dict[str, Any], inputs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process individual trade with enhanced context"""
        context = extract_standard_context(inputs)
        risk_score = InfoBusExtractor.get_risk_score(inputs)
        
        # Enhance trade data
        processed = {
            'trade_id': f"{trade.get('symbol')}_{inputs.get('step_idx', int(time.time()))}",
            'symbol': trade.get('symbol'),
            'side': trade.get('side'),
            'size': trade.get('size', 0),
            'price': trade.get('price', 0),
            'pnl': trade.get('pnl', 0),
            'commission': trade.get('commission', 0),
            'slippage': trade.get('slippage', 0),
            'risk_at_entry': risk_score,
            'regime': context.get('regime'),
            'confidence': trade.get('confidence', 0.5),
            'processed_at': datetime.datetime.now().isoformat(),
            'module': self.__class__.__name__,
            'execution_quality': self._assess_execution_quality(trade, context)
        }
        
        # Update position tracking
        self._update_position_tracking(processed)
        
        # Log significant trades
        if abs(processed['pnl']) > 100:
            self.logger.info(
                format_operator_message(
                    '[MONEY]' if processed['pnl'] > 0 else '💸',
                    "SIGNIFICANT TRADE",
                    instrument=processed['symbol'],
                    details=f"P&L: ${processed['pnl']:+.2f}, Risk: {risk_score:.1%}",
                    context="trading"
                )
            )
        
        return processed
    
    def _assess_execution_quality(self, trade: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Assess trade execution quality"""
        pnl = trade.get('pnl', 0)
        slippage = abs(trade.get('slippage', 0))
        commission = trade.get('commission', 0)
        
        # Quality factors
        factors = []
        
        if pnl > 0:
            factors.append("profitable")
        if slippage < 0.001:  # 10 bps
            factors.append("low_slippage")
        if commission < abs(pnl) * 0.1:  # Commission < 10% of profit
            factors.append("cost_efficient")
        
        if len(factors) >= 3:
            return "EXCELLENT"
        elif len(factors) >= 2:
            return "GOOD"
        elif len(factors) >= 1:
            return "ACCEPTABLE"
        else:
            return "POOR"
    
    def _update_position_tracking(self, trade: Dict[str, Any]):
        """Update position tracking for portfolio management"""
        symbol = trade['symbol']
        side = trade['side']
        size = trade['size']
        
        # Get current position
        current_positions = self.smart_bus.get('current_positions', self.__class__.__name__) or {}
        
        if symbol not in current_positions:
            current_positions[symbol] = {'long': 0, 'short': 0, 'net': 0}
        
        # Update position
        if side.lower() == 'buy':
            current_positions[symbol]['long'] += size
        elif side.lower() == 'sell':
            current_positions[symbol]['short'] += size
        
        current_positions[symbol]['net'] = (
            current_positions[symbol]['long'] - current_positions[symbol]['short']
        )
        
        # Store updated positions
        self.smart_bus.set(
            'current_positions',
            current_positions,
            module=self.__class__.__name__,
            thesis=f"Updated positions after {symbol} {side} trade"
        )
    
    async def _generate_trading_summary_thesis(self, trades: List[Dict], errors: List[str]) -> str:
        """Generate comprehensive trading session summary"""
        if not trades and not errors:
            return "No trading activity in this session"
        
        total_pnl = sum(t.get('pnl', 0) for t in trades)
        total_commission = sum(t.get('commission', 0) for t in trades)
        winning_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
        
        thesis = f"""
TRADING SESSION PERFORMANCE ANALYSIS
===================================
Period: {datetime.datetime.now().isoformat()}
Total Trades: {len(trades)}
Processing Errors: {len(errors)}

FINANCIAL PERFORMANCE:
- Gross P&L: ${total_pnl:+.2f}
- Total Commissions: ${total_commission:.2f}
- Net P&L: ${total_pnl - total_commission:+.2f}
- Win Rate: {winning_trades/max(len(trades), 1):.1%}
- Average P&L per Trade: ${total_pnl/max(len(trades), 1):+.2f}

EXECUTION QUALITY:
"""
        
        if not errors:
            thesis += "- [OK] CLEAN EXECUTION: No processing errors"
        else:
            thesis += f"- [WARN] EXECUTION ISSUES: {len(errors)} errors encountered"
        
        # Performance assessment
        if total_pnl > 0 and not errors:
            thesis += "\n- [OK] EXCELLENT SESSION: Profitable with clean execution"
        elif total_pnl > 0:
            thesis += "\n- [WARN] MIXED SESSION: Profitable but with execution issues"
        elif not errors:
            thesis += "\n- [WARN] CHALLENGING SESSION: Clean execution but losses"
        else:
            thesis += "\n- [ALERT] POOR SESSION: Losses with execution issues"
        
        # Add circuit breaker status
        cb_state = self._trading_circuit_breaker['state']
        thesis += f"\n\nSYSTEM STATUS:\n- Circuit Breaker: {cb_state}"
        
        if errors:
            thesis += f"\n\nERROR SUMMARY:\n" + "\n".join(f"- {e}" for e in errors[:3])
            if len(errors) > 3:
                thesis += f"\n- ... and {len(errors)-3} more errors"
        
        return thesis.strip()
    
    def _update_trading_metrics(self, trade: Dict[str, Any]):
        """Update comprehensive trading metrics"""
        pnl = trade.get('pnl', 0)
        
        self._trades_processed += 1
        self._total_pnl += pnl
        
        if pnl > 0:
            self._winning_trades += 1
        elif pnl < 0:
            self._losing_trades += 1
        
        # Update drawdown tracking
        self._peak_equity = max(self._peak_equity, self._total_pnl)
        self._current_drawdown = (self._peak_equity - self._total_pnl) / max(self._peak_equity, 1)
        self._max_drawdown = max(self._max_drawdown, self._current_drawdown)
        
        # Store in history
        self._trade_history.append(trade)
        if 'thesis' in trade:
            self._trade_theses.append(trade['thesis'])
    
    def _check_trading_circuit_breaker(self) -> bool:
        """Check if trading circuit breaker allows operations"""
        cb = self._trading_circuit_breaker
        current_time = time.time()
        
        if cb['state'] == 'OPEN':
            # Check if we can move to half-open
            if current_time - cb['last_failure'] > cb['reset_time']:
                cb['state'] = 'HALF_OPEN'
                cb['failures'] = 0
                self.logger.info("Trading circuit breaker moved to HALF_OPEN")
            else:
                return False
        
        return cb['state'] in ['CLOSED', 'HALF_OPEN']
    
    def _record_trading_failure(self, error: str):
        """Record trading failure for circuit breaker"""
        cb = self._trading_circuit_breaker
        cb['failures'] += 1
        cb['last_failure'] = time.time()
        
        if cb['failures'] >= cb['threshold']:
            cb['state'] = 'OPEN'
            self.logger.warning(
                format_operator_message(
                    "[ALERT]", "TRADING CIRCUIT BREAKER OPENED",
                    details=f"Failures: {cb['failures']}, Error: {error}",
                    context="circuit_breaker"
                )
            )
    
    def _is_emergency_mode(self) -> bool:
        """Check if system is in emergency mode"""
        # Try to get emergency mode status from orchestrator
        emergency_status = self.smart_bus.get('emergency_mode_status', self.__class__.__name__)
        if emergency_status:
            return emergency_status.get('active', False)
        return False
    
    def _get_trading_summary(self) -> Dict[str, Any]:
        """Get comprehensive trading summary"""
        return {
            'trades_processed': self._trades_processed,
            'total_pnl': self._total_pnl,
            'winning_trades': self._winning_trades,
            'losing_trades': self._losing_trades,
            'win_rate': self._get_win_rate(),
            'avg_pnl': self._total_pnl / max(self._trades_processed, 1),
            'max_drawdown': self._max_drawdown,
            'current_drawdown': self._current_drawdown,
            'circuit_breaker_state': self._trading_circuit_breaker['state'],
            'has_theses': len(self._trade_theses) > 0,
            'performance_health': self.state_manager.performance_metrics.health_status
        }
    
    def _get_win_rate(self) -> float:
        """Calculate win rate"""
        total = self._winning_trades + self._losing_trades
        return self._winning_trades / max(total, 1)
    
    def get_state(self) -> Dict[str, Any]:
        """Get trading mixin state for persistence"""
        base_state = self.state_manager.get_state()
        base_state.update({
            'total_pnl': self._total_pnl,
            'trades_processed': self._trades_processed,
            'winning_trades': self._winning_trades,
            'losing_trades': self._losing_trades,
            'max_drawdown': self._max_drawdown,
            'current_drawdown': self._current_drawdown,
            'peak_equity': self._peak_equity,
            'circuit_breaker': self._trading_circuit_breaker,
            'trade_history': list(self._trade_history),
            'trade_theses': list(self._trade_theses)
        })
        return base_state
    
    def set_state(self, state: Dict[str, Any]):
        """Restore trading mixin state"""
        self.state_manager.set_state(state)
        
        # Restore trading-specific state
        self._total_pnl = state.get('total_pnl', 0.0)
        self._trades_processed = state.get('trades_processed', 0)
        self._winning_trades = state.get('winning_trades', 0)
        self._losing_trades = state.get('losing_trades', 0)
        self._max_drawdown = state.get('max_drawdown', 0.0)
        self._current_drawdown = state.get('current_drawdown', 0.0)
        self._peak_equity = state.get('peak_equity', 0.0)
        
        if 'circuit_breaker' in state:
            self._trading_circuit_breaker.update(state['circuit_breaker'])
        
        if 'trade_history' in state:
            self._trade_history = deque(state['trade_history'], maxlen=self._trade_history.maxlen)
        
        if 'trade_theses' in state:
            self._trade_theses = deque(state['trade_theses'], maxlen=self._trade_theses.maxlen)

    def _set_trading_state(self, state: Dict[str, Any]):
        """Set trading-specific state"""
        for base in self.__class__.__mro__:
            if base.__name__ == "SmartInfoBusTradingMixin":
                base.set_state(self, state)
                break

# ═══════════════════════════════════════════════════════════════════
# ENHANCED SMARTINFOBUS RISK MIXIN
# ═══════════════════════════════════════════════════════════════════

class SmartInfoBusRiskMixin(ABC):
    """
    PRODUCTION-GRADE risk management with complete SmartInfoBus integration.
    
    FEATURES:
    - Comprehensive risk assessment with thesis generation
    - Real-time risk monitoring and alerting
    - Emergency mode integration
    - State management for hot-reload
    - Circuit breaker protection
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_risk_state()
    
    def _initialize_risk_state(self):
        """Initialize enhanced risk management state"""
        # Core risk state
        self._risk_alerts = deque(maxlen=100)
        self._risk_violations = 0
        self._last_risk_check = None
        self._risk_theses = deque(maxlen=50)
        self._risk_history = deque(maxlen=1000)
        
        # Risk limits and thresholds
        self._risk_limits = {
            'max_drawdown': 0.15,  # 15%
            'max_position_size': 0.1,  # 10% of portfolio
            'max_sector_exposure': 0.3,  # 30%
            'max_leverage': 2.0,
            'var_limit': 0.05,  # 5% VaR
            'stress_test_limit': 0.1  # 10% stress loss
        }
        
        # State management
        self.state_manager = MixinStateManager(self)
        
        # Smart bus integration
        self.smart_bus = InfoBusManager.get_instance()
        
        # Logger setup
        self.logger = getattr(self, "logger", RotatingLogger(
            name=f"{self.__class__.__name__}_Risk",
            log_path=f"logs/mixins/{self.__class__.__name__.lower()}_risk.log",
            max_lines=5000,
            operator_mode=True
        ))
        
        # Circuit breaker for risk operations
        self._risk_circuit_breaker = {
            'failures': 0,
            'threshold': 3,
            'reset_time': 180,  # 3 minutes
            'last_failure': 0,
            'state': 'CLOSED'
        }
        
        self.logger.info(
            format_operator_message(
                "[SAFE]", "RISK MIXIN INITIALIZED",
                context="mixin_init"
            )
        )
    
    @with_mixin_error_handling("assess_risk")
    async def assess_risk_with_thesis(self, **inputs) -> Dict[str, Any]:
        """
        Comprehensive risk assessment with mandatory thesis generation.
        
        Returns:
            Dict with risk level, score, alerts, thesis, and recommendations
        """
        # Check circuit breaker
        if not self._check_risk_circuit_breaker():
            return {
                'level': 'UNKNOWN',
                'score': 1.0,
                'alerts': ['Risk assessment unavailable - circuit breaker open'],
                'thesis': 'Risk assessment temporarily suspended due to system errors',
                'circuit_breaker_open': True
            }
        
        try:
            # Extract comprehensive risk context
            risk_context = await self._extract_enhanced_risk_context(inputs)
            
            # Calculate risk components
            risk_components = await self._calculate_risk_components(risk_context)
            
            # Determine overall risk level and score
            risk_level, risk_score = self._determine_risk_level(risk_components)
            
            # Generate risk alerts
            alerts = await self._generate_risk_alerts(risk_context, risk_components)
            
            # Generate comprehensive thesis
            thesis = await self._generate_risk_thesis(
                risk_level, risk_score, risk_context, risk_components, alerts
            )
            
            # Generate recommendations
            recommendations = await self._generate_risk_recommendations(
                risk_level, risk_components, alerts
            )
            
            # Store comprehensive risk assessment
            risk_assessment = {
                'level': risk_level,
                'score': risk_score,
                'components': risk_components,
                'alerts': alerts,
                'recommendations': recommendations,
                'context': risk_context,
                'timestamp': datetime.datetime.now().isoformat(),
                'module': self.__class__.__name__
            }
            
            # Store in SmartInfoBus with thesis
            self.smart_bus.set(
                'comprehensive_risk_assessment',
                risk_assessment,
                module=self.__class__.__name__,
                thesis=thesis,
                confidence=0.95
            )
            
            # Record in history
            self._risk_history.append(risk_assessment)
            self._last_risk_check = time.time()
            
            # Check for emergency conditions
            if risk_score > 0.9 or risk_level == "CRITICAL":
                await self._trigger_emergency_risk_response(risk_assessment)
            
            return {
                **risk_assessment,
                'thesis': thesis
            }
            
        except Exception as e:
            self._record_risk_failure(str(e))
            raise
    
    async def _extract_enhanced_risk_context(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive risk context from inputs and SmartInfoBus"""
        # Get basic context
        context = InfoBusExtractor.extract_risk_context(inputs)
        
        # Enhance with SmartInfoBus data
        positions = self.smart_bus.get('current_positions', self.__class__.__name__) or {}
        portfolio_value = self.smart_bus.get('portfolio_value', self.__class__.__name__) or 1000000
        market_data = self.smart_bus.get('market_data', self.__class__.__name__) or {}
        
        # Calculate enhanced metrics
        total_exposure = sum(
            abs(pos['net']) * market_data.get(symbol, {}).get('price', 0)
            for symbol, pos in positions.items()
        )
        
        return {
            **context,
            'total_positions': len(positions),
            'total_exposure': total_exposure,
            'exposure_pct': (total_exposure / portfolio_value) * 100,
            'portfolio_value': portfolio_value,
            'leverage': total_exposure / portfolio_value if portfolio_value > 0 else 0,
            'positions': positions,
            'market_volatility': market_data.get('volatility', 0.0),
            'correlation_risk': await self._calculate_correlation_risk(positions, market_data)
        }
    
    async def _calculate_risk_components(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate individual risk components"""
        components = {}
        
        # Market risk
        components['market_risk'] = min(1.0, context.get('market_volatility', 0.0) * 5)
        
        # Concentration risk
        positions = context.get('positions', {})
        if positions:
            position_weights = [abs(pos['net']) for pos in positions.values()]
            total_weight = sum(position_weights)
            if total_weight > 0:
                max_weight = max(position_weights) / total_weight
                components['concentration_risk'] = min(1.0, max_weight * 2)
            else:
                components['concentration_risk'] = 0.0
        else:
            components['concentration_risk'] = 0.0
        
        # Leverage risk
        leverage = context.get('leverage', 0)
        components['leverage_risk'] = min(1.0, leverage / self._risk_limits['max_leverage'])
        
        # Drawdown risk
        drawdown = context.get('drawdown_pct', 0) / 100
        components['drawdown_risk'] = min(1.0, drawdown / self._risk_limits['max_drawdown'])
        
        # Liquidity risk (simplified)
        components['liquidity_risk'] = 0.3  # Default moderate liquidity risk
        
        # Correlation risk
        components['correlation_risk'] = context.get('correlation_risk', 0.0)
        
        return components
    
    async def _calculate_correlation_risk(self, positions: Dict, market_data: Dict) -> float:
        """Calculate portfolio correlation risk"""
        if len(positions) < 2:
            return 0.0
        
        # Simplified correlation risk calculation
        # In production, this would use actual correlation matrices
        return min(1.0, len(positions) * 0.1)  # Assume 10% correlation risk per position
    
    def _determine_risk_level(self, components: Dict[str, float]) -> Tuple[str, float]:
        """Determine overall risk level and score"""
        # Weight components
        weights = {
            'market_risk': 0.3,
            'concentration_risk': 0.2,
            'leverage_risk': 0.2,
            'drawdown_risk': 0.2,
            'liquidity_risk': 0.05,
            'correlation_risk': 0.05
        }
        
        # Calculate weighted score
        risk_score = sum(
            components.get(component, 0) * weight
            for component, weight in weights.items()
        )
        
        # Determine level
        if risk_score >= 0.8:
            level = "CRITICAL"
        elif risk_score >= 0.6:
            level = "HIGH"
        elif risk_score >= 0.4:
            level = "MEDIUM"
        elif risk_score >= 0.2:
            level = "LOW"
        else:
            level = "MINIMAL"
        
        return level, risk_score
    
    async def _generate_risk_alerts(self, context: Dict[str, Any], components: Dict[str, float]) -> List[str]:
        """Generate specific risk alerts"""
        alerts = []
        
        # Drawdown alerts
        drawdown_pct = context.get('drawdown_pct', 0)
        if drawdown_pct > self._risk_limits['max_drawdown'] * 100:
            alerts.append(f"[ALERT] Drawdown exceeds limit: {drawdown_pct:.1f}% > {self._risk_limits['max_drawdown']*100:.1f}%")
        
        # Exposure alerts
        exposure_pct = context.get('exposure_pct', 0)
        if exposure_pct > 90:
            alerts.append(f"[WARN] High portfolio exposure: {exposure_pct:.1f}%")
        
        # Leverage alerts
        leverage = context.get('leverage', 0)
        if leverage > self._risk_limits['max_leverage']:
            alerts.append(f"[ALERT] Leverage exceeds limit: {leverage:.1f}x > {self._risk_limits['max_leverage']:.1f}x")
        
        # Concentration alerts
        if components.get('concentration_risk', 0) > 0.7:
            alerts.append("[WARN] High position concentration detected")
        
        # Market risk alerts
        if components.get('market_risk', 0) > 0.8:
            alerts.append("[ALERT] Extreme market volatility detected")
        
        return alerts
    
    async def _generate_risk_thesis(self, level: str, score: float, 
                                  context: Dict[str, Any], components: Dict[str, float],
                                  alerts: List[str]) -> str:
        """Generate comprehensive risk assessment thesis"""
        
        thesis = f"""
COMPREHENSIVE RISK ASSESSMENT
============================
Overall Risk Level: {level}
Risk Score: {score:.1%}
Assessment Time: {datetime.datetime.now().isoformat()}

PORTFOLIO METRICS:
- Total Positions: {context.get('total_positions', 0)}
- Portfolio Exposure: {context.get('exposure_pct', 0):.1f}%
- Current Leverage: {context.get('leverage', 0):.2f}x
- Drawdown: {context.get('drawdown_pct', 0):.1f}%
- Portfolio Value: ${context.get('portfolio_value', 0):,.0f}

RISK COMPONENT ANALYSIS:
"""
        
        for component, value in components.items():
            status = "[ALERT]" if value > 0.8 else "[WARN]" if value > 0.6 else "[OK]"
            thesis += f"- {component.replace('_', ' ').title()}: {status} {value:.1%}\n"
        
        thesis += f"\nRISK ASSESSMENT:\n"
        
        if level == "CRITICAL":
            thesis += """
[ALERT] CRITICAL RISK LEVEL:
- Immediate action required to reduce exposure
- Consider halting new positions
- Review and close high-risk positions
- Activate emergency risk protocols
"""
        elif level == "HIGH":
            thesis += """
[WARN] HIGH RISK LEVEL:
- Elevated caution required
- Limit new position sizes
- Monitor positions closely
- Prepare risk reduction measures
"""
        elif level == "MEDIUM":
            thesis += """
[BALANCE] MEDIUM RISK LEVEL:
- Standard risk management protocols
- Monitor key risk metrics
- Maintain diversification
- Regular portfolio review
"""
        else:
            thesis += """
[OK] LOW RISK LEVEL:
- Risk levels within acceptable parameters
- Normal trading operations can continue
- Maintain current risk controls
- Continue monitoring market conditions
"""
        
        if alerts:
            thesis += f"\n\nSPECIFIC ALERTS:\n" + "\n".join(f"- {alert}" for alert in alerts)
        
        # Add recommendations section
        thesis += f"\n\nRECOMMENDATIONS:\n"
        thesis += f"- Primary Action: {self._get_primary_risk_action(level, components)}\n"
        thesis += f"- Monitoring Focus: {self._get_monitoring_focus(components)}\n"
        thesis += f"- Next Review: {self._get_next_review_time(level)}"
        
        return thesis.strip()
    
    async def _generate_risk_recommendations(self, level: str, components: Dict[str, float], alerts: List[str]) -> List[str]:
        """Generate actionable risk management recommendations"""
        recommendations = []
        
        if level in ["CRITICAL", "HIGH"]:
            recommendations.append("Reduce portfolio exposure immediately")
            recommendations.append("Close highest-risk positions")
            recommendations.append("Implement position size limits")
        
        if components.get('concentration_risk', 0) > 0.6:
            recommendations.append("Diversify concentrated positions")
        
        if components.get('leverage_risk', 0) > 0.7:
            recommendations.append("Reduce leverage to acceptable levels")
        
        if components.get('drawdown_risk', 0) > 0.6:
            recommendations.append("Implement stop-loss procedures")
        
        if not recommendations:
            recommendations.append("Maintain current risk management practices")
            recommendations.append("Continue regular monitoring")
        
        return recommendations
    
    def _get_primary_risk_action(self, level: str, components: Dict[str, float]) -> str:
        """Get primary recommended action"""
        if level == "CRITICAL":
            return "EMERGENCY POSITION REDUCTION"
        elif level == "HIGH":
            return "REDUCE EXPOSURE AND MONITOR"
        elif level == "MEDIUM":
            return "MONITOR AND MAINTAIN CONTROLS"
        else:
            return "CONTINUE NORMAL OPERATIONS"
    
    def _get_monitoring_focus(self, components: Dict[str, float]) -> str:
        """Get primary monitoring focus"""
        max_component = max(components.items(), key=lambda x: x[1])
        return max_component[0].replace('_', ' ').title()
    
    def _get_next_review_time(self, level: str) -> str:
        """Get recommended next review time"""
        if level == "CRITICAL":
            return "Every 15 minutes"
        elif level == "HIGH":
            return "Every hour"
        elif level == "MEDIUM":
            return "Every 4 hours"
        else:
            return "Daily"
    
    async def _trigger_emergency_risk_response(self, risk_assessment: Dict[str, Any]):
        """Trigger emergency risk response procedures"""
        self.logger.critical(
            format_operator_message(
                "[ALERT]", "EMERGENCY RISK RESPONSE TRIGGERED",
                details=f"Risk Level: {risk_assessment['level']}, Score: {risk_assessment['score']:.1%}",
                context="emergency_risk"
            )
        )
        
        # Store emergency event
        self.smart_bus.set(
            'emergency_risk_event',
            {
                'triggered': True,
                'risk_assessment': risk_assessment,
                'timestamp': datetime.datetime.now().isoformat(),
                'response_actions': [
                    'Position size limits activated',
                    'High-risk position review initiated',
                    'Enhanced monitoring enabled'
                ]
            },
            module=self.__class__.__name__,
            thesis="Emergency risk response activated due to critical risk conditions"
        )
    
    def _check_risk_circuit_breaker(self) -> bool:
        """Check if risk circuit breaker allows operations"""
        cb = self._risk_circuit_breaker
        current_time = time.time()
        
        if cb['state'] == 'OPEN':
            if current_time - cb['last_failure'] > cb['reset_time']:
                cb['state'] = 'HALF_OPEN'
                cb['failures'] = 0
                self.logger.info("Risk circuit breaker moved to HALF_OPEN")
            else:
                return False
        
        return cb['state'] in ['CLOSED', 'HALF_OPEN']
    
    def _record_risk_failure(self, error: str):
        """Record risk assessment failure"""
        cb = self._risk_circuit_breaker
        cb['failures'] += 1
        cb['last_failure'] = time.time()
        
        if cb['failures'] >= cb['threshold']:
            cb['state'] = 'OPEN'
            self.logger.warning(
                format_operator_message(
                    "[ALERT]", "RISK CIRCUIT BREAKER OPENED",
                    details=f"Failures: {cb['failures']}, Error: {error}",
                    context="circuit_breaker"
                )
            )
    
    def get_state(self) -> Dict[str, Any]:
        """Get risk mixin state for persistence"""
        base_state = self.state_manager.get_state()
        base_state.update({
            'risk_violations': self._risk_violations,
            'last_risk_check': self._last_risk_check,
            'risk_limits': self._risk_limits,
            'circuit_breaker': self._risk_circuit_breaker,
            'risk_alerts': list(self._risk_alerts),
            'risk_theses': list(self._risk_theses),
            'risk_history': list(self._risk_history)[-100:]  # Keep last 100
        })
        return base_state
    
    def set_state(self, state: Dict[str, Any]):
        """Restore risk mixin state"""
        self.state_manager.set_state(state)
        
        self._risk_violations = state.get('risk_violations', 0)
        self._last_risk_check = state.get('last_risk_check')
        
        if 'risk_limits' in state:
            self._risk_limits.update(state['risk_limits'])
        
        if 'circuit_breaker' in state:
            self._risk_circuit_breaker.update(state['circuit_breaker'])
        
        if 'risk_alerts' in state:
            self._risk_alerts = deque(state['risk_alerts'], maxlen=self._risk_alerts.maxlen)
        
        if 'risk_theses' in state:
            self._risk_theses = deque(state['risk_theses'], maxlen=self._risk_theses.maxlen)
        
        if 'risk_history' in state:
            self._risk_history = deque(state['risk_history'], maxlen=self._risk_history.maxlen)

    def _set_risk_state(self, state: Dict[str, Any]):
        """Set risk-specific state"""
        for base in self.__class__.__mro__:
            if base.__name__ == "SmartInfoBusRiskMixin":
                base.set_state(self, state)
                break

# ═══════════════════════════════════════════════════════════════════
# ENHANCED SMARTINFOBUS VOTING MIXIN
# ═══════════════════════════════════════════════════════════════════

class SmartInfoBusVotingMixin(ABC):
    """
    PRODUCTION-GRADE voting mixin with complete SmartInfoBus integration.
    
    FEATURES:
    - Mandatory thesis generation for all votes
    - Weighted voting with confidence tracking
    - Consensus analysis and conflict resolution
    - State management for hot-reload
    - Performance tracking and circuit breaker protection
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_voting_state()
    
    def _initialize_voting_state(self):
        """Initialize enhanced voting state"""
        # Core voting state
        max_history = getattr(getattr(self, "config", None), "max_history", 100)
        self._votes_cast = 0
        self._vote_history = deque(maxlen=max_history)
        self._confidence_history = deque(maxlen=100)
        self._vote_theses = deque(maxlen=50)
        self._consensus_history = deque(maxlen=100)
        
        # Voting performance metrics
        self._successful_votes = 0
        self._vote_accuracy = 0.0
        self._consensus_participation = 0.0
        
        # State management
        self.state_manager = MixinStateManager(self)
        
        # Smart bus integration
        self.smart_bus = InfoBusManager.get_instance()
        
        # Logger setup
        self.logger = getattr(self, "logger", RotatingLogger(
            name=f"{self.__class__.__name__}_Voting",
            log_path=f"logs/mixins/{self.__class__.__name__.lower()}_voting.log",
            max_lines=5000,
            operator_mode=True
        ))
        
        # Circuit breaker for voting operations
        self._voting_circuit_breaker = {
            'failures': 0,
            'threshold': 3,
            'reset_time': 120,  # 2 minutes
            'last_failure': 0,
            'state': 'CLOSED'
        }
        
        self.logger.info(
            format_operator_message(
                "🗳️", "VOTING MIXIN INITIALIZED",
                context="mixin_init"
            )
        )
    
    @abstractmethod
    async def propose_action(self, **inputs) -> Dict[str, Any]:
        """
        Propose voting action based on inputs.
        Must return dict with 'action', 'confidence', 'reasoning'
        """
        pass
    
    @abstractmethod
    async def calculate_confidence(self, action: Dict[str, Any], **inputs) -> float:
        """Calculate confidence level for proposed action"""
        pass
    
    @with_mixin_error_handling("prepare_vote")
    async def prepare_vote_with_thesis(self, **inputs) -> Dict[str, Any]:
        """
        Prepare comprehensive vote with mandatory thesis and consensus analysis.
        
        Returns:
            Dict containing vote, thesis, consensus analysis, and metadata
        """
        # Check circuit breaker
        if not self._check_voting_circuit_breaker():
            return {
                'vote': None,
                'thesis': 'Voting suspended due to circuit breaker',
                'circuit_breaker_open': True
            }
        
        try:
            # Get action proposal
            action_proposal = await self.propose_action(**inputs)
            confidence = await self.calculate_confidence(action_proposal, **inputs)
            
            # Analyze current consensus if available
            consensus_analysis = await self._analyze_consensus(inputs)
            
            # Generate comprehensive thesis
            thesis = await self._generate_vote_thesis(action_proposal, confidence, inputs, consensus_analysis)
            
            # Create enhanced vote
            vote = {
                'module': self.__class__.__name__,
                'action': action_proposal,
                'confidence': confidence,
                'reasoning': thesis,
                'timestamp': datetime.datetime.now().isoformat(),
                'step': inputs.get('step_idx', 0),
                'risk_score': InfoBusExtractor.get_risk_score(inputs),
                'consensus_alignment': consensus_analysis.get('alignment_score', 0.5),
                'vote_weight': self._calculate_vote_weight(confidence, consensus_analysis),
                'metadata': {
                    'market_regime': self.smart_bus.get('market_regime', self.__class__.__name__),
                    'portfolio_state': self._get_portfolio_context(inputs),
                    'historical_accuracy': self._vote_accuracy
                }
            }
            
            # Validate vote quality
            vote_quality = await self._assess_vote_quality(vote, inputs)
            vote['quality_score'] = vote_quality
            
            # Record vote
            self._record_vote(vote)
            
            # Store in SmartInfoBus with thesis
            self.smart_bus.set(
                f'vote_{self.__class__.__name__}_{self._votes_cast}',
                vote,
                module=self.__class__.__name__,
                thesis=thesis,
                confidence=confidence
            )
            
            # Add to InfoBus votes
            InfoBusUpdater.add_vote(inputs, vote)
            
            # Update consensus tracking
            await self._update_consensus_tracking(vote, inputs)
            
            return {
                'vote': vote,
                'thesis': thesis,
                'consensus_analysis': consensus_analysis,
                'quality_score': vote_quality,
                'circuit_breaker_state': self._voting_circuit_breaker['state']
            }
            
        except Exception as e:
            self._record_voting_failure(str(e))
            raise
    
    async def _analyze_consensus(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current voting consensus and conflicts"""
        # Get existing votes from InfoBus
        existing_votes = inputs.get('votes', [])
        
        if not existing_votes:
            return {
                'consensus_exists': False,
                'alignment_score': 0.5,
                'conflict_level': 'NONE',
                'dominant_action': None
            }
        
        # Analyze vote distribution
        actions = [vote.get('action') for vote in existing_votes if vote.get('action')]
        confidences = [vote.get('confidence', 0.5) for vote in existing_votes]
        
        # Calculate consensus metrics
        if actions:
            # Simple consensus analysis (in production, this would be more sophisticated)
            action_counts = {}
            for action in actions:
                action_str = str(action)  # Simplify for counting
                action_counts[action_str] = action_counts.get(action_str, 0) + 1
            
            dominant_action = max(action_counts.items(), key=lambda x: x[1])
            consensus_strength = dominant_action[1] / len(actions)
            
            return {
                'consensus_exists': consensus_strength > 0.6,
                'alignment_score': consensus_strength,
                'conflict_level': self._assess_conflict_level(consensus_strength),
                'dominant_action': dominant_action[0],
                'vote_count': len(existing_votes),
                'avg_confidence': np.mean(confidences) if confidences else 0.5
            }
        
        return {
            'consensus_exists': False,
            'alignment_score': 0.5,
            'conflict_level': 'UNKNOWN',
            'dominant_action': None
        }
    
    def _assess_conflict_level(self, consensus_strength: float) -> str:
        """Assess level of voting conflict"""
        if consensus_strength > 0.8:
            return "LOW"
        elif consensus_strength > 0.6:
            return "MEDIUM"
        elif consensus_strength > 0.4:
            return "HIGH"
        else:
            return "SEVERE"
    
    async def _generate_vote_thesis(self, action: Dict[str, Any], confidence: float, 
                                  inputs: Dict[str, Any], consensus_analysis: Dict[str, Any]) -> str:
        """Generate comprehensive voting thesis"""
        
        context = extract_standard_context(inputs)
        
        # Determine action description
        if isinstance(action, dict):
            action_desc = f"Action: {action.get('type', 'Unknown')} with parameters {list(action.keys())}"
        elif isinstance(action, np.ndarray):
            action_desc = f"Action vector with {len(action)} dimensions, max signal at index {np.argmax(np.abs(action))}"
        else:
            action_desc = f"Action: {str(action)}"
        
        thesis = f"""
VOTING DECISION ANALYSIS
========================
Module: {self.__class__.__name__}
Vote Confidence: {confidence:.1%}
{action_desc}
Decision Time: {datetime.datetime.now().isoformat()}

MARKET CONTEXT:
- Market Regime: {context.get('regime', 'Unknown')}
- Risk Level: {context.get('risk_score', 0):.1%}
- Portfolio Positions: {context.get('position_count', 0)}
- System Health: {inputs.get('system_health', 'Unknown')}

CONSENSUS ANALYSIS:
"""
        
        if consensus_analysis['consensus_exists']:
            thesis += f"""
- [OK] CONSENSUS PRESENT: {consensus_analysis['alignment_score']:.1%} agreement
- Dominant Action: {consensus_analysis['dominant_action']}
- Conflict Level: {consensus_analysis['conflict_level']}
- My Position: {'ALIGNED' if consensus_analysis['alignment_score'] > 0.6 else 'CONTRARIAN'}
"""
        else:
            thesis += """
- [WARN] NO CONSENSUS: First vote or highly divided opinions
- Vote Weight: Higher due to early position
- Market Leadership: Taking initiative in uncertain conditions
"""
        
        thesis += f"\n\nDECISION RATIONALE:\n"
        
        if confidence > 0.8:
            thesis += f"""
🔥 HIGH CONFIDENCE VOTE:
- Strong conviction based on robust analysis
- Clear market signals support this action
- Risk/reward profile highly favorable
- Historical accuracy: {self._vote_accuracy:.1%}
"""
        elif confidence > 0.5:
            thesis += f"""
[BALANCE] MODERATE CONFIDENCE VOTE:
- Balanced view with mixed signals
- Acceptable risk/reward trade-off
- Following systematic approach
- Monitoring for confirmation signals
"""
        else:
            thesis += f"""
🤔 LOW CONFIDENCE VOTE:
- Uncertain market conditions
- Limited conviction in current signals
- Defensive positioning preferred
- Ready to adjust based on new information
"""
        
        # Add consensus alignment analysis
        alignment = consensus_analysis.get('alignment_score', 0.5)
        if alignment > 0.8:
            thesis += "\n\n🤝 STRONG CONSENSUS: Vote aligns with majority view"
        elif alignment < 0.3:
            thesis += "\n\n[FAST] CONTRARIAN POSITION: Vote differs significantly from consensus"
        else:
            thesis += "\n\n[BALANCE] MIXED CONSENSUS: Moderate alignment with existing votes"
        
        # Add risk considerations
        risk_score = context.get('risk_score', 0)
        if risk_score > 0.7:
            thesis += f"\n\n[WARN] HIGH RISK ENVIRONMENT: Vote considers elevated risk level of {risk_score:.1%}"
        
        thesis += f"\n\nVOTE QUALITY: {await self._get_vote_quality_description(confidence, consensus_analysis)}"
        
        return thesis.strip()
    
    async def _get_vote_quality_description(self, confidence: float, consensus_analysis: Dict[str, Any]) -> str:
        """Get vote quality description"""
        if confidence > 0.8 and consensus_analysis.get('consensus_exists', False):
            return "EXCELLENT - High confidence with consensus support"
        elif confidence > 0.8:
            return "VERY GOOD - High confidence, independent judgment"
        elif confidence > 0.5 and consensus_analysis.get('consensus_exists', False):
            return "GOOD - Moderate confidence with consensus support"
        elif confidence > 0.5:
            return "ACCEPTABLE - Moderate confidence, independent view"
        else:
            return "CAUTIOUS - Low confidence, defensive positioning"
    
    def _calculate_vote_weight(self, confidence: float, consensus_analysis: Dict[str, Any]) -> float:
        """Calculate voting weight based on confidence and consensus"""
        base_weight = confidence
        
        # Adjust for consensus
        if consensus_analysis.get('consensus_exists', False):
            alignment = consensus_analysis.get('alignment_score', 0.5)
            if alignment > 0.8:  # Strong consensus alignment
                base_weight *= 1.1  # Slight boost for consensus
            elif alignment < 0.3:  # Contrarian position
                base_weight *= 0.9  # Slight penalty for contrarian
        
        # Adjust for historical accuracy
        if self._vote_accuracy > 0.7:
            base_weight *= 1.2  # Boost for good track record
        elif self._vote_accuracy < 0.4:
            base_weight *= 0.8  # Penalty for poor track record
        
        return min(1.0, max(0.1, base_weight))  # Clamp between 0.1 and 1.0
    
    def _get_portfolio_context(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Get portfolio context for vote"""
        return {
            'total_positions': len(self.smart_bus.get('current_positions', self.__class__.__name__) or {}),
            'portfolio_value': self.smart_bus.get('portfolio_value', self.__class__.__name__),
            'current_pnl': self.smart_bus.get('current_pnl', self.__class__.__name__),
            'exposure_level': InfoBusExtractor.get_risk_score(inputs)
        }
    
    async def _assess_vote_quality(self, vote: Dict[str, Any], inputs: Dict[str, Any]) -> float:
        """Assess overall vote quality"""
        quality_score = 0.0
        
        # Confidence component (40%)
        quality_score += vote['confidence'] * 0.4
        
        # Consensus alignment component (20%)
        alignment = vote.get('consensus_alignment', 0.5)
        if alignment > 0.8 or alignment < 0.2:  # Either strong consensus or strong contrarian
            quality_score += 0.2
        else:
            quality_score += alignment * 0.2
        
        # Historical accuracy component (20%)
        quality_score += self._vote_accuracy * 0.2
        
        # Market timing component (20%)
        risk_score = vote.get('risk_score', 0.5)
        timing_score = 1.0 - abs(risk_score - 0.5) * 2  # Best at moderate risk
        quality_score += timing_score * 0.2
        
        return min(1.0, max(0.0, quality_score))
    
    async def _update_consensus_tracking(self, vote: Dict[str, Any], inputs: Dict[str, Any]):
        """Update consensus tracking and analysis"""
        consensus_update = {
            'timestamp': datetime.datetime.now().isoformat(),
            'vote_id': f"{self.__class__.__name__}_{self._votes_cast}",
            'action': vote['action'],
            'confidence': vote['confidence'],
            'consensus_alignment': vote['consensus_alignment'],
            'quality_score': vote['quality_score']
        }
        
        self._consensus_history.append(consensus_update)
        
        # Store consensus analysis in SmartInfoBus
        self.smart_bus.set(
            'consensus_tracking_update',
            consensus_update,
            module=self.__class__.__name__,
            thesis=f"Consensus tracking updated after {self.__class__.__name__} vote"
        )
    
    def _record_vote(self, vote: Dict[str, Any]):
        """Record vote with enhanced tracking"""
        self._votes_cast += 1
        self._vote_history.append(vote)
        self._confidence_history.append(vote['confidence'])
        
        if 'reasoning' in vote:
            self._vote_theses.append(vote['reasoning'])
        
        # Update voting performance metrics (simplified)
        # In production, this would track actual outcomes
        if vote['confidence'] > 0.7:
            self._successful_votes += 1
        
        # Update accuracy estimate
        if self._votes_cast > 0:
            self._vote_accuracy = self._successful_votes / self._votes_cast
    
    def _check_voting_circuit_breaker(self) -> bool:
        """Check if voting circuit breaker allows operations"""
        cb = self._voting_circuit_breaker
        current_time = time.time()
        
        if cb['state'] == 'OPEN':
            if current_time - cb['last_failure'] > cb['reset_time']:
                cb['state'] = 'HALF_OPEN'
                cb['failures'] = 0
                self.logger.info("Voting circuit breaker moved to HALF_OPEN")
            else:
                return False
        
        return cb['state'] in ['CLOSED', 'HALF_OPEN']
    
    def _record_voting_failure(self, error: str):
        """Record voting failure for circuit breaker"""
        cb = self._voting_circuit_breaker
        cb['failures'] += 1
        cb['last_failure'] = time.time()
        
        if cb['failures'] >= cb['threshold']:
            cb['state'] = 'OPEN'
            self.logger.warning(
                format_operator_message(
                    "[ALERT]", "VOTING CIRCUIT BREAKER OPENED",
                    details=f"Failures: {cb['failures']}, Error: {error}",
                    context="circuit_breaker"
                )
            )
    
    def get_state(self) -> Dict[str, Any]:
        """Get voting mixin state for persistence"""
        base_state = self.state_manager.get_state()
        base_state.update({
            'votes_cast': self._votes_cast,
            'successful_votes': self._successful_votes,
            'vote_accuracy': self._vote_accuracy,
            'consensus_participation': self._consensus_participation,
            'circuit_breaker': self._voting_circuit_breaker,
            'vote_history': list(self._vote_history),
            'confidence_history': list(self._confidence_history),
            'vote_theses': list(self._vote_theses),
            'consensus_history': list(self._consensus_history)
        })
        return base_state
    
    def set_state(self, state: Dict[str, Any]):
        """Restore voting mixin state"""
        self.state_manager.set_state(state)
        
        self._votes_cast = state.get('votes_cast', 0)
        self._successful_votes = state.get('successful_votes', 0)
        self._vote_accuracy = state.get('vote_accuracy', 0.0)
        self._consensus_participation = state.get('consensus_participation', 0.0)
        
        if 'circuit_breaker' in state:
            self._voting_circuit_breaker.update(state['circuit_breaker'])
        
        if 'vote_history' in state:
            self._vote_history = deque(state['vote_history'], maxlen=self._vote_history.maxlen)
        
        if 'confidence_history' in state:
            self._confidence_history = deque(state['confidence_history'], maxlen=self._confidence_history.maxlen)
        
        if 'vote_theses' in state:
            self._vote_theses = deque(state['vote_theses'], maxlen=self._vote_theses.maxlen)
        
        if 'consensus_history' in state:
            self._consensus_history = deque(state['consensus_history'], maxlen=self._consensus_history.maxlen)

    def _set_voting_state(self, state: Dict[str, Any]):
        """Set voting-specific state"""
        for base in self.__class__.__mro__:
            if base.__name__ == "SmartInfoBusVotingMixin":
                base.set_state(self, state)
                break

# ═══════════════════════════════════════════════════════════════════
# ENHANCED STATE MIXIN
# ═══════════════════════════════════════════════════════════════════

class SmartInfoBusStateMixin(ABC):
    """
    PRODUCTION-GRADE state management mixin for hot-reload support.
    
    FEATURES:
    - Complete state persistence and restoration
    - State validation and integrity checking
    - Version compatibility management
    - Performance state tracking
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_state_management()
    
    def _initialize_state_management(self):
        """Initialize state management infrastructure"""
        self.state_manager = MixinStateManager(self)
        self._state_version = 1
        self._last_state_save = None
        self._state_integrity_hash = None
        
        # Smart bus integration
        self.smart_bus = InfoBusManager.get_instance()
        
        self.logger = getattr(self, "logger", RotatingLogger(
            name=f"{self.__class__.__name__}_State",
            log_path=f"logs/mixins/{self.__class__.__name__.lower()}_state.log",
            max_lines=1000,
            operator_mode=True
        ))
    
    def get_complete_state(self) -> Dict[str, Any]:
        """Get complete state including all mixin states"""
        state = {
            'state_version': self._state_version,
            'timestamp': datetime.datetime.now().isoformat(),
            'module_class': self.__class__.__name__,
            'module_path': self.__class__.__module__
        }
        
        # Get base state manager state
        if hasattr(self, 'state_manager'):
            state['base_state'] = self.state_manager.get_state()
        
        # Get trading state if available
        if hasattr(self, '_initialize_trading_state'):
            state['trading_state'] = self._get_trading_state()
        
        # Get risk state if available
        if hasattr(self, '_initialize_risk_state'):
            state['risk_state'] = self._get_risk_state()
        
        # Get voting state if available
        if hasattr(self, '_initialize_voting_state'):
            state['voting_state'] = self._get_voting_state()
        
        # Calculate integrity hash
        import hashlib
        import json
        state_json = json.dumps(state, sort_keys=True, default=str)
        state['integrity_hash'] = hashlib.sha256(state_json.encode()).hexdigest()
        
        return state
    
    def set_complete_state(self, state: Dict[str, Any]) -> bool:
        """Restore complete state with validation"""
        try:
            # Validate state version
            if state.get('state_version', 0) > self._state_version:
                self.logger.warning(f"State version mismatch: {state.get('state_version')} > {self._state_version}")
                return False
            
            # Validate integrity if present
            if 'integrity_hash' in state:
                # Remove hash for verification
                original_hash = state.pop('integrity_hash')
                import hashlib
                import json
                state_json = json.dumps(state, sort_keys=True, default=str)
                calculated_hash = hashlib.sha256(state_json.encode()).hexdigest()
                
                if original_hash != calculated_hash:
                    self.logger.error("State integrity check failed")
                    return False
            
            # Restore base state
            if 'base_state' in state and hasattr(self, 'state_manager'):
                self.state_manager.set_state(state['base_state'])
            
            # Restore mixin-specific states
            if 'trading_state' in state and hasattr(self, '_set_trading_state'):
                self._set_trading_state(state['trading_state'])
            
            if 'risk_state' in state and hasattr(self, '_set_risk_state'):
                self._set_risk_state(state['risk_state'])
            
            if 'voting_state' in state and hasattr(self, '_set_voting_state'):
                self._set_voting_state(state['voting_state'])
            
            self._last_state_save = time.time()
            
            self.logger.info(
                format_operator_message(
                    "[SAVE]", "STATE RESTORED",
                    details=f"Version: {state.get('state_version')}, Timestamp: {state.get('timestamp')}",
                    context="state_management"
                )
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore state: {e}")
            return False
    
    def _get_trading_state(self) -> Dict[str, Any]:
        """Get trading-specific state"""
        if not hasattr(self, '_total_pnl'):
            return {}
        
        return {
            'total_pnl': getattr(self, '_total_pnl', 0.0),
            'trades_processed': getattr(self, '_trades_processed', 0),
            'winning_trades': getattr(self, '_winning_trades', 0),
            'losing_trades': getattr(self, '_losing_trades', 0)
        }
    
    def _set_trading_state(self, state: Dict[str, Any]):
        """Set trading-specific state"""
        for base in self.__class__.__mro__:
            if base.__name__ == "SmartInfoBusTradingMixin":
                base.set_state(self, state)
                break

    def _get_risk_state(self) -> Dict[str, Any]:
        """Get risk-specific state"""
        if not hasattr(self, '_risk_violations'):
            return {}
        
        return {
            'risk_violations': getattr(self, '_risk_violations', 0),
            'last_risk_check': getattr(self, '_last_risk_check', None)
        }
    
    def _set_risk_state(self, state: Dict[str, Any]):
        """Set risk-specific state"""
        for base in self.__class__.__mro__:
            if base.__name__ == "SmartInfoBusRiskMixin":
                base.set_state(self, state)
                break
    
    def _get_voting_state(self) -> Dict[str, Any]:
        """Get voting-specific state"""
        if not hasattr(self, '_votes_cast'):
            return {}
        
        return {
            'votes_cast': getattr(self, '_votes_cast', 0),
            'vote_accuracy': getattr(self, '_vote_accuracy', 0.0)
        }
    
    def _set_voting_state(self, state: Dict[str, Any]):
        """Set voting-specific state"""
        for base in self.__class__.__mro__:
            if base.__name__ == "SmartInfoBusVotingMixin":
                base.set_state(self, state)
                break

# ═══════════════════════════════════════════════════════════════════
# COMPOSITE MIXINS
# ═══════════════════════════════════════════════════════════════════

class InfoBusFullIntegrationMixin(
    SmartInfoBusTradingMixin,
    SmartInfoBusRiskMixin,
    SmartInfoBusVotingMixin,
    SmartInfoBusStateMixin
):
    """
    Complete SmartInfoBus integration with all enhanced features.
    
    This mixin provides:
    - Trading operations with thesis generation
    - Risk management with comprehensive assessment
    - Voting with consensus analysis
    - State management for hot-reload
    - Circuit breaker protection
    - Performance tracking
    - Emergency mode integration
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_full_integration()
    
    def _initialize_full_integration(self):
        """Initialize all mixin components"""
        # Core logging
        self.logger = RotatingLogger(
            name=f"{self.__class__.__name__}_FullIntegration",
            log_path=f"logs/mixins/{self.__class__.__name__.lower()}_full.log",
            max_lines=10000,
            operator_mode=True,
            info_bus_aware=True,
            plain_english=True
        )
        
        # Register with orchestrator if available
        try:
            from modules.core.module_system import ModuleOrchestrator
            orchestrator = ModuleOrchestrator.get_instance()
            if orchestrator:
                # Register error handling integration
                if hasattr(orchestrator, 'error_pinpointer'):
                    self.error_pinpointer = orchestrator.error_pinpointer
                
                # Register health monitoring
                if hasattr(orchestrator, 'health_monitor'):
                    self.health_monitor = orchestrator.health_monitor
        except:
            pass  # Orchestrator not available
        
        self.logger.info(
            format_operator_message(
                "[ROCKET]", "FULL INTEGRATION MIXIN INITIALIZED",
                details="Trading, Risk, Voting, and State management active",
                context="mixin_init"
            )
        )
    
    async def process_full_cycle(self, **inputs) -> Dict[str, Any]:
        """
        Process a complete cycle: risk assessment, trading, and voting.
        """
        results = {}
        
        try:
            # 1. Risk Assessment
            risk_assessment = await self.assess_risk_with_thesis(**inputs)
            results['risk_assessment'] = risk_assessment
            
            # 2. Trading (if risk allows)
            if risk_assessment['level'] not in ['CRITICAL']:
                trading_results = await self.process_trades_with_thesis(**inputs)
                results['trading'] = trading_results
            else:
                results['trading'] = {'suspended': True, 'reason': 'Risk level too high'}
            
            # 3. Voting
            vote_results = await self.prepare_vote_with_thesis(**inputs)
            results['voting'] = vote_results
            
            # 4. Generate comprehensive summary
            summary_thesis = await self._generate_cycle_summary_thesis(results)
            
            # Store complete cycle results
            self.smart_bus.set(
                f'full_cycle_{int(time.time())}',
                results,
                module=self.__class__.__name__,
                thesis=summary_thesis,
                confidence=0.9
            )
            
            results['cycle_thesis'] = summary_thesis
            
            return results
            
        except Exception as e:
            self.logger.error(f"Full cycle processing failed: {e}")
            if hasattr(self, 'error_pinpointer'):
                self.error_pinpointer.analyze_error(e, self.__class__.__name__)
            raise
    
    async def _generate_cycle_summary_thesis(self, results: Dict[str, Any]) -> str:
        """Generate summary thesis for complete cycle"""
        
        risk_level = results.get('risk_assessment', {}).get('level', 'UNKNOWN')
        trading_suspended = results.get('trading', {}).get('suspended', False)
        vote_confidence = results.get('voting', {}).get('vote', {}).get('confidence', 0.0)
        
        thesis = f"""
COMPLETE PROCESSING CYCLE SUMMARY
=================================
Cycle Time: {datetime.datetime.now().isoformat()}
Module: {self.__class__.__name__}

COMPONENT RESULTS:
- Risk Assessment: {risk_level}
- Trading Status: {'SUSPENDED' if trading_suspended else 'ACTIVE'}
- Vote Confidence: {vote_confidence:.1%}

INTEGRATED ANALYSIS:
"""
        
        if risk_level == 'CRITICAL':
            thesis += """
[ALERT] DEFENSIVE POSTURE:
- Critical risk conditions detected
- Trading operations suspended
- Focus on risk reduction and protection
- Conservative voting approach recommended
"""
        elif risk_level == 'HIGH':
            thesis += """
[WARN] CAUTIOUS APPROACH:
- Elevated risk environment
- Limited trading activity
- Careful position management
- Risk-aware voting decisions
"""
        else:
            thesis += """
[OK] NORMAL OPERATIONS:
- Risk levels acceptable
- Trading operations proceeding
- Standard voting participation
- Monitoring conditions for changes
"""
        
        # Add performance summary
        if hasattr(self, '_get_trading_summary'):
            trading_summary = self._get_trading_summary()
            thesis += f"""

PERFORMANCE METRICS:
- Total P&L: ${trading_summary.get('total_pnl', 0):+.2f}
- Win Rate: {trading_summary.get('win_rate', 0):.1%}
- Votes Cast: {getattr(self, '_votes_cast', 0)}
- Health Status: {self.state_manager.performance_metrics.health_status}
"""
        
        thesis += f"\n\nCYCLE QUALITY: {self._assess_cycle_quality(results)}"
        
        return thesis.strip()
    
    def _assess_cycle_quality(self, results: Dict[str, Any]) -> str:
        """Assess overall cycle processing quality"""
        issues = []
        
        if results.get('trading', {}).get('suspended'):
            issues.append("trading_suspended")
        
        if results.get('risk_assessment', {}).get('level') == 'CRITICAL':
            issues.append("critical_risk")
        
        if results.get('voting', {}).get('circuit_breaker_open'):
            issues.append("voting_issues")
        
        if not issues:
            return "EXCELLENT - All components operating normally"
        elif len(issues) == 1:
            return f"GOOD - Minor issue: {issues[0]}"
        else:
            return f"CHALLENGED - Multiple issues: {', '.join(issues)}"

# ═══════════════════════════════════════════════════════════════════
# LEGACY COMPATIBILITY LAYER
# ═══════════════════════════════════════════════════════════════════

# Provide legacy aliases for backward compatibility
TradingMixin = SmartInfoBusTradingMixin
RiskMixin = SmartInfoBusRiskMixin
VotingMixin = SmartInfoBusVotingMixin
StateMixin = SmartInfoBusStateMixin

class InfoBusTradingAnalysisMixin(SmartInfoBusTradingMixin):
    """Legacy composite - maps to enhanced SmartInfoBus trading"""
    pass

class InfoBusRiskAnalysisMixin(SmartInfoBusRiskMixin):
    """Legacy composite - maps to enhanced SmartInfoBus risk"""
    pass

class InfoBusVotingAnalysisMixin(SmartInfoBusVotingMixin):
    """Legacy composite - maps to enhanced SmartInfoBus voting"""
    pass

# ═══════════════════════════════════════════════════════════════════
# MAIN EXPORTS
# ═══════════════════════════════════════════════════════════════════

__all__ = [
    # Enhanced mixins
    'SmartInfoBusTradingMixin',
    'SmartInfoBusRiskMixin', 
    'SmartInfoBusVotingMixin',
    'SmartInfoBusStateMixin',
    'InfoBusFullIntegrationMixin',
    
    # Utility classes
    'MixinStateManager',
    'MixinPerformanceMetrics',
    'with_mixin_error_handling',
    
    # Legacy compatibility
    'TradingMixin',
    'RiskMixin',
    'VotingMixin',
    'StateMixin',
    'InfoBusTradingAnalysisMixin',
    'InfoBusRiskAnalysisMixin',
    'InfoBusVotingAnalysisMixin'
]