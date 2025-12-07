# ─────────────────────────────────────────────────────────────
# File: modules/core/mixins.py
# [CLEANED] SmartInfoBus Core Mixins - Infrastructure Only
# Removed: Unused orchestration methods (1500+ lines of dead code)
# Kept: State management, contracts, utilities actually used by modules
# ─────────────────────────────────────────────────────────────

from __future__ import annotations
from abc import ABC, abstractmethod
import time
import threading
from typing import Dict, Any, List, Optional, TYPE_CHECKING, Literal
from collections import deque
from dataclasses import dataclass

from modules.utils.info_bus import (
    SmartInfoBus, InfoBusManager
)
from modules.utils.audit_utils import RotatingLogger, format_operator_message

if TYPE_CHECKING:
    from modules.core.module_base import BaseModule

# ═══════════════════════════════════════════════════════════════════
# MIXIN STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════

HealthStatus = Literal["OK", "DEGRADED", "FAILED"]

@dataclass
class MixinPerformanceMetrics:
    """Performance metrics for mixin operations"""
    operation_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_latency_ms: float = 0.0
    last_execution: Optional[float] = None
    health_status: HealthStatus = "OK"


class MixinStateManager:
    """State management for mixins with hot-reload support"""

    def __init__(self, mixin_instance: Any):
        self.mixin_instance = mixin_instance
        self.state_lock = threading.RLock()
        self.performance_metrics = MixinPerformanceMetrics()

    def get_state(self) -> Dict[str, Any]:
        """Get complete mixin state for persistence."""
        with self.state_lock:
            return {
                'performance_metrics': {
                    'operation_count': self.performance_metrics.operation_count,
                    'success_count': self.performance_metrics.success_count,
                    'failure_count': self.performance_metrics.failure_count,
                    'avg_latency_ms': self.performance_metrics.avg_latency_ms,
                    'last_execution': self.performance_metrics.last_execution,
                    'health_status': self.performance_metrics.health_status,
                }
            }

    def set_state(self, state: Dict[str, Any]):
        """Restore mixin state."""
        with self.state_lock:
            metrics = state.get('performance_metrics', {})
            self.performance_metrics.operation_count = metrics.get('operation_count', 0)
            self.performance_metrics.success_count = metrics.get('success_count', 0)
            self.performance_metrics.failure_count = metrics.get('failure_count', 0)
            self.performance_metrics.avg_latency_ms = metrics.get('avg_latency_ms', 0.0)
            self.performance_metrics.last_execution = metrics.get('last_execution')
            self.performance_metrics.health_status = metrics.get('health_status', 'OK')  # type: ignore[assignment]

    def record_operation(self, operation_name: str, duration_ms: float, success: bool):
        """Record operation performance (thread-safe)."""
        with self.state_lock:
            self.performance_metrics.operation_count += 1
            self.performance_metrics.last_execution = time.time()

            if success:
                self.performance_metrics.success_count += 1
                if self.performance_metrics.health_status in ("DEGRADED", "FAILED"):
                    success_rate = self.performance_metrics.success_count / max(self.performance_metrics.operation_count, 1)
                    if success_rate > 0.8:
                        self.performance_metrics.health_status = "OK"
            else:
                self.performance_metrics.failure_count += 1
                failure_rate = self.performance_metrics.failure_count / max(self.performance_metrics.operation_count, 1)
                if failure_rate > 0.5:
                    self.performance_metrics.health_status = "FAILED"
                elif failure_rate > 0.3:
                    self.performance_metrics.health_status = "DEGRADED"

            # Online average latency
            total_ops = self.performance_metrics.operation_count
            current_avg = self.performance_metrics.avg_latency_ms
            self.performance_metrics.avg_latency_ms = ((current_avg * (total_ops - 1)) + duration_ms) / total_ops


# ═══════════════════════════════════════════════════════════════════
# SMARTINFOBUS TRADING MIXIN
# ═══════════════════════════════════════════════════════════════════

class SmartInfoBusTradingMixin(ABC):
    """
    Trading mixin providing state management and abstract contracts.

    FEATURES:
    - Trading state initialization and persistence
    - Abstract method contracts for propose_action and calculate_confidence
    - Performance metrics tracking
    - Helper utilities for metrics updates

    USAGE:
    - Modules inherit this and implement propose_action() and calculate_confidence()
    - Modules call _initialize_trading_state() in their __init__
    - Modules use helper methods like _update_trading_metrics() as needed
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_trading_state()

    def _initialize_trading_state(self):
        """Initialize trading state (idempotent)."""
        max_history = getattr(getattr(self, "config", None), "max_history", 100)
        self._trade_history = deque(maxlen=max_history)
        self._trade_theses = deque(maxlen=max_history)

        # Trading metrics
        self._total_pnl = getattr(self, "_total_pnl", 0.0)
        self._trades_processed = getattr(self, "_trades_processed", 0)
        self._winning_trades = getattr(self, "_winning_trades", 0)
        self._losing_trades = getattr(self, "_losing_trades", 0)
        self._max_drawdown = getattr(self, "_max_drawdown", 0.0)
        self._current_drawdown = getattr(self, "_current_drawdown", 0.0)
        self._peak_equity = getattr(self, "_peak_equity", 0.0)

        # State management
        if not hasattr(self, 'state_manager'):
            self.state_manager = MixinStateManager(self)

        # Smart bus
        self.smart_bus: SmartInfoBus = getattr(self, 'smart_bus', InfoBusManager.get_instance())

        # Logger setup
        if not getattr(self, 'logger', None):
            self.logger = RotatingLogger(
                name=f"{self.__class__.__name__}_Trading",
                log_path=f"logs/mixins/{self.__class__.__name__.lower()}_trading.log",
                max_lines=5000,
                operator_mode=True
            )

        self.logger.info(
            format_operator_message("🏗️", "TRADING MIXIN INITIALIZED", context="mixin_init")
        )

    @abstractmethod
    async def propose_action(self, **inputs) -> Optional[Dict[str, Any]]:
        """
        ABSTRACT: Propose a trading action based on inputs.
        Modules MUST implement this method.

        Returns:
            Dict with action details or None
        """
        pass

    @abstractmethod
    async def calculate_confidence(self, action: Dict[str, Any], **inputs) -> Optional[float]:
        """
        ABSTRACT: Calculate confidence for a proposed action.
        Modules MUST implement this method.

        Returns:
            Float confidence score (0.0-1.0) or None
        """
        pass

    def _update_trading_metrics(self, trade: Dict[str, Any]):
        """Helper: Update comprehensive trading metrics"""
        pnl = float(trade.get('pnl', 0) or 0)

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

    def _get_trading_summary(self) -> Dict[str, Any]:
        """Helper: Get comprehensive trading summary"""
        return {
            'trades_processed': self._trades_processed,
            'total_pnl': self._total_pnl,
            'winning_trades': self._winning_trades,
            'losing_trades': self._losing_trades,
            'win_rate': self._get_win_rate(),
            'avg_pnl': self._total_pnl / max(self._trades_processed, 1),
            'max_drawdown': self._max_drawdown,
            'current_drawdown': self._current_drawdown,
            'has_theses': len(self._trade_theses) > 0,
            'performance_health': self.state_manager.performance_metrics.health_status
        }

    def _get_win_rate(self) -> float:
        """Helper: Calculate win rate"""
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
            'trade_history': list(self._trade_history),
            'trade_theses': list(self._trade_theses)
        })
        return base_state

    def set_state(self, state: Dict[str, Any]):
        """Restore trading mixin state"""
        self.state_manager.set_state(state)

        self._total_pnl = state.get('total_pnl', 0.0)
        self._trades_processed = state.get('trades_processed', 0)
        self._winning_trades = state.get('winning_trades', 0)
        self._losing_trades = state.get('losing_trades', 0)
        self._max_drawdown = state.get('max_drawdown', 0.0)
        self._current_drawdown = state.get('current_drawdown', 0.0)
        self._peak_equity = state.get('peak_equity', 0.0)

        if 'trade_history' in state:
            self._trade_history = deque(state['trade_history'], maxlen=self._trade_history.maxlen)

        if 'trade_theses' in state:
            self._trade_theses = deque(state['trade_theses'], maxlen=self._trade_theses.maxlen)


# ═══════════════════════════════════════════════════════════════════
# SMARTINFOBUS RISK MIXIN
# ═══════════════════════════════════════════════════════════════════

class SmartInfoBusRiskMixin(ABC):
    """
    Risk management mixin providing state management and abstract contracts.

    FEATURES:
    - Risk state initialization and persistence
    - Risk limits and threshold management
    - Performance metrics tracking

    USAGE:
    - Modules inherit this and implement their own risk assessment logic
    - Modules call _initialize_risk_state() in their __init__
    - Modules manage their own risk calculations in process()
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_risk_state()

    def _initialize_risk_state(self):
        """Initialize risk management state (idempotent)."""
        # Core risk state
        self._risk_alerts = deque(maxlen=100)
        self._risk_violations = getattr(self, "_risk_violations", 0)
        self._last_risk_check = getattr(self, "_last_risk_check", None)
        self._risk_theses = deque(maxlen=50)
        self._risk_history = deque(maxlen=1000)

        # Load risk limits from risk_policy.yaml
        defaults = self._load_risk_limits_from_yaml()
        self._risk_limits = {**defaults, **getattr(self, "_risk_limits", {})}

        # State manager
        if not hasattr(self, 'state_manager'):
            self.state_manager = MixinStateManager(self)

        # Smart bus
        self.smart_bus: SmartInfoBus = getattr(self, 'smart_bus', InfoBusManager.get_instance())

        # Logger
        if not getattr(self, 'logger', None):
            self.logger = RotatingLogger(
                name=f"{self.__class__.__name__}_Risk",
                log_path=f"logs/mixins/{self.__class__.__name__.lower()}_risk.log",
                max_lines=5000,
                operator_mode=True
            )

        self.logger.info(
            format_operator_message("[SAFE]", "RISK MIXIN INITIALIZED", context="mixin_init")
        )
    
    def _load_risk_limits_from_yaml(self) -> Dict[str, Any]:
        """Load risk limits from risk_policy.yaml with hardcoded fallbacks."""
        import yaml
        import os
        
        # Hardcoded fallbacks (prop firm safe)
        defaults = {
            'max_drawdown': 0.085,
            'max_position_size': 0.05,
            'max_sector_exposure': 0.10,
            'max_leverage': 5.0,
            'var_limit': 0.015,
            'stress_test_limit': 0.05
        }
        
        try:
            config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "risk_policy.yaml")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    policy = yaml.safe_load(f) or {}
                
                limits = policy.get("limits", {})
                lot_sizing = policy.get("lot_sizing", {})
                
                defaults['max_drawdown'] = float(limits.get("max_drawdown", defaults['max_drawdown']))
                defaults['max_position_size'] = float(limits.get("max_position_size", defaults['max_position_size']))
                defaults['max_sector_exposure'] = float(limits.get("max_exposure_pct", defaults['max_sector_exposure']) * 2)
                defaults['max_leverage'] = float(limits.get("max_leverage", defaults['max_leverage']))
                defaults['var_limit'] = float(limits.get("max_portfolio_var", defaults['var_limit']))
        except Exception:
            pass  # Use fallback defaults
        
        return defaults

    def get_state(self) -> Dict[str, Any]:
        """Get risk mixin state for persistence"""
        base_state = self.state_manager.get_state()
        base_state.update({
            'risk_violations': self._risk_violations,
            'last_risk_check': self._last_risk_check,
            'risk_limits': self._risk_limits,
            'risk_alerts': list(self._risk_alerts),
            'risk_theses': list(self._risk_theses),
            'risk_history': list(self._risk_history)[-100:]
        })
        return base_state

    def set_state(self, state: Dict[str, Any]):
        """Restore risk mixin state"""
        self.state_manager.set_state(state)

        self._risk_violations = state.get('risk_violations', 0)
        self._last_risk_check = state.get('last_risk_check')

        if 'risk_limits' in state:
            self._risk_limits.update(state['risk_limits'])

        if 'risk_alerts' in state:
            self._risk_alerts = deque(state['risk_alerts'], maxlen=self._risk_alerts.maxlen)

        if 'risk_theses' in state:
            self._risk_theses = deque(state['risk_theses'], maxlen=self._risk_theses.maxlen)

        if 'risk_history' in state:
            self._risk_history = deque(state['risk_history'], maxlen=self._risk_history.maxlen)


# ═══════════════════════════════════════════════════════════════════
# SMARTINFOBUS VOTING MIXIN
# ═══════════════════════════════════════════════════════════════════

class SmartInfoBusVotingMixin(ABC):
    """
    Voting mixin providing state management and abstract contracts.

    FEATURES:
    - Voting state initialization and persistence
    - Vote history and confidence tracking
    - Performance analytics

    USAGE:
    - Modules inherit this and implement propose_action and calculate_confidence
    - Modules call _initialize_voting_state() in their __init__
    - Modules manage their own voting logic in process()
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_voting_state()

    def _initialize_voting_state(self):
        """Initialize voting state (idempotent)."""
        max_history = getattr(getattr(self, "config", None), "max_history", 100)
        self._votes_cast = getattr(self, "_votes_cast", 0)
        self._vote_history = deque(maxlen=max_history)
        self._confidence_history = deque(maxlen=100)
        self._vote_theses = deque(maxlen=50)
        self._consensus_history = deque(maxlen=100)

        self._successful_votes = getattr(self, "_successful_votes", 0)
        self._vote_accuracy = getattr(self, "_vote_accuracy", 0.0)
        self._consensus_participation = getattr(self, "_consensus_participation", 0.0)

        if not hasattr(self, 'state_manager'):
            self.state_manager = MixinStateManager(self)

        self.smart_bus: SmartInfoBus = getattr(self, 'smart_bus', InfoBusManager.get_instance())

        if not getattr(self, 'logger', None):
            self.logger = RotatingLogger(
                name=f"{self.__class__.__name__}_Voting",
                log_path=f"logs/mixins/{self.__class__.__name__.lower()}_voting.log",
                max_lines=5000,
                operator_mode=True
            )

        self.logger.info(
            format_operator_message("🗳️", "VOTING MIXIN INITIALIZED", context="mixin_init")
        )

    @abstractmethod
    async def propose_action(self, **inputs) -> Optional[Dict[str, Any]]:
        """
        ABSTRACT: Propose voting action based on inputs.
        Modules MUST implement this method.

        Returns:
            Dict with 'action', 'confidence', 'reasoning' or None
        """
        pass

    @abstractmethod
    async def calculate_confidence(self, action: Dict[str, Any], **inputs) -> Optional[float]:
        """
        ABSTRACT: Calculate confidence for proposed action.
        Modules MUST implement this method.

        Returns:
            Float confidence (0.0-1.0) or None
        """
        pass

    def _record_vote(self, vote: Dict[str, Any]):
        """Helper: Record vote with enhanced tracking"""
        self._votes_cast += 1
        self._vote_history.append(vote)
        self._confidence_history.append(float(vote.get('confidence', 0.0) or 0.0))

        if 'reasoning' in vote:
            self._vote_theses.append(str(vote['reasoning']))

        if float(vote.get('confidence', 0.0) or 0.0) > 0.7:
            self._successful_votes += 1

        if self._votes_cast > 0:
            self._vote_accuracy = self._successful_votes / self._votes_cast

    def get_state(self) -> Dict[str, Any]:
        """Get voting mixin state for persistence"""
        base_state = self.state_manager.get_state()
        base_state.update({
            'votes_cast': self._votes_cast,
            'successful_votes': self._successful_votes,
            'vote_accuracy': self._vote_accuracy,
            'consensus_participation': self._consensus_participation,
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

        if 'vote_history' in state:
            self._vote_history = deque(state['vote_history'], maxlen=self._vote_history.maxlen)

        if 'confidence_history' in state:
            self._confidence_history = deque(state['confidence_history'], maxlen=self._confidence_history.maxlen)

        if 'vote_theses' in state:
            self._vote_theses = deque(state['vote_theses'], maxlen=self._vote_theses.maxlen)

        if 'consensus_history' in state:
            self._consensus_history = deque(state['consensus_history'], maxlen=self._consensus_history.maxlen)


# ═══════════════════════════════════════════════════════════════════
# STATE MIXIN
# ═══════════════════════════════════════════════════════════════════

class SmartInfoBusStateMixin(ABC):
    """
    State management mixin for hot-reload support.

    FEATURES:
    - Complete state persistence and restoration
    - State validation and integrity checking
    - Version compatibility management

    USAGE:
    - Modules inherit this for state management capabilities
    - Modules call _initialize_state_management() in their __init__
    - Modules can use get_complete_state() and set_complete_state() for persistence
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_state_management()

    def _initialize_state_management(self):
        """Initialize state management infrastructure (non-destructive)."""
        if not hasattr(self, 'state_manager'):
            self.state_manager = MixinStateManager(self)
        self._state_version = getattr(self, "_state_version", 1)
        self._last_state_save = getattr(self, "_last_state_save", None)
        self._state_integrity_hash = getattr(self, "_state_integrity_hash", None)

        self.smart_bus: SmartInfoBus = getattr(self, 'smart_bus', InfoBusManager.get_instance())

        if not getattr(self, 'logger', None):
            self.logger = RotatingLogger(
                name=f"{self.__class__.__name__}_State",
                log_path=f"logs/mixins/{self.__class__.__name__.lower()}_state.log",
                max_lines=1000,
                operator_mode=True
            )

    def get_complete_state(self) -> Dict[str, Any]:
        """Get complete state including all mixin states"""
        import datetime
        import hashlib
        import json

        state = {
            'state_version': self._state_version,
            'timestamp': datetime.datetime.now().isoformat(),
            'module_class': self.__class__.__name__,
            'module_path': self.__class__.__module__
        }

        if hasattr(self, 'state_manager'):
            state['base_state'] = self.state_manager.get_state()

        if hasattr(self, '_initialize_trading_state'):
            state['trading_state'] = self._get_trading_state()

        if hasattr(self, '_initialize_risk_state'):
            state['risk_state'] = self._get_risk_state()

        if hasattr(self, '_initialize_voting_state'):
            state['voting_state'] = self._get_voting_state()

        # Calculate integrity hash
        state_json = json.dumps(state, sort_keys=True, default=str)
        state['integrity_hash'] = hashlib.sha256(state_json.encode()).hexdigest()

        return state

    def set_complete_state(self, state: Dict[str, Any]) -> bool:
        """Restore complete state with validation"""
        import hashlib
        import json

        try:
            incoming_version = state.get('state_version', 0)
            if incoming_version > getattr(self, '_state_version', 1):
                self.logger.warning(f"State version mismatch: {incoming_version} > {self._state_version}")
                return False

            if 'integrity_hash' in state:
                original_hash = state['integrity_hash']
                state_copy = {k: v for k, v in state.items() if k != 'integrity_hash'}
                state_json = json.dumps(state_copy, sort_keys=True, default=str)
                calculated_hash = hashlib.sha256(state_json.encode()).hexdigest()
                if original_hash != calculated_hash:
                    self.logger.error("State integrity check failed")
                    return False

            if 'base_state' in state and hasattr(self, 'state_manager'):
                self.state_manager.set_state(state['base_state'])

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
# COMPOSITE MIXIN
# ═══════════════════════════════════════════════════════════════════

class InfoBusFullIntegrationMixin(
    SmartInfoBusTradingMixin,
    SmartInfoBusRiskMixin,
    SmartInfoBusVotingMixin,
    SmartInfoBusStateMixin
):
    """
    Complete SmartInfoBus integration combining all mixin features.

    Provides:
    - Trading state and metrics
    - Risk management state
    - Voting state and tracking
    - State persistence for hot-reload

    USAGE:
    - Modules can inherit this single class to get all mixin features
    - Equivalent to inheriting all four mixins individually
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Consolidated logger if not already set
        if not getattr(self, 'logger', None):
            self.logger = RotatingLogger(
                name=f"{self.__class__.__name__}_FullIntegration",
                log_path=f"logs/mixins/{self.__class__.__name__.lower()}_full.log",
                max_lines=10000,
                operator_mode=True,
                info_bus_aware=True,
                plain_english=True
            )

        self.logger.info(
            format_operator_message(
                "[ROCKET]", "FULL INTEGRATION MIXIN INITIALIZED",
                details="Trading, Risk, Voting, and State management active",
                context="mixin_init"
            )
        )
