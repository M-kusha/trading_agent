# -------------------------------------------------------------
# File: modules/position/position_debug.py
# Position Manager Debug System (Hardened & Mirrored)
#
# - Multi-level verbosity with pretty, human-readable blocks
# - Thread-safe, atomic file writes with immediate flush
# - Mirrors all key logs into a shared RotatingLogger (if attached)
# - Optional console output (disabled in your PositionManagerBase)
# - CSV/JSON/TXT artifacts for forensics
# -------------------------------------------------------------

from __future__ import annotations

import datetime
import json
import threading
import time
import traceback
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# ENUMERATIONS & DATA MODELS
# =============================================================================

class DebugLevel(Enum):
    """Debug verbosity levels with priorities."""
    TRACE = ("TRACE", "🔍", 0)        # Ultra-detailed internal state
    DEBUG = ("DEBUG", "🐛", 10)       # Detailed debugging info
    INFO = ("INFO", "ℹ️", 20)         # General informational messages
    SUCCESS = ("SUCCESS", "✅", 25)   # Successful operations
    WARNING = ("WARNING", "⚠️", 30)   # Warning conditions
    ERROR = ("ERROR", "❌", 40)       # Error conditions
    CRITICAL = ("CRITICAL", "🚨", 50) # Critical failures

    def __init__(self, label: str, icon: str, priority: int):
        self.label = label
        self.icon = icon
        self.priority = priority


class ActionType(Enum):
    """Trading action types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    SCALE_UP = "ADD_MORE"
    SCALE_DOWN = "REDUCE"
    CLOSE_POSITION = "CLOSE"
    EMERGENCY_EXIT = "EMERGENCY"


@dataclass
class DecisionSnapshot:
    """Complete snapshot of a position decision with full context."""
    # Core identification
    timestamp: str
    instrument: str

    # Action details
    action: str
    is_buying: bool
    direction: str  # LONG/SHORT/NEUTRAL

    # Sizing and confidence
    size_eur: float
    confidence: float
    market_intensity: float
    current_price: float

    # Technical metrics
    signal_strength: float
    trend_direction: str
    volatility: float

    # Portfolio state
    portfolio_health: float
    risk_score: float

    # Explanations
    plain_english_reason: str
    technical_factors: List[str]
    risk_factors: List[str]

    # Execution
    will_execute: bool
    execution_blocked_reason: Optional[str] = None

    # Extended context (for detailed debugging)
    raw_context: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_plain_english(self) -> str:
        """Generate human-readable explanation."""
        action_word = self._format_action()
        lines = [
            f"{'='*80}",
            f"📍 {self.instrument} - {action_word}",
            f"{'='*80}",
            f"💰 Size: €{self.size_eur:,.2f}",
            f"📊 Confidence: {self.confidence:.1%}",
            f"📈 Signal: {self.signal_strength:.2f} ({self.trend_direction})",
            f"⚡ Volatility: {self.volatility:.4f}",
            f"🏥 Portfolio Health: {self.portfolio_health:.1%}",
            f"⚠️  Risk Score: {self.risk_score:.1%}",
            f"💵 Price: {self.current_price:.5f}",
            "",
            "📝 Reason:",
            f"  {self.plain_english_reason}",
            "",
            "Technical Factors:",
        ]
        for factor in self.technical_factors:
            lines.append(f"  • {factor}")
        lines.append("")
        lines.append("Risk Considerations:")
        for factor in self.risk_factors:
            lines.append(f"  • {factor}")
        lines.append("")
        status = '✅ WILL EXECUTE' if self.will_execute else f'❌ BLOCKED: {self.execution_blocked_reason}'
        lines.append(f"Status: {status}")
        lines.append(f"{'='*80}")
        return "\n".join(lines)

    def _format_action(self) -> str:
        """Format action as readable string."""
        if self.is_buying and self.action == "BUY":
            return "🟢 BUYING (LONG)"
        elif self.is_buying and self.action == "SCALE_UP":
            return "🟢 ADDING TO LONG"
        elif not self.is_buying and self.action == "SELL":
            return "🔴 SELLING (SHORT)"
        elif self.action == "CLOSE_POSITION":
            return "🔒 CLOSING POSITION"
        elif self.action == "EMERGENCY_EXIT":
            return "🚨 EMERGENCY EXIT"
        elif self.action == "SCALE_DOWN":
            return "📉 REDUCING POSITION"
        else:
            return f"⏸️  {self.action}"


@dataclass
class ErrorSnapshot:
    """Detailed error information with full context."""
    timestamp: str
    error_type: str
    error_message: str
    component: str
    stack_trace: str
    context: Dict[str, Any]
    recovery_action: Optional[str] = None

    def to_plain_english(self) -> str:
        lines = [
            f"{'!'*80}",
            f"🚨 ERROR DETECTED",
            f"{'!'*80}",
            f"Time: {self.timestamp}",
            f"Component: {self.component}",
            f"Type: {self.error_type}",
            "",
            "What went wrong:",
            f"  {self.error_message}",
            "",
            "Stack trace:",
            f"  {self.stack_trace[:300]}...",
            "",
            "Context:",
            f"  {json.dumps(self.context, indent=2)[:300]}...",
            "",
            f"Recovery: {self.recovery_action or 'Manual intervention needed'}",
            f"{'!'*80}",
        ]
        return "\n".join(lines)


@dataclass
class TradeSignal:
    """Real-time trade signal tracking."""
    timestamp: datetime.datetime
    instrument: str
    action: str
    size_eur: float
    confidence: float
    executed: bool
    price: float
    pnl: Optional[float] = None


@dataclass
class PerformanceMetric:
    """Performance tracking data point."""
    timestamp: str
    metric_name: str
    value: float
    unit: str
    context: Dict[str, Any]


# =============================================================================
# MAIN DEBUG SYSTEM
# =============================================================================

class PositionDebugSystem:
    """
    Unified debug system for PositionManager.

    Key guarantees:
    - All pretty blocks (decisions/errors) are mirrored into a shared RotatingLogger if attached.
    - File writes are serialized and flushed immediately.
    - Console output is optional and independent from file mirroring.
    """

    def __init__(
        self,
        log_dir: str = "logs/debug",
        enable: bool = True,
        verbosity: DebugLevel = DebugLevel.INFO,
        console_output: bool = True,
        file_output: bool = True,
        max_memory_items: int = 10000,
    ):
        self.enabled = enable
        self.verbosity = verbosity
        self.console_output = console_output
        self.file_output = file_output
        self.max_memory_items = max_memory_items

        # Thread safety
        self._lock = threading.RLock()

        # Optional external sink (RotatingLogger), attached by module base
        self._shared_logger = None  # set via attach_shared_logger()

        # In-memory storage
        self.decision_history: Deque[DecisionSnapshot] = deque(maxlen=max_memory_items)
        self.error_history: Deque[ErrorSnapshot] = deque(maxlen=1000)
        self.performance_metrics: Deque[PerformanceMetric] = deque(maxlen=5000)
        self.trade_signals: Deque[TradeSignal] = deque(maxlen=1000)

        # Statistics
        self.stats = {
            "total_decisions": 0,
            "buy_decisions": 0,
            "sell_decisions": 0,
            "hold_decisions": 0,
            "successful_executions": 0,
            "blocked_executions": 0,
            "total_errors": 0,
            "start_time": datetime.datetime.utcnow(),
        }

        # Trade signal tracking
        self.active_signals: Dict[str, TradeSignal] = {}
        self.signal_stats = defaultdict(lambda: {
            'total_buys': 0,
            'total_sells': 0,
            'total_holds': 0,
            'buy_volume': 0.0,
            'sell_volume': 0.0,
            'win_count': 0,
            'loss_count': 0,
            'total_pnl': 0.0,
        })

        # File paths
        self.log_dir: Optional[Path] = None
        self.decision_csv: Optional[Path] = None
        self.decision_summary: Optional[Path] = None
        self.error_log: Optional[Path] = None
        self.debug_log: Optional[Path] = None
        self.buy_signals_json: Optional[Path] = None
        self.sell_signals_json: Optional[Path] = None
        self.metrics_json: Optional[Path] = None
        self.signals_csv: Optional[Path] = None

        if self.enabled and self.file_output:
            self._init_log_files(log_dir)

    # -------------------------------------------------------------------------
    # Shared RotatingLogger wiring
    # -------------------------------------------------------------------------
    def attach_shared_logger(self, logger: Any) -> None:
        """Attach shared RotatingLogger to mirror pretty blocks and lines."""
        with self._lock:
            self._shared_logger = logger

    # -------------------------------------------------------------------------
    # File initialization
    # -------------------------------------------------------------------------
    def _init_log_files(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.log_dir / "decisions").mkdir(exist_ok=True)
        (self.log_dir / "errors").mkdir(exist_ok=True)
        (self.log_dir / "signals").mkdir(exist_ok=True)
        (self.log_dir / "metrics").mkdir(exist_ok=True)

        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # Main files
        self.decision_csv = self.log_dir / "decisions" / f"decisions_{timestamp}.csv"
        self.decision_summary = self.log_dir / "decisions" / f"summary_{timestamp}.txt"
        self.error_log = self.log_dir / "errors" / f"errors_{timestamp}.log"
        self.debug_log = self.log_dir / f"debug_{timestamp}.log"

        # Signal files
        self.buy_signals_json = self.log_dir / "signals" / f"buy_signals_{timestamp}.json"
        self.sell_signals_json = self.log_dir / "signals" / f"sell_signals_{timestamp}.json"
        self.signals_csv = self.log_dir / "signals" / f"signals_{timestamp}.csv"

        # Metrics file
        self.metrics_json = self.log_dir / "metrics" / f"metrics_{timestamp}.json"

        # Initialize CSV headers
        self._write_csv_header(
            self.decision_csv,
            "timestamp,instrument,action,is_buying,direction,size_eur,confidence,"
            "signal_strength,volatility,portfolio_health,risk_score,will_execute,"
            "blocked_reason,plain_english_reason\n"
        )
        self._write_csv_header(
            self.signals_csv,
            "timestamp,instrument,action,size_eur,confidence,executed,price,pnl\n"
        )

        # Write session header
        header = f"""
{'='*80}
🎯 POSITION MANAGER DEBUG SESSION
{'='*80}
Session Started: {timestamp}
Verbosity Level: {self.verbosity.label} {self.verbosity.icon}
Console Output: {'Enabled' if self.console_output else 'Disabled'}
File Output: {'Enabled' if self.file_output else 'Disabled'}
Log Directory: {self.log_dir}
{'='*80}

"""
        self._log_to_file(self.debug_log, header)

    # -------------------------------------------------------------------------
    # Core logging API
    # -------------------------------------------------------------------------
    def log_decision(
        self,
        instrument: str,
        decision: str,
        intensity: float,
        size: float,
        confidence: float,
        context: Dict[str, Any],
        rationale: Dict[str, Any],
        portfolio_health: float,
    ) -> Optional[DecisionSnapshot]:
        if not self.enabled:
            return None

        try:
            # Parse decision
            action, is_buying, direction = self._parse_decision(decision, intensity)

            # Extract metrics
            signal_strength = abs(float(intensity))
            trend_direction = "BULLISH" if intensity > 0 else "BEARISH" if intensity < 0 else "NEUTRAL"
            volatility = max(float(context.get('volatility', 0.02)), 1e-6)
            current_price = float(context.get('current_price', 0.0))

            # Risk score
            risk_score = self._calculate_risk_score(context)

            # Generate explanations
            plain_english = self._generate_plain_english_reason(
                action, is_buying, instrument, signal_strength, trend_direction, confidence, rationale
            )
            technical_factors = self._extract_technical_factors(context)
            risk_factors = self._extract_risk_factors(context, rationale.get('risk_factors', {}))

            # Check execution viability
            will_execute, blocked_reason = self._check_execution_viability(size, confidence, context)

            # Snapshot
            snapshot = DecisionSnapshot(
                timestamp=datetime.datetime.utcnow().isoformat() + "Z",
                instrument=instrument,
                action=action,
                is_buying=is_buying,
                direction=direction,
                size_eur=float(size),
                confidence=float(np.clip(confidence, 0.0, 1.0)),
                market_intensity=float(intensity),
                current_price=current_price,
                signal_strength=signal_strength,
                trend_direction=trend_direction,
                volatility=volatility,
                portfolio_health=float(np.clip(portfolio_health, 0.0, 1.0)),
                risk_score=risk_score,
                plain_english_reason=plain_english,
                technical_factors=technical_factors,
                risk_factors=risk_factors,
                will_execute=will_execute,
                execution_blocked_reason=blocked_reason,
                raw_context=context if self.verbosity.priority <= DebugLevel.DEBUG.priority else None,
            )

            # Store & write
            with self._lock:
                self.decision_history.append(snapshot)
                self._update_stats(snapshot)

                if self.file_output:
                    self._write_decision_to_csv(snapshot)
                    self._write_decision_to_summary(snapshot)
                    self._write_decision_to_debug_log(snapshot)

                if self.console_output and self.verbosity.priority <= DebugLevel.INFO.priority:
                    self._print_decision(snapshot)

                self._track_trade_signal(snapshot)

            # Skip compact line - unified logger handles this
            # Only log at TRACE level for deep debugging
            if self.verbosity.priority <= DebugLevel.TRACE.priority:
                self._log_debug(
                    DebugLevel.TRACE,
                    f"CSV: {instrument} | {action} | Size: €{size:.2f} | Conf: {confidence:.1%}"
                )

            return snapshot

        except Exception as e:
            self._log_debug(DebugLevel.ERROR, f"Error logging decision: {e}")
            return None

    def log_error(
        self,
        error: Exception,
        component: str,
        context: Dict[str, Any],
        recovery_action: Optional[str] = None,
    ) -> Optional[ErrorSnapshot]:
        if not self.enabled:
            return None
        try:
            snapshot = ErrorSnapshot(
                timestamp=datetime.datetime.utcnow().isoformat() + "Z",
                error_type=type(error).__name__,
                error_message=str(error),
                component=component,
                stack_trace=traceback.format_exc(),
                context=context,
                recovery_action=recovery_action,
            )

            with self._lock:
                self.error_history.append(snapshot)
                self.stats["total_errors"] += 1

                if self.file_output:
                    self._write_error_to_file(snapshot)
                    self._write_error_to_debug_log(snapshot)

                if self.console_output and self.verbosity.priority <= DebugLevel.ERROR.priority:
                    self._print_error(snapshot)

            self._log_debug(
                DebugLevel.ERROR,
                f"Error in {component}: {type(error).__name__} - {str(error)[:100]}"
            )

            return snapshot

        except Exception as e:
            # Last-resort print to avoid swallowing critical failures
            print(f"Critical: Error while logging error: {e}")
            return None

    def log_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = "",
        context: Optional[Dict[str, Any]] = None,
    ):
        if not self.enabled:
            return
        try:
            metric = PerformanceMetric(
                timestamp=datetime.datetime.utcnow().isoformat() + "Z",
                metric_name=metric_name,
                value=float(value),
                unit=unit,
                context=context or {},
            )
            with self._lock:
                self.performance_metrics.append(metric)
                if self.file_output:
                    self._write_metric_to_file(metric)
            # Trace-level mirror
            self._log_debug(DebugLevel.TRACE, f"Metric logged: {metric_name}={value}{unit}")
        except Exception as e:
            self._log_debug(DebugLevel.ERROR, f"Error logging metric: {e}")

    def log_trace(self, message: str, **kwargs):
        self._log_debug(DebugLevel.TRACE, message, **kwargs)

    def log_debug(self, message: str, **kwargs):
        self._log_debug(DebugLevel.DEBUG, message, **kwargs)

    def log_info(self, message: str, **kwargs):
        self._log_debug(DebugLevel.INFO, message, **kwargs)

    def log_warning(self, message: str, **kwargs):
        self._log_debug(DebugLevel.WARNING, message, **kwargs)

    def log_success(self, message: str, **kwargs):
        self._log_debug(DebugLevel.SUCCESS, message, **kwargs)

    # -------------------------------------------------------------------------
    # INTERNAL HELPERS
    # -------------------------------------------------------------------------
    def _log_debug(self, level: DebugLevel, message: str, **kwargs):
        """Internal debug logging to file (if enabled) and mirrored to shared logger."""
        if not self.enabled or level.priority < self.verbosity.priority:
            return

        timestamp = datetime.datetime.utcnow().isoformat() + "Z"
        formatted = f"[{timestamp}] {level.icon} {level.label}: {message}"
        if kwargs:
            formatted += f" | {kwargs}"

        # Console (optional)
        if self.console_output:
            print(formatted)

        # File (optional)
        if self.file_output and self.debug_log:
            self._log_to_file(self.debug_log, formatted + "\n")

        # Mirror to shared logger
        sl = getattr(self, "_shared_logger", None)
        if sl:
            try:
                if level.priority >= DebugLevel.ERROR.priority and hasattr(sl, "error"):
                    sl.error(formatted)
                elif level.priority >= DebugLevel.WARNING.priority and hasattr(sl, "warning"):
                    sl.warning(formatted)
                elif level.priority >= DebugLevel.INFO.priority and hasattr(sl, "info"):
                    sl.info(formatted)
                elif hasattr(sl, "debug"):
                    sl.debug(formatted)
                if hasattr(sl, "flush"):
                    sl.flush()
            except Exception:
                pass

    def _parse_decision(self, decision: str, intensity: float) -> Tuple[str, bool, str]:
        d = (decision or "").lower()
        if "open_long" in d:
            return "BUY", True, "LONG"
        elif "open_short" in d:
            return "SELL", False, "SHORT"
        elif "scale_up" in d:
            is_buying = intensity > 0
            return "SCALE_UP", is_buying, "LONG" if is_buying else "SHORT"
        elif "scale_down" in d:
            return "SCALE_DOWN", False, "REDUCING"
        elif "emergency_close" in d:
            return "EMERGENCY_EXIT", False, "EXIT"
        elif "close" in d:
            return "CLOSE_POSITION", False, "CLOSING"
        else:
            return "HOLD", False, "NEUTRAL"

    def _calculate_risk_score(self, context: Dict[str, Any]) -> float:
        drawdown = float(context.get('drawdown', 0.0) or 0.0)
        exposure = float(context.get('current_exposure', 0.0) or 0.0)
        volatility = float(context.get('volatility', 0.02) or 0.02)
        drawdown_risk = min(drawdown * 5, 1.0)   # 20% drawdown -> 1.0
        exposure_risk = min(exposure * 2, 1.0)   # 50% exposure  -> 1.0
        vol_risk = min(volatility / 0.05, 1.0)   # 5% volatility -> 1.0
        return float(min(drawdown_risk * 0.4 + exposure_risk * 0.3 + vol_risk * 0.3, 1.0))

    def _generate_plain_english_reason(
        self,
        action: str,
        is_buying: bool,
        instrument: str,
        signal_strength: float,
        trend: str,
        confidence: float,
        rationale: Dict[str, Any],
    ) -> str:
        stage = rationale.get('stage', 'unknown')
        factors = rationale.get('factors', [])
        if action == "BUY":
            reason = (
                f"Opening LONG position on {instrument} because market shows "
                f"strong {trend} signal ({signal_strength:.2f}) with {confidence:.1%} confidence"
            )
        elif action == "SELL":
            reason = (
                f"Opening SHORT position on {instrument} because market shows "
                f"strong {trend} signal ({signal_strength:.2f}) with {confidence:.1%} confidence"
            )
        elif action == "SCALE_UP":
            reason = (
                f"Adding to existing position on {instrument} as trend continues "
                f"to be {trend} and aligns with our position"
            )
        elif action == "SCALE_DOWN":
            reason = (
                f"Reducing position on {instrument} to manage risk as "
                f"market conditions have changed"
            )
        elif action == "CLOSE_POSITION":
            if "risk" in stage:
                reason = f"Closing {instrument} position due to risk management rules"
            elif "reverse" in stage:
                reason = f"Closing {instrument} position because market has reversed"
            else:
                reason = f"Closing {instrument} position to lock in results"
        elif action == "EMERGENCY_EXIT":
            reason = f"EMERGENCY EXIT from {instrument} - critical risk conditions detected"
        else:
            reason = (
                f"Holding {instrument} position - market signals not strong "
                f"enough for action"
            )
        if factors:
            reason += f". Key factors: {'; '.join(factors[:3])}"
        return reason

    def _extract_technical_factors(self, context: Dict[str, Any]) -> List[str]:
        factors = []
        trend_strength = float(context.get('trend_strength', 0.0) or 0.0)
        if abs(trend_strength) > 0.5:
            direction = "uptrend" if trend_strength > 0 else "downtrend"
            factors.append(f"Strong {direction} detected (strength: {abs(trend_strength):.2f})")
        momentum = float(context.get('momentum', 0.0) or 0.0)
        if abs(momentum) > 0.3:
            momentum_dir = "positive" if momentum > 0 else "negative"
            factors.append(f"Momentum is {momentum_dir} ({momentum:.2f})")
        rsi = float(context.get('rsi', 50.0) or 50.0)
        if rsi > 70:
            factors.append(f"Market is overbought (RSI: {rsi:.0f})")
        elif rsi < 30:
            factors.append(f"Market is oversold (RSI: {rsi:.0f})")
        volatility = float(context.get('volatility', 0.02) or 0.02)
        if volatility > 0.04:
            factors.append(f"High volatility detected ({volatility:.4f})")
        elif volatility < 0.01:
            factors.append(f"Low volatility environment ({volatility:.4f})")
        volume = float(context.get('volume_profile', 1.0) or 1.0)
        if volume > 1.5:
            factors.append(f"High trading volume ({volume:.1f}x normal)")
        elif volume < 0.5:
            factors.append(f"Low trading volume ({volume:.1f}x normal)")
        return factors if factors else ["Normal market conditions"]

    def _extract_risk_factors(self, context: Dict[str, Any], risk_factors: Dict[str, float]) -> List[str]:
        factors = []
        drawdown = float(context.get('drawdown', 0.0) or 0.0)
        if drawdown > 0.05:
            factors.append(f"Portfolio drawdown at {drawdown:.1%}")
        exposure = float(context.get('current_exposure', 0.0) or 0.0)
        if exposure > 0.5:
            factors.append(f"High exposure level ({exposure:.1%} of capital)")
        correlation = float(risk_factors.get('correlation', 0.0) or 0.0)
        if correlation > 0.5:
            factors.append(f"High correlation with other positions ({correlation:.1%})")
        session = context.get('session', 'unknown')
        if session == 'closed':
            factors.append("Market is closed - higher spread risk")
        elif session == 'asian':
            factors.append("Asian session - lower liquidity")
        vol_risk = float(risk_factors.get('volatility', 0.0) or 0.0)
        if vol_risk > 0.3:
            factors.append(f"Elevated volatility risk ({vol_risk:.1%})")
        return factors if factors else ["Risk levels are acceptable"]

    def _check_execution_viability(self, size: float, confidence: float, context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        if size <= 0:
            return False, "Position size is zero or negative"
        balance = float(context.get('balance', 0.0) or 0.0)
        if balance > 0 and size > balance * 0.5:
            return False, f"Position size (€{size:.2f}) exceeds 50% of balance"
        if confidence < 0.3:
            return False, f"Confidence too low ({confidence:.1%})"
        risk_score = self._calculate_risk_score(context)
        if risk_score > 0.8:
            return False, f"Risk score too high ({risk_score:.1%})"
        drawdown = float(context.get('drawdown', 0.0) or 0.0)
        if drawdown > 0.15:
            return False, f"Drawdown limit exceeded ({drawdown:.1%})"
        return True, None

    def _track_trade_signal(self, snapshot: DecisionSnapshot):
        if snapshot.action in ("BUY", "SELL", "SCALE_UP"):
            signal = TradeSignal(
                timestamp=datetime.datetime.utcnow(),
                instrument=snapshot.instrument,
                action=snapshot.action,
                size_eur=snapshot.size_eur,
                confidence=snapshot.confidence,
                executed=snapshot.will_execute,
                price=snapshot.current_price,
                pnl=None,
            )
            self.active_signals[snapshot.instrument] = signal
            self.trade_signals.append(signal)
            if snapshot.action in ("BUY", "SCALE_UP") and snapshot.is_buying:
                self.signal_stats[snapshot.instrument]['total_buys'] += 1
                self.signal_stats[snapshot.instrument]['buy_volume'] += snapshot.size_eur
            elif snapshot.action in ("SELL", "SCALE_UP") and not snapshot.is_buying:
                self.signal_stats[snapshot.instrument]['total_sells'] += 1
                self.signal_stats[snapshot.instrument]['sell_volume'] += snapshot.size_eur

    def _update_stats(self, snapshot: DecisionSnapshot):
        self.stats["total_decisions"] += 1
        if snapshot.action == "BUY":
            self.stats["buy_decisions"] += 1
        elif snapshot.action == "SELL":
            self.stats["sell_decisions"] += 1
        elif snapshot.action == "HOLD":
            self.stats["hold_decisions"] += 1
        if snapshot.will_execute:
            self.stats["successful_executions"] += 1
        else:
            self.stats["blocked_executions"] += 1

    # -------------------------------------------------------------------------
    # FILE I/O (hardened + atomic)
    # -------------------------------------------------------------------------
    def _write_csv_header(self, filepath: Optional[Path], header: str):
        if not self.file_output or filepath is None:
            return
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                new_file = not filepath.exists()
                with open(filepath, 'a', encoding='utf-8', newline='') as f:
                    if new_file:
                        f.write(header)
                        f.flush()
        except Exception:
            pass

    def _log_to_file(self, filepath: Optional[Path], content: str):
        if not self.file_output or filepath is None:
            return
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                with open(filepath, 'a', encoding='utf-8', newline='') as f:
                    f.write(content)
                    f.flush()
        except Exception:
            pass

    def _write_decision_to_csv(self, snapshot: DecisionSnapshot):
        if not self.file_output or self.decision_csv is None:
            return
        try:
            blocked_reason = (snapshot.execution_blocked_reason or '').replace('"', "'")
            plain_reason = snapshot.plain_english_reason.replace('"', "'")
            line = (
                f"{snapshot.timestamp},{snapshot.instrument},{snapshot.action},"
                f"{snapshot.is_buying},{snapshot.direction},{snapshot.size_eur:.2f},"
                f"{snapshot.confidence:.4f},{snapshot.signal_strength:.4f},"
                f"{snapshot.volatility:.6f},{snapshot.portfolio_health:.4f},"
                f"{snapshot.risk_score:.4f},{snapshot.will_execute},"
                f'"{blocked_reason}","{plain_reason}"\n'
            )
            self._log_to_file(self.decision_csv, line)
        except Exception:
            pass

    def _write_decision_to_summary(self, snapshot: DecisionSnapshot):
        if not self.file_output or self.decision_summary is None:
            return
        try:
            self._log_to_file(self.decision_summary, snapshot.to_plain_english() + "\n\n")
        except Exception:
            pass

    def _write_decision_to_debug_log(self, snapshot: DecisionSnapshot):
        """Write to debug log only if verbosity allows (CSV-only mode doesn't write here)."""
        # Skip pretty block output - unified logger handles this now
        # Only log if verbosity is DEBUG or lower (not CRITICAL)
        if self.verbosity.priority > DebugLevel.DEBUG.priority:
            return

        block = "\n" + snapshot.to_plain_english() + "\n"
        try:
            if self.file_output and self.debug_log is not None:
                self._log_to_file(self.debug_log, block)
        except Exception:
            pass

    def _write_error_to_file(self, snapshot: ErrorSnapshot):
        if not self.file_output or self.error_log is None:
            return
        try:
            self._log_to_file(self.error_log, snapshot.to_plain_english() + "\n\n")
        except Exception:
            pass

    def _write_error_to_debug_log(self, snapshot: ErrorSnapshot):
        block = "\n" + snapshot.to_plain_english() + "\n"
        try:
            if self.file_output and self.debug_log is not None:
                self._log_to_file(self.debug_log, block)
        except Exception:
            pass
        sl = getattr(self, "_shared_logger", None)
        if sl:
            try:
                sl.error(block)
                if hasattr(sl, "flush"):
                    sl.flush()
            except Exception:
                pass

    def _write_metric_to_file(self, metric: PerformanceMetric):
        if not self.file_output or self.debug_log is None:
            return
        try:
            line = f"[METRIC] {metric.timestamp} | {metric.metric_name}: {metric.value:.6f} {metric.unit}"
            if metric.context:
                context_str = " | ".join([f"{k}={v}" for k, v in list(metric.context.items())[:3]])
                line += f" | {context_str}"
            line += "\n"
            self._log_to_file(self.debug_log, line)
            # Mirror compact line at DEBUG level to shared logger
            sl = getattr(self, "_shared_logger", None)
            if sl:
                try:
                    if hasattr(sl, "debug"):
                        sl.debug(line.rstrip("\n"))
                    else:
                        sl.info(line.rstrip("\n"))
                except Exception:
                    pass
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # CONSOLE OUTPUT
    # -------------------------------------------------------------------------
    def _print_decision(self, snapshot: DecisionSnapshot):
        if self.verbosity.priority <= DebugLevel.INFO.priority:
            print(snapshot.to_plain_english())

    def _print_error(self, snapshot: ErrorSnapshot):
        print(snapshot.to_plain_english())

    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------
    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            runtime = (datetime.datetime.utcnow() - self.stats["start_time"]).total_seconds()
            return {
                "runtime_seconds": runtime,
                "runtime_hours": runtime / 3600,
                "total_decisions": self.stats["total_decisions"],
                "buy_decisions": self.stats["buy_decisions"],
                "sell_decisions": self.stats["sell_decisions"],
                "hold_decisions": self.stats["hold_decisions"],
                "successful_executions": self.stats["successful_executions"],
                "blocked_executions": self.stats["blocked_executions"],
                "execution_rate": self.stats["successful_executions"] / max(self.stats["total_decisions"], 1) * 100,
                "block_rate": self.stats["blocked_executions"] / max(self.stats["total_decisions"], 1) * 100,
                "total_errors": self.stats["total_errors"],
                "decisions_per_hour": self.stats["total_decisions"] / max(runtime / 3600, 0.001),
                "signal_stats": dict(self.signal_stats),
            }

    def print_summary(self):
        stats = self.get_statistics()
        print("\n" + "="*80)
        print("📊 POSITION DEBUG SYSTEM SUMMARY")
        print("="*80)
        print(f"Runtime: {stats['runtime_hours']:.2f} hours")
        print(f"Total Decisions: {stats['total_decisions']}")
        print(f"  • Buy Decisions: {stats['buy_decisions']}")
        print(f"  • Sell Decisions: {stats['sell_decisions']}")
        print(f"  • Hold Decisions: {stats['hold_decisions']}")
        print(f"Execution Rate: {stats['execution_rate']:.1f}%")
        print(f"Block Rate: {stats['block_rate']:.1f}%")
        print(f"Total Errors: {stats['total_errors']}")
        print(f"Decision Rate: {stats['decisions_per_hour']:.1f}/hour")
        print("="*80 + "\n")

    def flush(self) -> None:
        """Flush shared logger (files are flushed per write)."""
        sl = getattr(self, "_shared_logger", None)
        try:
            if sl and hasattr(sl, "flush"):
                sl.flush()
        except Exception:
            pass

    def flush_signals_to_json(self):
        if not self.enabled or not self.file_output:
            return
        try:
            buy_signals = [
                s.__dict__ for s in self.trade_signals
                if s.action in ("BUY", "SCALE_UP") and s.timestamp
            ]
            sell_signals = [
                s.__dict__ for s in self.trade_signals
                if s.action == "SELL" and s.timestamp
            ]
            # Convert timestamps
            for sig in buy_signals + sell_signals:
                if isinstance(sig.get('timestamp'), datetime.datetime):
                    sig['timestamp'] = sig['timestamp'].isoformat() + "Z"

            if self.buy_signals_json and buy_signals:
                with open(self.buy_signals_json, 'w', encoding='utf-8', newline='') as f:
                    json.dump(buy_signals, f, indent=2, default=str)
            if self.sell_signals_json and sell_signals:
                with open(self.sell_signals_json, 'w', encoding='utf-8', newline='') as f:
                    json.dump(sell_signals, f, indent=2, default=str)
        except Exception as e:
            self._log_debug(DebugLevel.ERROR, f"Error flushing signals: {e}")

    def cleanup(self):
        if not self.enabled:
            return
        self._log_debug(DebugLevel.INFO, "Cleaning up debug system...")
        self.flush_signals_to_json()
        stats = self.get_statistics()

        # Session footer
        if self.file_output and self.log_dir and self.debug_log:
            footer = f"""

{'='*80}
📊 SESSION SUMMARY
{'='*80}
Runtime: {stats['runtime_hours']:.2f} hours
Total Decisions: {stats['total_decisions']}
  🟢 Buy Decisions: {stats['buy_decisions']}
  🔴 Sell Decisions: {stats['sell_decisions']}
  ⏸️  Hold Decisions: {stats['hold_decisions']}
Execution Rate: {stats['execution_rate']:.1f}%
Block Rate: {stats['block_rate']:.1f}%
Total Errors: {stats['total_errors']}
Decision Rate: {stats['decisions_per_hour']:.1f}/hour
{'='*80}
🎯 Session ended successfully
{'='*80}

"""
            self._log_to_file(self.debug_log, footer)
            # JSON summary
            summary_path = self.log_dir / "final_summary.json"
            try:
                with open(summary_path, 'w', encoding='utf-8', newline='') as f:
                    json.dump(stats, f, indent=2, default=str)
            except Exception:
                pass

        self._log_debug(DebugLevel.SUCCESS, "✅ Debug system cleanup complete")
