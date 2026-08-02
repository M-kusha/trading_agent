# -------------------------------------------------------------
# File: modules/position/position_debug.py
# Position Manager Debug System (Hardened & Mirrored)
#
# Responsibilities:
#   - Track every decision with full context (per instrument)
#   - Emit human-readable, plain-English explanations
#   - Persist CSV / TXT / JSON artifacts for forensics
#   - Mirror messages into a shared RotatingLogger (if attached)
#   - Stay thread-safe and cheap in the hot path
# -------------------------------------------------------------

from __future__ import annotations

import datetime
import json
import threading
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
    """Debug verbosity levels with priorities (lower = more verbose)."""

    TRACE = ("TRACE", "🔍", 0)        # Ultra-detailed internal state
    DEBUG = ("DEBUG", "🐛", 10)       # Detailed debugging info
    INFO = ("INFO", "ℹ️", 20)        # General informational messages
    SUCCESS = ("SUCCESS", "✅", 25)   # Successful operations
    WARNING = ("WARNING", "⚠️", 30)   # Warning conditions
    ERROR = ("ERROR", "❌", 40)       # Error conditions
    CRITICAL = ("CRITICAL", "🚨", 50) # Critical failures

    def __init__(self, label: str, icon: str, priority: int):
        self.label = label
        self.icon = icon
        self.priority = priority


@dataclass
class DecisionSnapshot:
    """
    Complete snapshot of a position decision with full context.

    This is the “atomic record” of what the PositionManager decided
    for a single instrument at a single time step.
    """

    # Identification
    timestamp: str
    instrument: str

    # Action details
    action: str                       # e.g. BUY / SELL / HOLD / CLOSE_POSITION
    is_buying: bool                  # True if action is in BUY direction
    direction: str                   # LONG / SHORT / NEUTRAL / EXIT / CLOSING

    # Sizing & confidence
    size_eur: float
    confidence: float
    market_intensity: float
    current_price: float

    # Technical metrics
    signal_strength: float
    trend_direction: str             # BULLISH / BEARISH / NEUTRAL
    volatility: float

    # Portfolio state
    portfolio_health: float          # 0–1
    risk_score: float                # 0–1

    # Explanations
    plain_english_reason: str
    technical_factors: List[str]
    risk_factors: List[str]

    # Execution
    will_execute: bool
    execution_blocked_reason: Optional[str] = None

    # Optional extended context (only at low verbosity)
    raw_context: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def _format_action(self) -> str:
        """Human-friendly short description of the action."""
        a = self.action
        if a == "BUY":
            return "🟢 OPEN LONG"
        if a == "SELL":
            return "🔴 OPEN SHORT"
        if a == "SCALE_UP":
            return "🟢 ADD TO POSITION" if self.is_buying else "🔴 ADD TO POSITION"
        if a == "SCALE_DOWN":
            return "📉 REDUCE POSITION"
        if a == "CLOSE_POSITION":
            return "🔒 CLOSE POSITION"
        if a == "EMERGENCY_EXIT":
            return "🚨 EMERGENCY EXIT"
        if a == "HOLD":
            return "⏸ HOLD"
        return f"⏸ {a}"

    def to_plain_english(self) -> str:
        """Compact human-readable explanation block."""
        action_line = self._format_action()
        status = (
            "✅ WILL EXECUTE"
            if self.will_execute
            else f"❌ BLOCKED: {self.execution_blocked_reason}"
        )

        lines = [
            "=" * 80,
            f"📍 {self.instrument}  |  {action_line}",
            "=" * 80,
            f"💰 Size:          €{self.size_eur:,.2f}",
            f"📊 Confidence:    {self.confidence:.1%}",
            f"📈 Signal:        {self.signal_strength:.3f} ({self.trend_direction})",
            f"⚡ Volatility:    {self.volatility:.4f}",
            f"🏥 Port. Health:  {self.portfolio_health:.1%}",
            f"⚠️  Risk Score:    {self.risk_score:.1%}",
            f"💵 Price:         {self.current_price:.5f}",
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
        lines.append(f"Status: {status}")
        lines.append("=" * 80)
        return "\n".join(lines)


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
        try:
            ctx_str = json.dumps(self.context, indent=2, default=str)[:300]
        except Exception:
            ctx_str = str(self.context)[:300]

        lines = [
            "!" * 80,
            "🚨 ERROR DETECTED",
            "!" * 80,
            f"Time:      {self.timestamp}",
            f"Component: {self.component}",
            f"Type:      {self.error_type}",
            "",
            "What went wrong:",
            f"  {self.error_message}",
            "",
            "Stack trace (truncated):",
            f"{self.stack_trace[:300]}...",
            "",
            "Context (truncated):",
            f"{ctx_str}...",
            "",
            f"Recovery: {self.recovery_action or 'Manual intervention needed'}",
            "!" * 80,
        ]
        return "\n".join(lines)


@dataclass
class TradeSignal:
    """Simple record of a trade signal that could be executed."""

    timestamp: datetime.datetime
    instrument: str
    action: str          # BUY / SELL / SCALE_UP
    size_eur: float
    confidence: float
    executed: bool
    price: float
    pnl: Optional[float] = None


@dataclass
class PerformanceMetric:
    """One metric sample for monitoring (drawdown, exposure, etc.)."""

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

    Design goals:
      - Very easy to reason about decisions when reading logs.
      - Safe in hot paths (thread-safe, cheap when disabled).
      - Minimal API surface from the outside:
          * log_decision(...)
          * log_error(...)
          * log_metric(...)
          * get_statistics()
          * cleanup()
    """

    # CSV header constants – kept here for easy grepping
    _DECISION_CSV_HEADER = (
        "timestamp,instrument,action,is_buying,direction,size_eur,confidence,"
        "signal_strength,volatility,portfolio_health,risk_score,will_execute,"
        "blocked_reason,plain_english_reason\n"
    )
    _SIGNAL_CSV_HEADER = (
        "timestamp,instrument,action,size_eur,confidence,executed,price,pnl\n"
    )

    def __init__(
        self,
        log_dir: str = "logs/debug",
        enable: bool = True,
        verbosity: DebugLevel = DebugLevel.INFO,
        console_output: bool = True,
        file_output: bool = True,
        max_memory_items: int = 10000,
    ):
        # Configuration
        self.enabled = enable
        self.verbosity = verbosity
        self.console_output = console_output
        self.file_output = file_output
        self.max_memory_items = max_memory_items

        # Thread safety
        self._lock = threading.RLock()

        # Optional external sink (e.g. RotatingLogger), wired via attach_shared_logger()
        self._shared_logger = None

        # In-memory storage (bounded)
        self.decision_history: Deque[DecisionSnapshot] = deque(maxlen=max_memory_items)
        self.error_history: Deque[ErrorSnapshot] = deque(maxlen=1000)
        self.performance_metrics: Deque[PerformanceMetric] = deque(maxlen=5000)
        self.trade_signals: Deque[TradeSignal] = deque(maxlen=1000)

        # Stats
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

        # Per-instrument signal stats
        self.active_signals: Dict[str, TradeSignal] = {}
        self.signal_stats = defaultdict(
            lambda: {
                "total_buys": 0,
                "total_sells": 0,
                "total_holds": 0,
                "buy_volume": 0.0,
                "sell_volume": 0.0,
                "win_count": 0,
                "loss_count": 0,
                "total_pnl": 0.0,
            }
        )

        # File paths (initialised lazily)
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
        """Attach shared RotatingLogger to mirror lines and blocks."""
        with self._lock:
            self._shared_logger = logger

    # -------------------------------------------------------------------------
    # File initialization
    # -------------------------------------------------------------------------

    def _init_log_files(self, log_dir: str) -> None:
        """Create directories and base files for this debug session."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories
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

        # CSV headers
        self._write_csv_header(self.decision_csv, self._DECISION_CSV_HEADER)
        self._write_csv_header(self.signals_csv, self._SIGNAL_CSV_HEADER)

        # Session header
        header = f"""
{'=' * 80}
🎯 POSITION MANAGER DEBUG SESSION
{'=' * 80}
Session Started: {timestamp}
Verbosity Level: {self.verbosity.label} {self.verbosity.icon}
Console Output: {'Enabled' if self.console_output else 'Disabled'}
File Output: {'Enabled' if self.file_output else 'Disabled'}
Log Directory: {self.log_dir}
{'=' * 80}

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
        """
        Log a decision taken by the PositionManager.

        Parameters are intentionally minimal: instrument, decision enum value,
        a scalar intensity, size in EUR, confidence, and a small context dict.
        """
        if not self.enabled:
            return None

        try:
            # 1) Parse high-level action semantics
            action, is_buying, direction_label = self._parse_decision(decision, intensity)

            # 2) Extract metrics from context
            signal_strength = abs(float(intensity))
            trend_direction = (
                "BULLISH" if intensity > 0 else "BEARISH" if intensity < 0 else "NEUTRAL"
            )
            volatility = max(float(context.get("volatility", 0.02) or 0.02), 1e-6)
            current_price = float(context.get("current_price", 0.0) or 0.0)

            risk_score = self._calculate_risk_score(context)

            # 3) Generate explanations
            plain_english = self._generate_plain_english_reason(
                action=action,
                is_buying=is_buying,
                instrument=instrument,
                signal_strength=signal_strength,
                trend=trend_direction,
                confidence=confidence,
                rationale=rationale,
            )
            technical_factors = self._extract_technical_factors(context)
            risk_factors_list = self._extract_risk_factors(
                context, rationale.get("risk_factors", {})
            )

            # 4) Check if this decision should be executed at all (log-level check)
            will_execute, blocked_reason = self._check_execution_viability(
                size, confidence, context, risk_score=risk_score
            )

            # 5) Build snapshot
            snapshot = DecisionSnapshot(
                timestamp=datetime.datetime.utcnow().isoformat() + "Z",
                instrument=instrument,
                action=action,
                is_buying=is_buying,
                direction=direction_label,
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
                risk_factors=risk_factors_list,
                will_execute=will_execute,
                execution_blocked_reason=blocked_reason,
                raw_context=(
                    context if self.verbosity.priority <= DebugLevel.DEBUG.priority else None
                ),
            )

            # 6) Persist & mirror
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

            # TRACE-only compact mirror
            if self.verbosity.priority <= DebugLevel.TRACE.priority:
                self._log_debug(
                    DebugLevel.TRACE,
                    f"Decision: {instrument} | {action} | Size €{size:.2f} | Conf {confidence:.1%}",
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
        """Log an error with a full snapshot (stack trace + context)."""
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
                f"Error in {component}: {type(error).__name__} - {str(error)[:100]}",
            )
            return snapshot

        except Exception as e:
            # Final fallback – do not swallow meta-errors
            print(f"Critical: Error while logging error: {e}")
            return None

    def log_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a performance / health metric (lightweight)."""
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

            self._log_debug(
                DebugLevel.TRACE, f"Metric logged: {metric_name}={value}{unit}"
            )
        except Exception as e:
            self._log_debug(DebugLevel.ERROR, f"Error logging metric: {e}")

    # Convenience wrappers
    def log_trace(self, message: str, **kwargs) -> None:
        self._log_debug(DebugLevel.TRACE, message, **kwargs)

    def log_debug(self, message: str, **kwargs) -> None:
        self._log_debug(DebugLevel.DEBUG, message, **kwargs)

    def log_info(self, message: str, **kwargs) -> None:
        self._log_debug(DebugLevel.INFO, message, **kwargs)

    def log_warning(self, message: str, **kwargs) -> None:
        self._log_debug(DebugLevel.WARNING, message, **kwargs)

    def log_success(self, message: str, **kwargs) -> None:
        self._log_debug(DebugLevel.SUCCESS, message, **kwargs)

    # -------------------------------------------------------------------------
    # INTERNAL HELPERS – logging + formatting
    # -------------------------------------------------------------------------

    def _log_debug(self, level: DebugLevel, message: str, **kwargs) -> None:
        """
        Internal debug logger.

        - Applies verbosity filter.
        - Writes to console (if enabled).
        - Writes to debug log file (if enabled).
        - Mirrors into shared RotatingLogger (if attached).
        """
        if not self.enabled or level.priority < self.verbosity.priority:
            return

        timestamp = datetime.datetime.utcnow().isoformat() + "Z"
        formatted = f"[{timestamp}] {level.icon} {level.label}: {message}"
        if kwargs:
            formatted += f" | {kwargs}"

        # Console
        if self.console_output:
            print(formatted)

        # File
        if self.file_output and self.debug_log:
            self._log_to_file(self.debug_log, formatted + "\n")

        # Mirror into shared logger
        sl = getattr(self, "_shared_logger", None)
        if not sl:
            return
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
            # Never allow debug propagation to break hot path
            pass

    @staticmethod
    def _parse_decision(decision: str, intensity: float) -> Tuple[str, bool, str]:
        """
        Map PositionDecision enum value to a normalized action string.

        Returns: (action, is_buying, direction_label)
        """
        d = (decision or "").lower()
        if "open_long" in d:
            return "BUY", True, "LONG"
        if "open_short" in d:
            return "SELL", False, "SHORT"
        if "scale_up" in d:
            is_buying = intensity > 0
            return "SCALE_UP", is_buying, "LONG" if is_buying else "SHORT"
        if "scale_down" in d:
            return "SCALE_DOWN", False, "REDUCING"
        if "emergency_close" in d:
            return "EMERGENCY_EXIT", False, "EXIT"
        if "close" in d:
            return "CLOSE_POSITION", False, "CLOSING"
        return "HOLD", False, "NEUTRAL"

    @staticmethod
    def _calculate_risk_score(context: Dict[str, Any]) -> float:
        """
        Aggregate a simple 0–1 risk score from drawdown, exposure, volatility.

        This is intentionally simple and monotonic:
          - high drawdown, exposure, or volatility pushes risk towards 1.
        """
        drawdown = float(context.get("drawdown", 0.0) or 0.0)
        exposure = float(context.get("current_exposure", 0.0) or 0.0)
        volatility = float(context.get("volatility", 0.02) or 0.02)

        drawdown_risk = min(drawdown * 5.0, 1.0)         # 20% dd -> 1.0
        exposure_risk = min(exposure * 2.0, 1.0)         # 50% exposure -> 1.0
        vol_risk = min(volatility / 0.05, 1.0)           # 5% vol -> 1.0

        score = drawdown_risk * 0.4 + exposure_risk * 0.3 + vol_risk * 0.3
        return float(min(score, 1.0))

    @staticmethod
    def _generate_plain_english_reason(
        action: str,
        is_buying: bool,
        instrument: str,
        signal_strength: float,
        trend: str,
        confidence: float,
        rationale: Dict[str, Any],
    ) -> str:
        """Generate a short plain-English summary sentence for the decision."""
        stage = rationale.get("stage", "unknown")
        factors = rationale.get("factors", [])

        if action == "BUY":
            reason = (
                f"Opening LONG on {instrument} because the signal is {trend.lower()} "
                f"({signal_strength:.2f}) with {confidence:.1%} confidence"
            )
        elif action == "SELL":
            reason = (
                f"Opening SHORT on {instrument} because the signal is {trend.lower()} "
                f"({signal_strength:.2f}) with {confidence:.1%} confidence"
            )
        elif action == "SCALE_UP":
            direction = "LONG" if is_buying else "SHORT"
            reason = (
                f"Adding to existing {direction} position on {instrument} "
                f"as conditions continue to support the trade"
            )
        elif action == "SCALE_DOWN":
            reason = (
                f"Reducing position on {instrument} to manage risk under current conditions"
            )
        elif action == "CLOSE_POSITION":
            if "risk" in stage:
                reason = f"Closing {instrument} position due to risk management rules"
            elif "reverse" in stage:
                reason = f"Closing {instrument} position because the market reversed"
            else:
                reason = f"Closing {instrument} position to lock in results or exit exposure"
        elif action == "EMERGENCY_EXIT":
            reason = f"Emergency exit from {instrument} due to critical risk conditions"
        else:
            reason = (
                f"Holding {instrument} position because signals are not strong "
                f"enough to justify action"
            )

        if factors:
            reason += f". Key factors: {'; '.join(factors[:3])}"
        return reason

    @staticmethod
    def _extract_technical_factors(context: Dict[str, Any]) -> List[str]:
        """Translate raw context values into compact, human-readable tags."""
        factors: List[str] = []

        trend_strength = float(context.get("trend_strength", 0.0) or 0.0)
        if abs(trend_strength) > 0.5:
            direction = "uptrend" if trend_strength > 0 else "downtrend"
            factors.append(f"Strong {direction} (trend={abs(trend_strength):.2f})")

        momentum = float(context.get("momentum", 0.0) or 0.0)
        if abs(momentum) > 0.3:
            momentum_dir = "positive" if momentum > 0 else "negative"
            factors.append(f"Momentum is {momentum_dir} ({momentum:.2f})")

        rsi = float(context.get("rsi", 50.0) or 50.0)
        if rsi > 70:
            factors.append(f"Overbought (RSI={rsi:.0f})")
        elif rsi < 30:
            factors.append(f"Oversold (RSI={rsi:.0f})")

        volatility = float(context.get("volatility", 0.02) or 0.02)
        if volatility > 0.04:
            factors.append(f"High volatility ({volatility:.4f})")
        elif volatility < 0.01:
            factors.append(f"Low volatility ({volatility:.4f})")

        volume = float(context.get("volume_profile", 1.0) or 1.0)
        if volume > 1.5:
            factors.append(f"High volume ({volume:.1f}x normal)")
        elif volume < 0.5:
            factors.append(f"Low volume ({volume:.1f}x normal)")

        return factors if factors else ["Normal market conditions"]

    @staticmethod
    def _extract_risk_factors(
        context: Dict[str, Any],
        risk_factors: Dict[str, float],
    ) -> List[str]:
        """Translate risk context into human-readable risk notes."""
        factors: List[str] = []

        drawdown = float(context.get("drawdown", 0.0) or 0.0)
        if drawdown > 0.05:
            factors.append(f"Portfolio drawdown at {drawdown:.1%}")

        exposure = float(context.get("current_exposure", 0.0) or 0.0)
        if exposure > 0.5:
            factors.append(f"High exposure ({exposure:.1%} of capital)")

        correlation = float(risk_factors.get("correlation", 0.0) or 0.0)
        if correlation > 0.5:
            factors.append(f"High correlation with other positions ({correlation:.1%})")

        session = context.get("session", "unknown")
        if session == "closed":
            factors.append("Market closed – spread / liquidity risk")
        elif session == "asian":
            factors.append("Asian session – typically lower liquidity")

        vol_risk = float(risk_factors.get("volatility", 0.0) or 0.0)
        if vol_risk > 0.3:
            factors.append(f"Elevated volatility risk ({vol_risk:.1%})")

        return factors if factors else ["Risk levels are acceptable"]

    def _check_execution_viability(
        self,
        size: float,
        confidence: float,
        context: Dict[str, Any],
        risk_score: Optional[float] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Decide whether this decision should realistically be executed.

        This is an additional safety layer used only for logging:
          - zero / tiny size → block,
          - too large relative to balance → block,
          - low confidence → block,
          - very high risk score or drawdown → block.
        """
        if size <= 0:
            return False, "Position size is zero or negative"

        balance = float(context.get("balance", 0.0) or 0.0)
        if balance > 0 and size > balance * 0.5:
            return False, f"Position size (€{size:.2f}) exceeds 50% of balance"

        if confidence < 0.3:
            return False, f"Confidence too low ({confidence:.1%})"

        risk = float(risk_score) if risk_score is not None else self._calculate_risk_score(context)
        if risk > 0.8:
            return False, f"Risk score too high ({risk:.1%})"

        drawdown = float(context.get("drawdown", 0.0) or 0.0)
        if drawdown > 0.15:
            return False, f"Drawdown limit exceeded ({drawdown:.1%})"

        return True, None

    # -------------------------------------------------------------------------
    # INTERNAL HELPERS – signal tracking / stats
    # -------------------------------------------------------------------------

    def _track_trade_signal(self, snapshot: DecisionSnapshot) -> None:
        """Track BUY / SELL / SCALE_UP decisions for later signal stats."""
        if snapshot.action not in ("BUY", "SELL", "SCALE_UP"):
            return

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

        stats = self.signal_stats[snapshot.instrument]
        if snapshot.action in ("BUY", "SCALE_UP") and snapshot.is_buying:
            stats["total_buys"] += 1
            stats["buy_volume"] += snapshot.size_eur
        elif snapshot.action in ("SELL", "SCALE_UP") and not snapshot.is_buying:
            stats["total_sells"] += 1
            stats["sell_volume"] += snapshot.size_eur

        # Persist to signals CSV for fast forensic analysis
        if self.file_output and self.signals_csv is not None:
            try:
                ts = signal.timestamp.isoformat() + "Z"
                pnl_val = "" if signal.pnl is None else f"{signal.pnl:.2f}"
                line = (
                    f"{ts},{signal.instrument},{signal.action},"
                    f"{signal.size_eur:.2f},{signal.confidence:.4f},"
                    f"{signal.executed},{signal.price:.5f},{pnl_val}\n"
                )
                self._log_to_file(self.signals_csv, line)
            except Exception:
                # Never let debug I/O break trading
                pass

    def _update_stats(self, snapshot: DecisionSnapshot) -> None:
        """Update global statistics from a new decision snapshot."""
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

    def _write_csv_header(self, filepath: Optional[Path], header: str) -> None:
        if not self.file_output or filepath is None:
            return
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                new_file = not filepath.exists()
                if not new_file:
                    return
                with open(filepath, "a", encoding="utf-8", newline="") as f:
                    f.write(header)
                    f.flush()
        except Exception:
            pass

    def _log_to_file(self, filepath: Optional[Path], content: str) -> None:
        if not self.file_output or filepath is None:
            return
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                with open(filepath, "a", encoding="utf-8", newline="") as f:
                    f.write(content)
                    f.flush()
        except Exception:
            pass

    def _write_decision_to_csv(self, snapshot: DecisionSnapshot) -> None:
        if not self.file_output or self.decision_csv is None:
            return
        try:
            blocked_reason = (snapshot.execution_blocked_reason or "").replace('"', "'")
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

    def _write_decision_to_summary(self, snapshot: DecisionSnapshot) -> None:
        if not self.file_output or self.decision_summary is None:
            return
        try:
            self._log_to_file(self.decision_summary, snapshot.to_plain_english() + "\n\n")
        except Exception:
            pass

    def _write_decision_to_debug_log(self, snapshot: DecisionSnapshot) -> None:
        """
        Write the pretty-block decision only when verbosity is high enough.

        The unified logger already produces high-level, human-readable summaries;
        this block is for deeper inspection sessions.
        """
        if self.verbosity.priority > DebugLevel.DEBUG.priority:
            return
        if not self.file_output or self.debug_log is None:
            return

        block = "\n" + snapshot.to_plain_english() + "\n"
        self._log_to_file(self.debug_log, block)

    def _write_error_to_file(self, snapshot: ErrorSnapshot) -> None:
        if not self.file_output or self.error_log is None:
            return
        try:
            self._log_to_file(self.error_log, snapshot.to_plain_english() + "\n\n")
        except Exception:
            pass

    def _write_error_to_debug_log(self, snapshot: ErrorSnapshot) -> None:
        block = "\n" + snapshot.to_plain_english() + "\n"
        try:
            if self.file_output and self.debug_log is not None:
                self._log_to_file(self.debug_log, block)
            sl = getattr(self, "_shared_logger", None)
            if sl:
                try:
                    sl.error(block)
                    if hasattr(sl, "flush"):
                        sl.flush()
                except Exception:
                    pass
        except Exception:
            pass

    def _write_metric_to_file(self, metric: PerformanceMetric) -> None:
        if not self.file_output:
            return
        try:
            # Human-readable line into debug log
            line = (
                f"[METRIC] {metric.timestamp} | {metric.metric_name}: "
                f"{metric.value:.6f} {metric.unit}"
            )
            if metric.context:
                context_str = " | ".join(
                    [f"{k}={v}" for k, v in list(metric.context.items())[:3]]
                )
                line += f" | {context_str}"
            line += "\n"
            if self.debug_log is not None:
                self._log_to_file(self.debug_log, line)

            # Structured JSON into metrics file
            if self.metrics_json is not None:
                payload = {
                    "timestamp": metric.timestamp,
                    "metric_name": metric.metric_name,
                    "value": metric.value,
                    "unit": metric.unit,
                    "context": metric.context,
                }
                self._log_to_file(
                    self.metrics_json,
                    json.dumps(payload, default=str) + "\n",
                )

            # Mirror compact line to shared logger
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

    @staticmethod
    def _print_decision(snapshot: DecisionSnapshot) -> None:
        print(snapshot.to_plain_english())

    @staticmethod
    def _print_error(snapshot: ErrorSnapshot) -> None:
        print(snapshot.to_plain_english())

    # -------------------------------------------------------------------------
    # PUBLIC API – stats / lifecycle
    # -------------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return a snapshot of high-level debug statistics."""
        with self._lock:
            runtime = (datetime.datetime.utcnow() - self.stats["start_time"]).total_seconds()
            total_decisions = max(self.stats["total_decisions"], 1)

            return {
                "runtime_seconds": runtime,
                "runtime_hours": runtime / 3600.0,
                "total_decisions": self.stats["total_decisions"],
                "buy_decisions": self.stats["buy_decisions"],
                "sell_decisions": self.stats["sell_decisions"],
                "hold_decisions": self.stats["hold_decisions"],
                "successful_executions": self.stats["successful_executions"],
                "blocked_executions": self.stats["blocked_executions"],
                "execution_rate": (
                    self.stats["successful_executions"] / total_decisions * 100.0
                ),
                "block_rate": (
                    self.stats["blocked_executions"] / total_decisions * 100.0
                ),
                "total_errors": self.stats["total_errors"],
                "decisions_per_hour": (
                    self.stats["total_decisions"] / max(runtime / 3600.0, 0.001)
                ),
                "signal_stats": dict(self.signal_stats),
            }

    def print_summary(self) -> None:
        """Pretty-print a short stats summary to stdout."""
        stats = self.get_statistics()
        print("\n" + "=" * 80)
        print("📊 POSITION DEBUG SYSTEM SUMMARY")
        print("=" * 80)
        print(f"Runtime:           {stats['runtime_hours']:.2f} hours")
        print(f"Total Decisions:   {stats['total_decisions']}")
        print(f"  • Buy Decisions: {stats['buy_decisions']}")
        print(f"  • Sell Decisions:{stats['sell_decisions']}")
        print(f"  • Hold Decisions:{stats['hold_decisions']}")
        print(f"Execution Rate:    {stats['execution_rate']:.1f}%")
        print(f"Block Rate:        {stats['block_rate']:.1f}%")
        print(f"Total Errors:      {stats['total_errors']}")
        print(f"Decision Rate:     {stats['decisions_per_hour']:.1f} / hour")
        print("=" * 80 + "\n")

    def flush(self) -> None:
        """Flush shared logger (file writes are already flushed per write)."""
        sl = getattr(self, "_shared_logger", None)
        try:
            if sl and hasattr(sl, "flush"):
                sl.flush()
        except Exception:
            pass

    def flush_signals_to_json(self) -> None:
        """Persist trade signal history into JSON artifacts (BUY / SELL)."""
        if not self.enabled or not self.file_output:
            return
        try:
            buy_signals = [
                s.__dict__
                for s in self.trade_signals
                if s.action in ("BUY", "SCALE_UP") and s.timestamp
            ]
            sell_signals = [
                s.__dict__
                for s in self.trade_signals
                if s.action == "SELL" and s.timestamp
            ]

            # Normalise timestamps
            for sig in buy_signals + sell_signals:
                ts = sig.get("timestamp")
                if isinstance(ts, datetime.datetime):
                    sig["timestamp"] = ts.isoformat() + "Z"

            if self.buy_signals_json and buy_signals:
                with open(self.buy_signals_json, "w", encoding="utf-8", newline="") as f:
                    json.dump(buy_signals, f, indent=2, default=str)

            if self.sell_signals_json and sell_signals:
                with open(self.sell_signals_json, "w", encoding="utf-8", newline="") as f:
                    json.dump(sell_signals, f, indent=2, default=str)
        except Exception as e:
            self._log_debug(DebugLevel.ERROR, f"Error flushing signals: {e}")

    def cleanup(self) -> None:
        """
        Finalise the debug session:
          - flush signals into JSON,
          - write a final summary footer to logs,
          - dump a JSON summary file.
        """
        if not self.enabled:
            return

        self._log_debug(DebugLevel.INFO, "Cleaning up PositionDebugSystem...")
        self.flush_signals_to_json()
        stats = self.get_statistics()

        # Session footer
        if self.file_output and self.log_dir and self.debug_log:
            footer = f"""

{'=' * 80}
📊 SESSION SUMMARY
{'=' * 80}
Runtime:           {stats['runtime_hours']:.2f} hours
Total Decisions:   {stats['total_decisions']}
  🟢 Buy Decisions:   {stats['buy_decisions']}
  🔴 Sell Decisions:  {stats['sell_decisions']}
  ⏸ Hold Decisions:   {stats['hold_decisions']}
Execution Rate:    {stats['execution_rate']:.1f}%
Block Rate:        {stats['block_rate']:.1f}%
Total Errors:      {stats['total_errors']}
Decision Rate:     {stats['decisions_per_hour']:.1f} / hour
{'=' * 80}
🎯 Session ended successfully
{'=' * 80}

"""
            self._log_to_file(self.debug_log, footer)

            # JSON summary
            summary_path = self.log_dir / "final_summary.json"
            try:
                with open(summary_path, "w", encoding="utf-8", newline="") as f:
                    json.dump(stats, f, indent=2, default=str)
            except Exception:
                pass

        self._log_debug(DebugLevel.SUCCESS, "✅ Debug system cleanup complete")
