# -------------------------------------------------------------
# File: modules/position/position_logger.py
# Unified Position Manager Logging System
#
# Consolidates all position decision logging into clean,
# professional, well-organized summaries with voting signals
# -------------------------------------------------------------

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from modules.utils.info_bus import InfoBusManager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.utils.info_bus import SmartInfoBus


@dataclass
class PositionLogEntry:
    """Complete position decision log entry with all context"""
    # Core decision
    instrument: str
    decision: str
    intensity: float
    size_eur: float
    confidence: float

    # Market context
    signal_strength: float
    volatility: float
    trend_strength: float
    current_price: float

    # Portfolio state
    portfolio_health: float
    exposure_ratio: float
    balance: float
    drawdown: float
    risk_score: float

    # Voting signals (optional)
    committee_consensus: Optional[Dict[str, Any]] = None
    trade_vote: Optional[Dict[str, Any]] = None
    consensus_strength: Optional[float] = None

    # Decision rationale
    stage: str = "unknown"
    factors: Optional[List[str]] = None
    risk_factors: Optional[Dict[str, float]] = None

    # Execution
    will_execute: bool = True
    blocked_reason: Optional[str] = None

    def __post_init__(self):
        if self.factors is None:
            self.factors = []
        if self.risk_factors is None:
            self.risk_factors = {}


class UnifiedPositionLogger:
    """
    Unified logging system for position decisions.

    Creates clean, professional log summaries with:
    - Decision summary
    - Market analysis
    - Portfolio state
    - Voting signals
    - Risk assessment
    - Execution status
    """

    def __init__(self, logger, smart_bus: Optional[Any] = None):
        self.logger = logger
        self.smart_bus = smart_bus if smart_bus is not None else InfoBusManager.get_instance()
        self.session_start = dt.datetime.utcnow()
        self.decision_count = 0

    def log_decision_summary(self, entry: PositionLogEntry) -> None:
        """
        Log a complete, unified decision summary with all context.

        Format:
        ╔══════════════════════════════════════════════════════════════════╗
        ║                    POSITION DECISION SUMMARY                     ║
        ╠══════════════════════════════════════════════════════════════════╣
        ║ [sections with clean formatting]                                 ║
        ╚══════════════════════════════════════════════════════════════════╝
        """
        self.decision_count += 1
        timestamp = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        # Build the unified log
        lines = []
        lines.append("")
        lines.append("╔" + "═" * 78 + "╗")
        lines.append("║" + "POSITION DECISION SUMMARY".center(78) + "║")
        lines.append("╠" + "═" * 78 + "╣")

        # Section 1: Decision Overview
        lines.append("║ " + "📊 DECISION OVERVIEW".ljust(77) + "║")
        lines.append("║ " + "─" * 77 + "║")
        lines.append("║ " + f"Instrument:  {entry.instrument}".ljust(77) + "║")
        lines.append("║ " + f"Decision:    {self._format_decision(entry.decision)} ({entry.decision})".ljust(77) + "║")
        lines.append("║ " + f"Size:        €{entry.size_eur:,.2f}".ljust(77) + "║")
        lines.append("║ " + f"Confidence:  {entry.confidence:.1%} {'█' * int(entry.confidence * 20)}".ljust(77) + "║")
        lines.append("║ " + f"Timestamp:   {timestamp}".ljust(77) + "║")
        lines.append("║" + " " * 78 + "║")

        # Section 2: Market Analysis
        lines.append("║ " + "📈 MARKET ANALYSIS".ljust(77) + "║")
        lines.append("║ " + "─" * 77 + "║")
        lines.append("║ " + f"Signal Strength:  {entry.signal_strength:+.3f} {self._get_signal_bar(entry.signal_strength)}".ljust(77) + "║")
        lines.append("║ " + f"Trend Strength:   {entry.trend_strength:+.3f}".ljust(77) + "║")
        lines.append("║ " + f"Volatility:       {entry.volatility:.4f} ({self._volatility_level(entry.volatility)})".ljust(77) + "║")
        lines.append("║ " + f"Current Price:    {entry.current_price:.5f}".ljust(77) + "║")
        lines.append("║" + " " * 78 + "║")

        # Section 3: Voting Signals (if available)
        if entry.committee_consensus or entry.trade_vote or entry.consensus_strength is not None:
            lines.append("║ " + "🗳️  VOTING SIGNALS".ljust(77) + "║")
            lines.append("║ " + "─" * 77 + "║")

            if entry.committee_consensus:
                cc = entry.committee_consensus
                exists = cc.get('consensus_exists', False)
                strength = cc.get('consensus_strength', 0.0)
                action = cc.get('consensus_action', 'N/A')
                lines.append("║ " + f"Committee:        {'✅ CONSENSUS' if exists else '❌ NO CONSENSUS'}".ljust(77) + "║")
                lines.append("║ " + f"  └─ Strength:    {strength:.1%} {'█' * int(strength * 20)}".ljust(77) + "║")
                lines.append("║ " + f"  └─ Action:      {action}".ljust(77) + "║")

            if entry.trade_vote:
                tv = entry.trade_vote
                vote_action = tv.get('action', 'HOLD')
                vote_conf = tv.get('confidence', 0.0)
                lines.append("║ " + f"Trade Vote:       {vote_action} (conf: {vote_conf:.1%})".ljust(77) + "║")

            if entry.consensus_strength is not None:
                lines.append("║ " + f"Consensus Str:    {entry.consensus_strength:.1%}".ljust(77) + "║")

            lines.append("║" + " " * 78 + "║")

        # Section 4: Portfolio Health
        lines.append("║ " + "💼 PORTFOLIO STATE".ljust(77) + "║")
        lines.append("║ " + "─" * 77 + "║")
        lines.append("║ " + f"Health Score:     {entry.portfolio_health:.1%} {self._health_indicator(entry.portfolio_health)}".ljust(77) + "║")
        lines.append("║ " + f"Exposure Ratio:   {entry.exposure_ratio:.1%}".ljust(77) + "║")
        lines.append("║ " + f"Balance:          €{entry.balance:,.2f}".ljust(77) + "║")
        lines.append("║ " + f"Drawdown:         {entry.drawdown:.1%} {self._drawdown_indicator(entry.drawdown)}".ljust(77) + "║")
        lines.append("║" + " " * 78 + "║")

        # Section 5: Risk Assessment
        lines.append("║ " + "⚠️  RISK ASSESSMENT".ljust(77) + "║")
        lines.append("║ " + "─" * 77 + "║")
        lines.append("║ " + f"Risk Score:       {entry.risk_score:.1%} {self._risk_indicator(entry.risk_score)}".ljust(77) + "║")
        if entry.risk_factors:
            for name, value in list(entry.risk_factors.items())[:4]:
                lines.append("║ " + f"  • {name.capitalize():15s} {value:.1%}".ljust(77) + "║")
        lines.append("║" + " " * 78 + "║")

        # Section 6: Decision Rationale
        lines.append("║ " + "💡 RATIONALE".ljust(77) + "║")
        lines.append("║ " + "─" * 77 + "║")
        lines.append("║ " + f"Stage:            {entry.stage}".ljust(77) + "║")
        if entry.factors:
            lines.append("║ " + "Key Factors:".ljust(77) + "║")
            for factor in entry.factors[:3]:
                # Split long factors into multiple lines
                factor_lines = self._wrap_text(f"  • {factor}", 75)
                for fline in factor_lines:
                    lines.append("║ " + fline.ljust(77) + "║")
        lines.append("║" + " " * 78 + "║")

        # Section 7: Execution Status
        lines.append("║ " + "⚡ EXECUTION STATUS".ljust(77) + "║")
        lines.append("║ " + "─" * 77 + "║")
        if entry.will_execute:
            lines.append("║ " + "Status:           ✅ WILL EXECUTE".ljust(77) + "║")
        else:
            lines.append("║ " + "Status:           ❌ BLOCKED".ljust(77) + "║")
            if entry.blocked_reason:
                reason_lines = self._wrap_text(f"Reason: {entry.blocked_reason}", 75)
                for rline in reason_lines:
                    lines.append("║ " + rline.ljust(77) + "║")

        lines.append("╚" + "═" * 78 + "╝")
        lines.append("")

        # Log the complete summary
        summary = "\n".join(lines)
        self.logger.info(summary)

    def log_order_build(self, instrument: str, order: Dict[str, Any]) -> None:
        """Log order build in clean format"""
        lines = []
        lines.append("")
        lines.append("┌─ ORDER BUILD " + "─" * 64)
        lines.append(f"│ Instrument:    {instrument}")
        lines.append(f"│ Side:          {self._format_side(order.get('side', 0))}")
        lines.append(f"│ Intent:        {order.get('intent', 'N/A').upper()}")
        lines.append(f"│ Size:          €{order.get('size_eur', 0):.2f}")
        lines.append(f"│ Confidence:    {order.get('confidence', 0):.1%}")
        lines.append(f"│ Reduce Only:   {'Yes' if order.get('reduce_only') else 'No'}")
        lines.append(f"│ Order ID:      {order.get('id', 'N/A')[:40]}")
        lines.append("└" + "─" * 78)
        lines.append("")

        self.logger.info("\n".join(lines))

    def log_portfolio_stats(self, health: Dict[str, float]) -> None:
        """Log portfolio statistics summary"""
        lines = []
        lines.append("")
        lines.append("┌─ PORTFOLIO STATISTICS " + "─" * 55)
        lines.append(f"│ Health Score:      {health.get('overall_health', 0):.1%} {self._health_indicator(health.get('overall_health', 0))}")
        lines.append(f"│ Exposure Ratio:    {health.get('exposure_ratio', 0):.1%}")
        lines.append(f"│ Total Exposure:    €{health.get('total_exposure', 0):,.2f}")
        lines.append(f"│ Balance:           €{health.get('balance', 0):,.2f}")
        lines.append(f"│ Drawdown:          {health.get('drawdown', 0):.1%}")
        lines.append(f"│ DD Health:         {health.get('drawdown_health', 0):.1%}")
        lines.append(f"│ Exposure Health:   {health.get('exposure_health', 0):.1%}")
        lines.append(f"│ Streak Health:     {health.get('streak_health', 0):.1%}")
        lines.append(f"│ Risk Health:       {health.get('risk_health', 0):.1%}")
        lines.append("└" + "─" * 78)
        lines.append("")

        self.logger.info("\n".join(lines))

    def log_signal_mapping(self, instrument: str, source: str, intensity: float,
                          volatility: float, trend: float, momentum: float) -> None:
        """Log signal mapping in clean format"""
        self.logger.debug(
            f"[SIGNAL] {instrument:10s} | "
            f"Src: {source:8s} | "
            f"Int: {intensity:+.3f} | "
            f"Vol: {volatility:.4f} | "
            f"Trend: {trend:+.3f} | "
            f"Mom: {momentum:+.3f}"
        )

    # Helper methods for formatting
    def _format_decision(self, decision: str) -> str:
        """Format decision with emoji"""
        mapping = {
            'open_long': '🟢 LONG',
            'open_short': '🔴 SHORT',
            'scale_up': '📈 ADD',
            'scale_down': '📉 REDUCE',
            'close': '🔒 CLOSE',
            'emergency_close': '🚨 EMERGENCY',
            'hold': '⏸️  HOLD'
        }
        return mapping.get(decision.lower(), decision.upper())

    def _format_side(self, side: int) -> str:
        """Format order side"""
        if side > 0:
            return "🟢 BUY"
        elif side < 0:
            return "🔴 SELL"
        return "⚪ NEUTRAL"

    def _get_signal_bar(self, signal: float) -> str:
        """Visual bar for signal strength"""
        bars = int(abs(signal) * 10)
        direction = "🟢" if signal > 0 else "🔴" if signal < 0 else "⚪"
        return f"{direction} {'█' * bars}"

    def _health_indicator(self, health: float) -> str:
        """Health indicator emoji"""
        if health >= 0.8:
            return "🟢 Excellent"
        elif health >= 0.6:
            return "🟡 Good"
        elif health >= 0.4:
            return "🟠 Fair"
        else:
            return "🔴 Poor"

    def _drawdown_indicator(self, drawdown: float) -> str:
        """Drawdown indicator"""
        if drawdown < 0.05:
            return "🟢 Minimal"
        elif drawdown < 0.10:
            return "🟡 Moderate"
        elif drawdown < 0.15:
            return "🟠 High"
        else:
            return "🔴 Critical"

    def _risk_indicator(self, risk: float) -> str:
        """Risk level indicator"""
        if risk < 0.3:
            return "🟢 Low"
        elif risk < 0.6:
            return "🟡 Medium"
        elif risk < 0.8:
            return "🟠 High"
        else:
            return "🔴 Very High"

    def _volatility_level(self, vol: float) -> str:
        """Volatility level description"""
        if vol < 0.015:
            return "Low"
        elif vol < 0.03:
            return "Normal"
        elif vol < 0.05:
            return "Elevated"
        else:
            return "High"

    def _wrap_text(self, text: str, max_width: int) -> List[str]:
        """Wrap long text into multiple lines"""
        if len(text) <= max_width:
            return [text]

        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            if len(current_line) + len(word) + 1 <= max_width:
                current_line += (word + " ")
            else:
                if current_line:
                    lines.append(current_line.rstrip())
                current_line = "  " + word + " "

        if current_line:
            lines.append(current_line.rstrip())

        return lines if lines else [text[:max_width]]

    def get_voting_signals(self) -> Dict[str, Any]:
        """Fetch current voting signals from InfoBus"""
        signals = {}

        try:
            bus = self.smart_bus
            if not hasattr(bus, 'get'):
                return signals

            # Committee consensus
            cc = bus.get("committee_consensus", "PositionManager")  # type: ignore
            if cc and isinstance(cc, dict):
                signals['committee_consensus'] = cc

            # Trade vote
            tv = bus.get("trade_vote_v2", "PositionManager") or bus.get("trade_vote", "PositionManager")  # type: ignore
            if tv and isinstance(tv, dict):
                signals['trade_vote'] = tv

            # Consensus strength
            cs = bus.get("consensus_score", "PositionManager")  # type: ignore
            if cs is not None:
                if isinstance(cs, dict):
                    signals['consensus_strength'] = cs.get('strength', cs.get('score', 0.0))
                elif isinstance(cs, (int, float)):
                    signals['consensus_strength'] = float(cs)

        except Exception:
            pass

        return signals
