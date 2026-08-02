

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from modules.utils.info_bus import InfoBusManager


@dataclass
class PositionLogEntry:


    instrument: str
    decision: str
    intensity: float
    size_eur: float
    confidence: float


    signal_strength: float
    volatility: float
    trend_strength: float
    current_price: float


    portfolio_health: float
    exposure_ratio: float
    balance: float
    drawdown: float
    risk_score: float


    committee_consensus: Optional[Dict[str, Any]] = None
    trade_vote: Optional[Dict[str, Any]] = None
    consensus_strength: Optional[float] = None


    stage: str = "unknown"
    factors: Optional[List[str]] = None
    risk_factors: Optional[Dict[str, float]] = None


    will_execute: bool = True
    blocked_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.factors is None:
            self.factors = []
        if self.risk_factors is None:
            self.risk_factors = {}


class UnifiedPositionLogger:

    BOX_WIDTH = 78
    PAD_WIDTH = 77

    def __init__(self, logger: Any, smart_bus: Optional[Any] = None) -> None:
        self.logger = logger
        self.smart_bus = smart_bus if smart_bus is not None else InfoBusManager.get_instance()
        self.session_start = dt.datetime.utcnow()
        self.decision_count = 0


    def log_decision_summary(self, entry: PositionLogEntry) -> None:
        self.decision_count += 1
        timestamp = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


        voting = self.get_voting_signals()
        cc = entry.committee_consensus or voting.get("committee_consensus")
        tv = entry.trade_vote or voting.get("trade_vote")
        cs = (
            entry.consensus_strength
            if entry.consensus_strength is not None
            else voting.get("consensus_strength")
        )

        lines: List[str] = []
        pad = self.PAD_WIDTH
        box = self.BOX_WIDTH


        lines.append("")
        lines.append("╔" + "═" * box + "╗")
        lines.append("║" + "POSITION DECISION SUMMARY".center(box) + "║")
        lines.append("╠" + "═" * box + "╣")


        self._section_title(lines, "📊 DECISION OVERVIEW")
        lines.append("║ " + f"Instrument:  {entry.instrument}".ljust(pad) + "║")
        lines.append(
            "║ "
            + f"Decision:    {self._format_decision(entry.decision)} ({entry.decision})".ljust(pad)
            + "║"
        )
        lines.append("║ " + f"Size:        €{entry.size_eur:,.2f}".ljust(pad) + "║")
        conf_bar = "█" * int(max(min(entry.confidence, 1.0), 0.0) * 20)
        lines.append(
            "║ "
            + f"Confidence:  {entry.confidence:.1%} {conf_bar}".ljust(pad)
            + "║"
        )
        lines.append("║ " + f"Timestamp:   {timestamp}".ljust(pad) + "║")
        self._section_blank(lines)


        self._section_title(lines, "📈 MARKET ANALYSIS")
        lines.append(
            "║ "
            + f"Signal Strength:  {entry.signal_strength:+.3f} {self._get_signal_bar(entry.signal_strength)}".ljust(pad)
            + "║"
        )
        lines.append(
            "║ "
            + f"Trend Strength:   {entry.trend_strength:+.3f}".ljust(pad)
            + "║"
        )
        lines.append(
            "║ "
            + f"Volatility:       {entry.volatility:.4f} ({self._volatility_level(entry.volatility)})".ljust(pad)
            + "║"
        )
        lines.append(
            "║ "
            + f"Current Price:    {entry.current_price:.5f}".ljust(pad)
            + "║"
        )
        self._section_blank(lines)


        if cc or tv or cs is not None:
            self._section_title(lines, "🗳️  VOTING SIGNALS")

            if cc:
                exists = bool(cc.get("consensus_exists", False))
                strength = float(cc.get("consensus_strength", 0.0))
                action = cc.get("consensus_action", "N/A")
                lines.append(
                    "║ "
                    + f"Committee:        {'✅ CONSENSUS' if exists else '❌ NO CONSENSUS'}".ljust(pad)
                    + "║"
                )
                lines.append(
                    "║ "
                    + f"  └─ Strength:    {strength:.1%} {'█' * int(strength * 20)}".ljust(pad)
                    + "║"
                )
                lines.append(
                    "║ "
                    + f"  └─ Action:      {action}".ljust(pad)
                    + "║"
                )

            if tv:
                vote_action = tv.get("action", "HOLD")
                vote_conf = float(tv.get("confidence", 0.0))
                lines.append(
                    "║ "
                    + f"Trade Vote:       {vote_action} (conf: {vote_conf:.1%})".ljust(pad)
                    + "║"
                )

            if cs is not None:
                lines.append(
                    "║ "
                    + f"Consensus Str:    {cs:.1%}".ljust(pad)
                    + "║"
                )

            self._section_blank(lines)


        self._section_title(lines, "💼 PORTFOLIO STATE")
        lines.append(
            "║ "
            + f"Health Score:     {entry.portfolio_health:.1%} {self._health_indicator(entry.portfolio_health)}".ljust(pad)
            + "║"
        )
        lines.append(
            "║ "
            + f"Exposure Ratio:   {entry.exposure_ratio:.1%}".ljust(pad)
            + "║"
        )
        lines.append(
            "║ "
            + f"Balance:          €{entry.balance:,.2f}".ljust(pad)
            + "║"
        )
        lines.append(
            "║ "
            + f"Drawdown:         {entry.drawdown:.1%} {self._drawdown_indicator(entry.drawdown)}".ljust(pad)
            + "║"
        )
        self._section_blank(lines)


        self._section_title(lines, "⚠️  RISK ASSESSMENT")
        lines.append(
            "║ "
            + f"Risk Score:       {entry.risk_score:.1%} {self._risk_indicator(entry.risk_score)}".ljust(pad)
            + "║"
        )

        if entry.risk_factors:
            for name, value in list(entry.risk_factors.items())[:4]:
                label = name.replace("_", " ").capitalize()
                lines.append(
                    "║ "
                    + f"  • {label:15s} {value:.1%}".ljust(pad)
                    + "║"
                )

        self._section_blank(lines)


        self._section_title(lines, "💡 RATIONALE")
        lines.append(
            "║ "
            + f"Stage:            {entry.stage}".ljust(pad)
            + "║"
        )

        if entry.factors:
            lines.append("║ " + "Key Factors:".ljust(pad) + "║")
            for factor in entry.factors[:3]:
                for fline in self._wrap_text(f"  • {factor}", pad):
                    lines.append("║ " + fline.ljust(pad) + "║")

        self._section_blank(lines)


        self._section_title(lines, "⚡ EXECUTION STATUS")
        if entry.will_execute:
            lines.append("║ " + "Status:           ✅ WILL EXECUTE".ljust(pad) + "║")
        else:
            lines.append("║ " + "Status:           ❌ BLOCKED".ljust(pad) + "║")
            if entry.blocked_reason:
                for rline in self._wrap_text(f"Reason: {entry.blocked_reason}", pad):
                    lines.append("║ " + rline.ljust(pad) + "║")


        lines.append("╚" + "═" * box + "╝")
        lines.append("")

        self.logger.info("\n".join(lines))


    def log_order_build(self, instrument: str, order: Dict[str, Any]) -> None:
        side = self._format_side(int(order.get("side", 0) or 0))
        intent = str(order.get("intent", "N/A")).upper()
        size_eur = float(order.get("size_eur", 0.0) or 0.0)
        confidence = float(order.get("confidence", 0.0) or 0.0)
        reduce_only = bool(order.get("reduce_only", False))
        order_id = str(order.get("id", "N/A"))[:40]

        lines = [
            "",
            "┌─ ORDER BUILD " + "─" * 64,
            f"│ Instrument:    {instrument}",
            f"│ Side:          {side}",
            f"│ Intent:        {intent}",
            f"│ Size:          €{size_eur:.2f}",
            f"│ Confidence:    {confidence:.1%}",
            f"│ Reduce Only:   {'Yes' if reduce_only else 'No'}",
            f"│ Order ID:      {order_id}",
            "└" + "─" * 78,
            "",
        ]

        self.logger.info("\n".join(lines))

    def log_portfolio_stats(self, health: Dict[str, float]) -> None:
        def h(key: str, default: float = 0.0) -> float:
            return float(health.get(key, default) or 0.0)

        lines = [
            "",
            "┌─ PORTFOLIO STATISTICS " + "─" * 55,
            f"│ Health Score:      {h('overall_health'):.1%} {self._health_indicator(h('overall_health'))}",
            f"│ Exposure Ratio:    {h('exposure_ratio'):.1%}",
            f"│ Total Exposure:    €{h('total_exposure'):,.2f}",
            f"│ Balance:           €{h('balance'):,.2f}",
            f"│ Drawdown:          {h('drawdown'):.1%}",
            f"│ DD Health:         {h('drawdown_health'):.1%}",
            f"│ Exposure Health:   {h('exposure_health'):.1%}",
            f"│ Streak Health:     {h('streak_health'):.1%}",
            f"│ Risk Health:       {h('risk_health'):.1%}",
            "└" + "─" * 78,
            "",
        ]

        self.logger.info("\n".join(lines))

    def log_instrument_stats(self, instrument: str, stats: Dict[str, Any]) -> None:

        def f(key: str, default: float = 0.0) -> float:
            try:
                return float(stats.get(key, default) or 0.0)
            except Exception:
                return default

        side_val = int(stats.get("side", 0) or 0)
        if side_val > 0:
            side_label = "LONG"
        elif side_val < 0:
            side_label = "SHORT"
        else:
            side_label = "FLAT"

        lots = f("lots", 0.0)
        size_eur = f("size_eur", stats.get("size", 0.0))
        pnl = f("unrealized_pnl", 0.0)
        age_h = f("age_hours", 0.0)
        exposure = f("exposure", 0.0)
        drawdown = f("drawdown", 0.0)

        if pnl > 0.0:
            pnl_icon = "🟢"
        elif pnl < 0.0:
            pnl_icon = "🔴"
        else:
            pnl_icon = "⚪"

        lines = [
            "",
            f"┌─ {instrument} ─ INSTRUMENT STATISTICS " + "─" * 40,
            f"│ Side:             {side_label}",
            f"│ Lots:             {lots:.2f}",
            f"│ Size:             €{size_eur:,.2f}",
            f"│ Unrealized P&L:   {pnl_icon} €{pnl:,.2f}",
            f"│ Age:              {age_h:.1f}h",
            f"│ Exposure:         {exposure:.1%}",
            f"│ Drawdown:         {drawdown:.1%}",
            "└" + "─" * 78,
            "",
        ]

        self.logger.info("\n".join(lines))

    def log_signal_mapping(
        self,
        instrument: str,
        source: str,
        intensity: float,
        volatility: float,
        trend: float,
        momentum: float,
    ) -> None:
        self.logger.debug(
            f"[SIGNAL] {instrument:10s} | "
            f"Src: {source:8s} | "
            f"Int: {intensity:+.3f} | "
            f"Vol: {volatility:.4f} | "
            f"Trend: {trend:+.3f} | "
            f"Mom: {momentum:+.3f}"
        )


    def _section_title(self, lines: List[str], title: str) -> None:
        pad = self.PAD_WIDTH
        lines.append("║ " + title.ljust(pad) + "║")
        lines.append("║ " + "─" * pad + "║")

    def _section_blank(self, lines: List[str]) -> None:
        pad = self.PAD_WIDTH
        lines.append("║ " + " " * pad + "║")

    def _format_decision(self, decision: str) -> str:
        mapping = {
            "open_long": "🟢 LONG",
            "open_short": "🔴 SHORT",
            "scale_up": "📈 ADD",
            "scale_down": "📉 REDUCE",
            "close": "🔒 CLOSE",
            "emergency_close": "🚨 EMERGENCY",
            "hold": "⏸️  HOLD",
        }
        return mapping.get(decision.lower(), decision.upper())

    def _format_side(self, side: int) -> str:
        if side > 0:
            return "🟢 BUY"
        if side < 0:
            return "🔴 SELL"
        return "⚪ NEUTRAL"

    def _get_signal_bar(self, signal: float) -> str:
        bars = int(min(abs(signal), 1.0) * 10)
        direction = "🟢" if signal > 0 else "🔴" if signal < 0 else "⚪"
        return f"{direction} {'█' * bars}"

    def _health_indicator(self, health: float) -> str:
        if health >= 0.8:
            return "🟢 Excellent"
        if health >= 0.6:
            return "🟡 Good"
        if health >= 0.4:
            return "🟠 Fair"
        return "🔴 Poor"

    def _drawdown_indicator(self, drawdown: float) -> str:
        if drawdown < 0.05:
            return "🟢 Minimal"
        if drawdown < 0.10:
            return "🟡 Moderate"
        if drawdown < 0.15:
            return "🟠 High"
        return "🔴 Critical"

    def _risk_indicator(self, risk: float) -> str:
        if risk < 0.3:
            return "🟢 Low"
        if risk < 0.6:
            return "🟡 Medium"
        if risk < 0.8:
            return "🟠 High"
        return "🔴 Very High"

    def _volatility_level(self, vol: float) -> str:
        if vol < 0.015:
            return "Low"
        if vol < 0.03:
            return "Normal"
        if vol < 0.05:
            return "Elevated"
        return "High"

    def _wrap_text(self, text: str, max_width: int) -> List[str]:
        if len(text) <= max_width:
            return [text]

        words = text.split()
        lines: List[str] = []
        current = ""

        for word in words:
            if len(current) + len(word) + 1 <= max_width:
                current += word + " "
            else:
                if current:
                    lines.append(current.rstrip())

                current = "  " + word + " "

        if current:
            lines.append(current.rstrip())

        return lines or [text[:max_width]]


    def get_voting_signals(self) -> Dict[str, Any]:
        signals: Dict[str, Any] = {}

        try:
            bus = self.smart_bus
            if not hasattr(bus, "get"):
                return signals


            cc = bus.get("committee_consensus", "PositionManager")
            if isinstance(cc, dict) and cc:
                signals["committee_consensus"] = cc


            tv = bus.get("trade_vote_v2", "PositionManager")
            if isinstance(tv, dict) and tv:
                signals["trade_vote"] = tv


            cs = bus.get("consensus_score", "PositionManager")
            if cs is not None:
                if isinstance(cs, dict):
                    signals["consensus_strength"] = float(
                        cs.get("strength", cs.get("score", 0.0))
                    )
                elif isinstance(cs, (int, float)):
                    signals["consensus_strength"] = float(cs)

        except Exception:

            pass

        return signals
