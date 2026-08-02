# ═══════════════════════════════════════════════════════════════════════════
# File: modules/executor/unified_logger.py
# Unified Executor Logging System - Ultra-Detailed Professional Format
#
# Creates clean, comprehensive execution logs with:
# - Order acceptance/rejection details
# - Fill execution summaries
# - Position state changes
# - P&L breakdowns (realized/unrealized)
# - Balance/equity tracking
# - Trade history
# - Execution quality metrics
# - Issue detection
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ExecutionCycleEntry:
    """Complete execution cycle log entry with all context"""
    # Step context
    step: int
    mode: str  # 'sim' or 'live'
    timestamp: str

    # Orders
    orders_received: int
    orders_accepted: int
    orders_rejected: int
    rejected_reasons: Dict[str, int]

    # Fills
    fills_count: int
    fills_by_instrument: Dict[str, int]
    total_notional: float

    # Positions
    positions_before: Dict[str, Dict[str, Any]]
    positions_after: Dict[str, Dict[str, Any]]
    positions_opened: List[str]
    positions_closed: List[str]
    positions_modified: List[str]

    # P&L
    balance_before: float
    balance_after: float
    equity_before: float
    equity_after: float
    realized_pnl: float
    unrealized_pnl: float
    step_pnl: float

    # Trades
    trades_this_step: List[Dict[str, Any]]

    # Quality
    execution_time_ms: float
    issues: List[str]

    # Details
    accepted_details: List[Dict[str, Any]]
    rejected_details: List[Dict[str, Any]]
    fill_details: List[Dict[str, Any]]


class UnifiedExecutorLogger:
    """
    Unified logging system for executor operations.

    Creates ultra-detailed, professional execution logs with:
    - Execution cycle summaries
    - Order processing details
    - Fill execution reports
    - Position state tracking
    - P&L breakdowns
    - Trade ledger updates
    - Performance metrics
    - Issue detection
    """

    def __init__(self, logger):
        self.logger = logger
        self.session_start = dt.datetime.utcnow()
        self.cycle_count = 0

        # Cumulative stats
        self.cumulative = {
            'total_orders': 0,
            'total_accepted': 0,
            'total_rejected': 0,
            'total_fills': 0,
            'total_trades': 0,
            'total_notional': 0.0,
            'rejection_reasons': defaultdict(int),
        }

    def log_execution_cycle(self, entry: ExecutionCycleEntry) -> None:
        """
        Log a complete, ultra-detailed execution cycle summary.

        Format: Multi-section detailed report with all execution context
        """
        self.cycle_count += 1

        # Update cumulative stats
        self.cumulative['total_orders'] += entry.orders_received
        self.cumulative['total_accepted'] += entry.orders_accepted
        self.cumulative['total_rejected'] += entry.orders_rejected
        self.cumulative['total_fills'] += entry.fills_count
        self.cumulative['total_trades'] += len(entry.trades_this_step)
        self.cumulative['total_notional'] += entry.total_notional
        for reason, count in entry.rejected_reasons.items():
            self.cumulative['rejection_reasons'][reason] += count

        # Build the detailed log
        lines = []
        lines.append("")
        lines.append("╔" + "═" * 98 + "╗")
        lines.append("║" + "EXECUTION CYCLE SUMMARY".center(98) + "║")
        lines.append("╠" + "═" * 98 + "╣")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Section 1: Cycle Overview
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        lines.append("║ " + "⏱️  CYCLE OVERVIEW".ljust(97) + "║")
        lines.append("║ " + "─" * 97 + "║")
        lines.append("║ " + f"Step:             {entry.step:,}".ljust(97) + "║")
        lines.append("║ " + f"Mode:             {self._format_mode(entry.mode)}".ljust(97) + "║")
        lines.append("║ " + f"Timestamp:        {entry.timestamp}".ljust(97) + "║")
        lines.append("║ " + f"Processing Time:  {entry.execution_time_ms:.2f}ms".ljust(97) + "║")
        lines.append("║" + " " * 98 + "║")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Section 2: Order Processing
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        lines.append("║ " + "📥 ORDER PROCESSING".ljust(97) + "║")
        lines.append("║ " + "─" * 97 + "║")
        lines.append("║ " + f"Received:         {entry.orders_received} order(s)".ljust(97) + "║")

        if entry.orders_received > 0:
            accept_pct = (entry.orders_accepted / entry.orders_received * 100) if entry.orders_received > 0 else 0
            lines.append("║ " + f"Accepted:         ✅ {entry.orders_accepted} ({accept_pct:.1f}%) {self._bar(accept_pct, 20)}".ljust(97) + "║")
            lines.append("║ " + f"Rejected:         ❌ {entry.orders_rejected} ({100-accept_pct:.1f}%)".ljust(97) + "║")

            if entry.rejected_reasons:
                lines.append("║ " + "Rejection Breakdown:".ljust(97) + "║")
                for reason, count in list(entry.rejected_reasons.items())[:5]:
                    lines.append("║ " + f"  • {reason:25s} {count:2d}x".ljust(97) + "║")
        else:
            lines.append("║ " + "No orders received this cycle".ljust(97) + "║")

        lines.append("║" + " " * 98 + "║")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Section 3: Accepted Orders Details
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if entry.accepted_details:
            lines.append("║ " + f"📋 ACCEPTED ORDERS DETAILS ({len(entry.accepted_details)})".ljust(97) + "║")
            lines.append("║ " + "─" * 97 + "║")

            for i, order in enumerate(entry.accepted_details[:10], 1):  # Show first 10
                inst = order.get('instrument', 'N/A')
                action = order.get('action', 'N/A')
                size_eur = order.get('size_eur', 0.0)
                units = order.get('units', 0.0)
                conf = order.get('confidence', 0.0)

                lines.append("║ " + f"  {i}. {inst:10s} │ {self._format_action(action):15s} │ "
                            f"€{size_eur:8.2f} │ {units:7.2f} units │ conf: {conf:.1%}".ljust(97) + "║")

            if len(entry.accepted_details) > 10:
                lines.append("║ " + f"  ... and {len(entry.accepted_details) - 10} more".ljust(97) + "║")

            lines.append("║" + " " * 98 + "║")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Section 4: Rejected Orders Details
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if entry.rejected_details:
            lines.append("║ " + f"⛔ REJECTED ORDERS DETAILS ({len(entry.rejected_details)})".ljust(97) + "║")
            lines.append("║ " + "─" * 97 + "║")

            for i, rej in enumerate(entry.rejected_details[:5], 1):  # Show first 5
                reason = rej.get('reason', 'unknown')
                intent = rej.get('intent', {})
                inst = intent.get('instrument', 'N/A')
                action = intent.get('action', 'N/A')

                lines.append("║ " + f"  {i}. {inst:10s} │ {action:12s} │ Reason: {reason}".ljust(97) + "║")

            if len(entry.rejected_details) > 5:
                lines.append("║ " + f"  ... and {len(entry.rejected_details) - 5} more".ljust(97) + "║")

            lines.append("║" + " " * 98 + "║")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Section 5: Fill Execution
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        lines.append("║ " + f"⚡ FILL EXECUTION ({entry.fills_count})".ljust(97) + "║")
        lines.append("║ " + "─" * 97 + "║")

        if entry.fills_count > 0:
            lines.append("║ " + f"Total Fills:      {entry.fills_count}".ljust(97) + "║")
            lines.append("║ " + f"Total Notional:   €{entry.total_notional:,.2f}".ljust(97) + "║")

            if entry.fills_by_instrument:
                lines.append("║ " + "By Instrument:".ljust(97) + "║")
                for inst, count in list(entry.fills_by_instrument.items())[:10]:
                    lines.append("║ " + f"  • {inst:15s} {count} fill(s)".ljust(97) + "║")

            # Detailed fill list
            if entry.fill_details:
                lines.append("║ " + "Fill Details:".ljust(97) + "║")
                for i, fill in enumerate(entry.fill_details[:15], 1):
                    inst = fill.get('instrument', 'N/A')
                    action = fill.get('action', 'N/A')
                    side = fill.get('side', 0)
                    units = fill.get('units', 0.0)
                    price = fill.get('price', 0.0)
                    notional = fill.get('notional_eur', 0.0)
                    realized = fill.get('realized_pnl', 0.0)

                    side_icon = "🟢" if side > 0 else "🔴" if side < 0 else "⚪"
                    pnl_icon = "💰" if realized > 0 else "📉" if realized < 0 else "⚪"

                    lines.append("║ " + f"  {i:2d}. {side_icon} {inst:10s} │ {action:15s} │ "
                                f"{units:7.2f}u @ {price:8.5f} │ €{notional:8.2f} │ "
                                f"{pnl_icon} P&L: {realized:+.2f}".ljust(97) + "║")

                if len(entry.fill_details) > 15:
                    lines.append("║ " + f"  ... and {len(entry.fill_details) - 15} more".ljust(97) + "║")
        else:
            lines.append("║ " + "No fills executed this cycle".ljust(97) + "║")

        lines.append("║" + " " * 98 + "║")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Section 6: Position State Changes
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        lines.append("║ " + "📊 POSITION STATE CHANGES".ljust(97) + "║")
        lines.append("║ " + "─" * 97 + "║")

        has_changes = entry.positions_opened or entry.positions_closed or entry.positions_modified

        if has_changes:
            if entry.positions_opened:
                lines.append("║ " + f"Opened:           🆕 {', '.join(entry.positions_opened)}".ljust(97) + "║")
            if entry.positions_closed:
                lines.append("║ " + f"Closed:           🔒 {', '.join(entry.positions_closed)}".ljust(97) + "║")
            if entry.positions_modified:
                lines.append("║ " + f"Modified:         ✏️  {', '.join(entry.positions_modified)}".ljust(97) + "║")

            lines.append("║" + " " * 98 + "║")

            # Position details before/after
            if entry.positions_after:
                lines.append("║ " + "Current Positions:".ljust(97) + "║")
                for inst, pos in list(entry.positions_after.items())[:10]:
                    side = pos.get('side', 0)
                    units = pos.get('units', 0.0)
                    entry_price = pos.get('entry_price', 0.0)
                    current_price = pos.get('current_price', entry_price)
                    notional = pos.get('notional_eur', 0.0)
                    unrealized = pos.get('unrealized_pnl', 0.0)

                    side_str = "LONG" if side > 0 else "SHORT" if side < 0 else "FLAT"
                    side_icon = "🟢" if side > 0 else "🔴" if side < 0 else "⚪"
                    pnl_icon = "💰" if unrealized > 0 else "📉" if unrealized < 0 else "⚪"

                    lines.append("║ " + f"  {side_icon} {inst:10s} │ {side_str:5s} │ "
                                f"{units:7.2f}u @ {entry_price:8.5f} → {current_price:8.5f} │ "
                                f"€{abs(notional):8.2f} │ {pnl_icon} {unrealized:+.2f}".ljust(97) + "║")
        else:
            lines.append("║ " + "No position changes this cycle".ljust(97) + "║")
            if entry.positions_after:
                count = len(entry.positions_after)
                lines.append("║ " + f"Holding {count} position(s)".ljust(97) + "║")

        lines.append("║" + " " * 98 + "║")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Section 7: P&L Breakdown
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        lines.append("║ " + "💰 P&L BREAKDOWN".ljust(97) + "║")
        lines.append("║ " + "─" * 97 + "║")

        balance_delta = entry.balance_after - entry.balance_before
        equity_delta = entry.equity_after - entry.equity_before

        lines.append("║ " + f"Balance:          {entry.balance_before:11,.2f} → {entry.balance_after:11,.2f}  "
                    f"({self._format_delta(balance_delta)})".ljust(97) + "║")
        lines.append("║ " + f"Equity:           {entry.equity_before:11,.2f} → {entry.equity_after:11,.2f}  "
                    f"({self._format_delta(equity_delta)})".ljust(97) + "║")
        lines.append("║" + " " * 98 + "║")
        lines.append("║ " + f"Realized P&L:     {self._format_pnl(entry.realized_pnl):>15s}  "
                    f"{self._pnl_indicator(entry.realized_pnl)}".ljust(97) + "║")
        lines.append("║ " + f"Unrealized P&L:   {self._format_pnl(entry.unrealized_pnl):>15s}  "
                    f"{self._unrealized_indicator(entry.unrealized_pnl)}".ljust(97) + "║")
        lines.append("║ " + f"Step P&L:         {self._format_pnl(entry.step_pnl):>15s}  "
                    f"{self._pnl_indicator(entry.step_pnl)} (Total equity change)".ljust(97) + "║")

        # P&L percentages
        if entry.balance_before > 0:
            realized_pct = (entry.realized_pnl / entry.balance_before) * 100
            unrealized_pct = (entry.unrealized_pnl / entry.balance_before) * 100
            step_pct = (entry.step_pnl / entry.equity_before) * 100 if entry.equity_before > 0 else 0

            lines.append("║" + " " * 98 + "║")
            lines.append("║ " + f"Realized %:       {realized_pct:+.3f}%".ljust(97) + "║")
            lines.append("║ " + f"Unrealized %:     {unrealized_pct:+.3f}%".ljust(97) + "║")
            lines.append("║ " + f"Step %:           {step_pct:+.3f}%".ljust(97) + "║")

        lines.append("║" + " " * 98 + "║")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Section 8: Trades This Step
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if entry.trades_this_step:
            lines.append("║ " + f"📝 TRADES EXECUTED ({len(entry.trades_this_step)})".ljust(97) + "║")
            lines.append("║ " + "─" * 97 + "║")

            for i, trade in enumerate(entry.trades_this_step[:10], 1):
                inst = trade.get('instrument', 'N/A')
                action = trade.get('action', 'N/A')
                side = trade.get('side', 0)
                units = trade.get('units', 0.0)
                price = trade.get('price', 0.0)
                realized = trade.get('realized_pnl', 0.0)
                comment = trade.get('comment', '')

                side_icon = "🟢" if side > 0 else "🔴" if side < 0 else "⚪"
                pnl_icon = "💰" if realized > 0 else "📉" if realized < 0 else "⚪"

                lines.append("║ " + f"  {i:2d}. {side_icon} {inst:10s} │ {action:15s} │ "
                            f"{units:7.2f}u @ {price:8.5f} │ {pnl_icon} {realized:+8.2f} │ {comment}".ljust(97) + "║")

            if len(entry.trades_this_step) > 10:
                lines.append("║ " + f"  ... and {len(entry.trades_this_step) - 10} more".ljust(97) + "║")

            lines.append("║" + " " * 98 + "║")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Section 9: Performance & Issues
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        lines.append("║ " + "⚙️  PERFORMANCE & QUALITY".ljust(97) + "║")
        lines.append("║ " + "─" * 97 + "║")
        lines.append("║ " + f"Processing Time:  {entry.execution_time_ms:.2f}ms".ljust(97) + "║")

        # Calculate fill rate
        fill_rate = (entry.fills_count / entry.orders_accepted * 100) if entry.orders_accepted > 0 else 0
        lines.append("║ " + f"Fill Rate:        {fill_rate:.1f}% ({entry.fills_count}/{entry.orders_accepted})".ljust(97) + "║")

        # Issues
        if entry.issues:
            lines.append("║ " + f"⚠️  Issues Detected: {len(entry.issues)}".ljust(97) + "║")
            for issue in entry.issues[:5]:
                lines.append("║ " + f"  • {issue}".ljust(97) + "║")
        else:
            lines.append("║ " + "✅ No issues detected".ljust(97) + "║")

        lines.append("║" + " " * 98 + "║")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Section 10: Session Summary
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        lines.append("║ " + "📊 SESSION CUMULATIVE".ljust(97) + "║")
        lines.append("║ " + "─" * 97 + "║")
        lines.append("║ " + f"Total Cycles:     {self.cycle_count:,}".ljust(97) + "║")
        lines.append("║ " + f"Total Orders:     {self.cumulative['total_orders']:,}".ljust(97) + "║")
        lines.append("║ " + f"Total Accepted:   {self.cumulative['total_accepted']:,}".ljust(97) + "║")
        lines.append("║ " + f"Total Rejected:   {self.cumulative['total_rejected']:,}".ljust(97) + "║")
        lines.append("║ " + f"Total Fills:      {self.cumulative['total_fills']:,}".ljust(97) + "║")
        lines.append("║ " + f"Total Trades:     {self.cumulative['total_trades']:,}".ljust(97) + "║")
        lines.append("║ " + f"Total Notional:   €{self.cumulative['total_notional']:,.2f}".ljust(97) + "║")

        # Session acceptance rate
        total_orders = self.cumulative['total_orders']
        if total_orders > 0:
            session_accept_rate = (self.cumulative['total_accepted'] / total_orders) * 100
            lines.append("║ " + f"Accept Rate:      {session_accept_rate:.1f}%".ljust(97) + "║")

        lines.append("╚" + "═" * 98 + "╝")
        lines.append("")

        # Log the complete summary
        summary = "\n".join(lines)
        self.logger.info(summary)

    def log_order_collection(self, queue_count: int, decisions_count: int,
                            accepted: List[Dict[str, Any]], rejected: List[Dict[str, Any]]) -> None:
        """Log order collection summary"""
        lines = []
        lines.append("")
        lines.append("┌─ ORDER COLLECTION " + "─" * 78)
        lines.append(f"│ Order Queue:      {queue_count} items")
        lines.append(f"│ Position Decisions: {decisions_count} items")
        lines.append(f"│ Accepted:         {len(accepted)} orders")
        lines.append(f"│ Rejected:         {len(rejected)} orders")

        if accepted:
            lines.append("│")
            lines.append("│ Accepted Preview:")
            for order in accepted[:5]:
                inst = order.get('instrument', 'N/A')
                action = order.get('action', 'N/A')
                size = order.get('size_eur', 0.0)
                lines.append(f"│   • {inst:10s} {action:15s} €{size:8.2f}")

        if rejected:
            lines.append("│")
            lines.append("│ Rejected Preview:")
            for rej in rejected[:3]:
                reason = rej.get('reason', 'unknown')
                intent = rej.get('intent', {})
                inst = intent.get('instrument', 'N/A')
                lines.append(f"│   • {inst:10s} - {reason}")

        lines.append("└" + "─" * 98)
        lines.append("")

        self.logger.debug("\n".join(lines))

    # ═══════════════════════════════════════════════════════════════════════════
    # Helper methods for formatting
    # ═══════════════════════════════════════════════════════════════════════════

    def _format_mode(self, mode: str) -> str:
        """Format execution mode"""
        if mode == "live":
            return "🔴 LIVE TRADING"
        elif mode == "sim":
            return "🟢 SIMULATION"
        return mode.upper()

    def _format_action(self, action: str) -> str:
        """Format order action"""
        mapping = {
            'open_long': '🟢 OPEN LONG',
            'open_short': '🔴 OPEN SHORT',
            'scale_up': '📈 SCALE UP',
            'scale_down': '📉 SCALE DOWN',
            'close': '🔒 CLOSE',
            'emergency_close': '🚨 EMERGENCY',
        }
        return mapping.get(action.lower(), action.upper())

    def _format_delta(self, delta: float) -> str:
        """Format delta with sign"""
        if delta > 0:
            return f"+€{delta:,.2f}"
        elif delta < 0:
            return f"-€{abs(delta):,.2f}"
        return "€0.00"

    def _format_pnl(self, pnl: float) -> str:
        """Format P&L with sign and color indicator"""
        if pnl > 0:
            return f"+€{pnl:,.2f}"
        elif pnl < 0:
            return f"-€{abs(pnl):,.2f}"
        return "€0.00"

    def _pnl_indicator(self, pnl: float) -> str:
        """Get P&L indicator"""
        if pnl > 0:
            return "💰 PROFIT"
        elif pnl < 0:
            return "📉 LOSS"
        return "⚪ NEUTRAL"

    def _unrealized_indicator(self, pnl: float) -> str:
        """Get unrealized P&L indicator"""
        if pnl > 0:
            return "🟢 POSITIVE"
        elif pnl < 0:
            return "🔴 NEGATIVE"
        return "⚪ NEUTRAL"

    def _bar(self, value: float, width: int = 20) -> str:
        """Visual progress bar"""
        filled = int((value / 100) * width)
        return "█" * filled + "░" * (width - filled)
