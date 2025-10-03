#!/usr/bin/env python3
"""
Live Trading Status Checker
----------------------------
Quick script to check current MT5 connection and account status
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def format_time(seconds: float) -> str:
    """Format seconds to human-readable time"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.0f}m"
    else:
        hours = int(seconds / 3600)
        mins = int((seconds % 3600) / 60)
        return f"{hours}h {mins}m"


def check_status():
    """Check current MT5 status"""
    print("\n" + "="*70)
    print("LIVE TRADING STATUS CHECK")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Check MT5 module
    try:
        import MetaTrader5 as mt5
        print("✓ MetaTrader5 module loaded")
    except ImportError:
        print("✗ MetaTrader5 module not installed")
        print("  Install with: pip install MetaTrader5")
        return

    # Load credentials
    try:
        from live.mt5_credentials import MT5Credentials
        print(f"✓ Credentials loaded (Account: {MT5Credentials.ACCOUNT})")
    except Exception as e:
        print(f"✗ Failed to load credentials: {e}")
        return

    # Check connection
    print("\nConnecting to MT5...")

    try:
        # Shutdown any existing connection
        mt5.shutdown()  # type: ignore[attr-defined]

        # Initialize
        connected = mt5.initialize(  # type: ignore[attr-defined]
            login=MT5Credentials.ACCOUNT,
            password=MT5Credentials.PASSWORD,
            server=MT5Credentials.SERVER,
            timeout=30000
        )

        if not connected:
            error = mt5.last_error()  # type: ignore[attr-defined]
            print(f"\n✗ Connection failed: {error}")
            print("\nPossible issues:")
            print("  - MT5 terminal not running")
            print("  - Incorrect credentials")
            print("  - Network issues")
            print("  - Server maintenance")
            return

        print("✓ Connected to MT5\n")

        # Get account info
        account = mt5.account_info()  # type: ignore[attr-defined]
        if not account:
            print("✗ Failed to get account info")
            mt5.shutdown()  # type: ignore[attr-defined]
            return

        # Get terminal info
        terminal = mt5.terminal_info()  # type: ignore[attr-defined]

        # Get positions
        positions = mt5.positions_get()  # type: ignore[attr-defined]
        num_positions = len(positions) if positions else 0

        # Get open orders
        orders = mt5.orders_get()  # type: ignore[attr-defined]
        num_orders = len(orders) if orders else 0

        # Print account info
        print("="*70)
        print("ACCOUNT INFORMATION")
        print("="*70)
        print(f"Account Number:     {account.login}")
        print(f"Account Name:       {account.name}")
        print(f"Server:             {account.server}")
        print(f"Currency:           {account.currency}")
        print(f"Leverage:           1:{account.leverage}")
        print(f"Company:            {account.company}")

        # Print balance info
        print("\n" + "="*70)
        print("BALANCE & EQUITY")
        print("="*70)
        print(f"Balance:            {account.balance:,.2f} {account.currency}")
        print(f"Equity:             {account.equity:,.2f} {account.currency}")
        print(f"Profit:             {account.profit:,.2f} {account.currency}")
        print(f"Margin Used:        {account.margin:,.2f} {account.currency}")
        print(f"Margin Free:        {account.margin_free:,.2f} {account.currency}")
        print(f"Margin Level:       {account.margin_level:.2f}%" if account.margin_level else "Margin Level:       N/A")

        # Calculate P&L
        unrealized_pnl = account.equity - account.balance
        pnl_pct = (unrealized_pnl / account.balance * 100) if account.balance > 0 else 0.0

        print(f"\nUnrealized P&L:     {unrealized_pnl:,.2f} {account.currency} ({pnl_pct:+.2f}%)")

        # Print position info
        print("\n" + "="*70)
        print("POSITIONS & ORDERS")
        print("="*70)
        print(f"Open Positions:     {num_positions}")
        print(f"Pending Orders:     {num_orders}")

        if num_positions > 0:
            print("\nCurrent Positions:")
            print("-" * 70)
            for pos in positions:
                pos_type = "BUY" if pos.type == 0 else "SELL"
                pos_profit = pos.profit
                pos_pnl_pct = (pos_profit / (pos.volume * pos.price_open) * 100) if pos.volume > 0 else 0.0

                print(f"  {pos.symbol:12s} {pos_type:4s} {pos.volume:.2f} lots @ {pos.price_open:.5f}")
                print(f"    Current: {pos.price_current:.5f}, P&L: {pos_profit:+.2f} {account.currency} ({pos_pnl_pct:+.2f}%)")

        if num_orders > 0:
            print("\nPending Orders:")
            print("-" * 70)
            for order in orders:
                order_type = mt5.order_type(order.type)  # type: ignore[attr-defined]
                print(f"  {order.symbol:12s} {order_type} {order.volume:.2f} lots @ {order.price_open:.5f}")

        # Print terminal info
        if terminal:
            print("\n" + "="*70)
            print("TERMINAL INFORMATION")
            print("="*70)
            print(f"Terminal Build:     {terminal.build}")
            print(f"Connected:          {'Yes' if terminal.connected else 'No'}")
            print(f"Trade Allowed:      {'Yes' if terminal.trade_allowed else 'No'}")
            print(f"Algo Trading:       {'Enabled' if terminal.tradeapi_disabled == 0 else 'Disabled'}")
            # maxorders is not always available
            if hasattr(terminal, 'maxorders'):
                print(f"Max Pending:        {terminal.maxorders}")

        # Trading status
        print("\n" + "="*70)
        print("TRADING STATUS")
        print("="*70)

        trade_mode = account.trade_mode
        trade_modes = {
            0: "DEMO",
            1: "CONTEST",
            2: "REAL"
        }
        mode_str = trade_modes.get(trade_mode, "UNKNOWN")

        print(f"Trading Mode:       {mode_str}")
        print(f"Trade Allowed:      {'Yes' if account.trade_allowed else 'No'}")
        print(f"Expert Allowed:     {'Yes' if account.trade_expert else 'No'}")

        # Risk assessment
        print("\n" + "="*70)
        print("RISK ASSESSMENT")
        print("="*70)

        if account.margin_level and account.margin_level < 200:
            print(f"⚠ WARNING: Low margin level ({account.margin_level:.2f}%)")
        elif account.margin_level:
            print(f"✓ Margin level healthy ({account.margin_level:.2f}%)")

        if num_positions > 10:
            print(f"⚠ WARNING: High number of open positions ({num_positions})")
        elif num_positions > 0:
            print(f"✓ Position count: {num_positions}")

        max_position_size = account.balance * 0.05  # 5% rule
        for pos in positions if positions else []:
            pos_size = pos.volume * pos.price_current
            if pos_size > max_position_size:
                print(f"⚠ WARNING: Large position in {pos.symbol} ({pos_size:.2f})")

        # Cleanup
        mt5.shutdown()  # type: ignore[attr-defined]

        print("\n" + "="*70)
        print("✓ Status check complete")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n✗ Error checking status: {e}")
        import traceback
        traceback.print_exc()
        try:
            mt5.shutdown()  # type: ignore[attr-defined]
        except:
            pass


def main():
    """Main entry point"""
    try:
        check_status()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
