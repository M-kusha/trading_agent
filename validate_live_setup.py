#!/usr/bin/env python3
"""
Live Trading Setup Validator
-----------------------------
Validates that everything is correctly configured for live trading
"""

import sys
import os
from pathlib import Path
from typing import List, Tuple

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class Colors:
    """Terminal colors"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def check_python_version() -> Tuple[bool, str]:
    """Check Python version"""
    if sys.version_info >= (3, 8):
        return True, f"Python {sys.version.split()[0]}"
    return False, f"Python {sys.version.split()[0]} (need >=3.8)"


def check_mt5_installation() -> Tuple[bool, str]:
    """Check if MT5 is installed"""
    try:
        import MetaTrader5 as mt5
        version = getattr(mt5, '__version__', 'unknown')
        return True, f"MetaTrader5 module installed (v{version})"
    except ImportError:
        return False, "MetaTrader5 module not installed"


def check_mt5_terminal() -> Tuple[bool, str]:
    """Check if MT5 terminal can be accessed"""
    try:
        import MetaTrader5 as mt5
        # Try to initialize (will fail if terminal not installed)
        mt5.initialize()  # type: ignore[attr-defined]
        terminal_info = mt5.terminal_info()  # type: ignore[attr-defined]
        mt5.shutdown()  # type: ignore[attr-defined]

        if terminal_info:
            return True, f"MT5 Terminal accessible (Build {terminal_info.build})"
        return False, "MT5 Terminal not responding"
    except Exception as e:
        return False, f"MT5 Terminal not accessible: {e}"


def check_credentials() -> Tuple[bool, str]:
    """Check if credentials are configured"""
    try:
        from live.mt5_credentials import MT5Credentials
        if MT5Credentials.ACCOUNT and MT5Credentials.PASSWORD and MT5Credentials.SERVER:
            return True, f"Credentials configured (Account: {MT5Credentials.ACCOUNT}, Server: {MT5Credentials.SERVER})"
        return False, "Credentials incomplete"
    except Exception as e:
        return False, f"Credentials error: {e}"


def check_connection() -> Tuple[bool, str]:
    """Test actual connection to MT5"""
    try:
        import MetaTrader5 as mt5
        from live.mt5_credentials import MT5Credentials

        # Try to connect
        mt5.shutdown()  # type: ignore[attr-defined]  # Close any existing connection
        result = mt5.initialize(  # type: ignore[attr-defined]
            login=MT5Credentials.ACCOUNT,
            password=MT5Credentials.PASSWORD,
            server=MT5Credentials.SERVER,
            timeout=30000
        )

        if result:
            account_info = mt5.account_info()  # type: ignore[attr-defined]
            mt5.shutdown()  # type: ignore[attr-defined]

            if account_info:
                return True, f"Connection successful (Balance: {account_info.balance:.2f} {account_info.currency})"
            return False, "Connected but account info unavailable"

        error = mt5.last_error()  # type: ignore[attr-defined]
        return False, f"Connection failed: {error}"

    except Exception as e:
        return False, f"Connection error: {e}"


def check_required_files() -> Tuple[bool, str]:
    """Check if required files exist"""
    required_files = [
        "backend/main.py",
        "modules/executor/executor.py",
        "modules/executor/adapters/mt5_adapter.py",
        "config/module_registry.yaml",
        "live/mt5_credentials.py",
        "live/mt5_connection_manager.py",
        "start_live_trading.py",
    ]

    missing = []
    for file_path in required_files:
        if not (project_root / file_path).exists():
            missing.append(file_path)

    if not missing:
        return True, f"All {len(required_files)} required files present"
    return False, f"Missing files: {', '.join(missing[:3])}"


def check_executor_config() -> Tuple[bool, str]:
    """Check executor configuration"""
    try:
        # Check if executor can be imported
        from modules.executor.executor import Executor, ExecutorConfig

        # Check if MT5 adapter can be imported
        from modules.executor.adapters.mt5_adapter import MT5Adapter

        return True, "Executor and MT5Adapter ready"
    except Exception as e:
        return False, f"Executor setup error: {e}"


def check_permissions() -> Tuple[bool, str]:
    """Check file permissions and write access"""
    try:
        # Check if we can write logs
        log_dir = project_root / "logs" / "live_trading"
        log_dir.mkdir(parents=True, exist_ok=True)

        test_file = log_dir / "test_write.tmp"
        test_file.write_text("test")
        test_file.unlink()

        return True, "File permissions OK"
    except Exception as e:
        return False, f"Permission error: {e}"


def print_header():
    """Print header"""
    print("\n" + "="*70)
    print(f"{Colors.BOLD}LIVE TRADING SETUP VALIDATOR{Colors.END}")
    print("="*70 + "\n")


def print_check(name: str, status: bool, message: str):
    """Print check result"""
    icon = f"{Colors.GREEN}✓{Colors.END}" if status else f"{Colors.RED}✗{Colors.END}"
    status_text = f"{Colors.GREEN}PASS{Colors.END}" if status else f"{Colors.RED}FAIL{Colors.END}"
    print(f"{icon} {name:30s} [{status_text}]")
    if message:
        indent = "  " if status else f"  {Colors.YELLOW}"
        end = "" if status else Colors.END
        print(f"{indent}{message}{end}")


def run_all_checks() -> List[Tuple[str, bool, str]]:
    """Run all validation checks"""
    checks = [
        ("Python Version", check_python_version),
        ("MT5 Module", check_mt5_installation),
        ("MT5 Terminal", check_mt5_terminal),
        ("Credentials", check_credentials),
        ("Required Files", check_required_files),
        ("Executor Config", check_executor_config),
        ("File Permissions", check_permissions),
        ("MT5 Connection", check_connection),
    ]

    results = []
    for name, check_func in checks:
        try:
            status, message = check_func()
            results.append((name, status, message))
        except Exception as e:
            results.append((name, False, f"Check failed: {e}"))

    return results


def print_summary(results: List[Tuple[str, bool, str]]):
    """Print summary"""
    passed = sum(1 for _, status, _ in results if status)
    total = len(results)

    print("\n" + "="*70)
    print(f"{Colors.BOLD}SUMMARY{Colors.END}")
    print("="*70)

    if passed == total:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ All checks passed! ({passed}/{total}){Colors.END}")
        print(f"\nYou can now start live trading with:")
        print(f"{Colors.BLUE}    python start_live_trading.py{Colors.END}")
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ Some checks failed ({passed}/{total} passed){Colors.END}")
        print(f"\nPlease fix the issues above before starting live trading.")
        print(f"\nCommon fixes:")
        print(f"  - Install MT5: pip install MetaTrader5")
        print(f"  - Check credentials in live/mt5_credentials.py")
        print(f"  - Ensure MT5 terminal is installed and running")

    print("="*70 + "\n")


def main():
    """Main entry point"""
    print_header()

    # Run all checks
    results = run_all_checks()

    # Print results
    print(f"{Colors.BOLD}Running validation checks...{Colors.END}\n")
    for name, status, message in results:
        print_check(name, status, message)

    # Print summary
    print_summary(results)

    # Return exit code
    all_passed = all(status for _, status, _ in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
