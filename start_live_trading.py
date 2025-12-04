#!/usr/bin/env python3
"""
Live Trading Launcher for MT5
-------------------------------
Comprehensive script to start live trading with:
- MT5 connection management
- Backend orchestrator initialization
- Health monitoring
- Auto-recovery
- Graceful shutdown
- Real-time status updates
"""

import os
import sys
import time
import signal
import logging
import argparse
import subprocess
import requests
import json
from typing import Optional, Any
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging with both file and console output"""
    log_dir = project_root / "logs" / "live_trading"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"live_trading_{timestamp}.log"

    # Create logger
    logger = logging.getLogger("LiveTrading")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


class LiveTradingLauncher:
    """Main launcher for live trading system"""

    def __init__(self, logger: logging.Logger, instruments: Optional[list] = None, backend_port: int = 8000):
        self.logger = logger
        self.running = False
        self.backend_process: Optional[Any] = None
        self.backend_url = f"http://localhost:{backend_port}"
        self.backend_port = backend_port
        self.instruments = instruments or ["EURUSD", "XAUUSD"]

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"\nReceived signal {signum}, initiating graceful shutdown...")
        self.shutdown()
        sys.exit(0)

    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met"""
        self.logger.info("Checking prerequisites...")

        # Check MT5 installation
        try:
            import MetaTrader5 as mt5
            self.logger.info("[OK] MetaTrader5 module found")
        except ImportError:
            self.logger.error("[FAIL] MetaTrader5 module not installed")
            self.logger.error("  Install with: pip install MetaTrader5")
            return False

        # Check credentials
        try:
            from live.mt5_credentials import MT5Credentials
            self.logger.info(f"[OK] Credentials loaded (Account: {MT5Credentials.ACCOUNT})")
        except Exception as e:
            self.logger.error(f"[FAIL] Failed to load credentials: {e}")
            return False

        # Check required files
        required_files = [
            "backend/main.py",
            "modules/executor/executor.py",
            "config/module_registry.yaml",
        ]

        for file_path in required_files:
            full_path = project_root / file_path
            if not full_path.exists():
                self.logger.error(f"[FAIL] Required file not found: {file_path}")
                return False

        self.logger.info("[OK] All required files found")
        self.logger.info("[OK] All prerequisites met")
        return True

    def setup_environment(self) -> bool:
        """Setup environment variables for live mode"""
        self.logger.info("Configuring live trading environment...")

        # Set execution mode to live
        os.environ["EXECUTION_MODE"] = "live"
        os.environ["TRADING_MODE"] = "live"

        self.logger.info("[OK] Environment configured for live trading")
        return True

    def start_backend(self) -> bool:
        """Start the backend server using uvicorn"""
        self.logger.info("Starting backend server...")

        try:
            # Start backend using uvicorn
            cmd = [
                sys.executable, "-m", "uvicorn",
                "backend.main:app",
                "--host", "0.0.0.0",
                "--port", str(self.backend_port),
                "--log-level", "info"
            ]

            # Set environment variables
            env = os.environ.copy()
            env["PYTHONPATH"] = str(project_root)
            env["PYTHONUNBUFFERED"] = "1"
            env["EXECUTION_MODE"] = "live"

            # Start backend as subprocess
            # Inherit stdio so we don't block on pipe buffers
            self.backend_process = subprocess.Popen(
                cmd,
                env=env,
                stdout=None,
                stderr=None,
            )

            # Wait for backend to be ready
            self.logger.info("Waiting for backend to be ready...")
            health_url = f"{self.backend_url}/health"
            start_time = time.time()
            timeout = 60

            while time.time() - start_time < timeout:
                if self.backend_process.poll() is not None:
                    self.logger.error("[FAIL] Backend process exited early")
                    return False

                try:
                    response = requests.get(health_url, timeout=2)
                    if response.status_code == 200:
                        self.logger.info("[OK] Backend server is ready")
                        time.sleep(2)  # Give it a moment to fully initialize
                        return True
                except:
                    pass

                time.sleep(1)

            self.logger.error(f"[FAIL] Backend failed to become ready within {timeout}s")
            return False

        except Exception as e:
            self.logger.error(f"[FAIL] Failed to start backend: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def connect_mt5(self) -> bool:
        """Connect to MT5 via backend API (login endpoint)"""
        self.logger.info("Connecting to MT5 via backend...")

        try:
            from live.mt5_credentials import MT5Credentials

            # Correct backend endpoint is /api/login
            url = f"{self.backend_url}/api/login"
            payload = {
                "login": MT5Credentials.ACCOUNT,
                "password": MT5Credentials.PASSWORD,
                "server": MT5Credentials.SERVER,
            }

            response = requests.post(url, json=payload, timeout=60)

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    acct = data.get("account", {})
                    self.logger.info("[OK] Connected to MT5 successfully")
                    self.logger.info(f"  Account: {acct.get('login')}")
                    bal = acct.get('balance')
                    if isinstance(bal, (int, float)):
                        self.logger.info(f"  Balance: {bal:.2f}")
                    return True
                else:
                    self.logger.error(f"[FAIL] MT5 connection failed: {data.get('error') or data.get('detail') or 'Unknown error'}")
                    return False
            else:
                detail = None
                try:
                    body = response.json()
                    detail = body.get('error') or body.get('detail')
                except Exception:
                    pass
                self.logger.error(f"[FAIL] MT5 login request failed: {response.status_code}{' - ' + str(detail) if detail else ''}")
                try:
                    if not detail:
                        self.logger.error(f"  Response: {response.text}")
                except Exception:
                    pass
                return False

        except Exception as e:
            self.logger.error(f"[FAIL] Failed to connect to MT5: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def start_trading(self) -> bool:
        """Start live trading (initializes orchestrator)"""
        self.logger.info("Starting live trading system...")

        try:
            # Call start trading endpoint
            url = f"{self.backend_url}/api/trading/start"
            payload = {
                "instruments": self.instruments,
                "timeframes": ["M15", "H1", "H4", "D1"],
                "update_interval": 5,
                "max_position_size": 0.05,
                "max_total_exposure": 0.30,
                "min_trade_interval": 60,
                "use_trailing_stop": True,
                "emergency_drawdown_limit": 0.25,
                "debug": False
            }

            self.logger.info(f"Trading instruments: {', '.join(self.instruments)}")
            response = requests.post(url, json=payload, timeout=30)

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    self.logger.info("[OK] Live trading started successfully")
                    self.logger.info("  Orchestrator initialized")
                    self.logger.info("  All modules loaded")
                    return True
                else:
                    self.logger.error(f"[FAIL] Failed to start trading: {data.get('message', 'Unknown error')}")
                    return False
            else:
                error_detail = response.json().get("detail", "Unknown error") if response.headers.get('content-type', '').startswith('application/json') else response.text
                self.logger.error(f"[FAIL] Start trading request failed: {error_detail}")
                return False

        except Exception as e:
            self.logger.error(f"[FAIL] Failed to start trading: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def print_status(self) -> None:
        """Print current system status from backend"""
        try:
            # Get system state
            response = requests.get(f"{self.backend_url}/api/system/state", timeout=5)
            if response.status_code != 200:
                return

            state = response.json()

            self.logger.info("\n" + "="*60)
            self.logger.info("LIVE TRADING STATUS")
            self.logger.info("="*60)
            self.logger.info(f"System Status:      {state.get('status', 'UNKNOWN')}")
            self.logger.info(f"MT5 Connected:      {'YES' if state.get('mt5_connected') else 'NO'}")

            metrics = state.get('performance_metrics', {})
            self.logger.info(f"Balance:            {metrics.get('current_balance', 0):.2f}")
            self.logger.info(f"Total P&L:          {metrics.get('total_pnl', 0):.2f}")
            self.logger.info(f"Total Trades:       {metrics.get('total_trades', 0)}")
            self.logger.info(f"Win Rate:           {metrics.get('win_rate', 0)*100:.1f}%")

            self.logger.info("="*60 + "\n")

        except Exception as e:
            self.logger.debug(f"Could not fetch status: {e}")

    def run(self) -> None:
        """Main run loop"""
        self.running = True

        self.logger.info("\n" + "="*60)
        self.logger.info("LIVE TRADING SYSTEM STARTING")
        self.logger.info("="*60 + "\n")

        # Check prerequisites
        if not self.check_prerequisites():
            self.logger.error("Prerequisites check failed. Exiting.")
            return

        # Setup environment
        if not self.setup_environment():
            self.logger.error("Environment setup failed. Exiting.")
            return

        # Start backend
        if not self.start_backend():
            self.logger.error("Backend startup failed. Exiting.")
            return

        # Explicitly connect MT5 before starting trading
        if not self.connect_mt5():
            self.logger.error("Failed to connect to MT5. Exiting.")
            return

        # Start trading (initializes orchestrator)
        if not self.start_trading():
            self.logger.error("Failed to start trading. Exiting.")
            return

        self.logger.info("\n" + "="*60)
        self.logger.info("LIVE TRADING SYSTEM RUNNING")
        self.logger.info("="*60)
        self.logger.info(f"Dashboard: {self.backend_url}")
        self.logger.info(f"API Docs:  {self.backend_url}/docs")
        self.logger.info("Press Ctrl+C to stop gracefully")
        self.logger.info("="*60 + "\n")

        # Main monitoring loop
        status_interval = 60  # Print status every 60 seconds
        last_status_time = time.time()

        try:
            while self.running:
                # Check if backend is still running
                if self.backend_process and self.backend_process.poll() is not None:
                    self.logger.error("Backend process died unexpectedly!")
                    break

                # Print status periodically
                if time.time() - last_status_time >= status_interval:
                    self.print_status()
                    last_status_time = time.time()

                # Sleep for a bit
                time.sleep(5)

        except KeyboardInterrupt:
            self.logger.info("\nReceived keyboard interrupt...")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Graceful shutdown"""
        if not self.running:
            return

        self.logger.info("\n" + "="*60)
        self.logger.info("SHUTTING DOWN LIVE TRADING SYSTEM")
        self.logger.info("="*60)

        self.running = False

        # Stop trading via API
        try:
            self.logger.info("Stopping trading system...")
            response = requests.post(f"{self.backend_url}/api/trading/stop", timeout=10)
            if response.status_code == 200:
                self.logger.info("[OK] Trading stopped")
        except:
            self.logger.warning("Could not stop trading via API")

        # Stop backend
        if self.backend_process:
            self.logger.info("Stopping backend server...")
            try:
                self.backend_process.terminate()
                self.backend_process.wait(timeout=10)
                self.logger.info("[OK] Backend stopped")
            except:
                self.logger.warning("Force killing backend...")
                self.backend_process.kill()

        self.logger.info("[OK] Shutdown complete\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Launch live trading system with MT5"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--instruments",
        nargs="+",
        default=["EURUSD", "XAUUSD"],
        help="Trading instruments (default: EURUSD XAUUSD)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Backend server port (default: 8000)"
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_level)

    logger.info("="*60)
    logger.info("MT5 LIVE TRADING LAUNCHER")
    logger.info("="*60)
    logger.info(f"Python: {sys.version.split()[0]}")
    logger.info(f"Project: {project_root}")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60 + "\n")

    # Create and run launcher
    launcher = LiveTradingLauncher(logger, instruments=args.instruments, backend_port=args.port)
    launcher.run()


if __name__ == "__main__":
    main()
