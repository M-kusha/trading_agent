"""
Enhanced MT5 Connection Manager with Auto-Recovery and Monitoring
Provides robust connection handling, automatic retry logic, and health monitoring
"""

from __future__ import annotations

import time
import threading
from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import logging

from config import get_logger

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False


@dataclass
class ConnectionConfig:
    """Configuration for MT5 connection"""
    account: int
    password: str
    server: str
    timeout: int = 60000  # milliseconds
    max_retries: int = 10
    retry_delay: float = 5.0  # seconds
    health_check_interval: float = 30.0  # seconds
    reconnect_on_error: bool = True
    auto_monitor: bool = True


@dataclass
class ConnectionStatus:
    """Current connection status"""
    connected: bool = False
    last_connect_attempt: Optional[float] = None
    last_success: Optional[float] = None
    connection_failures: int = 0
    total_reconnects: int = 0
    health_check_failures: int = 0
    uptime_seconds: float = 0.0
    last_error: Optional[str] = None


class MT5ConnectionManager:
    """
    Advanced MT5 connection manager with:
    - Automatic retry logic with exponential backoff
    - Connection health monitoring
    - Automatic reconnection on failure
    - Thread-safe operations
    - Connection statistics and logging
    """

    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.status = ConnectionStatus()
        self.logger = get_logger("MT5ConnectionManager")
        self._lock = threading.RLock()
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitor = threading.Event()
        self._callbacks: Dict[str, Optional[Callable]] = {
            "on_connect": None,
            "on_disconnect": None,
            "on_reconnect": None,
            "on_error": None,
        }
        self.logger.setLevel(logging.INFO)

    def register_callback(self, event: str, callback: Callable) -> None:
        """Register callback for connection events"""
        if event in self._callbacks:
            self._callbacks[event] = callback

    def _trigger_callback(self, event: str, *args, **kwargs) -> None:
        """Trigger registered callback"""
        callback = self._callbacks.get(event)
        if callback:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Callback {event} failed: {e}")

    def connect(self, retry: bool = True) -> bool:
        """
        Connect to MT5 with automatic retry logic

        Args:
            retry: Whether to retry on failure

        Returns:
            True if connected successfully
        """
        with self._lock:
            if not MT5_AVAILABLE:
                self.logger.error("MetaTrader5 module not available")
                self.status.last_error = "MT5 module not installed"
                return False

            attempt = 0
            max_attempts = self.config.max_retries if retry else 1

            while attempt < max_attempts:
                attempt += 1
                self.status.last_connect_attempt = time.time()

                try:
                    self.logger.info(
                        f"Connecting to MT5 (attempt {attempt}/{max_attempts})... "
                        f"Account: {self.config.account}, Server: {self.config.server}"
                    )

                    # Shutdown any existing connection
                    try:
                        if mt5:
                            mt5.shutdown()  # type: ignore[attr-defined]
                    except:
                        pass

                    # Initialize MT5
                    if not mt5 or not mt5.initialize(  # type: ignore[attr-defined]
                        login=self.config.account,
                        password=self.config.password,
                        server=self.config.server,
                        timeout=self.config.timeout
                    ):
                        error = mt5.last_error() if mt5 else (1, "MT5 not available")  # type: ignore[attr-defined]
                        error_msg = f"Initialize failed: {error}"
                        self.logger.warning(error_msg)
                        self.status.last_error = error_msg
                        self.status.connection_failures += 1

                        if attempt < max_attempts:
                            # Exponential backoff
                            delay = self.config.retry_delay * (2 ** (attempt - 1))
                            delay = min(delay, 60.0)  # Cap at 60 seconds
                            self.logger.info(f"Retrying in {delay:.1f} seconds...")
                            time.sleep(delay)
                            continue
                        else:
                            self._trigger_callback("on_error", error_msg)
                            return False

                    # Verify connection by getting account info
                    account_info = mt5.account_info() if mt5 else None  # type: ignore[attr-defined]
                    if not account_info:
                        error_msg = "Connected but account info unavailable"
                        self.logger.warning(error_msg)
                        self.status.last_error = error_msg
                        if mt5:
                            mt5.shutdown()  # type: ignore[attr-defined]

                        if attempt < max_attempts:
                            time.sleep(self.config.retry_delay)
                            continue
                        else:
                            return False

                    # Success!
                    self.status.connected = True
                    self.status.last_success = time.time()
                    self.status.connection_failures = 0
                    self.status.last_error = None

                    if self.status.total_reconnects > 0:
                        self.logger.info(f"Reconnected to MT5 successfully")
                        self._trigger_callback("on_reconnect", account_info)
                    else:
                        self.logger.info(f"Connected to MT5 successfully")
                        self._trigger_callback("on_connect", account_info)

                    self.logger.info(
                        f"Account: {account_info.login}, "
                        f"Balance: {account_info.balance:.2f} {account_info.currency}, "
                        f"Leverage: 1:{account_info.leverage}"
                    )

                    # Start monitoring if configured
                    if self.config.auto_monitor and not self._monitor_thread:
                        self.start_monitoring()

                    return True

                except Exception as e:
                    error_msg = f"Connection error: {str(e)}"
                    self.logger.error(error_msg)
                    self.status.last_error = error_msg
                    self.status.connection_failures += 1
                    self._trigger_callback("on_error", error_msg)

                    if attempt < max_attempts:
                        delay = self.config.retry_delay * (2 ** (attempt - 1))
                        delay = min(delay, 60.0)
                        time.sleep(delay)
                    else:
                        return False

            return False

    def disconnect(self) -> None:
        """Gracefully disconnect from MT5"""
        with self._lock:
            if self.config.auto_monitor:
                self.stop_monitoring()

            if MT5_AVAILABLE and mt5:
                try:
                    mt5.shutdown()  # type: ignore[attr-defined]
                    self.logger.info("Disconnected from MT5")
                except:
                    pass

            self.status.connected = False
            self._trigger_callback("on_disconnect")

    def is_connected(self) -> bool:
        """Check if currently connected"""
        with self._lock:
            return self.status.connected and MT5_AVAILABLE

    def check_health(self) -> bool:
        """
        Check connection health

        Returns:
            True if connection is healthy
        """
        with self._lock:
            if not self.is_connected():
                return False

            try:
                # Test connection by getting account info
                account_info = mt5.account_info() if mt5 else None  # type: ignore[attr-defined]
                if not account_info:
                    self.logger.warning("Health check failed: account info unavailable")
                    self.status.health_check_failures += 1
                    return False

                # Reset failure counter on success
                self.status.health_check_failures = 0
                return True

            except Exception as e:
                self.logger.warning(f"Health check failed: {e}")
                self.status.health_check_failures += 1
                return False

    def ensure_connection(self) -> bool:
        """
        Ensure connection is alive, reconnect if needed

        Returns:
            True if connected (or reconnected successfully)
        """
        with self._lock:
            if self.is_connected() and self.check_health():
                return True

            self.logger.warning("Connection lost or unhealthy, attempting reconnect...")
            self.status.connected = False
            self.status.total_reconnects += 1

            return self.connect(retry=True)

    def start_monitoring(self) -> None:
        """Start background connection monitoring thread"""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return

        self._stop_monitor.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="MT5ConnectionMonitor"
        )
        self._monitor_thread.start()
        self.logger.info("Connection monitoring started")

    def stop_monitoring(self) -> None:
        """Stop background connection monitoring"""
        if self._monitor_thread:
            self._stop_monitor.set()
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None
            self.logger.info("Connection monitoring stopped")

    def _monitor_loop(self) -> None:
        """Background monitoring loop"""
        connection_start = time.time()

        while not self._stop_monitor.is_set():
            try:
                # Update uptime
                if self.status.connected:
                    self.status.uptime_seconds = time.time() - connection_start

                # Check health
                if not self.check_health():
                    self.logger.warning(
                        f"Health check failed ({self.status.health_check_failures} times)"
                    )

                    # Reconnect if configured and health checks keep failing
                    if (self.config.reconnect_on_error and
                        self.status.health_check_failures >= 3):
                        self.logger.error("Multiple health check failures, reconnecting...")
                        self.status.connected = False
                        self.ensure_connection()
                        connection_start = time.time()

                # Sleep until next check
                self._stop_monitor.wait(self.config.health_check_interval)

            except Exception as e:
                self.logger.error(f"Monitor loop error: {e}")
                time.sleep(5.0)

    def get_status(self) -> Dict[str, Any]:
        """Get detailed connection status"""
        with self._lock:
            account_info = None
            if self.is_connected() and mt5:
                try:
                    account_info = mt5.account_info()  # type: ignore[attr-defined]
                except:
                    pass

            return {
                "connected": self.status.connected,
                "mt5_available": MT5_AVAILABLE,
                "last_connect_attempt": self.status.last_connect_attempt,
                "last_success": self.status.last_success,
                "connection_failures": self.status.connection_failures,
                "total_reconnects": self.status.total_reconnects,
                "health_check_failures": self.status.health_check_failures,
                "uptime_seconds": self.status.uptime_seconds,
                "last_error": self.status.last_error,
                "account_info": {
                    "login": account_info.login if account_info else None,
                    "balance": float(account_info.balance) if account_info else 0.0,
                    "equity": float(account_info.equity) if account_info else 0.0,
                    "currency": account_info.currency if account_info else None,
                } if account_info else None,
                "monitoring_active": self._monitor_thread and self._monitor_thread.is_alive(),
            }

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()
