# -------------------------------------------------------------
# File: modules/position/emergency_watchdog.py
# Emergency Position Watchdog - Independent safety net
#
# This watchdog runs in a background thread and monitors positions
# for emergency conditions, closing positions that exceed hard limits
# regardless of whether the main orchestrator loop is running.
#
# CRITICAL: This is a SAFETY NET, not a replacement for proper SL.
# Native MT5 stop-loss should be the primary protection!
# -------------------------------------------------------------

import time
import threading
from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import yaml

# Import MT5 if available
try:
    import MetaTrader5 as _mt5_mod  # type: ignore
    from typing import cast, Any as _Any
    mt5 = cast(_Any, _mt5_mod)
    _MT5_AVAILABLE = True
except ImportError:
    from typing import cast, Any as _Any
    mt5 = cast(_Any, None)
    _MT5_AVAILABLE = False

from modules.utils.audit_utils import RotatingLogger


@dataclass
class WatchdogConfig:
    """Configuration for emergency watchdog."""
    # Check interval (seconds)
    check_interval_s: float = 2.0
    
    # Hard stop limit (EUR) - close position if loss exceeds this
    hard_stop_eur: float = 150.0
    
    # Emergency stop (EUR) - absolutely never lose more than this
    emergency_stop_eur: float = 300.0
    
    # Account-level daily loss limit (EUR)
    daily_loss_limit_eur: float = 5000.0
    
    # Enable/disable watchdog
    enabled: bool = True
    
    # Log every check (verbose mode)
    verbose: bool = False


def load_watchdog_config() -> WatchdogConfig:
    """Load watchdog config from risk_policy.yaml."""
    try:
        config_path = Path("config/risk_policy.yaml")
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                policy = yaml.safe_load(f) or {}
            
            smart_pos = policy.get("smart_position", {})
            prop_firm = policy.get("prop_firm", {})
            
            # Calculate daily limit from prop firm config
            account_size = prop_firm.get("account_size", 100000.0)
            daily_dd_pct = prop_firm.get("daily_drawdown_limit", 0.05)
            daily_limit = account_size * daily_dd_pct * 0.7  # 70% of limit as safety margin
            
            return WatchdogConfig(
                hard_stop_eur=smart_pos.get("hard_stop_loss_eur", 150.0),
                emergency_stop_eur=smart_pos.get("hard_stop_loss_eur", 150.0) * 2,  # 2x hard stop
                daily_loss_limit_eur=daily_limit,
            )
    except Exception as e:
        print(f"[Watchdog] Failed to load config: {e}")
    
    return WatchdogConfig()


class EmergencyPositionWatchdog:
    """
    Independent watchdog thread that monitors positions for emergency conditions.
    
    This is a SAFETY NET that runs independently of the main orchestrator loop.
    It provides fast protection against:
    - Slow orchestrator loops
    - Orchestrator crashes/hangs
    - Missed exit signals
    
    WARNING: This is NOT a replacement for native MT5 stop-loss!
    Native SL is always faster and more reliable.
    """
    
    def __init__(
        self,
        config: Optional[WatchdogConfig] = None,
        on_emergency_close: Optional[Callable[[str, float, str], None]] = None,
    ):
        self.config = config or load_watchdog_config()
        self.on_emergency_close = on_emergency_close
        
        self.logger = RotatingLogger(
            "EmergencyWatchdog",
            log_path="logs/position/emergency_watchdog.log",
            operator_mode=True,
            max_lines=5000,
        )
        
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_running = False
        
        # Track daily losses
        self._daily_realized_loss: float = 0.0
        self._last_reset_day: int = 0
        
        # Track emergency closes
        self._emergency_closes: list = []
    
    def start(self) -> bool:
        """Start the watchdog thread."""
        if not self.config.enabled:
            self.logger.info("[Watchdog] Disabled by config")
            return False
        
        if not _MT5_AVAILABLE:
            self.logger.error("[Watchdog] MT5 not available - cannot start")
            return False
        
        if self._is_running:
            self.logger.warning("[Watchdog] Already running")
            return True
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._thread.start()
        self._is_running = True
        
        self.logger.info(
            f"[Watchdog] 🚨 Started - checking every {self.config.check_interval_s}s | "
            f"Hard stop: €{self.config.hard_stop_eur} | Emergency: €{self.config.emergency_stop_eur}"
        )
        return True
    
    def stop(self) -> None:
        """Stop the watchdog thread."""
        if not self._is_running:
            return
        
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        self._is_running = False
        self.logger.info("[Watchdog] Stopped")
    
    def _watchdog_loop(self) -> None:
        """Main watchdog loop - runs in background thread."""
        while not self._stop_event.is_set():
            try:
                self._check_positions()
            except Exception as e:
                self.logger.error(f"[Watchdog] Error in check loop: {e}")
            
            # Sleep in small increments to allow fast shutdown
            for _ in range(int(self.config.check_interval_s * 10)):
                if self._stop_event.is_set():
                    break
                time.sleep(0.1)
    
    def _check_positions(self) -> None:
        """Check all positions for emergency conditions."""
        try:
            # Ensure MT5 is initialized
            if not mt5.terminal_info():
                if not mt5.initialize():
                    return
            
            positions = mt5.positions_get()
            if not positions:
                return
            
            # Reset daily loss counter if new day
            current_day = time.localtime().tm_yday
            if current_day != self._last_reset_day:
                self._daily_realized_loss = 0.0
                self._last_reset_day = current_day
            
            # Check each position
            total_unrealized = 0.0
            for pos in positions:
                symbol = pos.symbol
                profit = pos.profit
                total_unrealized += profit
                
                if self.config.verbose:
                    self.logger.debug(f"[Watchdog] {symbol}: €{profit:.2f}")
                
                # Check hard stop
                if profit <= -self.config.hard_stop_eur:
                    self._emergency_close(pos, "HARD_STOP", f"Loss €{profit:.2f} exceeds hard stop €{self.config.hard_stop_eur}")
                
                # Check emergency stop
                elif profit <= -self.config.emergency_stop_eur:
                    self._emergency_close(pos, "EMERGENCY_STOP", f"Loss €{profit:.2f} exceeds emergency limit €{self.config.emergency_stop_eur}")
            
            # Check total unrealized + realized against daily limit
            total_loss = total_unrealized + self._daily_realized_loss
            if total_loss <= -self.config.daily_loss_limit_eur:
                self.logger.critical(
                    f"[Watchdog] 🚨🚨🚨 DAILY LIMIT BREACH! "
                    f"Total: €{total_loss:.2f} > limit €{self.config.daily_loss_limit_eur}"
                )
                # Close ALL positions
                for pos in positions:
                    self._emergency_close(pos, "DAILY_LIMIT", f"Daily loss limit breached")
        
        except Exception as e:
            self.logger.error(f"[Watchdog] Position check failed: {e}")
    
    def _emergency_close(self, position: Any, reason: str, details: str) -> bool:
        """Emergency close a position."""
        try:
            ticket = position.ticket
            symbol = position.symbol
            volume = position.volume
            profit = position.profit
            pos_type = position.type
            
            self.logger.critical(
                f"[Watchdog] 🚨 EMERGENCY CLOSE: {symbol} | "
                f"Reason: {reason} | {details}"
            )
            
            # Determine close direction (opposite of position)
            if pos_type == mt5.POSITION_TYPE_BUY:
                close_type = mt5.ORDER_TYPE_SELL
                tick = mt5.symbol_info_tick(symbol)
                price = tick.bid if tick else 0
            else:
                close_type = mt5.ORDER_TYPE_BUY
                tick = mt5.symbol_info_tick(symbol)
                price = tick.ask if tick else 0
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": close_type,
                "position": ticket,
                "price": price,
                "deviation": 50,  # Wider deviation for emergency
                "magic": 424243,  # Different magic for watchdog closes
                "comment": f"WATCHDOG:{reason}",
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result and result.retcode in (10009, 10008):  # Done or Placed
                self.logger.critical(
                    f"[Watchdog] ✅ Emergency close SUCCESS: {symbol} ticket {ticket}"
                )
                self._daily_realized_loss += profit
                self._emergency_closes.append({
                    "time": time.time(),
                    "symbol": symbol,
                    "ticket": ticket,
                    "profit": profit,
                    "reason": reason,
                })
                
                # Call callback if provided
                if self.on_emergency_close:
                    try:
                        self.on_emergency_close(symbol, profit, reason)
                    except Exception:
                        pass
                
                return True
            else:
                retcode = result.retcode if result else "None"
                self.logger.error(
                    f"[Watchdog] ❌ Emergency close FAILED: {symbol} | retcode={retcode}"
                )
                return False
        
        except Exception as e:
            self.logger.error(f"[Watchdog] Emergency close exception: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get watchdog statistics."""
        return {
            "is_running": self._is_running,
            "config": {
                "hard_stop_eur": self.config.hard_stop_eur,
                "emergency_stop_eur": self.config.emergency_stop_eur,
                "check_interval_s": self.config.check_interval_s,
            },
            "daily_realized_loss": self._daily_realized_loss,
            "emergency_closes_today": len([
                c for c in self._emergency_closes 
                if time.localtime(c["time"]).tm_yday == time.localtime().tm_yday
            ]),
            "total_emergency_closes": len(self._emergency_closes),
        }


# Singleton instance
_watchdog_instance: Optional[EmergencyPositionWatchdog] = None


def get_emergency_watchdog() -> EmergencyPositionWatchdog:
    """Get or create the singleton watchdog instance."""
    global _watchdog_instance
    if _watchdog_instance is None:
        _watchdog_instance = EmergencyPositionWatchdog()
    return _watchdog_instance


def start_emergency_watchdog() -> bool:
    """Start the emergency watchdog (convenience function)."""
    return get_emergency_watchdog().start()


def stop_emergency_watchdog() -> None:
    """Stop the emergency watchdog (convenience function)."""
    if _watchdog_instance:
        _watchdog_instance.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  EMERGENCY POSITION WATCHDOG")
    print("="*60)
    
    config = load_watchdog_config()
    print(f"\nConfiguration:")
    print(f"  Hard stop: €{config.hard_stop_eur}")
    print(f"  Emergency stop: €{config.emergency_stop_eur}")
    print(f"  Daily limit: €{config.daily_loss_limit_eur}")
    print(f"  Check interval: {config.check_interval_s}s")
    
    if _MT5_AVAILABLE:
        print("\n✅ MT5 available - watchdog can run")
        
        # Try to start watchdog
        watchdog = get_emergency_watchdog()
        if watchdog.start():
            print("✅ Watchdog started!")
            print("\nPress Ctrl+C to stop...")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping...")
                watchdog.stop()
        else:
            print("❌ Failed to start watchdog")
    else:
        print("\n❌ MT5 not available - watchdog cannot run")
        print("   Install MetaTrader5 package: pip install MetaTrader5")
