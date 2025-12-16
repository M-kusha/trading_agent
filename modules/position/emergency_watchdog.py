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
    """
    Configuration for emergency watchdog.

    NOTE: The watchdog is a BACKGROUND SAFETY NET that runs independently.
    It uses the same thresholds as ExitConfig (exit_engine.py) / prop_firm
    config for consistency.

    The watchdog should only trigger on truly critical conditions that the
    main loop missed due to being slow or hung.
    
    CRITICAL: Account closes at EXACTLY 10% drawdown. We MUST stay under!
    """

    # Check interval (seconds)
    check_interval_s: float = 2.0

    # Hard stop limit (EUR) - close position if loss exceeds this
    # Uses same value as ExitConfig.hard_stop_loss_eur by default.
    hard_stop_eur: float = 150.0

    # Emergency stop (EUR) - absolutely never lose more than this per position.
    # This is ABOVE hard stop - a true failsafe.
    emergency_stop_eur: float = 300.0

    # Account-level daily loss limit (EUR) - 5% of €100k = €5000
    daily_loss_limit_eur: float = 5000.0
    
    # Account-level MAX drawdown limit (EUR) - CRITICAL: 10% = account closed!
    # Set to 9% (€9000) to leave €1000 buffer for slippage
    max_loss_limit_eur: float = 9000.0
    
    # Starting balance for drawdown calculation
    starting_balance_eur: float = 100_000.0

    # Enable/disable watchdog
    enabled: bool = True

    # Log every check (verbose mode)
    verbose: bool = False


def load_watchdog_config() -> WatchdogConfig:
    """
    Load watchdog config from config/risk_policy.yaml.

    Priority:
    1) policy["watchdog"] overrides everything explicitly
    2) fall back to exit_strategies/smart_position for per-position stops
    3) derive daily loss limit from prop_firm block if present

    The goal is to stay consistent with:
      - ExitStrategyEngine config
      - prop_firm daily drawdown rules
    """
    try:
        config_path = Path("config/risk_policy.yaml")
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                policy = yaml.safe_load(f) or {}

            exit_cfg = policy.get("exit_strategies", {}) or {}
            smart_pos = policy.get("smart_position", {}) or {}
            prop_firm = policy.get("prop_firm", {}) or {}
            watchdog_cfg = policy.get("watchdog", {}) or {}

            # --- Hard stop per position -------------------------------------
            hard_stop = watchdog_cfg.get("hard_stop_eur")
            if hard_stop is None:
                hard_stop = (
                    exit_cfg.get("hard_stop_loss_eur")
                    or smart_pos.get("hard_stop_loss_eur")
                    or 150.0
                )
            hard_stop = float(hard_stop)

            # --- Emergency stop per position --------------------------------
            emergency_stop = watchdog_cfg.get("emergency_stop_eur")
            if emergency_stop is None:
                # Allow optional multiplier in config, else 2x hard stop
                mult = float(watchdog_cfg.get("emergency_multiplier", 2.0))
                emergency_stop = hard_stop * max(mult, 1.1)
            emergency_stop = float(emergency_stop)

            # Ensure emergency > hard stop to keep semantics sane
            if emergency_stop <= hard_stop:
                emergency_stop = hard_stop * 1.5

            # --- Daily limit from prop_firm block ---------------------------
            daily_limit = watchdog_cfg.get("daily_loss_limit_eur")
            if daily_limit is None:
                account_size = float(prop_firm.get("account_size", 100_000.0) or 100_000.0)
                daily_dd_pct = float(
                    prop_firm.get("daily_drawdown_limit", 0.05) or 0.05
                )  # e.g. 5%
                # Safety margin factor (e.g. 0.7 => trigger at 70% of official limit)
                safety_margin = float(prop_firm.get("watchdog_safety_margin", 0.7) or 0.7)
                daily_limit = account_size * daily_dd_pct * safety_margin
            daily_limit = float(daily_limit)

            # --- MAX DRAWDOWN limit (CRITICAL: 10% = account closed!) -------
            account_size = float(prop_firm.get("account_size", 100_000.0) or 100_000.0)
            max_limit = watchdog_cfg.get("max_loss_limit_eur")
            if max_limit is None:
                max_dd_pct = float(
                    prop_firm.get("max_drawdown_limit", 0.10) or 0.10
                )  # 10%
                # Safety margin: stop at 90% of limit (9% instead of 10%)
                max_limit = account_size * max_dd_pct * 0.90
            max_limit = float(max_limit)

            # --- Misc options -----------------------------------------------
            check_interval = float(watchdog_cfg.get("check_interval_s", 2.0))
            enabled = bool(watchdog_cfg.get("enabled", True))
            verbose = bool(watchdog_cfg.get("verbose", False))

            return WatchdogConfig(
                check_interval_s=check_interval,
                hard_stop_eur=hard_stop,
                emergency_stop_eur=emergency_stop,
                daily_loss_limit_eur=daily_limit,
                max_loss_limit_eur=max_limit,
                starting_balance_eur=account_size,
                enabled=enabled,
                verbose=verbose,
            )
    except Exception as e:  # pragma: no cover - defensive only
        print(f"[Watchdog] Failed to load config: {e}")

    # Fallback defaults – reasonable but conservative.
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

        # Track daily losses (realized PnL from watchdog-triggered closes)
        self._daily_realized_loss: float = 0.0
        self._last_reset_day: int = 0

        # Track all emergency closes (for diagnostics / dashboard)
        self._emergency_closes: list = []

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
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
            (
                f"[Watchdog] 🚨 Started - interval={self.config.check_interval_s:.1f}s | "
                f"Hard stop=€{self.config.hard_stop_eur:.2f} | "
                f"Emergency=€{self.config.emergency_stop_eur:.2f} | "
                f"Daily limit=€{self.config.daily_loss_limit_eur:.2f}"
            )
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

    # ------------------------------------------------------------------ #
    # Core loop
    # ------------------------------------------------------------------ #
    def _watchdog_loop(self) -> None:
        """Main watchdog loop - runs in background thread."""
        while not self._stop_event.is_set():
            try:
                self._check_positions()
            except Exception as e:  # pragma: no cover - defensive
                self.logger.error(f"[Watchdog] Error in check loop: {e}")

            # Sleep in small increments to allow fast shutdown
            remaining = float(self.config.check_interval_s)
            while remaining > 0.0 and not self._stop_event.is_set():
                step = min(0.1, remaining)
                time.sleep(step)
                remaining -= step

    def _check_positions(self) -> None:
        """Check all positions for emergency conditions (hard / emergency / daily)."""
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
                profit = float(pos.profit)
                total_unrealized += profit

                if self.config.verbose:
                    self.logger.debug(f"[Watchdog] {symbol}: €{profit:.2f}")

                # ----------------------------------------------------------
                # EMERGENCY_STOP has higher priority than HARD_STOP.
                # Check the stricter condition first.
                # ----------------------------------------------------------
                if profit <= -self.config.emergency_stop_eur:
                    self._emergency_close(
                        pos,
                        "EMERGENCY_STOP",
                        (
                            f"Loss €{profit:.2f} exceeds emergency limit "
                            f"€{self.config.emergency_stop_eur:.2f}"
                        ),
                    )
                elif profit <= -self.config.hard_stop_eur:
                    self._emergency_close(
                        pos,
                        "HARD_STOP",
                        (
                            f"Loss €{profit:.2f} exceeds hard stop "
                            f"€{self.config.hard_stop_eur:.2f}"
                        ),
                    )

            # Check total unrealized + realized against daily limit
            total_loss = total_unrealized + self._daily_realized_loss
            if total_loss <= -self.config.daily_loss_limit_eur:
                self.logger.critical(
                    "[Watchdog] 🚨🚨🚨 DAILY LIMIT BREACH! "
                    f"Total PnL: €{total_loss:.2f} (limit=€{self.config.daily_loss_limit_eur:.2f})"
                )
                # Close ALL positions
                for pos in positions:
                    self._emergency_close(pos, "DAILY_LIMIT", "Daily loss limit breached")
                return  # Don't check further after closing all

            # ══════════════════════════════════════════════════════════════
            # CHECK MAX DRAWDOWN (CRITICAL: 10% = ACCOUNT CLOSED!)
            # This is the absolute last line of defense before account death
            # ══════════════════════════════════════════════════════════════
            try:
                account_info = mt5.account_info()
                if account_info:
                    current_equity = float(account_info.equity)
                    current_balance = float(account_info.balance)
                    
                    # LIVE ADAPTATION: Use actual MT5 balance as baseline when
                    # config doesn't match (e.g., 100k config vs 10k demo account).
                    # This prevents false "90% drawdown" alerts on demo accounts.
                    configured_start = self.config.starting_balance_eur
                    if configured_start > 0 and current_balance > 0:
                        ratio = configured_start / current_balance
                        if ratio > 1.5 or ratio < 0.67:
                            # Mismatch detected - use live balance as baseline
                            starting_balance = current_balance
                            # Also scale max_loss_limit proportionally
                            max_loss_limit = self.config.max_loss_limit_eur * (current_balance / configured_start)
                        else:
                            starting_balance = configured_start
                            max_loss_limit = self.config.max_loss_limit_eur
                    else:
                        starting_balance = configured_start if configured_start > 0 else current_balance
                        max_loss_limit = self.config.max_loss_limit_eur
                    
                    total_drawdown = starting_balance - current_equity
                    
                    if total_drawdown >= max_loss_limit:
                        self.logger.critical(
                            f"[Watchdog] 🚨🚨🚨 MAX DRAWDOWN BREACH! "
                            f"Drawdown: €{total_drawdown:.2f} >= limit €{max_loss_limit:.2f} "
                            f"(Starting: €{starting_balance:.2f}, Current Equity: €{current_equity:.2f})"
                        )
                        # CLOSE ALL POSITIONS IMMEDIATELY
                        for pos in positions:
                            self._emergency_close(pos, "MAX_DRAWDOWN", 
                                f"Max drawdown €{total_drawdown:.2f} breached - PROTECTING ACCOUNT")
            except Exception as e:
                self.logger.warning(f"[Watchdog] Could not check max drawdown: {e}")

        except Exception as e:  # pragma: no cover - defensive
            self.logger.error(f"[Watchdog] Position check failed: {e}")

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    def _emergency_close(self, position: Any, reason: str, details: str) -> bool:
        """Emergency close a single position."""
        try:
            ticket = position.ticket
            symbol = position.symbol
            volume = float(position.volume)
            profit = float(position.profit)
            pos_type = position.type

            self.logger.critical(
                f"[Watchdog] 🚨 EMERGENCY CLOSE: {symbol} | "
                f"Reason={reason} | {details}"
            )

            # Determine close direction (opposite of position)
            if pos_type == mt5.POSITION_TYPE_BUY:
                close_type = mt5.ORDER_TYPE_SELL
                tick = mt5.symbol_info_tick(symbol)
                price = tick.bid if tick else 0.0
            else:
                close_type = mt5.ORDER_TYPE_BUY
                tick = mt5.symbol_info_tick(symbol)
                price = tick.ask if tick else 0.0

            request: Dict[str, Any] = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": close_type,
                "position": ticket,
                "price": price,
                "deviation": 50,  # Wider deviation for emergency
                "magic": 424243,  # Distinct magic for watchdog closes
                "comment": f"WATCHDOG:{reason}",
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)

            if result and result.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
                self.logger.critical(
                    f"[Watchdog] ✅ Emergency close SUCCESS: {symbol} ticket={ticket}"
                )
                # Profit is negative for losses; keep sign so daily check is consistent.
                self._daily_realized_loss += profit
                self._emergency_closes.append(
                    {
                        "time": time.time(),
                        "symbol": symbol,
                        "ticket": ticket,
                        "profit": profit,
                        "reason": reason,
                    }
                )

                # Callback hook for dashboard / alerting layer
                if self.on_emergency_close:
                    try:
                        self.on_emergency_close(symbol, profit, reason)
                    except Exception:
                        # Callback errors must not affect safety net
                        pass

                return True

            retcode = result.retcode if result else "None"
            self.logger.error(
                f"[Watchdog] ❌ Emergency close FAILED: {symbol} | retcode={retcode}"
            )
            return False

        except Exception as e:  # pragma: no cover - defensive
            self.logger.error(f"[Watchdog] Emergency close exception: {e}")
            return False

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def get_stats(self) -> Dict[str, Any]:
        """Expose basic watchdog statistics for dashboards / health checks."""
        today = time.localtime().tm_yday
        emergency_closes_today = len(
            [
                c
                for c in self._emergency_closes
                if time.localtime(c["time"]).tm_yday == today
            ]
        )
        return {
            "is_running": self._is_running,
            "config": {
                "check_interval_s": self.config.check_interval_s,
                "hard_stop_eur": self.config.hard_stop_eur,
                "emergency_stop_eur": self.config.emergency_stop_eur,
                "daily_loss_limit_eur": self.config.daily_loss_limit_eur,
                "enabled": self.config.enabled,
                "verbose": self.config.verbose,
            },
            "daily_realized_loss": self._daily_realized_loss,
            "emergency_closes_today": emergency_closes_today,
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
    print("\n" + "=" * 60)
    print("  EMERGENCY POSITION WATCHDOG")
    print("=" * 60)

    cfg = load_watchdog_config()
    print("\nConfiguration:")
    print(f"  Hard stop:      €{cfg.hard_stop_eur:.2f}")
    print(f"  Emergency stop: €{cfg.emergency_stop_eur:.2f}")
    print(f"  Daily limit:    €{cfg.daily_loss_limit_eur:.2f}")
    print(f"  Check interval: {cfg.check_interval_s:.2f}s")
    print(f"  Enabled:        {cfg.enabled}")
    print(f"  Verbose:        {cfg.verbose}")

    if _MT5_AVAILABLE:
        print("\n✅ MT5 available - watchdog can run")

        watchdog = get_emergency_watchdog()
        if watchdog.start():
            print("✅ Watchdog started! (Ctrl+C to stop)")
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                print("\nStopping...")
                watchdog.stop()
        else:
            print("❌ Failed to start watchdog")
    else:
        print("\n❌ MT5 not available - watchdog cannot run")
        print("   Install MetaTrader5 package: pip install MetaTrader5")
