#!/usr/bin/env python3
"""
Live Trading Script - Orchestrator Mode
========================================
The ModuleOrchestrator handles EVERYTHING including PPO model predictions.
PPOAgentShell module loads the model and makes all trading decisions.
This script just starts the orchestrator and feeds it market data.
"""

import os
import sys
import time
import signal
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import MetaTrader5 as mt5
    from modules.utils.info_bus import SmartInfoBus
    from modules.core.module_system import ModuleOrchestrator as OrchestratorType
    from live.live_connector import LiveDataConnector as ConnectorType

# Ensure directories exist BEFORE logging setup
Path("logs").mkdir(exist_ok=True)
Path("state").mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/live_orchestrated.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("LiveTrading")

# Global flag for graceful shutdown
running = True

def signal_handler(signum, frame):
    global running
    logger.info(f"\nReceived signal {signum}, shutting down gracefully...")
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class LiveTradingOrchestrated:
    """
    Live trading - Orchestrator handles ALL decisions.
    
    Architecture:
        run_live_simple.py (this file)
            → ModuleOrchestrator.execute_step()
                → PPOAgentShell.process()  ← Loads model, makes predictions
                → ArbiterLogic             ← Final decision making
                → Executor                 ← Executes trades via MT5
    """
    
    def __init__(self):
        self.mt5: Any = None
        self.connector: Any = None
        self.orchestrator: Any = None
        self.info_bus: Any = None
        self.account_info: Any = None
        self.session_tracker: Any = None  # v5.5: Live session tracking
        self.instruments = ["XAUUSD"]
        self.timeframes = ["M15", "H1", "H4", "D1"]
        
    def initialize(self) -> bool:
        """Initialize all components."""
        logger.info("=" * 70)
        logger.info("LIVE TRADING - Orchestrator Mode")
        logger.info("PPOAgentShell handles all model predictions")
        logger.info("=" * 70)
        
        # 1. Set trading mode
        os.environ["EXECUTION_MODE"] = "live"
        os.environ["TRADING_MODE"] = "live"
        
        try:
            from modules.core.trading_mode import TradingModeManager
            TradingModeManager.set_mode("LIVE")
            logger.info("[OK] TradingModeManager set to LIVE")
        except Exception as e:
            logger.warning(f"[WARN] Could not set TradingModeManager: {e}")
        
        # 2. Initialize InfoBus
        try:
            from modules.utils.info_bus import InfoBusManager
            self.info_bus = InfoBusManager.get_instance()
            
            # Set initial config
            env_config = {
                "instruments": [f"{i[:3]}/{i[3:]}" if len(i) == 6 else i for i in self.instruments],
                "initial_balance": 100000.0,
                "mode": "live",
                "max_steps": 100000,
                "bus_data_active": True,
            }
            self.info_bus.set("environment_config", env_config, module="LiveTrading", thesis="live trading startup")
            self.info_bus.set("execution_mode", "live", module="LiveTrading", thesis="live mode active")
            logger.info("[OK] InfoBus initialized")
        except Exception as e:
            logger.error(f"[FAIL] InfoBus initialization failed: {e}")
            return False
        
        # 2b. Initialize LiveSessionTracker for governor observation (v5.5)
        try:
            from modules.monitoring.live_session_tracker import LiveSessionTracker, LiveSessionConfig
            
            # Load session config from live config if available
            session_config = LiveSessionConfig(
                session_loss_limit_pct=0.03,       # Match LIVE_READY stage
                session_consecutive_loss_limit=3,  # Match LIVE_READY stage
                max_trades_per_session=4,
                loss_layer_stop=3,
                max_consecutive_losses=3,
            )
            self.session_tracker = LiveSessionTracker(self.info_bus, session_config)
            logger.info("[OK] LiveSessionTracker initialized (v5.5 governor observation)")
        except Exception as e:
            logger.warning(f"[WARN] LiveSessionTracker initialization failed: {e}")
            logger.warning("       Governor observation dims [76-83] will use defaults")
            self.session_tracker = None
        
        # 3. Load credentials and connect MT5
        try:
            from live.mt5_credentials import MT5Credentials
            import MetaTrader5 as mt5  # type: ignore[import]
            self.mt5 = mt5
            
            if not mt5.initialize():  # type: ignore[attr-defined]  # type: ignore[attr-defined]
                logger.error(f"[FAIL] MT5 initialization failed: {mt5.last_error()}")  # type: ignore[attr-defined]
                return False
            
            if not mt5.login(MT5Credentials.ACCOUNT, password=MT5Credentials.PASSWORD, server=MT5Credentials.SERVER):  # type: ignore[attr-defined]
                logger.error(f"[FAIL] MT5 login failed: {mt5.last_error()}")  # type: ignore[attr-defined]
                mt5.shutdown()  # type: ignore[attr-defined]
                return False
            
            self.account_info = mt5.account_info()  # type: ignore[attr-defined]
            logger.info(f"[OK] Connected to MT5 - Account: {MT5Credentials.ACCOUNT}, Balance: ${self.account_info.balance:.2f}")
            
            # Update InfoBus with real balance
            env_config["initial_balance"] = self.account_info.balance
            self.info_bus.set("environment_config", env_config, module="LiveTrading", thesis="updated with real balance")
            self.info_bus.set("account_balance", self.account_info.balance, module="LiveTrading", thesis="MT5 balance")
            
            # Initialize session tracker with real account balance (v5.5)
            if self.session_tracker:
                self.session_tracker.initialize_from_account(self.account_info.balance)
                self.session_tracker.update_and_publish()
            
        except Exception as e:
            logger.error(f"[FAIL] MT5 connection failed: {e}")
            return False
        
        # 4. Setup data connector (provides market data to orchestrator)
        try:
            from live.live_connector import LiveDataConnector
            
            self.connector = LiveDataConnector(instruments=self.instruments, timeframes=self.timeframes)
            self.connector.connect()
            
            # Fetch initial historical data
            hist_data = self.connector.get_historical_data(n_bars=1000)
            if not hist_data:
                logger.error("[FAIL] Could not get historical data")
                return False
            
            logger.info(f"[OK] LiveDataConnector initialized with {len(hist_data)} instruments")
            
            # Publish market data to InfoBus for modules to use
            self.info_bus.set("market_data", hist_data, module="LiveTrading", thesis="historical data")
            
        except Exception as e:
            logger.error(f"[FAIL] Data connector setup failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 5. Initialize ModuleOrchestrator (this loads PPOAgentShell which loads the model!)
        try:
            from modules.core.module_system import ModuleOrchestrator
            self.orchestrator = ModuleOrchestrator.get_instance()
            self.orchestrator.initialize()
            logger.info(f"[OK] ModuleOrchestrator initialized - {len(self.orchestrator.modules)} modules loaded")
            
            # Verify critical modules are loaded
            # NOTE: ArbiterLogic runs inside PPOAgentShell (it's not a standalone orchestrator module).
            critical_modules = ["PPOAgentShell", "Executor", "PositionManager"]
            optional_modules = ["ArbiterLogic"]
            for name in critical_modules + optional_modules:
                if name in self.orchestrator.modules:
                    logger.info(f"     ✓ {name}")
                elif name == "ArbiterLogic" and "PPOAgentShell" in self.orchestrator.modules:
                    logger.info("     ✓ ArbiterLogic (embedded in PPOAgentShell)")
                else:
                    logger.warning(f"     ✗ {name} NOT LOADED - trading may not work!")
            
            other_count = len(self.orchestrator.modules) - len([m for m in critical_modules if m in self.orchestrator.modules])
            if other_count > 0:
                logger.info(f"     + {other_count} additional modules")
                
        except Exception as e:
            logger.error(f"[FAIL] ModuleOrchestrator initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        logger.info("=" * 70)
        logger.info("[OK] All systems initialized!")
        logger.info("     → PPOAgentShell will handle all trading decisions")
        logger.info("     → Executor will execute trades via MT5")
        logger.info("=" * 70)
        return True
    
    async def run_async(self):
        """Main trading loop - orchestrator handles all decisions."""
        global running
        
        logger.info("Starting trading loop... Press Ctrl+C to stop")
        logger.info("-" * 70)
        
        step = 0
        start_balance = self.account_info.balance
        last_status_log = time.time()
        
        while running:
            step += 1
            loop_start = time.time()
            
            try:
                # Get fresh market data
                market_data = self.connector.get_historical_data(n_bars=100) or {}
                
                # Update market data in InfoBus
                if market_data:
                    self.info_bus.set("market_data", market_data, module="LiveTrading", thesis="market data update")
                
                # Update session tracker (v5.5 governor observation)
                if self.session_tracker:
                    self.session_tracker.update_and_publish()
                    
                    # Check for trade close events from Executor
                    try:
                        last_trade_result = self.info_bus.get("last_trade_result", module="LiveTrading", default=None)
                        if last_trade_result and isinstance(last_trade_result, dict):
                            trade_id = last_trade_result.get("trade_id")
                            # Only process if it's a new trade (track by id)
                            if trade_id and trade_id != getattr(self, "_last_processed_trade_id", None):
                                pnl = float(last_trade_result.get("pnl", 0))
                                self.session_tracker.on_trade_close(pnl)
                                self._last_processed_trade_id = trade_id
                                logger.info(f"[SessionTracker] Trade closed: pnl={pnl:.2f}")
                    except Exception:
                        pass  # Best-effort trade tracking
                
                # Execute orchestrator step - THIS RUNS EVERYTHING:
                #   - PPOAgentShell builds observations and runs model.predict()
                #   - ArbiterLogic makes final decisions
                #   - Executor executes trades via MT5
                await self.orchestrator.execute_step(market_data)
                
                # Log decisions from InfoBus (published by PPOAgentShell)
                try:
                    ppo_decision = self.info_bus.get("ppo_decision", module="LiveTrading", default=None)
                    if ppo_decision and step % 30 == 0:  # Log every 30 steps
                        logger.info(f"[PPO] Decision: {ppo_decision}")
                except Exception:
                    pass  # InfoBus get can fail, ignore
                
                # Periodic status (every 60 seconds)
                if time.time() - last_status_log > 60:
                    acc = self.mt5.account_info()
                    if acc:
                        pnl = acc.balance - start_balance
                        pnl_pct = (pnl / start_balance) * 100
                        emoji = "📈" if pnl >= 0 else "📉"
                        logger.info(f"[Status] Step {step} | {emoji} Balance: ${acc.balance:.2f} | P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
                        
                        # Update InfoBus
                        self.info_bus.set("account_balance", acc.balance, module="LiveTrading", thesis="balance update")
                        self.info_bus.set("account_equity", acc.equity, module="LiveTrading", thesis="equity update")
                        
                        # Check positions
                        positions = self.mt5.positions_get()
                        if positions:
                            logger.info(f"         📊 Open positions: {len(positions)}")
                            for pos in positions:
                                side = "LONG" if pos.type == 0 else "SHORT"
                                logger.info(f"            {pos.symbol} {side} {pos.volume} lots | P&L: ${pos.profit:+.2f}")
                    
                    last_status_log = time.time()
                
                # Maintain ~2 second loop time
                loop_time = time.time() - loop_start
                sleep_time = max(0, 2.0 - loop_time)
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(2)
        
        # Final stats
        logger.info("=" * 70)
        logger.info("TRADING SESSION ENDED")
        logger.info("=" * 70)
        logger.info(f"Total steps: {step}")
        
        acc = self.mt5.account_info()  # type: ignore[union-attr]
        if acc:
            pnl = acc.balance - start_balance
            pnl_pct = (pnl / start_balance) * 100
            logger.info(f"Final Balance: ${acc.balance:.2f}")
            logger.info(f"Session P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
    
    def shutdown(self):
        """Clean shutdown."""
        logger.info("Shutting down...")
        
        # Save orchestrator state
        if self.orchestrator:
            try:
                if hasattr(self.orchestrator, 'state_manager'):
                    results = self.orchestrator.state_manager.save_all_module_states(self.orchestrator)
                    saved = sum(1 for ok in results.values() if ok)
                    logger.info(f"[OK] Saved {saved}/{len(results)} module states")
            except Exception as e:
                logger.warning(f"[WARN] Could not save module states: {e}")
        
        # Disconnect
        if self.connector:
            try:
                self.connector.disconnect()
            except:
                pass
        
        if self.mt5:
            try:
                self.mt5.shutdown()  # type: ignore[union-attr]
                logger.info("[OK] MT5 disconnected")
            except:
                pass


def main():
    trader = LiveTradingOrchestrated()
    
    if not trader.initialize():
        logger.error("Initialization failed. Exiting.")
        return 1
    
    try:
        asyncio.run(trader.run_async())
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    finally:
        trader.shutdown()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
