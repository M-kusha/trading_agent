#!/usr/bin/env python3
"""
FIXED: Live trading system with full integration of all enhanced modules.
Now uses the complete 8-member voting committee and proper data flow.
"""

import os
import sys
# Ensure Windows console can print Unicode
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import time
import logging
import argparse
import signal
from datetime import datetime

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

# ==== Imports ====
from live.live_connector import LiveDataConnector
from envs.env import EnhancedTradingEnv, TradingConfig
from stable_baselines3 import PPO, SAC, TD3
from modules.strategy.voting import StrategyArbiter
from modules.risk.risk_controller import DynamicRiskController
from modules.memory.memory import MistakeMemory

# ==== Logging Config ====
logger = logging.getLogger("live_trading")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler("live_trading.log", encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)

# ==== Config ====
class Config:
    INSTRUMENTS = ["EURUSD", "XAUUSD"]
    TIMEFRAMES  = ["H1", "H4", "D1"]
    HIST_BARS   = 1000
    AGENT_MAP   = {
        "1": ("PPO", "models/ppo_final_model.zip"),
        "2": ("SAC", "models/sac_final_model.zip"),
        "3": ("TD3", "models/td3_final_model.zip"),
    }
    AGENT_ID    = "2"  # Default to SAC
    SLEEP_SECS  = 5
    
    # Risk parameters for live trading
    MIN_INTENSITY = 0.3       # Higher threshold for live
    CONSENSUS_MIN = 0.35      # Slightly higher for live
    MAX_POSITION_PCT = 0.15   # More conservative for live

# ==== Agent Factory ====
class AgentFactory:
    @staticmethod
    def load(agent_name, model_path):
        """Load the specified RL agent"""
        if agent_name == "PPO":
            return PPO.load(model_path, device="cpu")
        elif agent_name == "SAC":
            return SAC.load(model_path, device="cpu")
        elif agent_name == "TD3":
            return TD3.load(model_path, device="cpu")
        else:
            raise ValueError(f"Unsupported agent: {agent_name}")

# ==== Live Trading System ====
class LiveTradingSystem:
    def __init__(self):
        """Initialize the complete live trading system"""
        # 1) Initialize and authenticate MT5
        if not mt5.initialize():
            raise RuntimeError("MetaTrader5 initialize() failed")
            
        account_info = mt5.account_info()
        if account_info is None or not hasattr(account_info, "balance"):
            raise RuntimeError("Could not retrieve account_info from MT5")
            
        self.live_balance = float(account_info.balance)
        logger.info(f"Connected to MT5 – live balance=${self.live_balance:.2f}")

        # 2) Connect data feed and fetch history
        self.connector = LiveDataConnector(
            instruments=Config.INSTRUMENTS,
            timeframes=Config.TIMEFRAMES
        )
        self.connector.connect()
        hist = self.connector.get_historical_data(n_bars=Config.HIST_BARS)

        # 3) Create environment with live configuration
        cfg = TradingConfig(
            initial_balance=self.live_balance,
            max_steps=10_000_000,  # Effectively infinite for live trading
            live_mode=True,
            debug=False,  # Disable debug for production
            min_intensity=Config.MIN_INTENSITY,
            consensus_min=Config.CONSENSUS_MIN,
            no_trade_penalty=0.1,  # Light penalty for live
        )
        
        # Apply conservative position limits
        self.env = EnhancedTradingEnv(data_dict=hist, config=cfg)
        
        # Override position manager settings for live
        self.env.position_manager.max_pct = Config.MAX_POSITION_PCT
        self.env.position_manager.default_max_pct = Config.MAX_POSITION_PCT
        
        # 4) Load chosen RL agent
        agent_name, model_path = Config.AGENT_MAP[Config.AGENT_ID]
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model at {model_path}")
            
        self.model = AgentFactory.load(agent_name, model_path)
        logger.info(f"{agent_name} agent loaded from {model_path}")
        
        # 5) Initialize performance tracking
        self.performance_tracker = {
            "start_time": datetime.now(),
            "start_balance": self.live_balance,
            "total_trades": 0,
            "winning_trades": 0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
        }
        
        # 6) Verify committee is properly initialized
        self._verify_committee()

    def _verify_committee(self):
        """Verify all committee members are active"""
        if not hasattr(self.env, 'arbiter'):
            raise RuntimeError("Environment arbiter not initialized")
            
        expected_members = 8
        actual_members = len(self.env.arbiter.members)
        
        if actual_members < expected_members:
            logger.warning(
                f"Committee has {actual_members} members, expected {expected_members}. "
                f"Some experts may be missing."
            )
        else:
            logger.info(f"Committee verified with {actual_members} voting members")
            
        # Log member names
        member_names = [m.__class__.__name__ for m in self.env.arbiter.members]
        logger.info(f"Active committee members: {member_names}")

    @staticmethod
    def format_symbol(symbol):
        """Convert MT5 symbol format to internal format"""
        if "/" in symbol:
            return symbol
        if len(symbol) == 6:
            return symbol[:3] + "/" + symbol[3:]
        return symbol

    def append_new_bar(self):
        """Fetch and append the latest market data"""
        try:
            # Get latest bar for each instrument/timeframe
            new_data = self.connector.get_historical_data(n_bars=1)
            
            for sym_raw in Config.INSTRUMENTS:
                sym_internal = self.format_symbol(sym_raw)
                
                for tf in Config.TIMEFRAMES:
                    try:
                        # Get existing and new data
                        old_df = self.env.data[sym_internal][tf]
                        new_df = new_data[sym_internal][tf]
                        
                        if len(new_df) > 0:
                            # Concatenate and keep only the latest HIST_BARS
                            combined_df = pd.concat([old_df, new_df])
                            
                            # Remove duplicates based on index (timestamp)
                            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                            
                            # Keep only the latest bars
                            self.env.data[sym_internal][tf] = combined_df.iloc[-Config.HIST_BARS:]
                            
                            # Log the update
                            latest = new_df.iloc[-1]
                            logger.debug(
                                f"Updated {sym_internal} {tf}: "
                                f"Close={latest['close']:.5f}, Vol={latest['volatility']:.5f}"
                            )
                            
                    except Exception as e:
                        logger.error(f"Failed to update {sym_internal}/{tf}: {e}")
                        
        except Exception as e:
            logger.error(f"append_new_bar failed: {e}")

    def update_balance(self):
        """Sync balance with MT5 account"""
        try:
            account_info = mt5.account_info()
            if account_info and hasattr(account_info, "balance"):
                new_balance = float(account_info.balance)
                
                # Update environment balance
                self.env.market_state.balance = new_balance
                
                # Update peak balance if necessary
                if new_balance > self.env.market_state.peak_balance:
                    self.env.market_state.peak_balance = new_balance
                    
                # Recalculate drawdown
                if self.env.market_state.peak_balance > 0:
                    self.env.market_state.current_drawdown = (
                        self.env.market_state.peak_balance - new_balance
                    ) / self.env.market_state.peak_balance
                    
                return new_balance
        except Exception as e:
            logger.error(f"Failed to update balance: {e}")
            return self.env.market_state.balance

    def log_performance(self):
        """Log performance metrics"""
        elapsed = (datetime.now() - self.performance_tracker["start_time"]).total_seconds() / 3600
        current_balance = self.env.market_state.balance
        total_return = (current_balance - self.performance_tracker["start_balance"]) / self.performance_tracker["start_balance"]
        
        # Update tracker
        self.performance_tracker["total_pnl"] = current_balance - self.performance_tracker["start_balance"]
        self.performance_tracker["max_drawdown"] = max(
            self.performance_tracker["max_drawdown"],
            self.env.market_state.current_drawdown
        )
        
        # Calculate win rate
        if self.performance_tracker["total_trades"] > 0:
            win_rate = self.performance_tracker["winning_trades"] / self.performance_tracker["total_trades"]
        else:
            win_rate = 0.0
            
        logger.info(
            f"\n{'='*60}\n"
            f"PERFORMANCE SUMMARY (Runtime: {elapsed:.1f} hours)\n"
            f"{'='*60}\n"
            f"Current Balance: ${current_balance:.2f}\n"
            f"Total P&L: ${self.performance_tracker['total_pnl']:.2f} ({total_return:.2%})\n"
            f"Total Trades: {self.performance_tracker['total_trades']}\n"
            f"Win Rate: {win_rate:.1%}\n"
            f"Max Drawdown: {self.performance_tracker['max_drawdown']:.2%}\n"
            f"Open Positions: {len(self.env.position_manager.open_positions)}\n"
            f"{'='*60}"
        )

    def run(self):
        """Main live trading loop"""
        logger.info("Starting live trading loop")
        
        # Reset environment
        obs, info = self.env.reset()
        step_count = 0
        last_log_time = time.time()
        
        while True:
            try:
                # Update market data
                self.append_new_bar()
                
                # Update balance from broker
                current_balance = self.update_balance()
                
                # Prepare observation for model
                obs = np.asarray(obs, dtype=np.float32)
                
                # Handle observation dimension mismatch
                expected_dim = self.model.observation_space.shape[0]
                if obs.shape[0] != expected_dim:
                    if obs.shape[0] > expected_dim:
                        # Truncate if too large
                        obs = obs[:expected_dim]
                        logger.warning(f"Truncated observation from {obs.shape[0]} to {expected_dim}")
                    else:
                        # Pad if too small
                        pad_width = expected_dim - obs.shape[0]
                        obs = np.pad(obs, (0, pad_width), mode="constant", constant_values=0.0)
                        logger.warning(f"Padded observation from {obs.shape[0]} to {expected_dim}")
                
                # Get model prediction
                model_action, _ = self.model.predict(obs, deterministic=True)
                
                # The arbiter is already part of the environment's committee
                # No need for separate arbiter action - it's handled internally
                
                # Step the environment
                obs, reward, terminated, truncated, info = self.env.step(model_action)
                
                # Update performance tracking
                if self.env.trades:
                    for trade in self.env.trades:
                        self.performance_tracker["total_trades"] += 1
                        if trade.get("pnl", 0) > 0:
                            self.performance_tracker["winning_trades"] += 1
                
                # Log step info
                step_count += 1
                logger.info(
                    f"Step {step_count} | "
                    f"Balance=${current_balance:.2f} | "
                    f"PnL=${info.get('pnl', 0):.2f} | "
                    f"DD={info.get('drawdown', 0):.2%} | "
                    f"Trades={info.get('trades', 0)} | "
                    f"Consensus={info.get('consensus', 0):.3f} | "
                    f"Mode={info.get('mode', 'unknown')}"
                )
                
                # Get diagnostic info periodically
                if time.time() - last_log_time > 300:  # Every 5 minutes
                    self.log_performance()
                    
                    # Log arbiter statistics
                    arbiter_stats = self.env.arbiter.get_diagnostics()
                    logger.info(
                        f"Arbiter stats: Pass rate={arbiter_stats['gate_stats']['pass_rate']:.1%}, "
                        f"Bootstrap={arbiter_stats['bootstrap']}"
                    )
                    
                    last_log_time = time.time()
                
                # Handle episode termination (shouldn't happen in live mode)
                if terminated or truncated:
                    logger.warning("Episode terminated unexpectedly, resetting environment")
                    obs, info = self.env.reset()
                
                # Sleep before next iteration
                time.sleep(Config.SLEEP_SECS)

            except KeyboardInterrupt:
                logger.info("Live trading interrupted by user")
                break
                
            except Exception as e:
                logger.exception(f"Error in live trading loop: {e}")
                
                # Try to recover
                try:
                    # Re-sync with MT5
                    self.update_balance()
                    
                    # Log error but continue
                    logger.info("Attempting to continue after error...")
                    time.sleep(Config.SLEEP_SECS * 2)  # Wait longer after error
                    
                except Exception as recovery_error:
                    logger.error(f"Recovery failed: {recovery_error}")
                    break

        # Clean shutdown
        logger.info("Shutting down live trading system...")
        self.log_performance()
        self.connector.disconnect()

# ==== CLI & Entrypoint ====
def parse_args():
    parser = argparse.ArgumentParser(description="Live trading system for MT5")
    parser.add_argument(
        "--agent", 
        choices=["1", "2", "3"], 
        default="2",
        help="Agent selection: 1=PPO, 2=SAC, 3=TD3"
    )
    parser.add_argument(
        "--sleep", 
        type=int, 
        default=5,
        help="Seconds between trading steps"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Apply configuration
    Config.AGENT_ID = args.agent
    Config.SLEEP_SECS = args.sleep
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    try:
        # Initialize live trading system
        logger.info("Initializing live trading system...")
        system = LiveTradingSystem()
        
        # Optional: Start monitoring API in background
        try:
            from live.api import start_api
            import threading
            api_thread = threading.Thread(
                target=lambda: start_api(system), 
                daemon=True
            )
            api_thread.start()
            logger.info("Started monitoring API server")
        except ImportError:
            logger.warning("API module not found, skipping API server")
        
        # Setup graceful shutdown
        def graceful_exit(signum, frame):
            logger.info("Received exit signal, shutting down gracefully...")
            
            # Log final performance
            system.log_performance()
            
            # Disconnect from MT5
            system.connector.disconnect()
            
            # Close environment
            if hasattr(system.env, "close"):
                system.env.close()
                
            logger.info("Shutdown complete")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, graceful_exit)
        signal.signal(signal.SIGTERM, graceful_exit)
        
        # Start main trading loop
        logger.info("Starting live trading...")
        system.run()
        
    except Exception as e:
        logger.exception(f"Fatal error in live trading system: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()