#!/usr/bin/env python3
"""
FIXED: Simplified live trading runner with full system integration.
This is the main entry point for live trading with MT5.
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

# ==== Setup Logging ====
def setup_logging(debug: bool = False):
    """Configure logging for the application"""
    log_level = logging.DEBUG if debug else logging.INFO
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler("logs/live_trading.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Reduce noise from other loggers
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    return logging.getLogger("LiveTrading")

# ==== Configuration ====
class LiveTradingConfig:
    """Centralized configuration for live trading"""
    # Instruments and timeframes
    INSTRUMENTS = ["EURUSD", "XAUUSD"]
    TIMEFRAMES = ["H1", "H4", "D1"]
    HIST_BARS = 1000
    
    # Model selection
    MODELS = {
        "ppo": "models/ppo_final_model.zip",
        "sac": "models/sac_final_model.zip",
        "td3": "models/td3_final_model.zip",
    }
    
    # Trading parameters
    SLEEP_SECONDS = 5
    MIN_TRADE_INTERVAL = 60  # Minimum seconds between trades per instrument
    
    # Risk parameters for live trading
    LIVE_RISK_PARAMS = {
        "min_intensity": 0.3,      # Higher threshold for live
        "consensus_min": 0.35,     # Require more agreement
        "max_position_pct": 0.10,  # Max 10% per position
        "max_total_exposure": 0.30, # Max 30% total exposure
    }
    
    # Performance tracking
    LOG_INTERVAL = 300  # Log performance every 5 minutes
    SAVE_STATE_INTERVAL = 3600  # Save state every hour

# ==== Live Trading System ====
class SimpleLiveTradingSystem:
    """Simplified live trading system with full integration"""
    
    def __init__(self, model_name: str = "sac", debug: bool = False):
        self.logger = setup_logging(debug)
        self.logger.info("="*60)
        self.logger.info("INITIALIZING LIVE TRADING SYSTEM")
        self.logger.info("="*60)
        
        self.model_name = model_name
        self.debug = debug
        self.running = False
        
        # Initialize components
        self._initialize_mt5()
        self._initialize_data_connector()
        self._initialize_environment()
        self._initialize_model()
        self._initialize_tracking()
        
        self.logger.info("System initialization complete")
        
    def _initialize_mt5(self):
        """Initialize MT5 connection"""
        self.logger.info("Connecting to MetaTrader5...")
        
        if not mt5.initialize():
            raise RuntimeError("Failed to initialize MT5")
            
        account_info = mt5.account_info()
        if not account_info:
            raise RuntimeError("Failed to get MT5 account info")
            
        self.initial_balance = float(account_info.balance)
        self.logger.info(f"Connected to MT5 - Balance: ${self.initial_balance:.2f}")
        
    def _initialize_data_connector(self):
        """Initialize data connector"""
        self.logger.info("Initializing data connector...")
        
        self.connector = LiveDataConnector(
            instruments=LiveTradingConfig.INSTRUMENTS,
            timeframes=LiveTradingConfig.TIMEFRAMES
        )
        self.connector.connect()
        
        # Fetch initial historical data
        self.logger.info(f"Fetching {LiveTradingConfig.HIST_BARS} historical bars...")
        hist_data = self.connector.get_historical_data(n_bars=LiveTradingConfig.HIST_BARS)
        
        # Validate data
        for inst in hist_data:
            for tf in hist_data[inst]:
                bars = len(hist_data[inst][tf])
                self.logger.debug(f"{inst} {tf}: {bars} bars loaded")
                
        self.hist_data = hist_data
        
    def _initialize_environment(self):
        """Initialize trading environment"""
        self.logger.info("Creating trading environment...")
        
        # Create configuration
        config = TradingConfig(
            initial_balance=self.initial_balance,
            max_steps=10_000_000,  # Effectively infinite
            live_mode=True,
            debug=self.debug,
            **LiveTradingConfig.LIVE_RISK_PARAMS
        )
        
        # Create environment
        self.env = EnhancedTradingEnv(
            data_dict=self.hist_data,
            config=config
        )
        
        # Verify committee initialization
        if hasattr(self.env, 'arbiter'):
            members = len(self.env.arbiter.members)
            self.logger.info(f"Strategy committee initialized with {members} members")
            
            # Log member types
            member_names = [m.__class__.__name__ for m in self.env.arbiter.members]
            self.logger.debug(f"Committee members: {member_names}")
        else:
            self.logger.warning("Strategy arbiter not found in environment")
            
    def _initialize_model(self):
        """Load the trained model"""
        model_path = LiveTradingConfig.MODELS.get(self.model_name)
        
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.logger.info(f"Loading {self.model_name.upper()} model from {model_path}")
        
        # Load appropriate model type
        if self.model_name == "ppo":
            self.model = PPO.load(model_path, device="cpu")
        elif self.model_name == "sac":
            self.model = SAC.load(model_path, device="cpu")
        elif self.model_name == "td3":
            self.model = TD3.load(model_path, device="cpu")
        else:
            raise ValueError(f"Unknown model type: {self.model_name}")
            
        self.logger.info("Model loaded successfully")
        
    def _initialize_tracking(self):
        """Initialize performance tracking"""
        self.start_time = datetime.now()
        self.last_log_time = time.time()
        self.last_save_time = time.time()
        self.step_count = 0
        self.last_trade_time = {}  # Track last trade time per instrument
        
        self.performance = {
            "initial_balance": self.initial_balance,
            "peak_balance": self.initial_balance,
            "total_trades": 0,
            "winning_trades": 0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
        }
        
    def update_market_data(self):
        """Fetch and update latest market data"""
        try:
            # Get latest bar
            new_data = self.connector.get_historical_data(n_bars=1)
            
            # Update environment data
            for inst_raw in LiveTradingConfig.INSTRUMENTS:
                # Convert format
                inst = inst_raw[:3] + "/" + inst_raw[3:] if len(inst_raw) == 6 else inst_raw
                
                for tf in LiveTradingConfig.TIMEFRAMES:
                    if inst in new_data and tf in new_data[inst]:
                        old_df = self.env.data[inst][tf]
                        new_df = new_data[inst][tf]
                        
                        if len(new_df) > 0:
                            # Append and maintain history size
                            combined = pd.concat([old_df, new_df])
                            combined = combined[~combined.index.duplicated(keep='last')]
                            self.env.data[inst][tf] = combined.iloc[-LiveTradingConfig.HIST_BARS:]
                            
        except Exception as e:
            self.logger.error(f"Failed to update market data: {e}")
            
    def sync_with_broker(self):
        """Synchronize state with broker"""
        try:
            # Update balance
            account_info = self.connector.get_account_info()
            if account_info:
                new_balance = account_info['balance']
                self.env.market_state.balance = new_balance
                
                # Update peak and drawdown
                if new_balance > self.env.market_state.peak_balance:
                    self.env.market_state.peak_balance = new_balance
                    
                dd = (self.env.market_state.peak_balance - new_balance) / self.env.market_state.peak_balance
                self.env.market_state.current_drawdown = max(0, dd)
                
            # Sync positions
            self.connector.sync_positions_with_env(self.env)
            
        except Exception as e:
            self.logger.error(f"Failed to sync with broker: {e}")
            
    def check_trade_timing(self, instrument: str) -> bool:
        """Check if enough time has passed since last trade"""
        last_time = self.last_trade_time.get(instrument, 0)
        current_time = time.time()
        
        if current_time - last_time >= LiveTradingConfig.MIN_TRADE_INTERVAL:
            return True
        return False
        
    def log_performance(self):
        """Log current performance metrics"""
        current_balance = self.env.market_state.balance
        runtime_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        
        # Update performance metrics
        self.performance["total_pnl"] = current_balance - self.initial_balance
        self.performance["max_drawdown"] = max(
            self.performance["max_drawdown"],
            self.env.market_state.current_drawdown
        )
        
        # Calculate additional metrics
        roi = self.performance["total_pnl"] / self.initial_balance * 100
        win_rate = (
            self.performance["winning_trades"] / self.performance["total_trades"] * 100
            if self.performance["total_trades"] > 0 else 0
        )
        
        # Create performance summary
        self.logger.info("\n" + "="*60)
        self.logger.info(f"PERFORMANCE UPDATE (Runtime: {runtime_hours:.1f} hours)")
        self.logger.info("="*60)
        self.logger.info(f"Balance: ${current_balance:.2f} (ROI: {roi:+.2f}%)")
        self.logger.info(f"P&L: ${self.performance['total_pnl']:+.2f}")
        self.logger.info(f"Trades: {self.performance['total_trades']} (Win Rate: {win_rate:.1f}%)")
        self.logger.info(f"Max Drawdown: {self.performance['max_drawdown']:.2%}")
        self.logger.info(f"Open Positions: {len(self.env.position_manager.open_positions)}")
        
        # Log arbiter statistics
        if hasattr(self.env, 'arbiter'):
            stats = self.env.arbiter.get_diagnostics()
            self.logger.info(f"Gate Pass Rate: {stats['gate_stats']['pass_rate']:.1%}")
            
        self.logger.info("="*60 + "\n")
        
    def save_state(self):
        """Save current state to disk"""
        try:
            state = {
                "timestamp": datetime.now().isoformat(),
                "step_count": self.step_count,
                "performance": self.performance,
                "env_state": self.env.get_state(),
            }
            
            # Save to file
            state_file = f"logs/state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            import pickle
            with open(state_file, 'wb') as f:
                pickle.dump(state, f)
                
            self.logger.info(f"State saved to {state_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
            
    def trading_step(self, obs: np.ndarray) -> np.ndarray:
        """Execute one trading step"""
        # Ensure observation matches model expectations
        expected_dim = self.model.observation_space.shape[0]
        
        if obs.shape[0] != expected_dim:
            if obs.shape[0] > expected_dim:
                obs = obs[:expected_dim]
            else:
                obs = np.pad(obs, (0, expected_dim - obs.shape[0]), constant_values=0)
                
        # Get model prediction
        action, _ = self.model.predict(obs, deterministic=True)
        
        # Execute environment step
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update tracking
        if self.env.trades:
            for trade in self.env.trades:
                self.performance["total_trades"] += 1
                if trade.get("pnl", 0) > 0:
                    self.performance["winning_trades"] += 1
                    
                # Update trade timing
                inst = trade.get("instrument")
                if inst:
                    self.last_trade_time[inst] = time.time()
                    
        # Log step info
        self.logger.info(
            f"Step {self.step_count} | "
            f"Balance: ${info.get('balance', 0):.2f} | "
            f"P&L: ${info.get('pnl', 0):.2f} | "
            f"DD: {info.get('drawdown', 0):.1%} | "
            f"Trades: {info.get('trades', 0)} | "
            f"Mode: {info.get('mode', 'N/A')}"
        )
        
        return obs
        
    def run(self):
        """Main trading loop"""
        self.logger.info("\n" + "="*60)
        self.logger.info("STARTING LIVE TRADING")
        self.logger.info("="*60 + "\n")
        
        self.running = True
        obs, _ = self.env.reset()
        
        while self.running:
            try:
                # Update market data
                self.update_market_data()
                
                # Sync with broker
                self.sync_with_broker()
                
                # Execute trading step
                obs = self.trading_step(obs)
                
                # Increment counter
                self.step_count += 1
                
                # Periodic logging
                if time.time() - self.last_log_time >= LiveTradingConfig.LOG_INTERVAL:
                    self.log_performance()
                    self.last_log_time = time.time()
                    
                # Periodic state saving
                if time.time() - self.last_save_time >= LiveTradingConfig.SAVE_STATE_INTERVAL:
                    self.save_state()
                    self.last_save_time = time.time()
                    
                # Sleep before next iteration
                time.sleep(LiveTradingConfig.SLEEP_SECONDS)
                
            except KeyboardInterrupt:
                self.logger.info("\nReceived interrupt signal")
                break
                
            except Exception as e:
                self.logger.exception(f"Error in trading loop: {e}")
                
                # Try to continue after error
                self.logger.info("Attempting to recover...")
                time.sleep(LiveTradingConfig.SLEEP_SECONDS * 2)
                
        self.shutdown()
        
    def shutdown(self):
        """Clean shutdown"""
        self.logger.info("\nShutting down trading system...")
        
        self.running = False
        
        # Final performance log
        self.log_performance()
        
        # Save final state
        self.save_state()
        
        # Disconnect
        if hasattr(self, 'connector'):
            self.connector.disconnect()
            
        # Close environment
        if hasattr(self, 'env'):
            self.env.close()
            
        self.logger.info("Shutdown complete")

# ==== Main Entry Point ====
def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Live trading system for MetaTrader5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                    # Run with default SAC model
  python run.py --model ppo        # Run with PPO model
  python run.py --debug            # Run with debug logging
  python run.py --sleep 10         # Custom sleep interval
        """
    )
    
    parser.add_argument(
        "--model",
        choices=["ppo", "sac", "td3"],
        default="sac",
        help="Model to use for trading (default: sac)"
    )
    
    parser.add_argument(
        "--sleep",
        type=int,
        default=5,
        help="Seconds between trading steps (default: 5)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Apply configuration
    LiveTradingConfig.SLEEP_SECONDS = args.sleep
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        print("\nReceived signal, initiating shutdown...")
        if hasattr(signal_handler, 'system'):
            signal_handler.system.running = False
            
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Create and run trading system
        system = SimpleLiveTradingSystem(
            model_name=args.model,
            debug=args.debug
        )
        
        # Store reference for signal handler
        signal_handler.system = system
        
        # Run trading loop
        system.run()
        
    except Exception as e:
        logging.exception(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()