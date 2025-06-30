#!/usr/bin/env python3
"""
FIXED: Enhanced live data connector with better error handling and data validation.
Ensures compatibility with the enhanced trading system.
"""

import time
import datetime
import logging
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from typing import Any, Dict, List, Optional, Tuple

from live.mt5_credentials import MT5Credentials

logger = logging.getLogger(__name__)


from stable_baselines3.common.callbacks import BaseCallback




class LiveDataConnector:
    """
    Enhanced connector for MT5 live data with robust error handling.
    """
    
    def __init__(
        self,
        instruments: List[str],
        timeframes: List[str],
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ):
        """
        Initialize the live data connector.
        
        Args:
            instruments: List of MT5 symbols (e.g., ["EURUSD", "XAUUSD"])
            timeframes: List of timeframes (e.g., ["H1", "H4", "D1"])
            max_retries: Maximum connection retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.account = MT5Credentials.ACCOUNT
        self.password = MT5Credentials.PASSWORD
        self.server = MT5Credentials.SERVER

        self.instruments = instruments
        self.timeframes = timeframes
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Track connection state
        self.connected = False
        self.last_tick_time = {}

        # Map timeframe names to MT5 constants
        self._tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
            "W1": mt5.TIMEFRAME_W1,
            "MN1": mt5.TIMEFRAME_MN1,
        }
        
        # Validate timeframes
        for tf in timeframes:
            if tf not in self._tf_map:
                raise ValueError(f"Invalid timeframe '{tf}'. Valid options: {list(self._tf_map.keys())}")

    def connect(self) -> None:
        """Initialize MT5, log in, and select symbols."""
        for attempt in range(1, self.max_retries + 1):
            try:
                # Initialize MT5
                if not mt5.initialize():
                    error = mt5.last_error()
                    logger.error(f"MT5 initialize failed: {error}")
                    raise ConnectionError(f"MT5 initialization failed: {error}")
                
                # Login
                if not mt5.login(self.account, self.password, self.server):
                    error = mt5.last_error()
                    logger.error(f"MT5 login failed: {error}")
                    mt5.shutdown()
                    raise ConnectionError(f"MT5 login failed: {error}")
                
                # Select all required symbols
                failed_symbols = []
                for symbol in self.instruments:
                    if not mt5.symbol_select(symbol, True):
                        failed_symbols.append(symbol)
                        logger.warning(f"Failed to select symbol {symbol}")
                
                if failed_symbols:
                    logger.warning(f"Some symbols could not be selected: {failed_symbols}")
                    # Continue anyway - some symbols might not be available
                
                # Verify we can get data
                self._verify_data_access()
                
                self.connected = True
                logger.info(f"MT5 connected successfully (attempt {attempt})")
                return
                
            except Exception as e:
                logger.error(f"Connection attempt {attempt} failed: {e}")
                
                if attempt < self.max_retries:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    raise ConnectionError(f"Unable to connect to MT5 after {self.max_retries} attempts")

    def disconnect(self) -> None:
        """Clean shutdown of MT5."""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("MT5 disconnected")

    def _verify_data_access(self) -> None:
        """Verify we can actually get data from MT5"""
        for symbol in self.instruments:
            # Try to get a tick
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.warning(f"Cannot get tick data for {symbol}")
                continue
                
            # Try to get some historical data
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1)
            if rates is None or len(rates) == 0:
                logger.warning(f"Cannot get historical data for {symbol}")

    def fetch_historical(
        self, 
        symbol: str, 
        timeframe: str, 
        n_bars: int
    ) -> pd.DataFrame:
        """
        Fetch last `n_bars` of OHLCV data with calculated volatility.
        
        Args:
            symbol: MT5 symbol (e.g., "EURUSD")
            timeframe: Timeframe string (e.g., "H1")
            n_bars: Number of bars to fetch
            
        Returns:
            DataFrame with OHLCV + volatility data
        """
        if not self.connected:
            raise RuntimeError("Not connected to MT5")
            
        if timeframe not in self._tf_map:
            raise ValueError(f"Invalid timeframe '{timeframe}'")
            
        tf_constant = self._tf_map[timeframe]
        
        # Get current time
        now = datetime.datetime.now()
        
        # Fetch rates
        rates = mt5.copy_rates_from(symbol, tf_constant, now, n_bars)
        
        if rates is None or len(rates) == 0:
            logger.warning(f"No data returned for {symbol} {timeframe}")
            # Return empty DataFrame with correct structure
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume", "volatility"]
            )
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        
        # Convert time to datetime index
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        
        # Select and rename columns
        if "tick_volume" in df.columns:
            df = df[["open", "high", "low", "close", "tick_volume"]]
            df.rename(columns={"tick_volume": "volume"}, inplace=True)
        else:
            df = df[["open", "high", "low", "close", "volume"]]
        
        # Calculate volatility (multiple methods for robustness)
        # Method 1: High-Low range
        hl_vol = (df["high"] - df["low"]) / df["close"]
        
        # Method 2: Rolling standard deviation of returns
        returns = df["close"].pct_change()
        rolling_vol = returns.rolling(window=20).std()
        
        # Method 3: ATR-based volatility
        tr1 = df["high"] - df["low"]
        tr2 = abs(df["high"] - df["close"].shift())
        tr3 = abs(df["low"] - df["close"].shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_vol = true_range.rolling(window=14).mean() / df["close"]
        
        # Combine volatility measures
        df["volatility"] = rolling_vol.fillna(hl_vol).fillna(atr_vol).fillna(0.01)
        
        # Ensure no negative volatility
        df["volatility"] = df["volatility"].abs()
        
        # Final cleanup
        df = df[["open", "high", "low", "close", "volume", "volatility"]]
        
        # Validate data
        if df.isnull().any().any():
            logger.warning(f"NaN values found in {symbol} {timeframe} data, filling...")
            df = df.fillna(method='ffill').fillna(method='bfill')
            
        return df

    def get_historical_data(
        self, 
        n_bars: int = 1000
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Fetch historical bars for all instruments and timeframes.
        
        Args:
            n_bars: Number of bars to fetch
            
        Returns:
            Nested dict: {instrument -> {timeframe -> DataFrame}}
        """
        if not self.connected:
            raise RuntimeError("Not connected to MT5")
            
        data: Dict[str, Dict[str, pd.DataFrame]] = {}
        
        for symbol in self.instruments:
            # Convert MT5 symbol to internal format (EURUSD -> EUR/USD)
            if len(symbol) == 6 and "/" not in symbol:
                symbol_internal = symbol[:3] + "/" + symbol[3:]
            else:
                symbol_internal = symbol
                
            data[symbol_internal] = {}
            
            for timeframe in self.timeframes:
                try:
                    df = self.fetch_historical(symbol, timeframe, n_bars)
                    
                    if len(df) == 0:
                        logger.warning(f"Empty data for {symbol} {timeframe}")
                        
                    data[symbol_internal][timeframe] = df
                    
                    logger.debug(
                        f"Fetched {len(df)} bars for {symbol} {timeframe}, "
                        f"latest close: {df['close'].iloc[-1] if len(df) > 0 else 'N/A'}"
                    )
                    
                except Exception as e:
                    logger.error(f"Error fetching {symbol} {timeframe}: {e}")
                    # Create empty DataFrame with correct structure
                    data[symbol_internal][timeframe] = pd.DataFrame(
                        columns=["open", "high", "low", "close", "volume", "volatility"]
                    )
                    
        return data

    def get_latest_tick(self, symbol: str) -> Optional[mt5.Tick]:
        """
        Return the latest tick for a symbol.
        
        Args:
            symbol: MT5 symbol (e.g., "EURUSD")
            
        Returns:
            MT5 Tick object or None
        """
        if not self.connected:
            raise RuntimeError("Not connected to MT5")
            
        tick = mt5.symbol_info_tick(symbol)
        
        if tick is not None:
            # Track last tick time for monitoring
            self.last_tick_time[symbol] = datetime.datetime.now()
            
        return tick

    def get_account_info(self) -> Dict[str, float]:
        """
        Get current account information.
        
        Returns:
            Dict with account metrics
        """
        if not self.connected:
            raise RuntimeError("Not connected to MT5")
            
        info = mt5.account_info()
        
        if info is None:
            logger.error("Failed to get account info")
            return {}
            
        return {
            "balance": float(info.balance),
            "equity": float(info.equity),
            "margin": float(info.margin),
            "free_margin": float(info.margin_free),
            "margin_level": float(info.margin_level) if info.margin_level else 0.0,
            "profit": float(info.profit),
        }

    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get all open positions.
        
        Returns:
            List of position dictionaries
        """
        if not self.connected:
            raise RuntimeError("Not connected to MT5")
            
        positions = mt5.positions_get()
        
        if positions is None:
            return []
            
        position_list = []
        for pos in positions:
            position_list.append({
                "ticket": pos.ticket,
                "symbol": pos.symbol,
                "type": "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL",
                "volume": pos.volume,
                "price_open": pos.price_open,
                "price_current": pos.price_current,
                "swap": pos.swap,
                "profit": pos.profit,
                "magic": pos.magic,
                "comment": pos.comment,
            })
            
        return position_list

    def send_order(
        self,
        symbol: str,
        side: str,
        volume: float,
        deviation: int = 20,
        magic: int = 202406,
        comment: str = "AI Live Trade"
    ) -> mt5.OrderSendResult:
        """
        Send a market order with enhanced error handling.
        
        Args:
            symbol: MT5 symbol (e.g., "EURUSD")
            side: "buy" or "sell"
            volume: Position size in lots
            deviation: Maximum price deviation in points
            magic: Magic number for identifying trades
            comment: Trade comment
            
        Returns:
            MT5 OrderSendResult
        """
        if not self.connected:
            raise RuntimeError("Not connected to MT5")
            
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            raise ValueError(f"Symbol {symbol} not found")
            
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                raise ValueError(f"Failed to select symbol {symbol}")
                
        # Get latest tick
        tick = self.get_latest_tick(symbol)
        if tick is None:
            raise RuntimeError(f"No tick data available for {symbol}")
            
        # Validate tick data
        if not hasattr(tick, "ask") or not hasattr(tick, "bid"):
            raise RuntimeError(f"Invalid tick data for {symbol}")
            
        if tick.ask is None or tick.bid is None or tick.ask <= 0 or tick.bid <= 0:
            raise RuntimeError(f"Invalid prices for {symbol}: ask={tick.ask}, bid={tick.bid}")
            
        # Determine order type and price
        side_lower = side.lower()
        if side_lower not in ("buy", "sell"):
            raise ValueError(f"Invalid side '{side}'. Must be 'buy' or 'sell'.")
            
        order_type = mt5.ORDER_TYPE_BUY if side_lower == "buy" else mt5.ORDER_TYPE_SELL
        price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
        
        # Round volume to symbol requirements
        volume_step = symbol_info.volume_step
        volume = round(volume / volume_step) * volume_step
        
        # Ensure volume is within limits
        volume = max(symbol_info.volume_min, min(volume, symbol_info.volume_max))
        
        # Create order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "deviation": deviation,
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._get_filling_mode(symbol_info),
        }
        
        # Send order
        result = mt5.order_send(request)
        
        # Log result
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.warning(
                f"Order failed for {symbol}: "
                f"retcode={result.retcode}, "
                f"comment='{result.comment}', "
                f"request={request}"
            )
        else:
            logger.info(
                f"Order successful: {symbol} {side} {volume:.2f} lots @ {price:.5f}, "
                f"ticket={result.order}"
            )
            
        return result

    def _get_filling_mode(self, symbol_info) -> int:
        """
        Determine the appropriate filling mode for the symbol.
        
        Args:
            symbol_info: MT5 symbol info object
            
        Returns:
            MT5 filling mode constant
        """
        filling_modes = [
            (mt5.SYMBOL_FILLING_FOK, mt5.ORDER_FILLING_FOK),
            (mt5.SYMBOL_FILLING_IOC, mt5.ORDER_FILLING_IOC),
            (mt5.SYMBOL_FILLING_RETURN, mt5.ORDER_FILLING_RETURN),
        ]
        
        for flag, mode in filling_modes:
            if symbol_info.filling_mode & flag:
                return mode
                
        # Default to IOC
        return mt5.ORDER_FILLING_IOC

    def check_connection(self) -> bool:
        """
        Check if MT5 connection is still alive.
        
        Returns:
            True if connected and responsive
        """
        if not self.connected:
            return False
            
        try:
            # Try to get account info as a connection test
            info = mt5.account_info()
            return info is not None
        except Exception as e:
            logger.error(f"Connection check failed: {e}")
            return False

    def sync_positions_with_env(self, env) -> None:
        """
        Synchronize MT5 positions with environment position manager.
        
        Args:
            env: The trading environment
        """
        if not hasattr(env, 'position_manager'):
            logger.warning("Environment has no position_manager")
            return
            
        # Get current MT5 positions
        mt5_positions = self.get_positions()
        
        # Convert to internal format
        for pos in mt5_positions:
            # Convert symbol format
            symbol = pos['symbol']
            if len(symbol) == 6 and "/" not in symbol:
                symbol_internal = symbol[:3] + "/" + symbol[3:]
            else:
                symbol_internal = symbol
                
            # Update position manager
            if symbol_internal in env.instruments:
                env.position_manager.open_positions[symbol_internal] = {
                    "ticket": pos['ticket'],
                    "side": 1 if pos['type'] == "BUY" else -1,
                    "lots": pos['volume'],
                    "price_open": pos['price_open'],
                    "entry_step": env.market_state.current_step,
                    "instrument": symbol_internal,
                }
                
        logger.debug(f"Synced {len(mt5_positions)} positions with environment")



class LiveTradingCallback(BaseCallback):
    """
    SB3 callback to synchronize your env with MT5 during live‐mode training.
    Connects at start, syncs positions on every step, and disconnects at end.
    """
    def __init__(self, connector: LiveDataConnector, verbose: int = 0):
        super().__init__(verbose)
        self.connector = connector

    def _on_training_start(self) -> None:
        # establish MT5 connection once at training start
        self.connector.connect()

    def _on_step(self) -> bool:
        # sync any open positions back into the env’s position manager
        # assumes a single‐env DummyVecEnv
        env = self.training_env.envs[0]
        self.connector.sync_positions_with_env(env)
        return True

    def _on_training_end(self) -> None:
        # cleanly disconnect when training finishes
        self.connector.disconnect()

    def connect(self):
        self.connector.connect()

    def disconnect(self):
        self.connector.disconnect()