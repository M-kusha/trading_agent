# live_connector.py
#!/usr/bin/env python3
import time
import datetime
import logging
import pandas as pd
import MetaTrader5 as mt5

from live.mt5_credentials import MT5Credentials

logger = logging.getLogger(__name__)

class LiveDataConnector:
    def __init__(
        self,
        instruments: list[str],
        timeframes: list[str],
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ):
        self.account  = MT5Credentials.ACCOUNT
        self.password = MT5Credentials.PASSWORD
        self.server   = MT5Credentials.SERVER

        self.instruments = instruments
        self.timeframes  = timeframes
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # map timeframe names to MT5 constants
        self._tf_map = {
            "M1":  mt5.TIMEFRAME_M1,
            "M5":  mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "H1":  mt5.TIMEFRAME_H1,
            "H4":  mt5.TIMEFRAME_H4,
            "D1":  mt5.TIMEFRAME_D1,
            "W1":  mt5.TIMEFRAME_W1,
            "MN1": mt5.TIMEFRAME_MN1,
        }

    def connect(self) -> None:
        """Initialize MT5, log in, and select symbols."""
        for attempt in range(1, self.max_retries + 1):
            if not mt5.initialize():
                logger.error(f"MT5 initialize failed: {mt5.last_error()}")
            elif not mt5.login(self.account, self.password, self.server):
                logger.error(f"MT5 login failed: {mt5.last_error()}")
            else:
                for sym in self.instruments:
                    if not mt5.symbol_select(sym, True):
                        logger.warning(f"Failed to select symbol {sym}")
                logger.info("MT5 connected and symbols selected")
                return
            logger.info(f"Retrying MT5 connect in {self.retry_delay}s (#{attempt})")
            time.sleep(self.retry_delay)
        raise ConnectionError("Unable to connect to MT5 after retries")

    def disconnect(self) -> None:
        """Clean shutdown of MT5."""
        mt5.shutdown()
        logger.info("MT5 disconnected")

    def fetch_historical(
        self, symbol: str, timeframe: str, n_bars: int
    ) -> pd.DataFrame:
        """Fetch last `n_bars` of OHLC + tick_volume."""
        tf = self._tf_map[timeframe]
        now = datetime.datetime.now()
        rates = mt5.copy_rates_from(symbol, tf, now, n_bars)
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        return df[["open", "high", "low", "close", "tick_volume"]]

    def get_historical_data(self, n_bars: int = 1000) -> dict[str, dict[str, pd.DataFrame]]:
        """Fetch historical bars for all instruments and timeframes, and inject volatility."""
        data = {}
        for sym in self.instruments:
            sym_internal = sym[:3] + "/" + sym[3:]  # "XAUUSD"  "XAU/USD"
            data[sym_internal] = {}
            for tf in self.timeframes:
                df = self.fetch_historical(sym, tf, n_bars)
                # Volatility: high-low range for every bar, every timeframe
                df["volatility"] = df["high"] - df["low"]

                # Optional: If you want ATR for D1 only, uncomment below
                # if tf == "D1":
                #     tr = df[["high", "low", "close"]].copy()
                #     tr["prev_close"] = tr["close"].shift(1)
                #     tr["tr"] = tr[["high", "prev_close"]].max(axis=1) - tr[["low", "prev_close"]].min(axis=1)
                #     df["volatility"] = tr["tr"].rolling(10).mean().fillna(0.0)

                data[sym_internal][tf] = df
        return data

    def get_latest_tick(self, symbol: str) -> mt5.Tick:
        """Return the latest tick for `symbol`."""
        return mt5.symbol_info_tick(symbol)

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
        Send a market order.
        `side` is "buy" or "sell", `volume` in lots.
        """
        tick = self.get_latest_tick(symbol)
        if tick is None:
            raise RuntimeError(f"No tick available for {symbol}")

        order_type = mt5.ORDER_TYPE_BUY if side.lower() == "buy" else mt5.ORDER_TYPE_SELL
        price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid

        req = {
            "action":      mt5.TRADE_ACTION_DEAL,
            "symbol":      symbol,
            "volume":      volume,
            "type":        order_type,
            "price":       price,
            "deviation":   deviation,
            "magic":       magic,
            "comment":     comment,
            "type_time":   mt5.ORDER_TIME_GTC,
            "type_filling":mt5.ORDER_FILLING_IOC,
        }
        res = mt5.order_send(req)
        if res.retcode != mt5.TRADE_RETCODE_DONE:
            logger.warning(f"Order failed: {res.comment}")
        else:
            logger.info(f"Order sent: {symbol} {side} {volume}@{price}")
        return res
