"""
Unified Data Provider Interface
Clean abstraction for both CSV and live data sources
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, cast
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
from pathlib import Path
import logging


@dataclass
class DataProviderStatus:
    """Status information for data provider"""
    connected: bool = False
    last_update: Optional[datetime] = None
    error_count: int = 0
    last_error: Optional[str] = None
    data_quality_score: float = 100.0
    instruments_available: List[str] = field(default_factory=list)
    timeframes_available: List[str] = field(default_factory=list)


@dataclass
class MarketData:
    """Standardized market data structure"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    spread: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'spread': self.spread
        }

    @classmethod
    def from_series(cls, series: pd.Series, timestamp: Optional[datetime] = None) -> MarketData:
        """Create MarketData from pandas Series"""
        if timestamp is None:
            if series.name is not None:
                timestamp = pd.to_datetime(str(series.name))
            else:
                timestamp = datetime.now()

        return cls(
            timestamp=timestamp,
            open=float(series.get('open', 0)),
            high=float(series.get('high', 0)),
            low=float(series.get('low', 0)),
            close=float(series.get('close', 0)),
            volume=float(series.get('volume', 0)),
            spread=float(series.get('spread', 0))
        )


class DataProviderInterface(ABC):
    """Abstract interface for all data providers"""

    @abstractmethod
    def connect(self) -> bool:
        """Connect to data source"""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from data source"""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to data source"""
        pass

    @abstractmethod
    def get_status(self) -> DataProviderStatus:
        """Get current provider status"""
        pass

    @abstractmethod
    def get_available_instruments(self) -> List[str]:
        """Get list of available instruments"""
        pass

    @abstractmethod
    def get_available_timeframes(self) -> List[str]:
        """Get list of available timeframes"""
        pass

    @abstractmethod
    def get_historical_data(
        self,
        instrument: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        count: Optional[int] = None
    ) -> pd.DataFrame:
        """Get historical data for instrument and timeframe"""
        pass

    @abstractmethod
    def get_latest_data(self, instrument: str, timeframe: str, count: int = 1) -> pd.DataFrame:
        """Get latest data points"""
        pass

    @abstractmethod
    def get_current_price(self, instrument: str) -> Optional[MarketData]:
        """Get current price for instrument"""
        pass

    @abstractmethod
    def validate_data_integrity(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate data integrity and return issues"""
        pass


class CSVDataProvider(DataProviderInterface):
    """CSV-based data provider for offline training"""

    def __init__(self, data_directory: str, instruments: List[str], timeframes: List[str]):
        self.data_directory = Path(data_directory)
        self.instruments = instruments
        self.timeframes = timeframes
        self.logger = logging.getLogger(__name__)

        self._connected = False
        self._data_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
        self._status = DataProviderStatus()
        self._current_positions: Dict[str, int] = {}  # For simulation

    def connect(self) -> bool:
        """Load all CSV data into memory"""
        try:
            self.logger.info(f"Loading CSV data from {self.data_directory}")

            if not self.data_directory.exists():
                raise FileNotFoundError(f"Data directory not found: {self.data_directory}")

            loaded_count = 0
            total_expected = len(self.instruments) * len(self.timeframes)

            for instrument in self.instruments:
                self._data_cache[instrument] = {}

                for timeframe in self.timeframes:
                    # Convert instrument format to file format (EUR/USD -> EUR_USD)
                    file_instrument = instrument.replace("/", "_")
                    file_path = self.data_directory / f"{file_instrument}_{timeframe}.csv"

                    if file_path.exists():
                        try:
                            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                            df = self._standardize_dataframe(df)

                            is_valid, issues = self.validate_data_integrity(df)
                            if not is_valid:
                                self.logger.warning(f"Data integrity issues in {file_path}: {issues}")

                            self._data_cache[instrument][timeframe] = df
                            loaded_count += 1
                            self.logger.debug(f"Loaded {len(df)} bars from {file_path}")

                        except Exception as e:
                            self.logger.error(f"Failed to load {file_path}: {e}")
                            self._status.error_count += 1
                            self._status.last_error = str(e)
                    else:
                        self.logger.warning(f"CSV file not found: {file_path}")

            self._connected = loaded_count > 0
            self._status.connected = self._connected
            self._status.last_update = datetime.now()
            self._status.instruments_available = list(self._data_cache.keys())
            self._status.timeframes_available = self.timeframes
            self._status.data_quality_score = (loaded_count / total_expected) * 100 if total_expected > 0 else 0

            # Initialize positions for simulation
            for instrument in self._data_cache:
                self._current_positions[instrument] = 0

            self.logger.info(f"CSV provider connected: {loaded_count}/{total_expected} datasets loaded")
            return self._connected

        except Exception as e:
            self.logger.error(f"Failed to connect CSV provider: {e}")
            self._status.error_count += 1
            self._status.last_error = str(e)
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Clear cached data"""
        self._data_cache.clear()
        self._current_positions.clear()
        self._connected = False
        self._status.connected = False
        self.logger.info("CSV provider disconnected")

    def is_connected(self) -> bool:
        return self._connected

    def get_status(self) -> DataProviderStatus:
        return self._status

    def get_available_instruments(self) -> List[str]:
        return list(self._data_cache.keys()) if self._connected else []

    def get_available_timeframes(self) -> List[str]:
        if not self._connected or not self._data_cache:
            return []

        # Get timeframes from first instrument
        first_instrument = next(iter(self._data_cache.keys()))
        return list(self._data_cache[first_instrument].keys())

    def get_historical_data(
        self,
        instrument: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        count: Optional[int] = None
    ) -> pd.DataFrame:
        """Get historical data with optional filtering"""
        if not self._connected:
            raise RuntimeError("CSV provider not connected")

        if instrument not in self._data_cache:
            raise ValueError(f"Instrument not available: {instrument}")

        if timeframe not in self._data_cache[instrument]:
            raise ValueError(f"Timeframe not available: {timeframe}")

        df = self._data_cache[instrument][timeframe].copy()

        # Apply date filtering
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        # Apply count limiting
        if count and count > 0:
            df = df.tail(count)

        return df

    def get_latest_data(self, instrument: str, timeframe: str, count: int = 1) -> pd.DataFrame:
        """Get latest data points"""
        return self.get_historical_data(instrument, timeframe, count=count)

    def get_current_price(self, instrument: str) -> Optional[MarketData]:
        """Get current (latest) price"""
        try:
            # Use primary timeframe (H1) for current price
            primary_tf = "H1" if "H1" in self.get_available_timeframes() else self.get_available_timeframes()[0]
            latest_data = self.get_latest_data(instrument, primary_tf, count=1)

            if latest_data.empty:
                return None

            latest_row = latest_data.iloc[-1]
            return MarketData.from_series(latest_row, latest_data.index[-1])

        except Exception as e:
            self.logger.error(f"Failed to get current price for {instrument}: {e}")
            return None

    def advance_simulation(self, instrument: str, steps: int = 1) -> None:
        """Advance simulation position (for training mode)"""
        if instrument in self._current_positions:
            max_position = len(self._data_cache[instrument][self.timeframes[0]]) - 1
            self._current_positions[instrument] = min(
                self._current_positions[instrument] + steps,
                max_position
            )

    def get_simulation_data(self, instrument: str, timeframe: str, window_size: int = 100) -> pd.DataFrame:
        """Get data for current simulation position"""
        if instrument not in self._current_positions:
            return pd.DataFrame()

        current_pos = self._current_positions[instrument]
        start_pos = max(0, current_pos - window_size + 1)
        end_pos = current_pos + 1

        df = self._data_cache[instrument][timeframe]
        return df.iloc[start_pos:end_pos]

    def validate_data_integrity(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate CSV data integrity"""
        issues = []

        if data.empty:
            issues.append("Empty dataset")
            return False, issues

        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")

        # Check for NaN values
        if data.isnull().any().any():
            nan_cols = data.columns[data.isnull().any()].tolist()
            issues.append(f"NaN values in columns: {nan_cols}")

        # Check OHLC consistency
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            invalid_ohlc = (
                (data['high'] < data[['open', 'close']].max(axis=1)) |
                (data['low'] > data[['open', 'close']].min(axis=1))
            )
            if invalid_ohlc.any():
                issues.append(f"OHLC inconsistencies in {invalid_ohlc.sum()} rows")

        # Check for negative values where inappropriate
        if 'volume' in data.columns and (data['volume'] < 0).any():
            issues.append("Negative volume values found")

        return len(issues) == 0, issues

    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize DataFrame format"""
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']

        for col in required_columns:
            if col not in df.columns:
                df[col] = 0.0

        # Add derived columns
        if 'spread' not in df.columns:
            df['spread'] = 0.0

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Sort by timestamp
        df = df.sort_index()

        # Remove duplicates
        df = df[~df.index.duplicated(keep='last')]

        return df


class LiveDataProvider(DataProviderInterface):
    """Live data provider using MT5 (simplified for now)"""

    def __init__(
        self,
        login: int,
        password: str,
        server: str,
        instruments: List[str],
        timeframes: List[str],
        path: Optional[str] = None,  # Path to specific MT5 terminal (e.g., FTMO)
    ):
        self.login = login
        self.password = password
        self.server = server
        self.terminal_path = path
        self.instruments = instruments
        self.timeframes = timeframes
        self.logger = logging.getLogger(__name__)

        self._connected = False
        self._status = DataProviderStatus()

        # Import MT5 here to avoid dependency issues when not needed
        # Cast to Any to suppress Pylance warnings about dynamic attributes
        try:
            import MetaTrader5 as _mt5
            self.mt5: Any = cast(Any, _mt5)
        except ImportError:
            self.logger.error("MetaTrader5 package not available")
            self.mt5 = None

    def connect(self) -> bool:
        """Connect to MT5"""
        if not self.mt5:
            self.logger.error("MT5 not available")
            return False

        try:
            # Build initialization kwargs (include path if specified for FTMO terminal)
            init_kwargs: Dict[str, Any] = {}
            if self.terminal_path:
                init_kwargs["path"] = self.terminal_path
                self.logger.info(f"Using MT5 terminal at: {self.terminal_path}")

            if not self.mt5.initialize(**init_kwargs):
                error = self.mt5.last_error()
                raise Exception(f"MT5 initialization failed: {error}")

            if not self.mt5.login(self.login, self.password, self.server):
                error = self.mt5.last_error()
                raise Exception(f"MT5 login failed: {error}")

            # Verify instruments
            available_instruments = []
            for instrument in self.instruments:
                if self.mt5.symbol_select(instrument, True):
                    available_instruments.append(instrument)
                else:
                    self.logger.warning(f"Instrument not available: {instrument}")

            self._connected = len(available_instruments) > 0
            self._status.connected = self._connected
            self._status.last_update = datetime.now()
            self._status.instruments_available = available_instruments
            self._status.timeframes_available = self.timeframes

            self.logger.info(f"MT5 provider connected: {len(available_instruments)} instruments available")
            return self._connected

        except Exception as e:
            self.logger.error(f"Failed to connect to MT5: {e}")
            self._status.error_count += 1
            self._status.last_error = str(e)
            return False

    def disconnect(self) -> None:
        """Disconnect from MT5"""
        if self.mt5 and self._connected:
            self.mt5.shutdown()
        self._connected = False
        self._status.connected = False
        self.logger.info("MT5 provider disconnected")

    def is_connected(self) -> bool:
        return self._connected and self.mt5 is not None

    def get_status(self) -> DataProviderStatus:
        return self._status

    def get_available_instruments(self) -> List[str]:
        return self._status.instruments_available

    def get_available_timeframes(self) -> List[str]:
        return self._status.timeframes_available

    def get_historical_data(
        self,
        instrument: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        count: Optional[int] = None
    ) -> pd.DataFrame:
        """Get historical data from MT5"""
        if not self.is_connected():
            raise RuntimeError("MT5 provider not connected")

        # Convert timeframe to MT5 constant
        tf_map = {
            "M1": self.mt5.TIMEFRAME_M1,
            "M5": self.mt5.TIMEFRAME_M5,
            "M15": self.mt5.TIMEFRAME_M15,
            "M30": self.mt5.TIMEFRAME_M30,
            "H1": self.mt5.TIMEFRAME_H1,
            "H4": self.mt5.TIMEFRAME_H4,
            "D1": self.mt5.TIMEFRAME_D1,
        }

        if timeframe not in tf_map:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        try:
            if count:
                rates = self.mt5.copy_rates_from_pos(instrument, tf_map[timeframe], 0, count)
            elif end_date:
                rates = self.mt5.copy_rates_from(instrument, tf_map[timeframe], end_date, count or 1000)
            else:
                # Default: get last 1000 bars
                rates = self.mt5.copy_rates_from_pos(instrument, tf_map[timeframe], 0, 1000)

            if rates is None or len(rates) == 0:
                self.logger.warning(f"No data received for {instrument} {timeframe}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)

            # Rename columns to standard format
            if 'tick_volume' in df.columns:
                df['volume'] = df['tick_volume']
                df.drop('tick_volume', axis=1, inplace=True)

            # Add spread column
            df['spread'] = 0.0  # Could be calculated from bid/ask if available

            return df[['open', 'high', 'low', 'close', 'volume', 'spread']]

        except Exception as e:
            self.logger.error(f"Failed to get historical data: {e}")
            self._status.error_count += 1
            self._status.last_error = str(e)
            return pd.DataFrame()

    def get_latest_data(self, instrument: str, timeframe: str, count: int = 1) -> pd.DataFrame:
        """Get latest data points"""
        return self.get_historical_data(instrument, timeframe, count=count)

    def get_current_price(self, instrument: str) -> Optional[MarketData]:
        """Get current price from MT5"""
        if not self.is_connected():
            return None

        try:
            tick = self.mt5.symbol_info_tick(instrument)
            if tick is None:
                return None

            return MarketData(
                timestamp=datetime.fromtimestamp(tick.time),
                open=tick.bid,  # Use bid as proxy for open
                high=tick.bid,  # For tick data, OHLC are the same
                low=tick.bid,
                close=tick.bid,
                volume=0,  # Tick volume not available in tick info
                spread=tick.ask - tick.bid
            )

        except Exception as e:
            self.logger.error(f"Failed to get current price: {e}")
            return None

    def validate_data_integrity(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate live data integrity"""
        # Similar to CSV validation but with different expectations
        issues = []

        if data.empty:
            issues.append("Empty dataset")
            return False, issues

        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")

        # Check for reasonable price values (not zero or negative)
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in data.columns and (data[col] <= 0).any():
                issues.append(f"Invalid price values in {col}")

        return len(issues) == 0, issues


class DataProviderFactory:
    """Factory for creating data providers"""

    @staticmethod
    def create_provider(config: Dict[str, Any]) -> DataProviderInterface:
        """Create data provider based on configuration"""
        mode = config.get('mode', 'offline')

        if mode == 'offline':
            return CSVDataProvider(
                data_directory=config.get('csv_data_dir', 'data/processed'),
                instruments=config.get('instruments', []),
                timeframes=config.get('timeframes', [])
            )
        elif mode in ['online', 'hybrid']:
            if not all(key in config for key in ['mt5_login', 'mt5_password']):
                raise ValueError("MT5 credentials required for online mode")

            return LiveDataProvider(
                login=config['mt5_login'],
                password=config['mt5_password'],
                server=config.get('mt5_server', 'MetaQuotes-Demo'),
                instruments=config.get('instruments', []),
                timeframes=config.get('timeframes', []),
                path=config.get('mt5_path'),  # Support FTMO terminal path
            )
        else:
            raise ValueError(f"Unsupported data mode: {mode}")