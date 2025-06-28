# utils/mt5_data_collector.py
"""
Complete MT5 Data Collection and Cleaning System
Extracts, cleans, and prepares data for the trading environment
"""

import os
import sys
import logging
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════
# Configuration and Setup
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DataConfig:
    """Configuration for data collection"""
    symbols: List[str] = None
    timeframes: Dict[str, int] = None
    start_date: datetime = None
    end_date: datetime = None
    output_dir: str = "data"
    min_bars: int = 1000
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["EURUSD", "XAUUSD"]
        
        if self.timeframes is None:
            self.timeframes = {
                "H1": mt5.TIMEFRAME_H1,
                "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1
            }
        
        if self.start_date is None:
            # Default: 2 years of data
            self.start_date = datetime.now() - timedelta(days=730)
        
        if self.end_date is None:
            self.end_date = datetime.now()


class MT5DataCollector:
    """Complete MT5 data collection and cleaning system"""
    
    def __init__(self, config: DataConfig = None):
        self.config = config or DataConfig()
        self.setup_logging()
        self.data_cache = {}
        
    def setup_logging(self):
        """Setup logging for data collection"""
        log_dir = Path(self.config.output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "data_collection.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger("MT5DataCollector")
    
    # ═══════════════════════════════════════════════════════════════════
    # MT5 Connection and Setup
    # ═══════════════════════════════════════════════════════════════════
    
    def connect_mt5(self) -> bool:
        """Connect to MT5 terminal"""
        try:
            if not mt5.initialize():
                self.logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
            
            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                self.logger.warning("No account info available")
            else:
                self.logger.info(f"Connected to MT5 - Account: {account_info.login}, Server: {account_info.server}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"MT5 connection failed: {e}")
            return False
    
    def setup_symbols(self) -> bool:
        """Setup and verify all required symbols"""
        success = True
        
        for symbol in self.config.symbols:
            # Try to select symbol
            if not mt5.symbol_select(symbol, True):
                self.logger.error(f"Failed to select symbol: {symbol}")
                success = False
                continue
            
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self.logger.error(f"No symbol info for: {symbol}")
                success = False
                continue
            
            # Check if symbol is available for trading
            if not symbol_info.visible:
                self.logger.warning(f"Symbol {symbol} not visible in Market Watch")
            
            self.logger.info(
                f"✅ {symbol}: spread={symbol_info.spread}, "
                f"digits={symbol_info.digits}, point={symbol_info.point}"
            )
        
        return success
    
    # ═══════════════════════════════════════════════════════════════════
    # Data Extraction
    # ═══════════════════════════════════════════════════════════════════
    
    def get_symbol_data(self, symbol: str, timeframe_name: str, timeframe: int) -> Optional[pd.DataFrame]:
        """Extract data for a single symbol/timeframe combination"""
        try:
            self.logger.info(f"Fetching {symbol} {timeframe_name} data...")
            
            # Get data from MT5
            rates = mt5.copy_rates_range(
                symbol,
                timeframe,
                self.config.start_date,
                self.config.end_date
            )
            
            if rates is None or len(rates) == 0:
                self.logger.error(f"No data returned for {symbol} {timeframe_name}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            
            # Convert time to datetime
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Rename columns to standard format
            df = df.rename(columns={
                'time': 'timestamp',
                'open': 'open',
                'high': 'high', 
                'low': 'low',
                'close': 'close',
                'tick_volume': 'volume',
                'real_volume': 'real_volume'
            })
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Handle volume data
            if 'real_volume' in df.columns and df['real_volume'].sum() > 0:
                # Use real volume if available and non-zero
                df['volume'] = df['real_volume']
            # else keep tick_volume as volume
            
            self.logger.info(f"✅ {symbol} {timeframe_name}: {len(df)} bars from {df.index[0]} to {df.index[-1]}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting {symbol} {timeframe_name} data: {e}")
            return None
    
    def collect_all_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Collect data for all symbols and timeframes"""
        if not self.connect_mt5():
            return {}
        
        if not self.setup_symbols():
            self.logger.error("Symbol setup failed")
            return {}
        
        all_data = {}
        
        for symbol in self.config.symbols:
            symbol_data = {}
            
            for tf_name, tf_value in self.config.timeframes.items():
                df = self.get_symbol_data(symbol, tf_name, tf_value)
                
                if df is not None and len(df) >= self.config.min_bars:
                    symbol_data[tf_name] = df
                else:
                    self.logger.warning(f"Insufficient data for {symbol} {tf_name}")
            
            if symbol_data:
                # Convert symbol name to standard format
                standard_name = self.standardize_symbol_name(symbol)
                all_data[standard_name] = symbol_data
        
        mt5.shutdown()
        self.logger.info(f"✅ Data collection completed for {len(all_data)} symbols")
        
        return all_data
    
    # ═══════════════════════════════════════════════════════════════════
    # Data Cleaning and Processing
    # ═══════════════════════════════════════════════════════════════════
    
    def clean_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
        """Clean and validate OHLCV data"""
        original_len = len(df)
        df_clean = df.copy()
        
        # 1. Remove rows with missing OHLC data
        ohlc_cols = ['open', 'high', 'low', 'close']
        df_clean = df_clean.dropna(subset=ohlc_cols)
        
        # 2. Validate OHLC relationships
        invalid_ohlc = (
            (df_clean['high'] < df_clean['low']) |
            (df_clean['high'] < df_clean['open']) |
            (df_clean['high'] < df_clean['close']) |
            (df_clean['low'] > df_clean['open']) |
            (df_clean['low'] > df_clean['close'])
        )
        
        if invalid_ohlc.sum() > 0:
            self.logger.warning(f"{symbol} {timeframe}: Removing {invalid_ohlc.sum()} invalid OHLC bars")
            df_clean = df_clean[~invalid_ohlc]
        
        # 3. Remove extreme outliers (gaps > 10 standard deviations)
        if len(df_clean) > 20:
            close_returns = df_clean['close'].pct_change()
            std_threshold = 10 * close_returns.std()
            extreme_moves = abs(close_returns) > std_threshold
            
            if extreme_moves.sum() > 0:
                self.logger.warning(f"{symbol} {timeframe}: Removing {extreme_moves.sum()} extreme outliers")
                df_clean = df_clean[~extreme_moves]
        
        # 4. Handle zero/negative prices
        zero_prices = (df_clean[ohlc_cols] <= 0).any(axis=1)
        if zero_prices.sum() > 0:
            self.logger.warning(f"{symbol} {timeframe}: Removing {zero_prices.sum()} zero/negative price bars")
            df_clean = df_clean[~zero_prices]
        
        # 5. Handle volume data
        if 'volume' not in df_clean.columns or df_clean['volume'].isna().all():
            df_clean['volume'] = 1.0  # Default volume
        else:
            # Replace negative/zero volume with median
            median_vol = df_clean['volume'][df_clean['volume'] > 0].median()
            df_clean.loc[df_clean['volume'] <= 0, 'volume'] = median_vol
        
        # 6. Fill remaining NaN values
        df_clean['volume'] = df_clean['volume'].fillna(method='ffill').fillna(1.0)
        
        # 7. Sort by timestamp
        df_clean = df_clean.sort_index()
        
        # 8. Remove duplicates
        df_clean = df_clean[~df_clean.index.duplicated(keep='first')]
        
        removed_count = original_len - len(df_clean)
        if removed_count > 0:
            self.logger.info(f"🧹 {symbol} {timeframe}: Cleaned {removed_count} problematic bars ({len(df_clean)} remaining)")
        
        return df_clean
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators required by the trading system"""
        df_enhanced = df.copy()
        
        # 1. Volatility (rolling standard deviation of returns)
        returns = df_enhanced['close'].pct_change()
        df_enhanced['volatility'] = returns.rolling(window=20, min_periods=5).std()
        df_enhanced['volatility'] = df_enhanced['volatility'].fillna(method='ffill').fillna(0.01)
        
        # 2. Average True Range (ATR)
        high_low = df_enhanced['high'] - df_enhanced['low']
        high_close = abs(df_enhanced['high'] - df_enhanced['close'].shift(1))
        low_close = abs(df_enhanced['low'] - df_enhanced['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df_enhanced['atr'] = true_range.rolling(window=14, min_periods=5).mean()
        df_enhanced['atr'] = df_enhanced['atr'].fillna(method='ffill').fillna(0.001)
        
        # 3. Price momentum
        df_enhanced['momentum_5'] = df_enhanced['close'].pct_change(5)
        df_enhanced['momentum_20'] = df_enhanced['close'].pct_change(20)
        
        # 4. Volume profile
        df_enhanced['volume_ma'] = df_enhanced['volume'].rolling(window=20, min_periods=5).mean()
        df_enhanced['volume_ratio'] = df_enhanced['volume'] / df_enhanced['volume_ma']
        
        # Fill any remaining NaN values
        numeric_cols = df_enhanced.select_dtypes(include=[np.number]).columns
        df_enhanced[numeric_cols] = df_enhanced[numeric_cols].fillna(method='ffill').fillna(0)
        
        return df_enhanced
    
    def validate_data_quality(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, dict]]:
        """Validate data quality and return quality metrics"""
        quality_report = {}
        
        for symbol, timeframes in data.items():
            quality_report[symbol] = {}
            
            for tf, df in timeframes.items():
                # Basic stats
                stats = {
                    'total_bars': len(df),
                    'date_range': f"{df.index[0]} to {df.index[-1]}",
                    'missing_data_pct': df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100,
                    'zero_volume_pct': (df['volume'] == 0).sum() / len(df) * 100,
                    'avg_spread_pct': ((df['high'] - df['low']) / df['close']).mean() * 100,
                }
                
                # Data continuity check
                time_diff = df.index.to_series().diff()
                expected_interval = self.get_expected_interval(tf)
                continuity = (time_diff <= expected_interval * 1.5).sum() / len(df) * 100
                stats['continuity_pct'] = continuity
                
                quality_report[symbol][tf] = stats
                
                # Log quality summary
                self.logger.info(
                    f"📊 {symbol} {tf}: {stats['total_bars']} bars, "
                    f"{stats['continuity_pct']:.1f}% continuity, "
                    f"{stats['missing_data_pct']:.1f}% missing data"
                )
        
        return quality_report
    
    # ═══════════════════════════════════════════════════════════════════
    # Utility Functions
    # ═══════════════════════════════════════════════════════════════════
    
    def standardize_symbol_name(self, symbol: str) -> str:
        """Convert MT5 symbol name to standard format"""
        symbol_map = {
            "EURUSD": "EUR/USD",
            "XAUUSD": "XAU/USD", 
            "GBPUSD": "GBP/USD",
            "USDJPY": "USD/JPY",
            "USDCHF": "USD/CHF",
            "AUDUSD": "AUD/USD",
            "USDCAD": "USD/CAD",
            "NZDUSD": "NZD/USD",
        }
        return symbol_map.get(symbol, symbol)
    
    def get_expected_interval(self, timeframe: str) -> timedelta:
        """Get expected time interval for timeframe"""
        intervals = {
            'H1': timedelta(hours=1),
            'H4': timedelta(hours=4),
            'D1': timedelta(days=1),
        }
        return intervals.get(timeframe, timedelta(hours=1))
    
    def save_data(self, data: Dict[str, Dict[str, pd.DataFrame]], format: str = 'both') -> bool:
        """Save data in specified format(s) - Compatible with existing load_data function"""
        try:
            output_dir = Path(self.config.output_dir)
            
            # Create directories
            processed_dir = output_dir / "processed"
            raw_dir = output_dir / "raw"
            processed_dir.mkdir(parents=True, exist_ok=True)
            raw_dir.mkdir(parents=True, exist_ok=True)
            
            if format in ['pickle', 'both']:
                # Save as pickle (fast loading for Python)
                pickle_path = processed_dir / "market_data.pkl"
                with open(pickle_path, 'wb') as f:
                    pickle.dump(data, f)
                self.logger.info(f"💾 Data saved to {pickle_path}")
            
            if format in ['csv', 'both']:
                # Save CSV files compatible with existing load_data function
                for symbol, timeframes in data.items():
                    # Convert symbol format for filename
                    symbol_code = symbol.replace('/', '').replace('-', '')  # EUR/USD -> EURUSD
                    
                    for tf, df in timeframes.items():
                        # FIXED: Match existing naming convention
                        csv_filename = f"{symbol_code}_{tf}_features.csv"
                        csv_path = processed_dir / csv_filename
                        
                        # Prepare DataFrame for existing load_data function
                        df_export = df.copy()
                        
                        # FIXED: Reset index to make timestamp a column named 'time'
                        df_export.reset_index(inplace=True)
                        if 'timestamp' in df_export.columns:
                            df_export.rename(columns={'timestamp': 'time'}, inplace=True)
                        elif df_export.index.name == 'timestamp':
                            df_export['time'] = df_export.index
                        
                        # Ensure required columns exist
                        required_cols = ['time', 'open', 'high', 'low', 'close', 'volume', 'volatility']
                        missing_cols = set(required_cols) - set(df_export.columns)
                        if missing_cols:
                            self.logger.warning(f"Adding missing columns for {symbol} {tf}: {missing_cols}")
                            for col in missing_cols:
                                if col == 'volume':
                                    df_export[col] = 1.0
                                elif col == 'volatility':
                                    df_export[col] = 0.01
                                else:
                                    df_export[col] = 0.0
                        
                        # Save in the format expected by load_data
                        df_export.to_csv(csv_path, index=False)
                        self.logger.info(f"💾 {symbol} {tf} saved to {csv_path}")
                
                # Also save in nested structure for backup
                for symbol, timeframes in data.items():
                    symbol_dir = raw_dir / symbol.replace('/', '_')
                    symbol_dir.mkdir(exist_ok=True)
                    
                    for tf, df in timeframes.items():
                        backup_path = symbol_dir / f"{tf}.csv"
                        df.to_csv(backup_path)
            
            # Save metadata
            metadata = {
                'collection_date': datetime.now().isoformat(),
                'symbols': list(data.keys()),
                'timeframes': list(self.config.timeframes.keys()),
                'total_bars': sum(len(df) for tfs in data.values() for df in tfs.values()),
                'date_range': {
                    'start': self.config.start_date.isoformat(),
                    'end': self.config.end_date.isoformat()
                },
                'format_info': {
                    'csv_naming': '{SYMBOL}_{TF}_features.csv',
                    'timestamp_column': 'time',
                    'location': 'data/processed/',
                    'compatible_with': 'existing load_data function'
                }
            }
            
            metadata_path = processed_dir / "metadata.json"
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save data: {e}")
            return False
    
    # ═══════════════════════════════════════════════════════════════════
    # Main Collection Pipeline
    # ═══════════════════════════════════════════════════════════════════
    
    def run_collection_pipeline(self, save_format: str = 'both') -> Dict[str, Dict[str, pd.DataFrame]]:
        """Run the complete data collection and cleaning pipeline"""
        self.logger.info("🚀 Starting MT5 data collection pipeline")
        self.logger.info(f"Symbols: {self.config.symbols}")
        self.logger.info(f"Timeframes: {list(self.config.timeframes.keys())}")
        self.logger.info(f"Date range: {self.config.start_date} to {self.config.end_date}")
        
        # Step 1: Collect raw data
        raw_data = self.collect_all_data()
        if not raw_data:
            self.logger.error("❌ Data collection failed")
            return {}
        
        # Step 2: Clean and enhance data
        cleaned_data = {}
        for symbol, timeframes in raw_data.items():
            cleaned_data[symbol] = {}
            
            for tf, df in timeframes.items():
                self.logger.info(f"🧹 Cleaning {symbol} {tf}...")
                
                # Clean the data
                df_clean = self.clean_data(df, symbol, tf)
                
                # Add technical indicators
                df_enhanced = self.add_technical_indicators(df_clean)
                
                cleaned_data[symbol][tf] = df_enhanced
        
        # Step 3: Validate data quality
        quality_report = self.validate_data_quality(cleaned_data)
        
        # Step 4: Save data
        if self.save_data(cleaned_data, save_format):
            self.logger.info("✅ Data collection pipeline completed successfully")
        else:
            self.logger.error("❌ Failed to save data")
        
        return cleaned_data


# ═══════════════════════════════════════════════════════════════════
# Quick Usage Functions
# ═══════════════════════════════════════════════════════════════════

def quick_collect_data(
    symbols: List[str] = None,
    days_back: int = 365,
    output_dir: str = "data"
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Quick data collection with default settings"""
    
    if symbols is None:
        symbols = ["EURUSD", "XAUUSD"]
    
    config = DataConfig(
        symbols=symbols,
        start_date=datetime.now() - timedelta(days=days_back),
        end_date=datetime.now(),
        output_dir=output_dir
    )
    
    collector = MT5DataCollector(config)
    return collector.run_collection_pipeline()


def load_collected_data(data_dir: str = "data") -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load previously collected data"""
    pickle_path = Path(data_dir) / "processed" / "market_data.pkl"
    
    if pickle_path.exists():
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f"No data found at {pickle_path}")


# ═══════════════════════════════════════════════════════════════════
# Main Execution
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🔄 MT5 Data Collection System")
    print("=" * 50)
    
    # Configure data collection
    config = DataConfig(
        symbols=["EURUSD", "XAUUSD"],  # Your main trading pairs
        start_date=datetime(2022, 1, 1),  # 2+ years of data
        end_date=datetime.now(),
        output_dir="data",
        min_bars=500  # Minimum bars required
    )
    
    # Run collection
    collector = MT5DataCollector(config)
    data = collector.run_collection_pipeline(save_format='both')
    
    if data:
        print(f"\n✅ Successfully collected data for {len(data)} symbols:")
        for symbol, timeframes in data.items():
            print(f"  📈 {symbol}:")
            for tf, df in timeframes.items():
                print(f"    {tf}: {len(df):,} bars ({df.index[0]} to {df.index[-1]})")
        
        print(f"\n📁 Data saved to: {config.output_dir}")
        print("🚀 Ready for training!")
    else:
        print("❌ Data collection failed")