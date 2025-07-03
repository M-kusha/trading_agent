# ─────────────────────────────────────────────────────────────
# File: live/enhanced_live_connector.py
# InfoBus-Integrated Live Data Connector with Comprehensive Health Monitoring
# ─────────────────────────────────────────────────────────────

import time
import datetime
import logging
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from typing import Any, Dict, List, Optional, Tuple
from collections import deque, defaultdict
from stable_baselines3.common.callbacks import BaseCallback

# InfoBus and audit infrastructure
from modules.utils.info_bus import InfoBus, InfoBusUpdater, InfoBusExtractor
from modules.utils.audit_utils import RotatingLogger, AuditTracker, format_operator_message, system_audit
from modules.core.core import Module, ModuleConfig
from live.mt5_credentials import MT5Credentials


class InfoBusLiveDataConnector(Module):
    """
    Enhanced live data connector with comprehensive InfoBus integration.
    Provides real-time market data with health monitoring and audit trails.
    """
    
    def __init__(
        self,
        instruments: List[str],
        timeframes: List[str],
        max_retries: int = 3,
        retry_delay: float = 5.0,
        config: Optional[ModuleConfig] = None
    ):
        # Initialize base module with InfoBus support
        super().__init__(config or ModuleConfig())
        
        # Connection credentials
        self.account = MT5Credentials.ACCOUNT
        self.password = MT5Credentials.PASSWORD
        self.server = MT5Credentials.SERVER
        
        self.instruments = instruments
        self.timeframes = timeframes
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # ══════════════════════════════════════════════════════════════
        # Enhanced InfoBus Infrastructure
        # ══════════════════════════════════════════════════════════════
        
        # InfoBus-integrated logging with 2000-line rotation
        self.live_logger = RotatingLogger(
            name="LiveDataConnector",
            log_path=f"logs/live/infobus_live_{datetime.datetime.now().strftime('%Y%m%d')}.log",
            max_lines=2000,
            operator_mode=True
        )
        
        # Audit tracker for live trading events
        self.audit_tracker = AuditTracker("LiveTradingSystem")
        
        # ══════════════════════════════════════════════════════════════
        # Connection State Tracking
        # ══════════════════════════════════════════════════════════════
        
        self.connected = False
        self.last_connection_check = 0
        self.connection_failures = 0
        self.last_tick_time = {}
        self.connection_quality_score = 100.0
        
        # ══════════════════════════════════════════════════════════════
        # Data Quality Monitoring
        # ══════════════════════════════════════════════════════════════
        
        self.data_quality_history = deque(maxlen=100)
        self.tick_latency_history = defaultdict(lambda: deque(maxlen=50))
        self.data_gaps_detected = 0
        self.last_data_quality_check = 0
        
        # ══════════════════════════════════════════════════════════════
        # Enhanced MT5 Timeframe Mapping
        # ══════════════════════════════════════════════════════════════
        
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
        invalid_tfs = [tf for tf in timeframes if tf not in self._tf_map]
        if invalid_tfs:
            raise ValueError(f"Invalid timeframes: {invalid_tfs}. Valid: {list(self._tf_map.keys())}")
        
        # ══════════════════════════════════════════════════════════════
        # Performance Monitoring
        # ══════════════════════════════════════════════════════════════
        
        self.fetch_performance = defaultdict(lambda: deque(maxlen=50))
        self.total_fetches = 0
        self.failed_fetches = 0
        self.avg_fetch_time = 0.0
        
        self.live_logger.info(
            format_operator_message(
                "🚀", "LIVE_CONNECTOR_INITIALIZED",
                details=f"{len(instruments)} instruments, {len(timeframes)} timeframes",
                result="InfoBus integration enabled",
                context="live_startup"
            )
        )

    def reset(self) -> None:
        """Enhanced reset with InfoBus state clearing"""
        super().reset()
        
        # Reset connection state
        self.connected = False
        self.connection_failures = 0
        self.connection_quality_score = 100.0
        
        # Clear data quality tracking
        self.data_quality_history.clear()
        self.tick_latency_history.clear()
        self.data_gaps_detected = 0
        
        # Clear performance tracking
        self.fetch_performance.clear()
        self.total_fetches = 0
        self.failed_fetches = 0

    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration"""
        
        # Periodic connection health check
        current_time = time.time()
        if current_time - self.last_connection_check > 30:  # Every 30 seconds
            self._perform_connection_health_check(info_bus)
            self.last_connection_check = current_time
        
        # Periodic data quality check
        if current_time - self.last_data_quality_check > 60:  # Every minute
            self._perform_data_quality_check(info_bus)
            self.last_data_quality_check = current_time

    def _get_observation_impl(self) -> np.ndarray:
        """Get connection health as observation"""
        return np.array([
            self.connection_quality_score / 100.0,
            1.0 if self.connected else 0.0,
            min(self.connection_failures / 10.0, 1.0),
            min(self.data_gaps_detected / 50.0, 1.0)
        ], dtype=np.float32)

    def connect(self, info_bus: Optional[InfoBus] = None) -> bool:
        """Enhanced connection with InfoBus integration and comprehensive monitoring"""
        
        self.live_logger.info(
            format_operator_message(
                "🔗", "ATTEMPTING_MT5_CONNECTION",
                details=f"Server: {self.server}, Account: {self.account}",
                context="connection_attempt"
            )
        )
        
        for attempt in range(1, self.max_retries + 1):
            try:
                connection_start = time.time()
                
                # Initialize MT5
                if not mt5.initialize():
                    error = mt5.last_error()
                    raise ConnectionError(f"MT5 initialization failed: {error}")
                
                # Login with enhanced error handling
                if not mt5.login(self.account, self.password, self.server):
                    error = mt5.last_error()
                    mt5.shutdown()
                    raise ConnectionError(f"MT5 login failed: {error}")
                
                # Enhanced symbol selection with validation
                failed_symbols = []
                selected_symbols = []
                
                for symbol in self.instruments:
                    if mt5.symbol_select(symbol, True):
                        selected_symbols.append(symbol)
                        # Verify symbol info
                        symbol_info = mt5.symbol_info(symbol)
                        if symbol_info is None:
                            failed_symbols.append(f"{symbol} (no info)")
                    else:
                        failed_symbols.append(symbol)
                
                if not selected_symbols:
                    raise ConnectionError("No symbols could be selected")
                
                # Comprehensive connection verification
                verification_results = self._verify_comprehensive_connection()
                
                connection_time = time.time() - connection_start
                
                # Update connection state
                self.connected = True
                self.connection_failures = 0
                self.connection_quality_score = min(100.0, 100.0 - len(failed_symbols) * 10)
                
                # Log successful connection
                self.live_logger.info(
                    format_operator_message(
                        "✅", "MT5_CONNECTION_SUCCESSFUL",
                        details=f"Attempt {attempt}, Time: {connection_time:.2f}s",
                        result=f"Symbols: {len(selected_symbols)}/{len(self.instruments)}",
                        context="connection_success"
                    )
                )
                
                # Update InfoBus with connection status
                if info_bus is not None:
                    self._update_infobus_connection_status(info_bus, True, verification_results)
                
                # Record successful connection audit
                self.audit_tracker.record_event(
                    "connection_established",
                    "LiveDataConnector",
                    {
                        "attempt": attempt,
                        "connection_time": connection_time,
                        "selected_symbols": selected_symbols,
                        "failed_symbols": failed_symbols,
                        "quality_score": self.connection_quality_score
                    },
                    severity="info"
                )
                
                if failed_symbols:
                    self.live_logger.warning(
                        format_operator_message(
                            "⚠️", "SOME_SYMBOLS_FAILED",
                            details=f"Failed: {failed_symbols}",
                            context="connection_warning"
                        )
                    )
                
                return True
                
            except Exception as e:
                self.connection_failures += 1
                
                self.live_logger.error(
                    format_operator_message(
                        "❌", "CONNECTION_ATTEMPT_FAILED",
                        details=f"Attempt {attempt}/{self.max_retries}",
                        result=str(e),
                        context="connection_failure"
                    )
                )
                
                # Update InfoBus with failure
                if info_bus is not None:
                    InfoBusUpdater.add_alert(
                        info_bus,
                        f"MT5 connection attempt {attempt} failed: {e}",
                        severity="warning",
                        module="LiveDataConnector"
                    )
                
                if attempt < self.max_retries:
                    self.live_logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
        
        # All attempts failed
        self.connected = False
        self.connection_quality_score = 0.0
        
        if info_bus is not None:
            self._update_infobus_connection_status(info_bus, False, {"error": "All connection attempts failed"})
        
        # Record connection failure audit
        self.audit_tracker.record_event(
            "connection_failed",
            "LiveDataConnector",
            {
                "attempts": self.max_retries,
                "total_failures": self.connection_failures
            },
            severity="error"
        )
        
        raise ConnectionError(f"Unable to connect to MT5 after {self.max_retries} attempts")

    def disconnect(self, info_bus: Optional[InfoBus] = None) -> None:
        """Enhanced disconnect with InfoBus integration"""
        
        if self.connected:
            mt5.shutdown()
            self.connected = False
            
            self.live_logger.info(
                format_operator_message(
                    "🔌", "MT5_DISCONNECTED",
                    details="Clean shutdown completed",
                    context="disconnection"
                )
            )
            
            # Update InfoBus
            if info_bus is not None:
                self._update_infobus_connection_status(info_bus, False, {"reason": "clean_shutdown"})
            
            # Record disconnection audit
            self.audit_tracker.record_event(
                "disconnection",
                "LiveDataConnector",
                {"reason": "clean_shutdown"},
                severity="info"
            )

    def fetch_historical_with_infobus(
        self, 
        symbol: str, 
        timeframe: str, 
        n_bars: int,
        info_bus: Optional[InfoBus] = None
    ) -> pd.DataFrame:
        """Enhanced historical data fetching with InfoBus integration"""
        
        fetch_start = time.time()
        self.total_fetches += 1
        
        try:
            if not self.connected:
                raise RuntimeError("Not connected to MT5")
            
            if timeframe not in self._tf_map:
                raise ValueError(f"Invalid timeframe '{timeframe}'")
            
            tf_constant = self._tf_map[timeframe]
            
            # Get current time with timezone handling
            now = datetime.datetime.now()
            
            # Fetch rates with enhanced error handling
            rates = mt5.copy_rates_from(symbol, tf_constant, now, n_bars)
            
            if rates is None or len(rates) == 0:
                error_info = mt5.last_error()
                self.failed_fetches += 1
                
                self.live_logger.warning(
                    format_operator_message(
                        "⚠️", "NO_DATA_RECEIVED",
                        instrument=symbol,
                        details=f"{timeframe}, requested: {n_bars} bars",
                        result=f"MT5 error: {error_info}",
                        context="data_fetch"
                    )
                )
                
                # Update InfoBus with data gap alert
                if info_bus is not None:
                    InfoBusUpdater.add_alert(
                        info_bus,
                        f"No data for {symbol} {timeframe}",
                        severity="warning",
                        module="LiveDataConnector"
                    )
                
                self.data_gaps_detected += 1
                
                # Return empty DataFrame with correct structure
                return self._create_empty_dataframe()
            
            # Convert to enhanced DataFrame
            df = self._convert_rates_to_dataframe(rates, symbol, timeframe)
            
            # Calculate fetch performance
            fetch_time = time.time() - fetch_start
            self.fetch_performance[symbol].append(fetch_time)
            self.avg_fetch_time = np.mean([np.mean(list(perf)) for perf in self.fetch_performance.values()])
            
            # Update InfoBus with successful fetch
            if info_bus is not None:
                self._update_infobus_data_quality(info_bus, symbol, timeframe, df, fetch_time)
            
            self.live_logger.info(
                format_operator_message(
                    "📊", "DATA_FETCHED",
                    instrument=symbol,
                    details=f"{timeframe}: {len(df)} bars",
                    result=f"Latest: {df['close'].iloc[-1]:.5f}, Time: {fetch_time:.3f}s",
                    context="data_fetch"
                )
            )
            
            return df
            
        except Exception as e:
            self.failed_fetches += 1
            fetch_time = time.time() - fetch_start
            
            self.live_logger.error(
                format_operator_message(
                    "💥", "DATA_FETCH_FAILED",
                    instrument=symbol,
                    details=f"{timeframe}, {n_bars} bars",
                    result=str(e),
                    context="data_fetch_error"
                )
            )
            
            # Update InfoBus with error
            if info_bus is not None:
                InfoBusUpdater.add_alert(
                    info_bus,
                    f"Data fetch failed for {symbol}: {e}",
                    severity="error",
                    module="LiveDataConnector"
                )
            
            # Record fetch failure audit
            self.audit_tracker.record_event(
                "data_fetch_failed",
                "LiveDataConnector",
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "n_bars": n_bars,
                    "error": str(e),
                    "fetch_time": fetch_time
                },
                severity="error"
            )
            
            # Return empty DataFrame
            return self._create_empty_dataframe()

    def get_historical_data_with_infobus(
        self, 
        n_bars: int = 1000,
        info_bus: Optional[InfoBus] = None
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Enhanced historical data collection with comprehensive InfoBus integration"""
        
        if not self.connected:
            raise RuntimeError("Not connected to MT5")
        
        self.live_logger.info(
            format_operator_message(
                "📈", "FETCHING_HISTORICAL_DATA",
                details=f"{len(self.instruments)} instruments, {len(self.timeframes)} timeframes",
                result=f"{n_bars} bars each",
                context="data_collection"
            )
        )
        
        data: Dict[str, Dict[str, pd.DataFrame]] = {}
        fetch_summary = {
            "successful_fetches": 0,
            "failed_fetches": 0,
            "empty_datasets": 0,
            "total_bars": 0
        }
        
        collection_start = time.time()
        
        for symbol in self.instruments:
            # Convert MT5 symbol to internal format (EURUSD -> EUR/USD)
            symbol_internal = self._convert_symbol_format(symbol)
            data[symbol_internal] = {}
            
            for timeframe in self.timeframes:
                try:
                    df = self.fetch_historical_with_infobus(symbol, timeframe, n_bars, info_bus)
                    
                    if len(df) == 0:
                        fetch_summary["empty_datasets"] += 1
                        self.live_logger.warning(
                            format_operator_message(
                                "⚠️", "EMPTY_DATASET",
                                instrument=symbol_internal,
                                details=timeframe,
                                context="data_validation"
                            )
                        )
                    else:
                        fetch_summary["successful_fetches"] += 1
                        fetch_summary["total_bars"] += len(df)
                    
                    data[symbol_internal][timeframe] = df
                    
                except Exception as e:
                    fetch_summary["failed_fetches"] += 1
                    
                    self.live_logger.error(
                        format_operator_message(
                            "💥", "FETCH_ERROR",
                            instrument=symbol_internal,
                            details=f"{timeframe}: {e}",
                            context="data_collection_error"
                        )
                    )
                    
                    # Create empty DataFrame for failed fetch
                    data[symbol_internal][timeframe] = self._create_empty_dataframe()
        
        collection_time = time.time() - collection_start
        
        # Log collection summary
        self.live_logger.info(
            format_operator_message(
                "✅", "DATA_COLLECTION_COMPLETE",
                details=f"Time: {collection_time:.2f}s",
                result=f"Success: {fetch_summary['successful_fetches']}, Failed: {fetch_summary['failed_fetches']}, Bars: {fetch_summary['total_bars']}",
                context="data_collection"
            )
        )
        
        # Update InfoBus with collection summary
        if info_bus is not None:
            InfoBusUpdater.add_module_data(
                info_bus,
                "LiveDataConnector",
                {
                    "collection_summary": fetch_summary,
                    "collection_time": collection_time,
                    "data_quality_score": self._calculate_data_quality_score(fetch_summary),
                    "instruments_processed": len(self.instruments),
                    "timeframes_processed": len(self.timeframes)
                }
            )
        
        # Record collection audit
        self.audit_tracker.record_event(
            "data_collection_completed",
            "LiveDataConnector",
            {
                **fetch_summary,
                "collection_time": collection_time,
                "instruments": len(self.instruments),
                "timeframes": len(self.timeframes)
            },
            severity="info"
        )
        
        return data

    def sync_positions_with_env_infobus(self, env, info_bus: Optional[InfoBus] = None) -> None:
        """Enhanced position synchronization with InfoBus integration"""
        
        if not hasattr(env, 'position_manager'):
            self.live_logger.warning("Environment has no position_manager")
            return
        
        try:
            sync_start = time.time()
            
            # Get current MT5 positions
            mt5_positions = self.get_positions_enhanced()
            
            if not mt5_positions:
                self.live_logger.info("No MT5 positions to sync")
                return
            
            synced_positions = 0
            sync_errors = []
            
            # Convert and sync positions
            for pos in mt5_positions:
                try:
                    # Convert symbol format
                    symbol = pos['symbol']
                    symbol_internal = self._convert_symbol_format(symbol)
                    
                    # Only sync if instrument is in our trading list
                    if symbol_internal in env.instruments:
                        env.position_manager.open_positions[symbol_internal] = {
                            "ticket": pos['ticket'],
                            "side": 1 if pos['type'] == "BUY" else -1,
                            "lots": pos['volume'],
                            "price_open": pos['price_open'],
                            "current_price": pos['price_current'],
                            "unrealized_pnl": pos['profit'],
                            "entry_step": env.market_state.current_step,
                            "instrument": symbol_internal,
                            "mt5_sync_time": datetime.datetime.now().isoformat()
                        }
                        synced_positions += 1
                    
                except Exception as e:
                    sync_errors.append(f"{pos.get('symbol', 'unknown')}: {e}")
            
            sync_time = time.time() - sync_start
            
            self.live_logger.info(
                format_operator_message(
                    "🔄", "POSITIONS_SYNCED",
                    details=f"{synced_positions}/{len(mt5_positions)} positions",
                    result=f"Time: {sync_time:.3f}s",
                    context="position_sync"
                )
            )
            
            # Update InfoBus with sync results
            if info_bus is not None:
                InfoBusUpdater.add_module_data(
                    info_bus,
                    "PositionSync",
                    {
                        "mt5_positions": len(mt5_positions),
                        "synced_positions": synced_positions,
                        "sync_errors": sync_errors,
                        "sync_time": sync_time,
                        "last_sync": datetime.datetime.now().isoformat()
                    }
                )
            
            # Record sync audit
            self.audit_tracker.record_event(
                "position_sync",
                "LiveDataConnector",
                {
                    "mt5_positions": len(mt5_positions),
                    "synced_positions": synced_positions,
                    "sync_errors": len(sync_errors),
                    "sync_time": sync_time
                },
                severity="warning" if sync_errors else "info"
            )
            
            if sync_errors:
                self.live_logger.warning(f"Position sync errors: {sync_errors}")
                
        except Exception as e:
            self.live_logger.error(
                format_operator_message(
                    "💥", "POSITION_SYNC_FAILED",
                    details=str(e),
                    context="position_sync_error"
                )
            )

    def get_positions_enhanced(self) -> List[Dict[str, Any]]:
        """Enhanced position retrieval with error handling"""
        
        if not self.connected:
            raise RuntimeError("Not connected to MT5")
        
        try:
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
                    "time": pos.time,
                })
            
            return position_list
            
        except Exception as e:
            self.live_logger.error(f"Failed to get positions: {e}")
            return []

    # ══════════════════════════════════════════════════════════════
    # Private Helper Methods
    # ══════════════════════════════════════════════════════════════

    def _verify_comprehensive_connection(self) -> Dict[str, Any]:
        """Comprehensive connection verification"""
        
        verification = {
            "account_info": False,
            "symbols_accessible": [],
            "data_streams": [],
            "connection_quality": 0.0
        }
        
        try:
            # Test account info
            info = mt5.account_info()
            verification["account_info"] = info is not None
            
            # Test symbol accessibility
            for symbol in self.instruments:
                tick = mt5.symbol_info_tick(symbol)
                if tick is not None:
                    verification["symbols_accessible"].append(symbol)
            
            # Test data streams
            for symbol in verification["symbols_accessible"][:2]:  # Test first 2
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1)
                if rates is not None and len(rates) > 0:
                    verification["data_streams"].append(symbol)
            
            # Calculate quality score
            symbol_score = len(verification["symbols_accessible"]) / len(self.instruments) * 50
            data_score = len(verification["data_streams"]) / min(len(self.instruments), 2) * 30
            account_score = 20 if verification["account_info"] else 0
            
            verification["connection_quality"] = symbol_score + data_score + account_score
            
        except Exception as e:
            verification["error"] = str(e)
        
        return verification

    def _update_infobus_connection_status(self, info_bus: InfoBus, connected: bool, details: Dict[str, Any]):
        """Update InfoBus with connection status"""
        
        InfoBusUpdater.add_module_data(
            info_bus,
            "LiveDataConnector",
            {
                "connected": connected,
                "connection_quality": self.connection_quality_score,
                "connection_failures": self.connection_failures,
                "last_update": datetime.datetime.now().isoformat(),
                **details
            }
        )
        
        if not connected:
            InfoBusUpdater.add_alert(
                info_bus,
                "MT5 connection lost or failed",
                severity="error",
                module="LiveDataConnector"
            )

    def _update_infobus_data_quality(self, info_bus: InfoBus, symbol: str, timeframe: str, 
                                   df: pd.DataFrame, fetch_time: float):
        """Update InfoBus with data quality metrics"""
        
        quality_metrics = {
            "symbol": symbol,
            "timeframe": timeframe,
            "bars_received": len(df),
            "fetch_time": fetch_time,
            "latest_price": float(df['close'].iloc[-1]) if len(df) > 0 else 0,
            "data_freshness": (datetime.datetime.now() - pd.to_datetime(df.index[-1])).total_seconds() if len(df) > 0 else float('inf'),
            "has_gaps": self._detect_data_gaps(df),
            "quality_score": self._calculate_single_dataset_quality(df, fetch_time)
        }
        
        self.data_quality_history.append(quality_metrics)
        
        # Update InfoBus
        current_module_data = info_bus.get('module_data', {}).get('LiveDataConnector', {})
        if 'data_quality' not in current_module_data:
            current_module_data['data_quality'] = {}
        
        current_module_data['data_quality'][f"{symbol}_{timeframe}"] = quality_metrics
        
        InfoBusUpdater.add_module_data(info_bus, "LiveDataConnector", current_module_data)

    def _perform_connection_health_check(self, info_bus: Optional[InfoBus] = None):
        """Perform comprehensive connection health check"""
        
        if not self.connected:
            return
        
        health_issues = []
        
        try:
            # Test account info
            info = mt5.account_info()
            if info is None:
                health_issues.append("Cannot retrieve account info")
            
            # Test symbol access
            failed_symbols = 0
            for symbol in self.instruments:
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    failed_symbols += 1
            
            if failed_symbols > 0:
                health_issues.append(f"{failed_symbols}/{len(self.instruments)} symbols inaccessible")
            
            # Update connection quality
            quality_penalty = len(health_issues) * 20
            self.connection_quality_score = max(0, 100 - quality_penalty)
            
            if health_issues:
                self.live_logger.warning(
                    format_operator_message(
                        "⚠️", "CONNECTION_HEALTH_ISSUES",
                        details="; ".join(health_issues),
                        result=f"Quality: {self.connection_quality_score:.1f}%",
                        context="health_check"
                    )
                )
                
                if info_bus is not None:
                    for issue in health_issues:
                        InfoBusUpdater.add_alert(info_bus, issue, severity="warning", module="LiveDataConnector")
            
        except Exception as e:
            self.connection_quality_score = 0
            self.live_logger.error(f"Connection health check failed: {e}")

    def _perform_data_quality_check(self, info_bus: Optional[InfoBus] = None):
        """Perform data quality analysis"""
        
        if len(self.data_quality_history) < 5:
            return
        
        recent_quality = list(self.data_quality_history)[-10:]
        
        # Calculate quality metrics
        avg_quality = np.mean([q['quality_score'] for q in recent_quality])
        avg_fetch_time = np.mean([q['fetch_time'] for q in recent_quality])
        gap_rate = np.mean([q['has_gaps'] for q in recent_quality])
        
        quality_summary = {
            "avg_quality_score": avg_quality,
            "avg_fetch_time": avg_fetch_time,
            "data_gap_rate": gap_rate,
            "total_fetches": self.total_fetches,
            "failed_fetches": self.failed_fetches,
            "success_rate": (self.total_fetches - self.failed_fetches) / max(self.total_fetches, 1) * 100
        }
        
        # Check for quality issues
        quality_issues = []
        if avg_quality < 70:
            quality_issues.append(f"Low data quality: {avg_quality:.1f}%")
        if avg_fetch_time > 2.0:
            quality_issues.append(f"Slow data fetches: {avg_fetch_time:.2f}s")
        if gap_rate > 0.2:
            quality_issues.append(f"High gap rate: {gap_rate:.1%}")
        
        if quality_issues:
            self.live_logger.warning(
                format_operator_message(
                    "⚠️", "DATA_QUALITY_ISSUES",
                    details="; ".join(quality_issues),
                    context="quality_monitoring"
                )
            )
        
        # Update InfoBus
        if info_bus is not None:
            InfoBusUpdater.add_module_data(
                info_bus,
                "DataQualityMonitor",
                quality_summary
            )

    def _convert_rates_to_dataframe(self, rates, symbol: str, timeframe: str) -> pd.DataFrame:
        """Convert MT5 rates to enhanced DataFrame"""
        
        df = pd.DataFrame(rates)
        
        # Convert time to datetime index
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        
        # Handle volume column variations
        if "tick_volume" in df.columns:
            df = df[["open", "high", "low", "close", "tick_volume"]]
            df.rename(columns={"tick_volume": "volume"}, inplace=True)
        else:
            df = df[["open", "high", "low", "close", "volume"]]
        
        # Enhanced volatility calculation
        df = self._add_enhanced_volatility(df)
        
        # Data validation and cleanup
        df = self._validate_and_clean_data(df, symbol, timeframe)
        
        return df

    def _add_enhanced_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add enhanced volatility calculations"""
        
        # Method 1: High-Low range volatility
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
        
        # Combine methods with fallbacks
        df["volatility"] = rolling_vol.fillna(hl_vol).fillna(atr_vol).fillna(0.01)
        
        # Ensure non-negative volatility
        df["volatility"] = df["volatility"].abs()
        
        return df

    def _validate_and_clean_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
        """Validate and clean market data"""
        
        # Check for missing values
        if df.isnull().any().any():
            self.live_logger.warning(
                format_operator_message(
                    "⚠️", "DATA_GAPS_DETECTED",
                    instrument=symbol,
                    details=f"{timeframe}: NaN values found",
                    context="data_validation"
                )
            )
            df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Validate OHLC consistency
        invalid_rows = (df['high'] < df[['open', 'close']].max(axis=1)) | (df['low'] > df[['open', 'close']].min(axis=1))
        if invalid_rows.any():
            self.live_logger.warning(
                format_operator_message(
                    "⚠️", "OHLC_INCONSISTENCY",
                    instrument=symbol,
                    details=f"{timeframe}: {invalid_rows.sum()} invalid rows",
                    context="data_validation"
                )
            )
            # Fix inconsistencies
            df.loc[invalid_rows, 'high'] = df.loc[invalid_rows, ['open', 'high', 'close']].max(axis=1)
            df.loc[invalid_rows, 'low'] = df.loc[invalid_rows, ['open', 'low', 'close']].min(axis=1)
        
        return df

    def _create_empty_dataframe(self) -> pd.DataFrame:
        """Create empty DataFrame with correct structure"""
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "volatility"])

    def _convert_symbol_format(self, symbol: str) -> str:
        """Convert MT5 symbol format to internal format"""
        if len(symbol) == 6 and "/" not in symbol:
            return symbol[:3] + "/" + symbol[3:]
        return symbol

    def _detect_data_gaps(self, df: pd.DataFrame) -> bool:
        """Detect gaps in time series data"""
        if len(df) < 2:
            return False
        
        # Simple gap detection based on expected frequency
        time_diffs = df.index.to_series().diff()[1:]
        median_diff = time_diffs.median()
        
        # Look for gaps larger than 3x median difference
        large_gaps = time_diffs > (median_diff * 3)
        return large_gaps.any()

    def _calculate_single_dataset_quality(self, df: pd.DataFrame, fetch_time: float) -> float:
        """Calculate quality score for a single dataset"""
        score = 100.0
        
        # Penalize for empty data
        if len(df) == 0:
            return 0.0
        
        # Penalize for slow fetches
        if fetch_time > 1.0:
            score -= min(20, (fetch_time - 1.0) * 10)
        
        # Penalize for data gaps
        if self._detect_data_gaps(df):
            score -= 15
        
        # Penalize for missing data
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
        score -= missing_pct * 2
        
        return max(0.0, score)

    def _calculate_data_quality_score(self, fetch_summary: Dict[str, int]) -> float:
        """Calculate overall data quality score"""
        
        total_fetches = fetch_summary["successful_fetches"] + fetch_summary["failed_fetches"]
        if total_fetches == 0:
            return 0.0
        
        success_rate = fetch_summary["successful_fetches"] / total_fetches
        empty_rate = fetch_summary["empty_datasets"] / max(fetch_summary["successful_fetches"], 1)
        
        quality_score = success_rate * 80 + (1 - empty_rate) * 20
        
        return min(100.0, quality_score)


class InfoBusLiveTradingCallback(BaseCallback):
    """
    Enhanced live trading callback with comprehensive InfoBus integration.
    Provides seamless integration between training and live trading systems.
    """
    
    def __init__(self, connector: InfoBusLiveDataConnector, verbose: int = 0):
        super().__init__(verbose)
        self.connector = connector
        
        # InfoBus-integrated logging
        self.logger = RotatingLogger(
            name="LiveTradingCallback",
            log_path=f"logs/live/callback_{datetime.datetime.now().strftime('%Y%m%d')}.log",
            max_lines=2000,
            operator_mode=True
        )
        
        self.logger.info(
            format_operator_message(
                "🚀", "LIVE_TRADING_CALLBACK_INITIALIZED",
                details="InfoBus integration enabled",
                context="callback_startup"
            )
        )

    def _on_training_start(self) -> None:
        """Enhanced training start with InfoBus connection"""
        
        try:
            # Create initial InfoBus for connection
            info_bus = self._create_info_bus()
            
            # Establish MT5 connection with InfoBus integration
            self.connector.connect(info_bus)
            
            self.logger.info(
                format_operator_message(
                    "✅", "LIVE_TRAINING_STARTED",
                    details="MT5 connection established",
                    context="training_start"
                )
            )
            
        except Exception as e:
            self.logger.error(
                format_operator_message(
                    "❌", "LIVE_TRAINING_START_FAILED",
                    details=str(e),
                    context="training_start_error"
                )
            )
            raise

    def _on_step(self) -> bool:
        """Enhanced step with InfoBus position synchronization"""
        
        try:
            # Get environment reference
            env = self.training_env.envs[0]
            
            # Create InfoBus for this step
            info_bus = self._create_info_bus_from_env(env)
            
            # Sync positions with InfoBus integration
            self.connector.sync_positions_with_env_infobus(env, info_bus)
            
            # Update connector with InfoBus
            self.connector._step_impl(info_bus)
            
            return True
            
        except Exception as e:
            self.logger.error(
                format_operator_message(
                    "💥", "LIVE_STEP_ERROR",
                    details=str(e),
                    context="step_error"
                )
            )
            return True  # Continue training despite errors

    def _on_training_end(self) -> None:
        """Enhanced training end with InfoBus cleanup"""
        
        try:
            # Create final InfoBus
            info_bus = self._create_info_bus()
            
            # Clean disconnect with InfoBus
            self.connector.disconnect(info_bus)
            
            self.logger.info(
                format_operator_message(
                    "🏁", "LIVE_TRAINING_ENDED",
                    details="Clean disconnection completed",
                    context="training_end"
                )
            )
            
        except Exception as e:
            self.logger.error(
                format_operator_message(
                    "❌", "LIVE_TRAINING_END_ERROR",
                    details=str(e),
                    context="training_end_error"
                )
            )

    def _create_info_bus(self) -> InfoBus:
        """Create basic InfoBus for connector operations"""
        return {
            'timestamp': datetime.datetime.now().isoformat(),
            'step_idx': getattr(self, 'num_timesteps', 0),
            'module_data': {},
            'alerts': [],
            'training_context': {
                'callback': 'LiveTradingCallback',
                'connector_type': 'InfoBusLiveDataConnector'
            }
        }

    def _create_info_bus_from_env(self, env) -> InfoBus:
        """Create InfoBus from environment state"""
        
        try:
            # Use environment's InfoBus if available
            if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'info_bus'):
                base_info_bus = env.unwrapped.info_bus or {}
            else:
                base_info_bus = {}
            
            # Enhance with callback context
            enhanced_info_bus = base_info_bus.copy()
            enhanced_info_bus.update({
                'live_trading_context': {
                    'callback_active': True,
                    'connector_connected': self.connector.connected,
                    'connection_quality': self.connector.connection_quality_score
                }
            })
            
            return enhanced_info_bus
            
        except Exception as e:
            self.logger.warning(f"Failed to create InfoBus from env: {e}")
            return self._create_info_bus()

    def connect(self):
        """Legacy compatibility method"""
        info_bus = self._create_info_bus()
        self.connector.connect(info_bus)

    def disconnect(self):
        """Legacy compatibility method"""
        info_bus = self._create_info_bus()
        self.connector.disconnect(info_bus)