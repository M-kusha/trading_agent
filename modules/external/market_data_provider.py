# ─────────────────────────────────────────────────────────────
# File: modules/external/market_data_provider.py
# UNIFIED Market Data Provider (Live MT5 + Offline CSV)
#
# • Dual-mode: Live trading from MT5, training from CSV files
# • Pointer-driven, timestamp-aligned multi-timeframe windows (fast)
# • Contract-aware alias publication (no ownership drift)
# • Comprehensive, human-friendly logging to logs/external/
# • Optional NDJSON snapshot stream for deep forensics
# • Single-writer, data-only outputs
# • Thread-safe state mutations
# • Pylance-clean; typed config; low-GC hot path
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import glob
import json
import math
import time
import datetime
import threading
from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set, cast

import numpy as np
import pandas as pd

# Contracts & infra
from modules.contracts import module_args, CONTRACTS  # contract-aware aliasing
from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusStateMixin
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.info_bus import SmartInfoBus

# MT5 integration (optional, only required for live mode)
try:
    import MetaTrader5 as _MT5
    MT5_AVAILABLE = True
    mt5: Any = cast(Any, _MT5)
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None


# ─────────────────────────────────────────────────────────────
# Pretty logger + optional NDJSON
# ─────────────────────────────────────────────────────────────

class PrettyLogger:
    """
    Formats log lines for humans and (optionally) writes compact NDJSON.
    We keep this tiny and robust—if logging fails, it never breaks the hot path.
    """
    def __init__(
        self,
        name: str,
        pretty_sink: RotatingLogger,
        ndjson_path: Optional[Path] = None,
        ndjson_every_n: int = 0,
    ) -> None:
        self.name = name
        self.pretty = pretty_sink
        self.ndjson_path = ndjson_path
        self.ndjson_every_n = max(0, int(ndjson_every_n))
        if self.ndjson_path is not None:
            self.ndjson_path.parent.mkdir(parents=True, exist_ok=True)
            self.ndjson_path.touch(exist_ok=True)

    @staticmethod
    def _hms_with_ms(ts: float, tz: datetime.tzinfo | None = None) -> str:
        dt = datetime.datetime.fromtimestamp(ts, tz=tz)
        return dt.strftime("%H:%M:%S.") + f"{int(dt.microsecond/1000):03d}"

    def _fmt(self, level: str, pid: int, tag: str, msg: str, ts: Optional[float] = None) -> str:
        """
        Example:
        [LOG] [DEBUG  ] 14:23:03.985 [p#15] [PROCESS_START       ] Starting... at 12:23:03
        """
        now = time.time() if ts is None else ts
        local = self._hms_with_ms(now)
        utc = self._hms_with_ms(now, tz=datetime.timezone.utc)
        tag_padded = f"{tag:22s}"
        lvl_padded = f"{level:<7s}"
        return f"[LOG] [{lvl_padded}] {local} [p#{pid}] [{tag_padded}] {msg} at {utc}"

    def trace(self, pid: int, tag: str, msg: str) -> None:
        self.pretty.debug(self._fmt("TRACE", pid, tag, msg))

    def debug(self, pid: int, tag: str, msg: str) -> None:
        self.pretty.debug(self._fmt("DEBUG", pid, tag, msg))

    def info(self, pid: int, tag: str, msg: str) -> None:
        self.pretty.info(self._fmt("INFO", pid, tag, msg))

    def warn(self, pid: int, tag: str, msg: str) -> None:
        self.pretty.warning(self._fmt("WARN", pid, tag, msg))

    def error(self, pid: int, tag: str, msg: str) -> None:
        self.pretty.error(self._fmt("ERROR", pid, tag, msg))

    def ndjson(self, pid: int, kind: str, obj: Dict[str, Any]) -> None:
        if self.ndjson_path is None:
            return
        try:
            record = {
                "ts": datetime.datetime.utcnow().isoformat(),
                "pid": pid,
                "kind": kind,
                "data": obj,
            }
            with self.ndjson_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            # Logging must never break the provider.
            self.pretty.debug(f"[DBG] NDJSON write failed: {e}")


# ─────────────────────────────────────────────────────────────
# RewardDebugManager-style human logger (integrated for provider)
# ─────────────────────────────────────────────────────────────

class MarketDataDebugManager:
    """
    Human-friendly, bracketed-line logger for MarketDataProvider.
    Matches RewardDebugManager style with clear English reporting.

    Produces lines like:
      [LOG] [DEBUG  ] 14:40:43.981 [p#8] [PROCESS_START         ] ════════════════════════════════════════════════════════════
      [LOG] [DEBUG  ] 14:40:43.981 [p#8] [PROCESS_START         ] Starting data update p#8
      [LOG] [DEBUG  ] 14:40:43.981 [p#8] [INPUT_INSPECTION      ] Input summary: {keys: 5, step_idx: 8, has_actions: True}
    """

    def __init__(self, enabled: bool, level: str, logger: Optional[Any] = None) -> None:
        self.enabled = enabled
        self._levels = {"TRACE": 0, "DEBUG": 1, "INFO": 2, "WARNING": 3, "ERROR": 4}
        self.set_level(level)
        self.logger = logger
        self._lock = None  # no heavy threading needed in provider debug path
        self._proc_idx: Optional[int] = None
        self._last_log: Dict[Tuple[str, str], float] = {}
        self._op_timings: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Statistics tracking (like RewardDebugManager)
        self.total_processes: int = 0
        self.successful_processes: int = 0
        self.failed_processes: int = 0
        self.data_quality_issues: int = 0

    def set_level(self, level: str) -> None:
        lvl = self._levels.get(level.upper(), 2)
        self.level_name = level.upper()
        self.current_level = lvl

    def _should(self, level: str) -> bool:
        if not self.enabled:
            return False
        return self._levels.get(level, 5) >= self.current_level

    @staticmethod
    def _ts() -> str:
        return datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]

    def _emit(self, level: str, context: str, message: str, rate_limit_sec: Optional[float] = None) -> None:
        if not self._should(level):
            return
        if rate_limit_sec is not None:
            key = (level, context)
            now = time.time()
            last = self._last_log.get(key, 0.0)
            if now - last < rate_limit_sec:
                return
            self._last_log[key] = now

        pid = f"p#{self._proc_idx}" if self._proc_idx is not None else "p#0"
        lvl = f"{level:<7}"
        ctx = f"{context:<22}"
        line = f"[LOG] [{lvl}] {self._ts()} [{pid}] [{ctx}] {message}"
        try:
            # map TRACE->debug if logger lacks trace
            if level == "ERROR":
                self.logger.error(line) if self.logger else print(line)
            elif level == "WARNING":
                self.logger.warning(line) if self.logger else print(line)
            elif level in ("DEBUG", "TRACE"):
                if hasattr(self.logger, "debug") and self.logger:
                    self.logger.debug(line)
                elif self.logger:
                    self.logger.info(line)
                else:
                    print(line)
            else:
                self.logger.info(line) if self.logger else print(line)
        except Exception:
            print(line)

    # Convenience logs matching your samples
    def log_csv_loading_start(self) -> None:
        self._emit("DEBUG", "INIT", "Loading CSVs…")

    def log_csv_ok(self, symbol: str, timeframe: str, bars: int, filename: str) -> None:
        csv_info = {
            "symbol": symbol,
            "timeframe": timeframe,
            "bars": int(bars),
            "file": filename
        }
        self._emit("INFO", "CSV_LOADED", f"✓ Loaded {csv_info}")

    def log_csv_warn(self, symbol: str, timeframe: str, msg: str) -> None:
        self.data_quality_issues += 1
        self._emit("WARNING", "CSV_WARNING", f"⚠ {symbol}/{timeframe}: {msg}")

    def log_init_ok(self, symbols_count: int, total_bars: int) -> None:
        init_summary = {
            "symbols": int(symbols_count),
            "total_bars": f"{int(total_bars):,}",
            "quality_issues": self.data_quality_issues
        }
        self._emit("INFO", "INIT_SUMMARY", f"Initialization complete: {init_summary}")
        self._emit("INFO", "INIT_OK", "✓ MarketDataProvider ready")

    def log_process_start(self, idx: int) -> None:
        self._proc_idx = int(idx)
        self.total_processes += 1
        self._emit("DEBUG", "PROCESS_START", "═" * 60)
        self._emit("DEBUG", "PROCESS_START", f"Starting data update p#{idx}")

    def log_input_inspection(self, inputs: Dict[str, Any]) -> None:
        keys = list(inputs.keys())
        step_idx = inputs.get("step_idx", "not_provided")
        has_actions = 1 if "actions" in inputs else 0
        has_reward_inputs = 1 if "reward_inputs" in inputs else 0
        input_summary = {
            "keys_provided": len(keys),
            "step_idx": step_idx,
            "has_actions": bool(has_actions),
            "has_reward_inputs": bool(has_reward_inputs)
        }
        self._emit("DEBUG", "INPUT_INSPECTION", f"Input summary: {input_summary}")

    def log_snapshot_summary(self, snapshot: Dict[str, Any]) -> None:
        md = snapshot.get("market_data", {}) or {}
        try:
            prices = {s: round(md[s]["close"], 5) for s in md.keys()}
        except Exception:
            prices = {}
        
        # Build structured summary
        summary = {
            "tick": snapshot.get('step_idx', 0),
            "symbols": list(md.keys()),
            "prices": prices,
            "volatility": snapshot.get('volatility_level', 'unknown'),
            "session": snapshot.get('trading_session', 'unknown')
        }
        self._emit("INFO", "DATA_SUMMARY", f"Market snapshot: {summary}", rate_limit_sec=0.5)

    def log_alias_publish(self, keys: List[str]) -> None:
        if not keys:
            return
        self._emit("TRACE", "BUS_PUBLISH", f"Published {len(keys)} alias keys to InfoBus", rate_limit_sec=1.0)

    def log_bus_publish_fail(self, key: str, err: Exception) -> None:
        self._emit("WARNING", "BUS_PUBLISH_FAIL", f"Failed to publish '{key}': {type(err).__name__}: {str(err)[:140]}")

    def log_health(self, *, mode: str, quality: float, win_rate: float, circuit_breaker: str = "CLOSED") -> None:
        health_status = {
            "status": "HEALTHY",
            "circuit_breaker": circuit_breaker,
            "mode": mode,
            "quality": round(quality, 3),
            "win_rate": round(win_rate, 3)
        }
        self._emit("INFO", "HEALTH_CHECK", f"Health: {health_status}")

    def log_monitoring(self) -> None:
        self._emit("TRACE", "MONITORING", "Monitoring cycle executed")

    def log_process_end(self, success: bool = True) -> None:
        if success:
            self.successful_processes += 1
            self._emit("DEBUG", "PROCESS_END", f"✓ Data update p#{self._proc_idx} completed successfully")
        else:
            self.failed_processes += 1
            self._emit("DEBUG", "PROCESS_END", f"✗ Data update p#{self._proc_idx} failed")
        self._emit("DEBUG", "PROCESS_END", "═" * 60)
    
    def log_statistics(self) -> None:
        """Log cumulative statistics"""
        stats = {
            "total": self.total_processes,
            "success": self.successful_processes,
            "failed": self.failed_processes,
            "quality_issues": self.data_quality_issues,
            "success_rate": f"{(self.successful_processes / max(1, self.total_processes) * 100):.1f}%"
        }
        self._emit("INFO", "STATISTICS", f"Provider stats: {stats}")

    def log_error(self, context: str, e: Exception) -> None:
        error_info = {
            "type": type(e).__name__,
            "message": str(e)[:200],
            "context": context
        }
        self._emit("ERROR", context, f"✗ Error: {error_info}")

    def log_data_advancement(self, symbol: str, timeframe: str, new_idx: int, timestamp: Any) -> None:
        """Log data pointer advancement"""
        advance_info = {
            "symbol": symbol,
            "timeframe": timeframe,
            "pointer": new_idx,
            "timestamp": str(timestamp)[:19] if timestamp else "N/A"
        }
        self._emit("TRACE", "DATA_ADVANCE", f"Advanced: {advance_info}")

    def log_data_quality_check(self, symbol: str, bar: Dict[str, Any], issues: List[str]) -> None:
        """Log data quality validation"""
        if issues:
            self.data_quality_issues += len(issues)
            quality_report = {
                "symbol": symbol,
                "issues": issues,
                "bar_timestamp": str(bar.get("timestamp", "N/A"))[:19]
            }
            self._emit("WARNING", "DATA_QUALITY", f"⚠ Quality issues: {quality_report}")

    # Generic
    def log_generic(self, level: str, context: str, message: str) -> None:
        self._emit(level.upper(), context, message)


# ─────────────────────────────────────────────────────────────
# NumPy-backed per-timeframe store (top-level to satisfy Pylance)
# ─────────────────────────────────────────────────────────────

class _TFStore:
    """Lightweight, NumPy-backed time series store per timeframe."""
    __slots__ = ("ts", "open", "high", "low", "close", "volume", "bid", "ask", "n")

    def __init__(self, df: pd.DataFrame):
        self.ts = pd.to_datetime(df["timestamp"]).to_numpy(dtype="datetime64[ns]", copy=False)
        self.open = df["open"].to_numpy(dtype="float64", copy=False)
        self.high = df["high"].to_numpy(dtype="float64", copy=False)
        self.low  = df["low"].to_numpy(dtype="float64", copy=False)
        self.close= df["close"].to_numpy(dtype="float64", copy=False)
        self.volume = df["volume"].to_numpy(dtype="int64", copy=False)
        self.bid  = df["bid"].to_numpy(dtype="float64", copy=False) if "bid" in df.columns else None
        self.ask  = df["ask"].to_numpy(dtype="float64", copy=False) if "ask" in df.columns else None
        self.n = int(self.close.shape[0])


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

@dataclass
class MarketDataConfig:
    # Operating mode: 'training' (CSV) or 'live' (MT5)
    mode: str = "training"
    
    # CSV/offline settings
    data_directory: str = "data/processed"
    # Optional multi-source controls (align with system_config / module_registry defaults)
    data_sources: List[str] = field(default_factory=lambda: ["primary", "backup"])
    cache_duration: int = 60
    quality_threshold: float = 0.95
    
    # MT5/live settings  
    mt5_account: Optional[int] = None
    mt5_password: Optional[str] = None
    mt5_server: Optional[str] = None
    mt5_timeout: int = 60000
    timeout_ms: int = 5000
    mt5_reconnect_attempts: int = 3
    mt5_reconnect_delay: float = 5.0
    # Module-level breaker guard (module_system injects this)
    circuit_breaker_threshold: int = 3
    
    # Symbols and timeframes
    supported_symbols: List[str] = field(default_factory=lambda: ["XAU_USD", "EUR_USD"])
    supported_timeframes: List[str] = field(default_factory=lambda: ["M15", "H1", "H4", "D1"])
    primary_timeframe: str = "H4"        # drives time advancement
    
    # Data settings
    update_frequency: float = 1.0        # seconds between updates in live mode (0 = no throttling)
    buffer_size: int = 512               # rolling indicator buffer length
    live_bars_to_fetch: int = 200        # how many bars to fetch in live mode
    enable_technical_indicators: bool = True

    # Logging controls
    log_every_n: int = 250               # 1 = log every tick
    ndjson_every_n: int = 0              # 0 = off; 1 = every tick; N = every Nth tick

    # Window size caps (experts like MomentumExpert need 55+ bars)
    window_min: int = 100
    window_max: int = 200
    
    # Symbol format mapping (MT5 -> internal)
    # e.g., {"EURUSD": "EUR_USD", "XAUUSD": "XAU_USD"}
    symbol_mapping: Dict[str, str] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────
# Module
# ─────────────────────────────────────────────────────────────

@module(**module_args(
    "MarketDataProvider",
    description="Unified market data provider supporting both live MT5 and offline CSV modes.",
    error_handling=True,
    hot_reload=True,
    timeout_ms=5000,
    critical=True,  # run even in emergency mode
))
class MarketDataProvider(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    Unified Market Data Provider with dual-mode support.
    
    Modes:
      - 'training': Read data from CSV files, pointer-driven playback
      - 'live': Connect to MT5 and fetch real-time data
    
    Guarantees:
      - Advances on every `process()` call (with optional throttling in live mode).
      - Publishes ALL keys declared for MarketDataProvider in contracts.py.
      - Publishes symbol/TF alias keys only if present in contract provides.
      - Thread-safe state mutations.
      - Timestamp fields are ISO-8601 strings on the bus/snapshot.
      - Single-writer: only keys declared in contracts are published.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Typed config
        self.cfg = MarketDataConfig(**(config or {}))
        
        # Thread safety
        self._lock = threading.RLock()

        # Backing log files
        log_dir = Path("logs/external")
        log_dir.mkdir(parents=True, exist_ok=True)

        # Pretty sink (rotating file)
        self.logger = RotatingLogger("MarketDataProvider", log_path=str(log_dir / "market_data_provider.log"))

        # Optional NDJSON stream for snapshots/metrics
        self._pretty = PrettyLogger(
            name="MarketDataProvider",
            pretty_sink=self.logger,
            ndjson_path=(log_dir / "market_data_provider.ndjson") if self.cfg.ndjson_every_n > 0 else None,
            ndjson_every_n=self.cfg.ndjson_every_n,
        )

        # Human debug manager (RewardDebugManager flavor)
        self.debug = MarketDataDebugManager(enabled=True, level="TRACE", logger=self.logger)

        # Contract keys (for alias allowlist)
        mdp = CONTRACTS.get("MarketDataProvider")
        self._contract_provides: Set[str] = set(mdp.provides) if mdp else set()

        # Storage
        self.data_files: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.tfs: Dict[str, Dict[str, _TFStore]] = {}
        self.primary_tf: Dict[str, str] = {}             # per symbol chosen primary TF
        self.ptr_primary: Dict[str, int] = {}            # per symbol pointer on primary TF
        self.ptrs_by_tf: Dict[str, Dict[str, int]] = {}  # per symbol per TF pointer aligned to primary ts

        # Current bars & buffers
        self.current_bars: Dict[str, Dict[str, Any]] = {}
        self.price_buffers: Dict[str, Dict[str, deque]] = {}
        self.technical_indicators: Dict[str, Dict[str, float]] = {}

        # Session labels & time
        self.current_timestamp: Optional[datetime.datetime] = None
        self.trading_session: str = "london"
        self.session_type: str = "normal"

        # Health metrics
        self._last_update_ts: float = 0.0
        self._update_count: int = 0
        self._success: int = 0
        self._fail: int = 0
        self._ema_ms: float = 0.0
        
        # MT5 connection state (for live mode)
        self._mt5_connected: bool = False
        self._mt5_last_connect_attempt: float = 0.0
        self._mt5_connection_failures: int = 0
        
        # MT5 timeframe mapping
        self._tf_map: Dict[str, Any] = {}
        if MT5_AVAILABLE and mt5 is not None:
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

        # Bus
        self.smart_bus = SmartInfoBus()

        # Wire & initialize (BaseModule will call _initialize)
        super().__init__(config=asdict(self.cfg))

        # Operator-style boot line
        mode_label = "LIVE (MT5)" if self.cfg.mode == "live" else "TRAINING (CSV)"
        self.logger.info(format_operator_message(
            "[BOOT]", "MARKET_DATA_PROVIDER_INIT",
            details=f"Mode={mode_label}, Symbols={self.cfg.supported_symbols}, TF={self.cfg.supported_timeframes}",
            result="Provider ready",
            context="system_startup",
        ))

    # ─────────────────────────────────────────────────────────
    # Initialization (BaseModule hook)
    # ─────────────────────────────────────────────────────────
    def _initialize(self) -> None:
        try:
            self.debug.log_csv_loading_start()
            
            # Check InfoBus for execution_mode first (set by backend before orchestrator init)
            # This allows dynamic mode switching when starting live trading
            effective_mode = self.cfg.mode
            try:
                bus_mode = self.smart_bus.get("execution_mode", "MarketDataProvider", default=None)
                if bus_mode in ("live", "LIVE"):
                    effective_mode = "live"
                    self.cfg.mode = "live"  # Update config to match
                    self.debug.log_generic("INFO", "MODE_OVERRIDE", f"Mode overridden to LIVE from InfoBus (execution_mode={bus_mode})")
            except Exception:
                pass  # InfoBus not ready yet, use config default
            
            if effective_mode == "live":
                # Live mode: connect to MT5
                self._initialize_live_mode()
            else:
                # Training mode: load CSV files
                self._initialize_training_mode()
            
            self._initialize_technical_indicators()
            self._setup_initial_conditions()
            
            total_bars = sum(
                store.n 
                for sym_stores in self.tfs.values() 
                for store in sym_stores.values()
            )
            self.debug.log_init_ok(len(self.tfs), total_bars)
            
        except Exception as e:
            self.debug.log_error("INIT_FAIL", e)
            raise
    
    def _initialize_training_mode(self) -> None:
        """Initialize for offline/training mode from CSV files."""
        self._load_data_files()
        self._materialize_tfstores()
        self._init_pointers_and_buffers()
        self.debug.log_generic("INFO", "MODE", "Initialized in TRAINING mode (CSV)")
    
    def _initialize_live_mode(self) -> None:
        """Initialize for live mode with MT5 connection."""
        if not MT5_AVAILABLE:
            raise RuntimeError("MetaTrader5 package not installed. Install with: pip install MetaTrader5")
        
        # Build symbol mapping if not provided
        if not self.cfg.symbol_mapping:
            self.cfg.symbol_mapping = self._build_default_symbol_mapping()
        
        # Connect to MT5
        self._connect_mt5()
        
        # Initialize structures for live data
        self._init_live_structures()
        
        # Fetch initial data
        self._fetch_initial_live_data()
        
        self.debug.log_generic("INFO", "MODE", "Initialized in LIVE mode (MT5)")
    
    def _build_default_symbol_mapping(self) -> Dict[str, str]:
        """Build default MT5 symbol to internal symbol mapping."""
        mapping = {}
        for symbol in self.cfg.supported_symbols:
            # Convert internal format (EUR_USD) to MT5 format (EURUSD)
            mt5_symbol = symbol.replace("_", "").replace("/", "")
            mapping[mt5_symbol] = symbol
        return mapping
    
    def _get_mt5_symbol(self, internal_symbol: str) -> str:
        """Convert internal symbol to MT5 symbol format."""
        # Reverse lookup in mapping
        for mt5_sym, int_sym in self.cfg.symbol_mapping.items():
            if int_sym == internal_symbol:
                return mt5_sym
        # Default: remove separators
        return internal_symbol.replace("_", "").replace("/", "")
    
    def _get_internal_symbol(self, mt5_symbol: str) -> str:
        """Convert MT5 symbol to internal symbol format."""
        return self.cfg.symbol_mapping.get(mt5_symbol, mt5_symbol)
    
    def _connect_mt5(self) -> bool:
        """Connect to MT5 terminal with retry logic."""
        if self._mt5_connected:
            return True
        
        for attempt in range(1, self.cfg.mt5_reconnect_attempts + 1):
            self._mt5_last_connect_attempt = time.time()
            
            try:
                self.debug.log_generic("INFO", "MT5_CONNECT", 
                    f"Connecting to MT5 (attempt {attempt}/{self.cfg.mt5_reconnect_attempts})...")
                
                # Shutdown any existing connection
                try:
                    mt5.shutdown()
                except Exception:
                    pass
                
                # Initialize with credentials if provided
                init_kwargs: Dict[str, Any] = {}
                if self.cfg.mt5_account and self.cfg.mt5_password and self.cfg.mt5_server:
                    init_kwargs = {
                        "login": self.cfg.mt5_account,
                        "password": self.cfg.mt5_password,
                        "server": self.cfg.mt5_server,
                        "timeout": self.cfg.mt5_timeout,
                    }
                
                # FIX: Proper boolean check - the ternary was causing issues
                if init_kwargs:
                    init_result = mt5.initialize(**init_kwargs)
                else:
                    init_result = mt5.initialize()
                
                if not init_result:
                    error = mt5.last_error()
                    raise ConnectionError(f"MT5 initialization failed: {error}")
                
                # If credentials provided but not in init, do explicit login
                if self.cfg.mt5_account and not init_kwargs:
                    if not mt5.login(self.cfg.mt5_account, self.cfg.mt5_password, self.cfg.mt5_server):
                        error = mt5.last_error()
                        mt5.shutdown()
                        raise ConnectionError(f"MT5 login failed: {error}")
                
                # Verify connection
                account_info = mt5.account_info()
                if account_info is None:
                    raise ConnectionError("Cannot retrieve account info after connection")
                
                # Select symbols
                selected_symbols = []
                for symbol in self.cfg.supported_symbols:
                    mt5_symbol = self._get_mt5_symbol(symbol)
                    if mt5.symbol_select(mt5_symbol, True):
                        selected_symbols.append(symbol)
                    else:
                        self.debug.log_generic("WARNING", "SYMBOL_SELECT", 
                            f"Could not select symbol {mt5_symbol}")
                
                if not selected_symbols:
                    raise ConnectionError("No symbols could be selected in MT5")
                
                self._mt5_connected = True
                self._mt5_connection_failures = 0
                
                self.debug.log_generic("INFO", "MT5_CONNECTED", 
                    f"Connected to MT5. Account: {account_info.login}, "
                    f"Balance: {account_info.balance:.2f} {account_info.currency}")
                
                return True
                
            except Exception as e:
                self._mt5_connection_failures += 1
                self.debug.log_error("MT5_CONNECT_FAIL", e)
                
                if attempt < self.cfg.mt5_reconnect_attempts:
                    time.sleep(self.cfg.mt5_reconnect_delay)
        
        self._mt5_connected = False
        raise ConnectionError(f"Failed to connect to MT5 after {self.cfg.mt5_reconnect_attempts} attempts")
    
    def _disconnect_mt5(self) -> None:
        """Disconnect from MT5."""
        if MT5_AVAILABLE and self._mt5_connected:
            try:
                mt5.shutdown()
            except Exception:
                pass
            self._mt5_connected = False
            self.debug.log_generic("INFO", "MT5_DISCONNECT", "Disconnected from MT5")
    
    def _init_live_structures(self) -> None:
        """Initialize data structures for live mode."""
        for symbol in self.cfg.supported_symbols:
            self.tfs[symbol] = {}
            self.ptrs_by_tf[symbol] = {}
            self.primary_tf[symbol] = self.cfg.primary_timeframe
            self.ptr_primary[symbol] = 0
            
            self.price_buffers[symbol] = {
                "close": deque(maxlen=self.cfg.buffer_size),
                "high": deque(maxlen=self.cfg.buffer_size),
                "low": deque(maxlen=self.cfg.buffer_size),
                "volume": deque(maxlen=self.cfg.buffer_size),
            }
    
    def _fetch_initial_live_data(self) -> None:
        """Fetch initial historical data from MT5 for all symbols/timeframes."""
        for symbol in self.cfg.supported_symbols:
            for tf in self.cfg.supported_timeframes:
                try:
                    df = self._fetch_mt5_data(symbol, tf, self.cfg.live_bars_to_fetch)
                    if df is not None and not df.empty:
                        self.tfs[symbol][tf] = _TFStore(df)
                        self.ptrs_by_tf[symbol][tf] = self.tfs[symbol][tf].n - 1
                        
                        self.debug.log_csv_ok(symbol, tf, len(df), "MT5_LIVE")
                except Exception as e:
                    self.debug.log_error(f"FETCH_INITIAL_{symbol}_{tf}", e)
        
        # Set primary pointer to latest
        for symbol in self.cfg.supported_symbols:
            ptf = self.primary_tf.get(symbol, self.cfg.primary_timeframe)
            if symbol in self.tfs and ptf in self.tfs[symbol]:
                self.ptr_primary[symbol] = self.tfs[symbol][ptf].n - 1
    
    def _fetch_mt5_data(self, symbol: str, timeframe: str, count: int) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from MT5."""
        if not self._mt5_connected:
            if not self._connect_mt5():
                return None
        
        mt5_symbol = self._get_mt5_symbol(symbol)
        tf_constant = self._tf_map.get(timeframe)
        
        if tf_constant is None:
            self.debug.log_generic("WARNING", "INVALID_TF", f"Unknown timeframe: {timeframe}")
            return None
        
        try:
            rates = mt5.copy_rates_from_pos(mt5_symbol, tf_constant, 0, count)
            
            if rates is None or len(rates) == 0:
                error = mt5.last_error()
                self.debug.log_generic("WARNING", "NO_DATA", 
                    f"No data for {mt5_symbol}/{timeframe}: {error}")
                return None
            
            df = pd.DataFrame(rates)
            df["timestamp"] = pd.to_datetime(df["time"], unit="s")
            df = df.rename(columns={"tick_volume": "volume"})
            
            # Get bid/ask if available
            tick = mt5.symbol_info_tick(mt5_symbol)
            if tick is not None:
                df["bid"] = tick.bid
                df["ask"] = tick.ask
            
            df = df[["timestamp", "open", "high", "low", "close", "volume"] + 
                    (["bid", "ask"] if "bid" in df.columns else [])]
            
            return df
            
        except Exception as e:
            self.debug.log_error(f"FETCH_MT5_{symbol}_{timeframe}", e)
            return None
    
    def _refresh_live_data(self, symbol: str) -> bool:
        """Refresh data for a symbol from MT5 (live mode only)."""
        if self.cfg.mode != "live":
            return False
        
        if not self._mt5_connected:
            if not self._connect_mt5():
                return False
        
        updated = False
        for tf in self.cfg.supported_timeframes:
            try:
                # Fetch a full history window so downstream modules (Trend/Momentum/Theme)
                # have enough bars for their longest lookbacks. Use live_bars_to_fetch,
                # which is already sized to our rolling window configuration.
                df = self._fetch_mt5_data(symbol, tf, self.cfg.live_bars_to_fetch)
                if df is not None and not df.empty:
                    # Update store
                    self.tfs[symbol][tf] = _TFStore(df)
                    self.ptrs_by_tf[symbol][tf] = self.tfs[symbol][tf].n - 1
                    updated = True
            except Exception as e:
                self.debug.log_error(f"REFRESH_{symbol}_{tf}", e)
        
        # Update primary pointer
        if updated:
            ptf = self.primary_tf.get(symbol, self.cfg.primary_timeframe)
            if ptf in self.tfs.get(symbol, {}):
                self.ptr_primary[symbol] = self.tfs[symbol][ptf].n - 1
        
        return updated

    # ─────────────────────────────────────────────────────────
    # Data loading and normalization
    # ─────────────────────────────────────────────────────────
    def _load_data_files(self) -> None:
        data_dir = self.cfg.data_directory
        if not os.path.exists(data_dir):
            self.debug.log_generic("WARNING", "DATA_DIR", f"Not found: {data_dir}. Will emit empty structures.")
            return

        total_loaded = 0
        for symbol in self.cfg.supported_symbols:
            per_tf: Dict[str, pd.DataFrame] = {}
            for tf in self.cfg.supported_timeframes:
                path = self._find_file_for(symbol, tf, data_dir)
                if not path:
                    self.debug.log_csv_warn(symbol, tf, f"No CSV (expected {symbol.replace('_','')}{tf}[(_features)].csv).")
                    continue
                try:
                    df = pd.read_csv(path)

                    # Accept 'time' alias
                    if "timestamp" not in df.columns and "time" in df.columns:
                        df = df.rename(columns={"time": "timestamp"})
                    if "timestamp" not in df.columns:
                        self.debug.log_generic("WARNING", "BAD_CSV", f"{symbol}/{tf} missing 'timestamp'. Skip {os.path.basename(path)}.")
                        continue

                    # Safe OHLCV normalization; no artificial bid/ask generation.
                    have_close = "close" in df.columns
                    need_cols = {"open", "high", "low", "close", "volume"}
                    if not need_cols.issubset(df.columns):
                        if have_close:
                            df["open"] = df.get("open", df["close"])
                            df["high"] = df.get("high", df["close"])
                            df["low"]  = df.get("low",  df["close"])
                            if "volume" not in df.columns:
                                df["volume"] = 0
                            self.debug.log_generic("INFO", "FILL_OHLCV",
                                                   f"{symbol}/{tf}: close-only dataset → filled O/H/L from close, volume=0.")
                        else:
                            self.debug.log_generic("WARNING", "NO_CLOSE",
                                                   f"{symbol}/{tf} lacks OHLC and 'close'. Skipping.")
                            continue

                    # Types & cleaning
                    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

                    # Cast numerics carefully; Pylance-safe
                    for col in ["open", "high", "low", "close", "bid", "ask"]:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors="coerce")

                    # Proper volume normalization: always a Series
                    if "volume" in df.columns:
                        vol_series = pd.to_numeric(df["volume"], errors="coerce")
                    else:
                        vol_series = pd.Series(0, index=df.index, dtype="int64")
                    df["volume"] = vol_series.fillna(0).astype("int64")

                    df.dropna(subset=["timestamp", "close"], inplace=True)

                    # Order & dedupe
                    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
                    if df.empty:
                        self.debug.log_generic("WARNING", "EMPTY_AFTER_CLEAN",
                                               f"{symbol}/{tf} has no valid rows after cleaning. Skipping.")
                        continue

                    per_tf[tf] = df
                    total_loaded += len(df)
                    self.debug.log_csv_ok(symbol, tf, len(df), os.path.basename(path))
                except Exception as e:
                    self.debug.log_error("CSV_FAIL", e)

            if per_tf:
                self.data_files[symbol] = per_tf

        if not self.data_files:
            self.debug.log_generic("WARNING", "SUMMARY", "No valid CSVs for any symbol. Will return empty values.")
        else:
            self.debug.log_generic("INFO", "SUMMARY",
                                   f"Loaded {len(self.data_files)} symbols, ~{total_loaded:,} clean bars.")

    def _find_file_for(self, symbol: str, timeframe: str, data_dir: str) -> Optional[str]:
        try:
            sym_nosl = symbol.replace("/", "").replace("_", "").upper()
            tf = timeframe.upper()
            patterns = [
                f"{sym_nosl}_{tf}_features.csv",
                f"{sym_nosl}_{tf}.csv",
                f"{sym_nosl}{tf}_features.csv",
                f"{sym_nosl}{tf}.csv",
            ]
            candidates: List[str] = []
            for pat in patterns:
                candidates.extend(glob.glob(os.path.join(data_dir, pat)))
                candidates.extend(glob.glob(os.path.join(data_dir, pat.lower())))
            if candidates:
                candidates.sort(key=lambda p: ("_features" not in os.path.basename(p), os.path.basename(p)))
                return candidates[0]
            return None
        except Exception:
            return None

    def _materialize_tfstores(self) -> None:
        """Convert per-TF DataFrames to NumPy-backed stores once (fast hot path)."""
        for symbol, per_tf in self.data_files.items():
            self.tfs[symbol] = {}
            for tf, df in per_tf.items():
                self.tfs[symbol][tf] = _TFStore(df)

    def _init_pointers_and_buffers(self) -> None:
        for symbol in self.cfg.supported_symbols:
            available_tfs = list(self.tfs.get(symbol, {}).keys())
            if not available_tfs:
                continue
            self.primary_tf[symbol] = self.cfg.primary_timeframe if self.cfg.primary_timeframe in available_tfs else available_tfs[0]
            # start at min window so indicators can form
            self.ptr_primary[symbol] = max(0, self.cfg.window_min)
            self.ptrs_by_tf[symbol] = {}
            for tf in available_tfs:
                self.ptrs_by_tf[symbol][tf] = self.ptr_primary[symbol]

            self.price_buffers.setdefault(symbol, {
                "close": deque(maxlen=self.cfg.buffer_size),
                "high":  deque(maxlen=self.cfg.buffer_size),
                "low":   deque(maxlen=self.cfg.buffer_size),
                "volume":deque(maxlen=self.cfg.buffer_size),
            })

    def _initialize_technical_indicators(self) -> None:
        for symbol in self.cfg.supported_symbols:
            self.technical_indicators[symbol] = {
                "sma_20": 0.0, "sma_50": 0.0,
                "rsi": 0.0, "atr": 0.0,
                "bollinger_upper": 0.0, "bollinger_lower": 0.0,
                "macd": 0.0, "macd_signal": 0.0,
                "stochastic": 0.0,
            }

    def _setup_initial_conditions(self) -> None:
        self.current_timestamp = datetime.datetime.utcnow()
        # Prime one step so current_bars is populated
        for symbol in self.cfg.supported_symbols:
            self._advance_symbol_data(symbol)

    # ─────────────────────────────────────────────────────────
    # Internal mechanics
    # ─────────────────────────────────────────────────────────
    def _advance_symbol_data(self, symbol: str) -> bool:
        """
        Advance data for a symbol.
        - Training mode: advance pointer through historical data
        - Live mode: refresh data from MT5
        """
        with self._lock:
            if self.cfg.mode == "live":
                return self._advance_symbol_data_live(symbol)
            else:
                return self._advance_symbol_data_training(symbol)
    
    def _advance_symbol_data_live(self, symbol: str) -> bool:
        """Advance data in live mode - fetch fresh data from MT5."""
        # Check throttling
        if self.cfg.update_frequency > 0:
            time_since_last = time.time() - self._last_update_ts
            if time_since_last < self.cfg.update_frequency:
                # Return current data without refresh
                return self._update_current_bar_from_store(symbol)
        
        # Refresh from MT5
        if not self._refresh_live_data(symbol):
            return False
        
        return self._update_current_bar_from_store(symbol)
    
    def _advance_symbol_data_training(self, symbol: str) -> bool:
        """Advance data in training mode - move pointer through historical data."""
        if symbol not in self.tfs or not self.tfs[symbol]:
            return False

        ptf = self.primary_tf[symbol]
        store = self.tfs[symbol].get(ptf)
        if store is None or store.n == 0:
            return False

        # Advance primary pointer
        idx = self.ptr_primary[symbol] + 1
        
        # In training mode, wrap around to continue training
        # (this allows continuous training on historical data)
        if idx >= store.n:
            idx = max(0, self.cfg.window_min)  # Reset to beginning with buffer
            self.debug.log_generic("DEBUG", "DATA_WRAP", 
                f"{symbol}: Data wrapped to index {idx}")

        self.ptr_primary[symbol] = idx

        # Current primary timestamp
        ts_np = store.ts[idx]  # numpy datetime64[ns]
        
        # Align all TF pointers to <= current ts (rightmost equal/less)
        for tf, tstore in self.tfs[symbol].items():
            pos = int(np.searchsorted(tstore.ts, ts_np, side="right") - 1)
            if pos < 0:
                pos = 0
            self.ptrs_by_tf[symbol][tf] = min(pos, tstore.n - 1)

        return self._update_current_bar_from_store(symbol)
    
    def _update_current_bar_from_store(self, symbol: str) -> bool:
        """Update current bar and buffers from store."""
        ptf = self.primary_tf.get(symbol)
        if not ptf or symbol not in self.tfs or ptf not in self.tfs[symbol]:
            return False
        
        store = self.tfs[symbol][ptf]
        idx = self.ptr_primary.get(symbol, 0)
        
        if idx < 0 or idx >= store.n:
            return False

        # Update current bar from primary TF
        cur = self._bar_from_store(store, idx)
        self.current_bars[symbol] = cur

        # Data quality validation
        issues: List[str] = []
        if cur["high"] < cur["low"]:
            issues.append("high < low")
        if cur["close"] <= 0:
            issues.append("close <= 0")
        if cur["open"] <= 0:
            issues.append("open <= 0")
        
        if issues and self._update_count % 50 == 0:  # Log quality issues periodically
            self.debug.log_data_quality_check(symbol, cur, issues)

        # Update buffers for indicators
        pb = self.price_buffers.get(symbol)
        if pb:
            pb["close"].append(cur["close"])
            pb["high"].append(cur["high"])
            pb["low"].append(cur["low"])
            pb["volume"].append(cur["volume"])

        if self.cfg.enable_technical_indicators:
            self._update_technical_indicators(symbol)

        return True

    def _bar_from_store(self, store: _TFStore, i: int) -> Dict[str, Any]:
        ts_py = pd.Timestamp(store.ts[i]).to_pydatetime()
        bid = float(store.bid[i]) if store.bid is not None and not np.isnan(store.bid[i]) else None
        ask = float(store.ask[i]) if store.ask is not None and not np.isnan(store.ask[i]) else None
        return {
            "timestamp": ts_py,
            "open": float(store.open[i]),
            "high": float(store.high[i]),
            "low":  float(store.low[i]),
            "close":float(store.close[i]),
            "volume": int(store.volume[i]),
            "bid": bid,
            "ask": ask,
        }

    def _window_from_store(self, store: _TFStore, end_idx: int) -> Tuple[List[float], List[float], List[float], List[float], List[int]]:
        # Extract a bounded window [start:end] with safe limits
        win = min(max(self.cfg.window_min, 1), self.cfg.window_max)
        start = max(0, end_idx - (win - 1))
        sl = slice(start, end_idx + 1)
        return (
            store.open[sl].astype(np.float64).tolist(),
            store.high[sl].astype(np.float64).tolist(),
            store.low[sl].astype(np.float64).tolist(),
            store.close[sl].astype(np.float64).tolist(),
            store.volume[sl].astype(np.int64).tolist(),
        )

    def _update_technical_indicators(self, symbol: str) -> None:
        """Update technical indicators using proper calculations."""
        pb = self.price_buffers.get(symbol)
        if not pb:
            return
            
        closes = list(pb["close"])
        highs  = list(pb["high"])
        lows   = list(pb["low"])
        
        indicators = self.technical_indicators.setdefault(symbol, {})

        # SMA
        if len(closes) >= 20:
            indicators["sma_20"] = float(np.mean(closes[-20:]))
        if len(closes) >= 50:
            indicators["sma_50"] = float(np.mean(closes[-50:]))

        # RSI(14) using Wilder's smoothing (exponential)
        if len(closes) >= 15:
            deltas = np.diff(closes[-15:])
            gains = np.where(deltas > 0, deltas, 0.0)
            losses = np.where(deltas < 0, -deltas, 0.0)
            
            # Use Wilder's smoothing factor (1/14)
            alpha = 1.0 / 14.0
            avg_gain = float(gains[0]) if len(gains) > 0 else 0.0
            avg_loss = float(losses[0]) if len(losses) > 0 else 0.0
            
            for i in range(1, len(gains)):
                avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
                avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss
            
            if avg_loss <= 1e-12:
                rsi = 100.0 if avg_gain > 0 else 50.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100.0 - (100.0 / (1.0 + rs))
            
            indicators["rsi"] = float(np.clip(rsi, 0.0, 100.0))

        # ATR(14) - Average True Range
        if len(closes) >= 15 and len(highs) >= 15 and len(lows) >= 15:
            trs = []
            for i in range(1, min(15, len(closes))):
                pc = closes[-(i + 1)]
                hi = highs[-i]
                lo = lows[-i]
                tr = max(hi - lo, abs(hi - pc), abs(lo - pc))
                trs.append(tr)
            indicators["atr"] = float(np.mean(trs)) if trs else 0.0

        # Bollinger Bands (20-period, 2 std)
        if len(closes) >= 20:
            sma20 = np.mean(closes[-20:])
            std20 = np.std(closes[-20:])
            indicators["bollinger_upper"] = float(sma20 + 2 * std20)
            indicators["bollinger_lower"] = float(sma20 - 2 * std20)
            indicators["bollinger_middle"] = float(sma20)

        # MACD (12, 26, 9)
        if len(closes) >= 26:
            ema12 = self._ema(closes, 12)
            ema26 = self._ema(closes, 26)
            macd_line = ema12 - ema26
            indicators["macd"] = float(macd_line)
            
            # Signal line would need historical MACD values
            # For simplicity, use a smoothed version
            indicators["macd_signal"] = float(macd_line * 0.8)  # Approximation

        # Stochastic (14, 3, 3)
        if len(closes) >= 14 and len(highs) >= 14 and len(lows) >= 14:
            highest_high = max(highs[-14:])
            lowest_low = min(lows[-14:])
            current_close = closes[-1]
            
            if highest_high - lowest_low > 1e-12:
                stoch_k = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
            else:
                stoch_k = 50.0
            
            indicators["stochastic"] = float(np.clip(stoch_k, 0.0, 100.0))
    
    @staticmethod
    def _ema(data: List[float], period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(data) < period:
            return float(np.mean(data)) if data else 0.0
        
        alpha = 2.0 / (period + 1)
        ema = float(data[-period])  # Start with first value in window
        
        for price in data[-period + 1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        return ema

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────
    async def calculate_confidence(self, action: Optional[Dict[str, Any]] = None, **inputs) -> float:
        try:
            if not self.current_bars:
                return 0.0
            available = len(self.current_bars) / max(1, len(self.cfg.supported_symbols))
            age = time.time() - self._last_update_ts
            freshness = max(0.0, 1.0 - age / 60.0)
            quality = 1.0
            for bar in self.current_bars.values():
                if bar["high"] < bar["low"] or bar["close"] <= 0:
                    quality *= 0.5
            return float(np.clip(0.4 * available + 0.3 * freshness + 0.3 * quality, 0.0, 1.0))
        except Exception:
            return 0.0

    async def propose_action(self, **inputs) -> Dict[str, Any]:
        return {
            "update_data": True,
            "symbols_to_update": list(self.cfg.supported_symbols),
            "maintenance_required": (self._update_count > 0 and self._update_count % 2000 == 0),
            "maintenance_type": "buffer_cleanup" if (self._update_count > 0 and self._update_count % 2000 == 0) else None,
            "data_quality": await self.calculate_confidence(),
        }

    async def process(self, **inputs) -> Dict[str, Any]:
        t0 = time.time()
        pid = self._update_count + 1  # p# counter in logs

        # Monitoring + process start + inputs
        self.debug.log_monitoring()
        self.debug.log_process_start(pid)
        self.debug.log_input_inspection(inputs)

        errors: Dict[str, str] = {}

        try:
            # Check connection in live mode
            if self.cfg.mode == "live" and not self._mt5_connected:
                try:
                    self._connect_mt5()
                except Exception as conn_err:
                    errors["connection"] = str(conn_err)
                    self.debug.log_error("MT5_RECONNECT_FAIL", conn_err)
            
            # Check throttling in live mode
            should_update = True
            if self.cfg.mode == "live" and self.cfg.update_frequency > 0:
                time_since_last = time.time() - self._last_update_ts
                if time_since_last < self.cfg.update_frequency:
                    should_update = False
            
            # Advance data for all symbols
            if should_update:
                for sym in self.cfg.supported_symbols:
                    try:
                        self._advance_symbol_data(sym)
                    except Exception as sym_err:
                        errors[sym] = str(sym_err)
                        if pid % max(1, self.cfg.log_every_n) == 0:
                            self.debug.log_generic("WARNING", "ADVANCE_FAIL", f"{sym}: {sym_err}")

            with self._lock:
                self._update_count += 1
                self._last_update_ts = time.time()
                dt_ms = (self._last_update_ts - t0) * 1000.0
                self._ema_ms = dt_ms if self._ema_ms <= 0.0 else (0.9 * self._ema_ms + 0.1 * dt_ms)

            try:
                self._update_session_labels()
            except Exception as sess_e:
                if pid % max(1, self.cfg.log_every_n) == 0:
                    self.debug.log_generic("DEBUG", "SESSION_LABEL", f"Update failed: {sess_e}")

            # Build snapshot
            snapshot = self._build_snapshot()

            # Contract-aware alias publication
            alias_payload = self._build_alias_map(snapshot)
            snapshot.update(alias_payload)

            # Universe & watched instruments
            snapshot['universe'] = list(self.cfg.supported_symbols)
            snapshot['watched_instruments'] = list(self.cfg.supported_symbols)

            # Health/meta
            snapshot['provider_status'] = {
                'mode': self.cfg.mode,
                'update_count': int(self._update_count),
                'last_update_ts': float(self._last_update_ts),
                'ms_since_last': (time.time() - self._last_update_ts) * 1000.0 if self._last_update_ts else None,
                'symbol_errors': errors,
                'fail_count': int(self._fail),
                'success_count': int(self._success),
                'mt5_connected': self._mt5_connected if self.cfg.mode == "live" else None,
                'market_open': self._is_market_hours(),
            }

            # Publish ALL declared keys to SmartInfoBus (not just aliases)
            # This prevents "critically stale" warnings for keys like alerts, market_conditions, etc.
            core_keys = [
                'alerts', 'market_conditions', 'economic_calendar', 'environment',
                'learning_context', 'learning_status', 'macro_data', 'step_data', 'strategy_status',
                'market_data', 'prices', 'price_data', 'ohlcv_data', 'volatility_data',
                'volatility_level', 'multi_timeframe_data', 'historical_prices',
                'session_type', 'trading_session', 'timestamp', 'step_idx',
                'symbols', 'universe', 'watched_instruments',
                # FIX: Per-instrument volatility keys for HorizonAligner
                'volatility_level_by_instrument', 'volatility_by_instrument',
                # Publish technical indicators so voting & risk modules see live MT5-derived signals
                'technical_indicators',
            ]
            for key in core_keys:
                if key in snapshot:
                    try:
                        self.smart_bus.set(key, snapshot[key], module='MarketDataProvider', 
                                          thesis=f'Market data update {key}', confidence=0.8)
                    except Exception as e:
                        if pid % max(1, self.cfg.log_every_n) == 0:
                            self.debug.log_bus_publish_fail(key, e)

            # Publish aliases to SmartInfoBus
            if alias_payload:
                self.debug.log_alias_publish(list(alias_payload.keys()))
                for k, v in alias_payload.items():
                    try:
                        self.smart_bus.set(k, v, module='MarketDataProvider', thesis=f'Alias publish {k}', confidence=0.8)
                    except Exception as e:
                        if pid % max(1, self.cfg.log_every_n) == 0:
                            self.debug.log_bus_publish_fail(k, e)

            # Pretty summary line
            if pid % max(1, self.cfg.log_every_n) == 0:
                self.debug.log_snapshot_summary(snapshot)

                # Health-ish line (friendly)
                conf = await self.calculate_confidence()
                mode_label = "live" if self.cfg.mode == "live" else "training"
                self.debug.log_health(mode=mode_label, quality=float(conf), win_rate=0.0, circuit_breaker="CLOSED")

            # Optional NDJSON snapshots
            if self.cfg.ndjson_every_n > 0 and (pid % self.cfg.ndjson_every_n == 0):
                snap_meta = {
                    "tick": self._update_count,
                    "mode": self.cfg.mode,
                    "ema_ms": round(self._ema_ms, 3),
                    "symbols": list(snapshot.get("market_data", {}).keys()),
                    "volatility_level": snapshot.get("volatility_level"),
                }
                self._pretty.ndjson(pid, "snapshot_meta", snap_meta)

            self._success += 1

            # Log statistics periodically
            if pid % 100 == 0:
                self.debug.log_statistics()

            # neat footer line
            self.debug.log_process_end(success=True)
            return snapshot

        except Exception as e:
            self._fail += 1
            self.debug.log_error("PROCESS_FAIL", e)
            empty = self._empty_snapshot(error=str(e))
            # Ensure alias keys exist even in error paths
            for sym in self.cfg.supported_symbols:
                for tf in self.cfg.supported_timeframes:
                    alias_key = self._alias_key(sym, tf)
                    if alias_key in self._contract_provides:
                        empty[alias_key] = {}
            empty['universe'] = list(self.cfg.supported_symbols)
            empty['watched_instruments'] = list(self.cfg.supported_symbols)
            empty['provider_status'] = {
                'mode': self.cfg.mode,
                'update_count': int(self._update_count),
                'last_error': str(e),
                'fail_count': int(self._fail),
                'success_count': int(self._success),
                'mt5_connected': self._mt5_connected if self.cfg.mode == "live" else None,
            }
            self.debug.log_process_end(success=False)
            return empty

    # ─────────────────────────────────────────────────────────
    # Snapshot builders
    # ─────────────────────────────────────────────────────────
    def _build_snapshot(self) -> Dict[str, Any]:
        # multi_timeframe_data windowed by pointers
        multi_tf: Dict[str, Dict[str, Any]] = {}
        for symbol in self.cfg.supported_symbols:
            if symbol not in self.tfs:
                continue
            multi_tf[symbol] = {}
            for tf, store in self.tfs[symbol].items():
                end_idx = self.ptrs_by_tf.get(symbol, {}).get(tf, 0)
                # Allow publishing whatever history is available; consumers (like ThemeExpert)
                # will check sequence length against their own lookback requirements.
                if store.n <= 0 or end_idx < 0:
                    continue
                o, h, l, c, v = self._window_from_store(store, end_idx)
                cur = self._bar_from_store(store, end_idx)
                rec = {
                    "open":   o, "high": h, "low": l, "close": c, "volume": v,
                    "current_bar": {
                        k: (self._to_iso_ts(cur[k]) if k == "timestamp" else cur[k])
                        for k in cur.keys()
                    },
                    "timeframe": tf,
                    "bars_available": int(store.n),
                }
                multi_tf[symbol][tf] = rec

        # Core maps from current_bars
        market_data = {s: self._bar_with_iso(self.current_bars[s]) for s in self.current_bars}

        price_data = {
            s: {
                "last":  market_data[s]["close"],
                "close": market_data[s]["close"],
                "open":  market_data[s]["open"],
                "high":  market_data[s]["high"],
                "low":   market_data[s]["low"],
            } for s in market_data
        }

        ohlcv_data = {
            s: {
                "open":   market_data[s]["open"],
                "high":   market_data[s]["high"],
                "low":    market_data[s]["low"],
                "close":  market_data[s]["close"],
                "volume": market_data[s]["volume"],
            } for s in market_data
        }

        bid_ask_data = {}
        for s in market_data:
            bar = market_data[s]
            bid = bar.get("bid")
            ask = bar.get("ask")
            
            # Calculate spread, with fallback estimation
            if bid is not None and ask is not None:
                spread = ask - bid
            else:
                spread = self._calculate_spread(bar)
            
            bid_ask_data[s] = {
                "bid": bid,
                "ask": ask,
                "spread": spread,
            }

        # Volume & liquidity
        volume_data: Dict[str, Any] = {}
        liquidity_data: Dict[str, Any] = {}
        for s in self.cfg.supported_symbols:
            current_vol = self._safe_int(ohlcv_data.get(s, {}).get("volume", None))
            tf_series: Dict[str, List[int]] = {}
            if s in multi_tf:
                for tf, rec in multi_tf[s].items():
                    tf_series[tf] = list(rec.get("volume", []))
            volume_data[s] = {"current": current_vol, "timeframes": tf_series}

            bbo = bid_ask_data.get(s, {})
            liquidity_data[s] = {
                "bid": bbo.get("bid"),
                "ask": bbo.get("ask"),
                "spread": bbo.get("spread"),
                "volume": current_vol,
            }

        # Volatility & indicators
        vol_data = {}
        vol_level_by_instrument = {}  # FIX: Per-instrument volatility level
        vol_by_instrument = {}        # FIX: Per-instrument volatility value
        for s in self.cfg.supported_symbols:
            atr = float(self.technical_indicators[s].get("atr", 0.0))
            last = float(market_data[s]["close"]) if s in market_data else 0.0
            vol = float(atr / max(last, 1e-9)) if last > 0 else 0.0
            vol_data[s] = {"atr": atr, "volatility": vol}
            # Determine per-instrument volatility level
            inst_vol_level = "high" if vol > 0.02 else ("medium" if vol > 0.01 else "low")
            vol_level_by_instrument[s] = inst_vol_level
            vol_by_instrument[s] = vol
        vol_level = "high" if any(v.get("volatility", 0.0) > 0.02 for v in vol_data.values()) else \
                    ("medium" if any(v.get("volatility", 0.0) > 0.01 for v in vol_data.values()) else "low")

        # Timestamp
        self.current_timestamp = max(
            (v["timestamp"] for v in self.current_bars.values()),
            default=datetime.datetime.utcnow(),
        )
        ts_iso = self._to_iso_ts(self.current_timestamp)

        indicators_map = {s: {k: float(v) for k, v in d.items()} for s, d in self.technical_indicators.items()}

        # NOTE: Removed stale placeholder keys (alerts, economic_calendar, environment,
        # learning_context, learning_status, macro_data, market_conditions, step_data,
        # strategy_status) - these were empty dicts/lists causing stale data warnings.
        # Consumers handle missing keys with fallbacks.
        snapshot: Dict[str, Any] = {
            "bid_ask_data": bid_ask_data,
            "historical_prices": multi_tf,
            "indicators": indicators_map,
            "market_data": market_data,
            "market_liquidity": liquidity_data,  # legacy mirror
            "module_insights": {
                "provider": "MarketDataProvider",
                "symbols": list(self.cfg.supported_symbols),
                "timeframes": list(self.cfg.supported_timeframes),
                "update_count": int(self._update_count),
                "ema_latency_ms": float(self._ema_ms),
                "fps_estimate": float(1000.0 / max(self._ema_ms, 1e-6)),
                "volatility_level": vol_level,
            },
            "multi_timeframe_data": multi_tf,
            "ohlcv_data": ohlcv_data,
            "price_data": price_data,
            "prices": {s: price_data[s]["last"] for s in price_data},
            "session_type": self.session_type,
            "step_idx": int(self._update_count),
            "symbols": list(self.cfg.supported_symbols),
            "technical_indicators": indicators_map,
            "timestamp": ts_iso,
            "trading_session": self.trading_session,
            "volatility": {s: float(self.technical_indicators[s].get("atr", 0.0)) for s in self.cfg.supported_symbols},
            "volatility_data": vol_data,
            "volatility_level": vol_level,
            # FIX: Per-instrument volatility for HorizonAligner and Executor
            "volatility_level_by_instrument": vol_level_by_instrument,
            "volatility_by_instrument": vol_by_instrument,
            # UnifiedDataExtractor expectations
            "volume_data": volume_data,
            "liquidity_data": liquidity_data,
            # Restored bus keys with actual data
            "market_conditions": {
                "timestamp": ts_iso,
                "volatility_level": vol_level,
                "session_type": self.session_type,
                "trading_session": self.trading_session,
                "active_sessions": getattr(self, "active_sessions", []),
                "symbols": list(self.cfg.supported_symbols),
            },
            "environment": {
                "mode": self.cfg.mode,
                "provider": "MarketDataProvider",
                "timestamp": ts_iso,
                "session_type": self.session_type,
                "trading_session": self.trading_session,
                "market_open": self._is_market_hours(),
                "symbols": list(self.cfg.supported_symbols),
            },
            "step_data": {
                "step_idx": int(self._update_count),
                "timestamp": ts_iso,
                "mode": self.cfg.mode,
            },
            "alerts": [],
            "learning_status": {
                "phase": self.cfg.mode,
                "progress_step": int(self._update_count),
                "timestamp": ts_iso,
            },
            "learning_context": {
                "session_type": self.session_type,
                "volatility_level": vol_level,
                "symbols": list(self.cfg.supported_symbols),
            },
            "strategy_status": {
                "status": "running",
                "volatility_level": vol_level,
                "timestamp": ts_iso,
            },
        }
        return snapshot

    def _build_alias_map(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """Create symbol/TF 'market_data_SYM_TF' alias keys only if declared in contract."""
        out: Dict[str, Any] = {}
        mtd = snapshot.get("multi_timeframe_data", {}) or {}
        for sym in self.cfg.supported_symbols:
            per_tf = mtd.get(sym, {}) or {}
            for tf in self.cfg.supported_timeframes:
                alias_key = self._alias_key(sym, tf)
                if alias_key not in self._contract_provides:
                    continue  # strict single-writer: only publish declared keys
                cur_bar = per_tf.get(tf, {}).get("current_bar", {})
                out[alias_key] = cur_bar if isinstance(cur_bar, dict) else {}
        return out

    def _alias_key(self, symbol: str, timeframe: str) -> str:
        sym_clean = symbol.replace("/", "_")
        return f"market_data_{sym_clean}_{timeframe}"

    def _empty_snapshot(self, error: Optional[str] = None) -> Dict[str, Any]:
        now = self._to_iso_ts(datetime.datetime.utcnow())
        indicators_map = {s: dict(self.technical_indicators.get(s, {})) for s in self.cfg.supported_symbols}
        # NOTE: Removed stale placeholder keys (same as _build_snapshot)
        snapshot: Dict[str, Any] = {
            "bid_ask_data": {},
            "historical_prices": {},
            "indicators": indicators_map,
            "market_data": {},
            "market_liquidity": {},
            "module_insights": {
                "provider": "MarketDataProvider",
                "symbols": list(self.cfg.supported_symbols),
                "timeframes": list(self.cfg.supported_timeframes),
                "update_count": int(self._update_count),
                "last_error": error,
                "volatility_level": "low",
            },
            "multi_timeframe_data": {},
            "ohlcv_data": {},
            "price_data": {},
            "prices": {},
            "session_type": self.session_type,
            "step_idx": int(self._update_count),
            "symbols": list(self.cfg.supported_symbols),
            "technical_indicators": indicators_map,
            "timestamp": now,
            "trading_session": self.trading_session,
            "volatility": {},
            "volatility_data": {},
            "volatility_level": "low",
            "volume_data": {},
            "liquidity_data": {},
            "market_conditions": {
                "timestamp": now,
                "volatility_level": "low",
                "session_type": self.session_type,
                "trading_session": self.trading_session,
                "active_sessions": getattr(self, "active_sessions", []),
                "symbols": list(self.cfg.supported_symbols),
            },
            "environment": {
                "mode": self.cfg.mode,
                "provider": "MarketDataProvider",
                "timestamp": now,
                "session_type": self.session_type,
                "trading_session": self.trading_session,
                "market_open": False,
                "symbols": list(self.cfg.supported_symbols),
            },
            "step_data": {
                "step_idx": int(self._update_count),
                "timestamp": now,
                "mode": self.cfg.mode,
            },
            "alerts": [],
            "learning_status": {
                "phase": self.cfg.mode,
                "progress_step": int(self._update_count),
                "timestamp": now,
            },
            "learning_context": {
                "session_type": self.session_type,
                "volatility_level": "low",
                "symbols": list(self.cfg.supported_symbols),
            },
            "strategy_status": {
                "status": "initializing",
                "volatility_level": "low",
                "timestamp": now,
            },
        }
        return snapshot

    # ─────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────
    @staticmethod
    def _to_iso_ts(v: Any) -> Optional[str]:
        if isinstance(v, pd.Timestamp):
            v = v.to_pydatetime()
        if isinstance(v, datetime.datetime):
            return v.isoformat()
        if v is None:
            return None
        return str(v)

    @staticmethod
    def _safe_int(v: Any) -> Optional[int]:
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, float):
            return None if math.isnan(v) else int(v)
        return None

    def _bar_with_iso(self, bar: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(bar)
        out["timestamp"] = self._to_iso_ts(out.get("timestamp"))
        return out

    def _update_session_labels(self) -> None:
        """
        Update session labels based on current time.
        In live mode, use real time. In training mode, use data timestamp.
        """
        # Determine reference time
        if self.cfg.mode == "live":
            ref_time = datetime.datetime.utcnow()
        else:
            # Use the latest data timestamp if available
            ref_time = datetime.datetime.utcnow()  # Default
            if self.current_bars:
                timestamps = []
                for bar in self.current_bars.values():
                    ts = bar.get("timestamp")
                    if isinstance(ts, datetime.datetime):
                        timestamps.append(ts)
                if timestamps:
                    ref_time = max(timestamps)
        
        hour = ref_time.hour
        
        # Detect ALL active sessions (markets can overlap)
        # London: 08:00-16:00 UTC, NY: 13:00-21:00 UTC, Sydney: 21:00-06:00 UTC, Tokyo: 00:00-08:00 UTC
        active_sessions = []
        if 8 <= hour < 16:
            active_sessions.append("london")
        if 13 <= hour < 21:
            active_sessions.append("new_york")
        if 21 <= hour or hour < 6:
            active_sessions.append("sydney")
        if 0 <= hour < 8:
            active_sessions.append("tokyo")
        
        # Store active sessions list for consumers
        self.active_sessions = active_sessions if active_sessions else ["closed"]
        
        # Determine primary session (with overlap priority)
        if "london" in active_sessions and "new_york" in active_sessions:
            self.trading_session = "london_newyork_overlap"  # High liquidity overlap
        elif "london" in active_sessions:
            self.trading_session = "london"
        elif "new_york" in active_sessions:
            self.trading_session = "new_york"
        elif "sydney" in active_sessions and "tokyo" in active_sessions:
            self.trading_session = "asia_overlap"  # Asian overlap
        elif "tokyo" in active_sessions:
            self.trading_session = "tokyo"
        elif "sydney" in active_sessions:
            self.trading_session = "sydney"
        else:
            self.trading_session = "closed"

        # Session type
        if 13 <= hour < 16:
            self.session_type = "london_ny_overlap"  # Prime trading hours
        elif 9 <= hour < 17:
            self.session_type = "main"
        elif 17 <= hour < 21:
            self.session_type = "late"
        else:
            self.session_type = "overnight"

        # Only update current_timestamp in live mode
        if self.cfg.mode == "live":
            self.current_timestamp = ref_time

    def _is_market_hours(self) -> bool:
        """Check if market is open. FX is effectively 24/5."""
        if self.cfg.mode == "live":
            now = datetime.datetime.utcnow()
            weekday = now.weekday()
            # FX closed on weekends (Saturday = 5, Sunday = 6)
            if weekday >= 5:
                return False
            # Also closed around 21:00 Friday to 21:00 Sunday UTC
            if weekday == 4 and now.hour >= 21:  # Friday after 9 PM UTC
                return False
            return True
        # In training mode, always consider market open
        return True
    
    def _calculate_spread(self, bar: Dict[str, Any]) -> Optional[float]:
        """Calculate spread from bid/ask or estimate from typical values."""
        bid = bar.get("bid")
        ask = bar.get("ask")
        
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            return float(ask - bid)
        
        # Estimate spread based on typical values for instrument type
        close = bar.get("close", 0)
        if close <= 0:
            return None
        
        # Rough estimates (could be configured per instrument)
        if close > 1000:  # Likely gold (XAU/USD)
            return close * 0.0003  # ~30 cents per oz typical
        else:  # Likely FX pair
            return close * 0.0001  # ~1 pip typical
    
    def get_mode(self) -> str:
        """Return current operating mode."""
        return self.cfg.mode
    
    def is_live(self) -> bool:
        """Check if running in live mode."""
        return self.cfg.mode == "live"
    
    def is_connected(self) -> bool:
        """Check if connected to data source (always True for training, MT5 status for live)."""
        if self.cfg.mode == "live":
            return self._mt5_connected
        return True  # Training mode always "connected"
    
    def reconnect(self) -> bool:
        """Attempt to reconnect (only meaningful in live mode)."""
        if self.cfg.mode == "live":
            self._mt5_connected = False
            return self._connect_mt5()
        return True
    
    def shutdown(self) -> None:
        """Clean shutdown of the provider."""
        if self.cfg.mode == "live":
            self._disconnect_mt5()
        self.debug.log_statistics()
        self.debug.log_generic("INFO", "SHUTDOWN", "MarketDataProvider shut down")
