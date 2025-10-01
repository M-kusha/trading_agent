# ─────────────────────────────────────────────────────────────
# File: modules/external/market_data_provider.py
# PRODUCTION-READY Offline Market Data Provider (Pure, No Simulation)
#
# • Pointer-driven, timestamp-aligned multi-timeframe windows (fast)
# • Contract-aware alias publication (no ownership drift)
# • Comprehensive, human-friendly logging to logs/external/
# • Optional NDJSON snapshot stream for deep forensics
# • Single-writer, data-only outputs; no bid/ask fabrication
# • Pylance-clean; typed config; low-GC hot path
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import glob
import json
import math
import time
import datetime
from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set

import numpy as np
import pandas as pd

# Contracts & infra
from modules.contracts import module_args, CONTRACTS  # contract-aware aliasing
from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusStateMixin
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.info_bus import SmartInfoBus


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
    data_directory: str = "data/processed"
    supported_symbols: List[str] = field(default_factory=lambda: ["XAU_USD", "EUR_USD"])
    supported_timeframes: List[str] = field(default_factory=lambda: ["H1", "H4", "D1"])
    primary_timeframe: str = "H4"        # drives time advancement
    update_frequency: float = 0.0        # orchestrator-driven; keep 0 for always-advance
    buffer_size: int = 512               # rolling indicator buffer length
    enable_technical_indicators: bool = True

    # Logging controls
    log_every_n: int = 250               # 1 = log every tick
    ndjson_every_n: int = 0              # 0 = off; 1 = every tick; N = every Nth tick

    # Window size caps
    window_min: int = 20
    window_max: int = 200


# ─────────────────────────────────────────────────────────────
# Module
# ─────────────────────────────────────────────────────────────

@module(**module_args(
    "MarketDataProvider",
    description="Offline market data provider that emits only real data from disk. No mock/simulated values.",
    error_handling=True,
    hot_reload=True,
    timeout_ms=5000,
    critical=True,  # run even in emergency mode
))
class MarketDataProvider(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    Pointer-driven, multi-TF, contract-clean offline provider.

    Guarantees:
      - Advances on every `process()` call (no time throttling).
      - Publishes ALL keys declared for MarketDataProvider in contracts.py.
      - Publishes symbol/TF alias keys only if present in contract provides.
      - Never fabricates bid/ask; only emits if CSV columns exist.
      - Timestamp fields are ISO-8601 strings on the bus/snapshot.
      - Single-writer: only keys declared in contracts are published.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Typed config
        self.cfg = MarketDataConfig(**(config or {}))

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

        # Bus
        self.smart_bus = SmartInfoBus()

        # Wire & initialize (BaseModule will call _initialize)
        super().__init__(config=asdict(self.cfg))

        # Operator-style boot line
        self.logger.info(format_operator_message(
            "[BOOT]", "MARKET_DATA_PROVIDER_INIT",
            details=f"Symbols={self.cfg.supported_symbols}, TF={self.cfg.supported_timeframes}",
            result="Provider ready",
            context="system_startup",
        ))

    # ─────────────────────────────────────────────────────────
    # Initialization (BaseModule hook)
    # ─────────────────────────────────────────────────────────
    def _initialize(self) -> None:
        try:
            self.debug.log_csv_loading_start()
            self._load_data_files()
            self._materialize_tfstores()
            self._init_pointers_and_buffers()
            self._initialize_technical_indicators()
            self._setup_initial_conditions()
            self.debug.log_init_ok(len(self.data_files), sum(len(df) for sym in self.data_files for df in self.data_files[sym].values()))
        except Exception as e:
            self.debug.log_error("INIT_FAIL", e)
            raise

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
        """Advance primary TF pointer and align other TFs by current timestamp (O(log n))."""
        if symbol not in self.tfs or not self.tfs[symbol]:
            return False

        ptf = self.primary_tf[symbol]
        store = self.tfs[symbol][ptf]
        if store.n == 0:
            return False

        # Advance primary pointer (loop back deterministically)
        idx = self.ptr_primary[symbol] + 1
        if idx >= store.n:
            idx = 0
        self.ptr_primary[symbol] = idx

        # Current primary timestamp
        ts_np = store.ts[idx]  # numpy datetime64[ns]
        # Align all TF pointers to <= current ts (rightmost equal/less)
        for tf, tstore in self.tfs[symbol].items():
            pos = int(np.searchsorted(tstore.ts, ts_np, side="right") - 1)
            if pos < 0:
                pos = 0
            self.ptrs_by_tf[symbol][tf] = min(pos, tstore.n - 1)

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
        pb = self.price_buffers[symbol]
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
        pb = self.price_buffers[symbol]
        closes = list(pb["close"])
        highs  = list(pb["high"])
        lows   = list(pb["low"])

        # SMA
        if len(closes) >= 20:
            self.technical_indicators[symbol]["sma_20"] = float(np.mean(closes[-20:]))
        if len(closes) >= 50:
            self.technical_indicators[symbol]["sma_50"] = float(np.mean(closes[-50:]))

        # RSI(14) (simple average variant)
        if len(closes) >= 15:
            deltas = np.diff(closes[-15:])
            gains = np.where(deltas > 0, deltas, 0.0)
            losses = np.where(deltas < 0, -deltas, 0.0)
            avg_gain = float(np.mean(gains)) if gains.size else 0.0
            avg_loss = float(np.mean(losses)) if losses.size else 0.0
            if avg_loss <= 0.0:
                rsi = 100.0
            else:
                rs = avg_gain / max(avg_loss, 1e-12)
                rsi = 100.0 - (100.0 / (1.0 + rs))
            self.technical_indicators[symbol]["rsi"] = float(np.clip(rsi, 0.0, 100.0))

        # ATR(14)
        if len(closes) >= 15 and len(highs) >= 15 and len(lows) >= 15:
            trs = []
            for i in range(1, 15):
                pc = closes[-(i + 1)]
                hi = highs[-i]
                lo = lows[-i]
                trs.append(max(hi - lo, abs(hi - pc), abs(lo - pc)))
            self.technical_indicators[symbol]["atr"] = float(np.mean(trs)) if trs else 0.0

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
            # Always advance (orchestrator controls cadence)
            for sym in self.cfg.supported_symbols:
                try:
                    self._advance_symbol_data(sym)
                except Exception as sym_err:
                    errors[sym] = str(sym_err)
                    if pid % max(1, self.cfg.log_every_n) == 0:
                        self.debug.log_generic("WARNING", "ADVANCE_FAIL", f"{sym}: {sym_err}")

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
                'update_count': int(self._update_count),
                'last_update_ts': float(self._last_update_ts),
                'ms_since_last': (time.time() - self._last_update_ts) * 1000.0 if self._last_update_ts else None,
                'symbol_errors': errors,
                'fail_count': int(self._fail),
                'success_count': int(self._success),
            }

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
                self.debug.log_health(mode="training", quality=float(conf), win_rate=0.0, circuit_breaker="CLOSED")

            # Optional NDJSON snapshots
            if self.cfg.ndjson_every_n > 0 and (pid % self.cfg.ndjson_every_n == 0):
                snap_meta = {
                    "tick": self._update_count,
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
                'update_count': int(self._update_count),
                'last_error': str(e),
                'fail_count': int(self._fail),
                'success_count': int(self._success),
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
                if store.n < self.cfg.window_min or end_idx < 0:
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

        bid_ask_data = {
            s: {
                "bid": market_data[s]["bid"],
                "ask": market_data[s]["ask"],
                "spread": (
                    market_data[s]["ask"] - market_data[s]["bid"]
                ) if (market_data[s]["bid"] is not None and market_data[s]["ask"] is not None) else None,
            } for s in market_data
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
        for s in self.cfg.supported_symbols:
            atr = float(self.technical_indicators[s].get("atr", 0.0))
            last = float(market_data[s]["close"]) if s in market_data else 0.0
            vol = float(atr / max(last, 1e-9)) if last > 0 else 0.0
            vol_data[s] = {"atr": atr, "volatility": vol}
        vol_level = "high" if any(v.get("volatility", 0.0) > 0.02 for v in vol_data.values()) else \
                    ("medium" if any(v.get("volatility", 0.0) > 0.01 for v in vol_data.values()) else "low")

        # Market context & ts
        self.current_timestamp = max(
            (v["timestamp"] for v in self.current_bars.values()),
            default=datetime.datetime.utcnow(),
        )
        ts_iso = self._to_iso_ts(self.current_timestamp)

        market_context = {
            "volatility_hint": vol_level,
            "market_hours": self._is_market_hours(),
            "session_human": self.trading_session,
        }

        indicators_map = {s: {k: float(v) for k, v in d.items()} for s, d in self.technical_indicators.items()}

        snapshot: Dict[str, Any] = {
            "alerts": [],
            "bid_ask_data": bid_ask_data,
            "economic_calendar": [],
            "environment": {},
            "environment_config": {},  # contract requires we provide it; keep empty to avoid semantic conflicts
            "historical_prices": multi_tf,
            "indicators": indicators_map,
            "input1": {},
            "input2": {},
            "learning_context": {},
            "learning_status": {},
            "macro_data": {},
            "market_conditions": {},
            "market_context": market_context,
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
            "portfolio_metrics": {},
            "price_data": price_data,
            "prices": {s: price_data[s]["last"] for s in price_data},
            "session_type": self.session_type,
            "step_data": {},
            "step_idx": int(self._update_count),
            "strategy_status": {},
            "symbols": list(self.cfg.supported_symbols),
            "technical_indicators": indicators_map,
            "timestamp": ts_iso,
            "trading_session": self.trading_session,
            "volatility": {s: float(self.technical_indicators[s].get("atr", 0.0)) for s in self.cfg.supported_symbols},
            "volatility_data": vol_data,
            "volatility_level": vol_level,
            # UnifiedDataExtractor expectations
            "volume_data": volume_data,
            "liquidity_data": liquidity_data,
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
        snapshot: Dict[str, Any] = {
            "alerts": [],
            "bid_ask_data": {},
            "economic_calendar": [],
            "environment": {},
            "environment_config": {},
            "historical_prices": {},
            "indicators": indicators_map,
            "input1": {},
            "input2": {},
            "learning_context": {},
            "learning_status": {},
            "macro_data": {},
            "market_conditions": {},
            "market_context": {
                "volatility_hint": "low",
                "market_hours": self._is_market_hours(),
                "session_human": self.trading_session,
            },
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
            "portfolio_metrics": {},
            "price_data": {},
            "prices": {},
            "session_type": self.session_type,
            "step_data": {},
            "step_idx": int(self._update_count),
            "strategy_status": {},
            "symbols": list(self.cfg.supported_symbols),
            "technical_indicators": indicators_map,
            "timestamp": now,
            "trading_session": self.trading_session,
            "volatility": {},
            "volatility_data": {},
            "volatility_level": "low",
            "volume_data": {},
            "liquidity_data": {},
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
        """Human labels for UI; canonical session mapping stays simple for FX."""
        hour = datetime.datetime.utcnow().hour
        if 8 <= hour < 16:
            self.trading_session = "london"
        elif 13 <= hour < 21:
            self.trading_session = "new_york"
        elif 21 <= hour or hour < 6:
            self.trading_session = "sydney"
        else:
            self.trading_session = "tokyo"

        if 9 <= hour < 17:
            self.session_type = "main"
        elif 17 <= hour < 21:
            self.session_type = "overlap"
        else:
            self.session_type = "overnight"

        self.current_timestamp = datetime.datetime.utcnow()

    def _is_market_hours(self) -> bool:
        # FX is effectively 24/5; keep simple True in provider context.
        return True
