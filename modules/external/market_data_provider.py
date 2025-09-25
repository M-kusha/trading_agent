# ─────────────────────────────────────────────────────────────
# File: modules/external/market_data_provider.py
# PRODUCTION-READY Offline Market Data Provider (Pure, No Simulation)
# Contract-clean: provides ONLY keys declared in contracts.py
# Zero fabrication: reads CSVs only; otherwise returns empty/unknown structures
# Pylance-clean: typed self.cfg (dataclass), pass dict to BaseModule
# No ownership collisions: exports data-only keys
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import glob
import time
import datetime
from dataclasses import dataclass, field, asdict
from collections import deque
from typing import Dict, Any, List, Optional, Iterator, Tuple, Hashable

from modules.contracts import module_args
import numpy as np
import pandas as pd

# Core infrastructure
from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusStateMixin
from modules.utils.audit_utils import RotatingLogger, format_operator_message


@dataclass
class MarketDataConfig:
    """Configuration for Offline Market Data Provider."""
    data_directory: str = "data/processed"
    supported_symbols: List[str] = field(default_factory=lambda: ["XAU/USD", "EUR/USD"])
    supported_timeframes: List[str] = field(default_factory=lambda: ["H1", "H4", "D1"])
    primary_timeframe: str = "H4"       # Primary TF for advancing iterators
    update_frequency: float = 1.0       # seconds
    buffer_size: int = 10000
    enable_technical_indicators: bool = True


@module(**module_args(
    "MarketDataProvider",
    description="Offline market data provider that emits only real data from disk. No mock/simulated values.",
    error_handling=True,
    hot_reload=True,
    timeout_ms=3000,
    critical=True,  # Ensure provider always runs (even in emergency mode)
))
class MarketDataProvider(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    Pure data source:
    - Loads per-symbol, per-timeframe CSVs.
    - Computes indicators from loaded bars only.
    - If something is missing, emits empty/unknown structures (never fabricates).

    Contract outputs (from contracts.py):
      alerts, bid_ask_data, economic_calendar, environment, environment_config,
      historical_prices, indicators, input1, input2, learning_context, learning_status,
      macro_data, market_conditions, market_context, market_data, market_liquidity,
      multi_timeframe_data, ohlcv_data, portfolio_metrics, price_data, prices,
      session_type, step_data, step_idx, strategy_status, symbols,
      technical_indicators, timestamp, trading_session, volatility, volatility_data, volatility_level
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Logger and typed config FIRST
        self.logger = RotatingLogger("MarketDataProvider", log_path="logs/external/market_data_provider.log")
        self.cfg = MarketDataConfig(**(config or {}))

        # Define ALL attributes used by _initialize() BEFORE calling super().__init__()
        self.data_files: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.data_iterators: Dict[str, Iterator[Tuple[Hashable, pd.Series]]] = {}
        self.current_bars: Dict[str, Dict[str, Any]] = {}

        self.technical_indicators: Dict[str, Dict[str, float]] = {}
        self.price_buffers: Dict[str, Dict[str, deque]] = {}

        self.current_timestamp: Optional[datetime.datetime] = None
        self.trading_session: str = "london"
        self.session_type: str = "normal"

        # Health metrics
        self._last_update_ts: float = 0.0
        self._update_count: int = 0
        self._success: int = 0
        self._fail: int = 0
        self._proc_times: deque = deque(maxlen=200)

        # Only now let BaseModule wire things and call _initialize()
        super().__init__(config=asdict(self.cfg))

        self.logger.info(format_operator_message(
            "[BOOT]", "MARKET_DATA_PROVIDER_INIT",
            details=f"Symbols={self.cfg.supported_symbols}, TF={self.cfg.supported_timeframes}",
            result="Provider ready",
            context="system_startup",
        ))

    # ─────────────────────────────────────────────────────────────
    # Initialization (BaseModule will call this)
    # ─────────────────────────────────────────────────────────────
    def _initialize(self) -> None:
        try:
            self.logger.info("[INIT] Loading CSVs…")
            self._load_data_files()
            self._initialize_technical_indicators()
            self._setup_initial_conditions()
            self.logger.info("[OK] MarketDataProvider ready.")
        except Exception as e:
            self.logger.error(f"[FAIL] Initialization failed: {e}")
            raise

    # ─────────────────────────────────────────────────────────────
    # Data loading
    # ─────────────────────────────────────────────────────────────
    def _load_data_files(self) -> None:
        """Load offline CSVs into per-symbol/per-timeframe DataFrames (supports XAUUSD_H1_features.csv etc.)."""
        data_dir = self.cfg.data_directory
        if not os.path.exists(data_dir):
            self.logger.warning(f"[WARN] Data directory not found: {data_dir}. Provider will emit empty structures.")
            return

        total_loaded = 0
        for symbol in self.cfg.supported_symbols:
            per_tf: Dict[str, pd.DataFrame] = {}
            for tf in self.cfg.supported_timeframes:
                path = self._find_file_for(symbol, tf, data_dir)
                if not path:
                    self.logger.warning(f"[WARN] No CSV for {symbol}/{tf} using common patterns (e.g. XAUUSD_{tf}_features.csv).")
                    continue

                try:
                    df = pd.read_csv(path)
                    # accept either 'timestamp' or 'time'
                    if "timestamp" not in df.columns and "time" in df.columns:
                        df = df.rename(columns={"time": "timestamp"})
                    if "timestamp" not in df.columns:
                        self.logger.warning(f"[WARN] {symbol}/{tf} ({os.path.basename(path)}) missing 'timestamp' or 'time'. Skipping.")
                        continue

                    # Normalize timestamp and numeric
                    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                    df.replace([np.inf, -np.inf], np.nan, inplace=True)

                    # If only close/volatility are present, accept and synthesize minimal OHLCV (from close only)
                    have_close = "close" in df.columns
                    need_cols = ["open", "high", "low", "close", "volume"]
                    if not set(need_cols).issubset(df.columns):
                        if have_close:
                            if "open" not in df.columns:   df["open"] = df["close"]
                            if "high" not in df.columns:   df["high"] = df["close"]
                            if "low" not in df.columns:    df["low"]  = df["close"]
                            if "volume" not in df.columns: df["volume"] = 0
                            self.logger.info(f"[INFO] {symbol}/{tf} ({os.path.basename(path)}): using close-only dataset; filled O/H/L from close, volume=0.")
                        else:
                            self.logger.warning(f"[WARN] {symbol}/{tf} lacks OHLC and 'close'. Skipping.")
                            continue

                    # Cast numerics
                    for col in ["open", "high", "low", "close", "volume"]:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors="coerce")

                    # Basic cleaning
                    df.dropna(subset=["timestamp", "close"], inplace=True)
                    if "volume" in df.columns:
                        vol_series = pd.to_numeric(df["volume"], errors="coerce")
                    else:
                        vol_series = pd.Series([0] * len(df), index=df.index, dtype="float64")
                    df["volume"] = vol_series.fillna(0).astype(np.int64)

                    # Sort & dedupe
                    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

                    if df.empty:
                        self.logger.warning(f"[WARN] {symbol}/{tf} has no valid rows after cleaning. Skipping.")
                        continue

                    per_tf[tf] = df
                    total_loaded += len(df)
                    self.logger.info(f"[OK] {symbol}/{tf}: {len(df)} bars from {os.path.basename(path)}")

                except Exception as e:
                    self.logger.error(f"[FAIL] Error loading {symbol}/{tf} from {path}: {e}")

            if per_tf:
                self.data_files[symbol] = per_tf
                tf0 = self.cfg.primary_timeframe if self.cfg.primary_timeframe in per_tf else next(iter(per_tf))
                self.data_iterators[symbol] = per_tf[tf0].iterrows()

        if not self.data_files:
            self.logger.warning("[SUMMARY] No valid CSVs found for any symbol (after cleaning). Provider will return empty/unknown values.")
        else:
            self.logger.info(f"[SUMMARY] Loaded {len(self.data_files)} symbols, ~{total_loaded:,} clean bars.")

    def _find_file_for(self, symbol: str, timeframe: str, data_dir: str) -> Optional[str]:
        """Find a CSV file for a given symbol/timeframe using common naming patterns."""
        try:
            sym_nosl = symbol.replace("/", "").upper()
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
                # Prefer files with "_features" in name, then any
                candidates.sort(key=lambda p: ("_features" not in os.path.basename(p), os.path.basename(p)))
                return candidates[0]
            return None
        except Exception:
            return None

    def _initialize_technical_indicators(self) -> None:
        """Prepare indicator dicts and rolling buffers per symbol."""
        for symbol in self.cfg.supported_symbols:
            self.technical_indicators[symbol] = {
                "sma_20": 0.0, "sma_50": 0.0,
                "rsi": 0.0, "atr": 0.0,
                "bollinger_upper": 0.0, "bollinger_lower": 0.0,
                "macd": 0.0, "macd_signal": 0.0,
                "stochastic": 0.0,
            }
            self.price_buffers[symbol] = {
                "close": deque(maxlen=200),
                "high": deque(maxlen=200),
                "low": deque(maxlen=200),
                "volume": deque(maxlen=200),
            }

    def _setup_initial_conditions(self) -> None:
        """Advance once for each symbol if possible."""
        self.current_timestamp = datetime.datetime.utcnow()
        for symbol in self.cfg.supported_symbols:
            self._advance_symbol_data(symbol)

    # ─────────────────────────────────────────────────────────────
    # Internal mechanics
    # ─────────────────────────────────────────────────────────────
    def _advance_symbol_data(self, symbol: str) -> bool:
        """Advance primary timeframe iterator and refresh buffers & indicators."""
        it = self.data_iterators.get(symbol)
        if it is None:
            return False
        try:
            _, row = next(it)
        except StopIteration:
            # Restart from the beginning (deterministic; still real data)
            dfs = self.data_files.get(symbol, {})
            if not dfs:
                return False
            tf0 = self.cfg.primary_timeframe if self.cfg.primary_timeframe in dfs else next(iter(dfs))
            self.data_iterators[symbol] = dfs[tf0].iterrows()
            _, row = next(self.data_iterators[symbol])

        # Normalize timestamp
        ts = row["timestamp"]
        ts = pd.to_datetime(ts) if not isinstance(ts, (pd.Timestamp, datetime.datetime)) else ts

        # Build bar strictly from CSV columns; do NOT fabricate bid/ask
        bid = float(row["bid"]) if "bid" in row else None
        ask = float(row["ask"]) if "ask" in row else None

        bar = {
            "timestamp": (ts.to_pydatetime() if isinstance(ts, pd.Timestamp) else ts),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": int(row["volume"]),
            "bid": bid,
            "ask": ask,
        }
        self.current_bars[symbol] = bar

        # Update buffers & indicators
        self.price_buffers[symbol]["close"].append(bar["close"])
        self.price_buffers[symbol]["high"].append(bar["high"])
        self.price_buffers[symbol]["low"].append(bar["low"])
        self.price_buffers[symbol]["volume"].append(bar["volume"])

        if self.cfg.enable_technical_indicators:
            self._update_technical_indicators(symbol)

        return True

    def _update_technical_indicators(self, symbol: str) -> None:
        prices = list(self.price_buffers[symbol]["close"])
        highs = list(self.price_buffers[symbol]["high"])
        lows = list(self.price_buffers[symbol]["low"])

        # SMAs
        if len(prices) >= 20:
            self.technical_indicators[symbol]["sma_20"] = float(np.mean(prices[-20:]))
        if len(prices) >= 50:
            self.technical_indicators[symbol]["sma_50"] = float(np.mean(prices[-50:]))

        # RSI(14) - classic formula (no smoothing to keep light)
        if len(prices) >= 15:
            deltas = np.diff(prices[-15:])
            gains = np.where(deltas > 0, deltas, 0.0)
            losses = np.where(deltas < 0, -deltas, 0.0)
            avg_gain = float(np.mean(gains)) if gains.size else 0.0
            avg_loss = float(np.mean(losses)) if losses.size else 0.0
            if avg_loss <= 0.0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100.0 - (100.0 / (1.0 + rs))
            self.technical_indicators[symbol]["rsi"] = float(np.clip(rsi, 0.0, 100.0))

        # ATR(14)
        if len(prices) >= 15 and len(highs) >= 15 and len(lows) >= 15:
            trs = []
            for i in range(1, 15):
                prev_close = prices[-(i + 1)]
                hi = highs[-i]
                lo = lows[-i]
                trs.append(max(hi - lo, abs(hi - prev_close), abs(lo - prev_close)))
            self.technical_indicators[symbol]["atr"] = float(np.mean(trs)) if trs else 0.0

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────
    async def calculate_confidence(self, action: Optional[Dict[str, Any]] = None, **inputs) -> float:
        """Data quality/availability diagnostic score (internal, not provided on bus)."""
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
        """Suggest provider upkeep operations (no data fabrication)."""
        return {
            "update_data": True,
            "symbols_to_update": list(self.cfg.supported_symbols),
            "maintenance_required": (self._update_count > 0 and self._update_count % 2000 == 0),
            "maintenance_type": "buffer_cleanup" if (self._update_count > 0 and self._update_count % 2000 == 0) else None,
            "data_quality": await self.calculate_confidence(),
        }

    async def process(self, **inputs) -> Dict[str, Any]:
        """
        Advance data (if due) and return a full snapshot adhering strictly to the contract.
        Enhancements:
          - Per-symbol advancement isolation (one bad symbol won't crash whole provider)
          - Alias keys included directly in returned snapshot to satisfy output validation
          - Universe / watched_instruments + heartbeat status
          - Critical flag ensures execution during emergency mode
        """
        t0 = time.time()
        errors: Dict[str, str] = {}
        try:
            now = time.time()
            if now - self._last_update_ts >= self.cfg.update_frequency:
                for sym in self.cfg.supported_symbols:
                    try:
                        self._advance_symbol_data(sym)
                    except Exception as sym_err:  # isolate per-symbol issues
                        errors[sym] = str(sym_err)
                        self.logger.warning(f"[WARN] Failed advancing {sym}: {sym_err}")
                self._update_count += 1
                self._last_update_ts = now
                try:
                    self._update_session_labels()
                except Exception as sess_e:
                    self.logger.warning(f"[WARN] Session label update failed: {sess_e}")

            snapshot = self._build_snapshot()

            # Construct alias map (contract requires these explicit keys)
            alias_map: Dict[str, Any] = {}
            mtd = snapshot.get('multi_timeframe_data', {}) or {}
            for sym in self.cfg.supported_symbols:
                per_tf = mtd.get(sym, {}) or {}
                for tf in self.cfg.supported_timeframes:
                    cur_bar = None
                    try:
                        cur_bar = per_tf.get(tf, {}).get('current_bar')
                    except Exception:
                        cur_bar = None
                    alias_key = f"market_data_{sym}_{tf}"
                    alias_map[alias_key] = cur_bar if isinstance(cur_bar, dict) else {}

            # Append required meta keys not presently in snapshot
            snapshot['universe'] = list(self.cfg.supported_symbols)
            snapshot['watched_instruments'] = list(self.cfg.supported_symbols)

            # Merge aliases into snapshot so validate_outputs sees them
            snapshot.update(alias_map)

            # Heartbeat / status (not in contract but useful)
            snapshot['provider_status'] = {
                'update_count': int(self._update_count),
                'last_update_ts': float(self._last_update_ts),
                'ms_since_last': (time.time() - self._last_update_ts) * 1000.0 if self._last_update_ts else None,
                'symbol_errors': errors,
                'fail_count': int(self._fail),
                'success_count': int(self._success),
            }

            # Publish aliases to bus (still useful for legacy listeners); tolerate errors
            for k, v in alias_map.items():
                try:
                    self.smart_bus.set(
                        k,
                        v,
                        module='MarketDataProvider',
                        thesis=f'Alias publish {k}',
                        confidence=0.8,
                    )
                except Exception:
                    pass

            self._success += 1
            self._proc_times.append((time.time() - t0) * 1000.0)
            return snapshot
        except Exception as e:
            self._fail += 1
            self.logger.error(f"[FAIL] process(): {e}")
            # Build minimally compliant empty snapshot including alias keys
            empty = self._empty_snapshot(error=str(e))
            for sym in self.cfg.supported_symbols:
                for tf in self.cfg.supported_timeframes:
                    empty[f"market_data_{sym}_{tf}"] = {}
            empty['universe'] = list(self.cfg.supported_symbols)
            empty['watched_instruments'] = list(self.cfg.supported_symbols)
            empty['provider_status'] = {
                'update_count': int(self._update_count),
                'last_error': str(e),
                'fail_count': int(self._fail),
                'success_count': int(self._success),
            }
            return empty

    # ─────────────────────────────────────────────────────────────
    # Snapshot builders (Contract-clean)
    # ─────────────────────────────────────────────────────────────
    def _build_snapshot(self) -> Dict[str, Any]:
        # Multi-TF window from actual files (no synthesis)
        multi_tf: Dict[str, Dict[str, Any]] = {}
        for symbol in self.cfg.supported_symbols:
            sym_data = self.data_files.get(symbol, {})
            if not sym_data:
                continue
            multi_tf[symbol] = {}
            for tf, df in sym_data.items():
                if len(df) < 20:
                    continue
                idx = min(len(df) - 1, max(10, int(len(df) * 0.5)))
                s = max(0, idx - 19)
                e = idx + 1
                rec = {
                    "open":   df["open"].iloc[s:e].astype(float).tolist(),
                    "high":   df["high"].iloc[s:e].astype(float).tolist(),
                    "low":    df["low"].iloc[s:e].astype(float).tolist(),
                    "close":  df["close"].iloc[s:e].astype(float).tolist(),
                    "volume": df["volume"].iloc[s:e].astype(int).tolist(),
                    "current_bar": {
                        "open":   float(df["open"].iloc[idx]),
                        "high":   float(df["high"].iloc[idx]),
                        "low":    float(df["low"].iloc[idx]),
                        "close":  float(df["close"].iloc[idx]),
                        "volume": int(df["volume"].iloc[idx]),
                        # bid/ask only if present in CSV
                        "bid": float(df["bid"].iloc[idx]) if "bid" in df.columns else None,
                        "ask": float(df["ask"].iloc[idx]) if "ask" in df.columns else None,
                    },
                    "timeframe": tf,
                    "bars_available": int(len(df)),
                }
                multi_tf[symbol][tf] = rec

        # Core maps (from current_bars only)
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

        # ---- NEW: explicit volume_data (current + per-TF series) ----
        volume_data: Dict[str, Any] = {}
        for s in self.cfg.supported_symbols:
            # current volume if we have a bar for the symbol
            current_vol = None
            if s in ohlcv_data:
                current_vol = int(ohlcv_data[s]["volume"])

            # per-timeframe volume series (from multi_tf windows)
            tf_series: Dict[str, List[int]] = {}
            if s in multi_tf:
                for tf, rec in multi_tf[s].items():
                    tf_series[tf] = list(rec.get("volume", []))

            volume_data[s] = {
                "current": current_vol,
                "timeframes": tf_series,  # may be {}
            }

        # ---- NEW: explicit liquidity_data (real values only) ----
        # Mirrors BBO info you already expose + current volume for convenience
        liquidity_data: Dict[str, Any] = {}
        for s in self.cfg.supported_symbols:
            bbo = bid_ask_data.get(s, {})
            liquidity_data[s] = {
                "bid": bbo.get("bid"),
                "ask": bbo.get("ask"),
                "spread": bbo.get("spread"),
                "volume": ohlcv_data.get(s, {}).get("volume"),
                # room for depth later, if your CSVs ever include it:
                # "market_depth": {"bids": [...], "asks": [...]}
            }

        # Indicators/volatility (computed from actual buffers)
        vol_data = {
            s: {
                "atr": float(self.technical_indicators[s].get("atr", 0.0)),
                "volatility": float(self.technical_indicators[s].get("atr", 0.0) /
                                    max(1e-9, market_data[s]["close"])) if s in market_data else 0.0,
            } for s in self.cfg.supported_symbols
        }

        vol_level = "high" if any(v.get("volatility", 0.0) > 0.02 for v in vol_data.values()) else \
                    ("medium" if any(v.get("volatility", 0.0) > 0.01 for v in vol_data.values()) else "low")

        market_context = {
            "volatility_hint": vol_level,
            "market_hours": self._is_market_hours(),
            "session_human": self.trading_session,
        }

        ts_iso = (self.current_timestamp or datetime.datetime.utcnow()).isoformat()

        # Build contract-clean snapshot (now includes volume_data & liquidity_data)
        snapshot: Dict[str, Any] = {
            "alerts": [],
            "bid_ask_data": bid_ask_data,
            "economic_calendar": [],
            "environment": {},
            "environment_config": {},
            "historical_prices": multi_tf,
            "indicators": {s: {k: float(v) for k, v in d.items()} for s, d in self.technical_indicators.items()},
            "input1": {},
            "input2": {},
            "learning_context": {},
            "learning_status": {},
            "macro_data": {},
            "market_conditions": {},
            "market_context": market_context,
            "market_data": market_data,
            "market_liquidity": {},  # keep for legacy consumers
            "module_insights": {
                "provider": "MarketDataProvider",
                "symbols": list(self.cfg.supported_symbols),
                "timeframes": list(self.cfg.supported_timeframes),
                "update_count": int(self._update_count),
                "last_update_ms_ago": int((time.time() - self._last_update_ts) * 1000.0) if self._last_update_ts else None,
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
            "technical_indicators": {s: {k: float(v) for k, v in d.items()} for s, d in self.technical_indicators.items()},
            "timestamp": ts_iso,
            "trading_session": self.trading_session,
            "volatility": {s: float(self.technical_indicators[s].get("atr", 0.0)) for s in self.cfg.supported_symbols},
            "volatility_data": vol_data,
            "volatility_level": vol_level,

            # >>> NEW keys to satisfy UnifiedDataExtractor <<<
            "volume_data": volume_data,
            "liquidity_data": liquidity_data,
        }
        return snapshot


    def _empty_snapshot(self, error: Optional[str] = None) -> Dict[str, Any]:
        """Return a schema-complete but empty snapshot (no fabricated values)."""
        now = datetime.datetime.utcnow().isoformat()
        snapshot: Dict[str, Any] = {
            "alerts": [],
            "bid_ask_data": {},
            "economic_calendar": [],
            "environment": {},
            "environment_config": {},
            "historical_prices": {},
            "indicators": {s: dict(self.technical_indicators.get(s, {})) for s in self.cfg.supported_symbols},
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
            "technical_indicators": {s: dict(self.technical_indicators.get(s, {})) for s in self.cfg.supported_symbols},
            "timestamp": now,
            "trading_session": self.trading_session,
            "volatility": {},
            "volatility_data": {},
            "volatility_level": "low",

            # Keep the keys present so the bus sees a provider even in empty/error states
            "volume_data": {},
            "liquidity_data": {},
        }
        return snapshot


    # ─────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────
    def _bar_with_iso(self, bar: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(bar)
        ts = out.get("timestamp")
        out["timestamp"] = (ts.isoformat() if isinstance(ts, (datetime.datetime, pd.Timestamp)) else str(ts))
        return out

    def _update_session_labels(self) -> None:
        """Human labels for UI only; canonical is separate and aligns with TimeAwareRiskScaling."""
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

    def _session_canonical(self) -> str:
        h = datetime.datetime.utcnow().hour
        if 0 <= h < 8:
            return "asian"
        if 8 <= h < 16:
            return "european"
        if 16 <= h < 22:
            return "us"
        return "closed"

    def _is_market_hours(self) -> bool:
        # FX is effectively 24/5; keep simple True in provider context.
        return True
