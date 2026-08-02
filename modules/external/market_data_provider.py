

from __future__ import annotations

import datetime
import glob
import json
import math
import os
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, cast

import numpy as np
import pandas as pd

from modules.contracts import CONTRACTS, module_args
from modules.core.mixins import SmartInfoBusStateMixin, SmartInfoBusTradingMixin
from modules.core.module_base import BaseModule, module
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.info_bus import SmartInfoBus

try:
    import MetaTrader5 as _MT5
    MT5_AVAILABLE = True
    mt5: Any = cast(Any, _MT5)
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None


def _to_float(x: Any, default: float = 0.0) -> float:
    if x is None:
        return float(default)
    try:
        return float(x)
    except Exception:
        return float(default)


def _to_int(x: Any, default: int = 0) -> int:
    if x is None:
        return int(default)
    try:
        return int(x)
    except Exception:
        return int(default)


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float, np.floating)) and math.isfinite(float(x))


def _now_utc_naive() -> datetime.datetime:
    return datetime.datetime.utcnow().replace(tzinfo=None)


def _iso_ts(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, pd.Timestamp):
        v = v.to_pydatetime()
    if isinstance(v, datetime.datetime):
        return v.isoformat()
    return str(v)


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, default=str, ensure_ascii=False)
    except Exception:
        return str(obj)


class AuditLogger:
    def __init__(
        self,
        sink: RotatingLogger,
        *,
        debug: bool,
        log_every_n: int,
        ndjson_path: Optional[Path] = None,
        ndjson_every_n: int = 0,
    ) -> None:
        self.sink = sink
        self.debug_enabled = bool(debug)
        self.log_every_n = max(1, int(log_every_n))
        self.ndjson_path = ndjson_path
        self.ndjson_every_n = max(0, int(ndjson_every_n))

        if self.ndjson_path is not None:
            try:
                self.ndjson_path.parent.mkdir(parents=True, exist_ok=True)
                self.ndjson_path.touch(exist_ok=True)
            except Exception:
                self.ndjson_path = None

    def set_debug(self, enabled: bool) -> None:
        self.debug_enabled = bool(enabled)

    def _emit(self, level: str, msg: str) -> None:
        try:
            if level == "DEBUG":
                self.sink.debug(msg)
            elif level == "INFO":
                self.sink.info(msg)
            elif level == "WARNING":
                self.sink.warning(msg)
            else:
                self.sink.error(msg)
        except Exception:

            pass

    def debug(self, msg: str) -> None:
        if self.debug_enabled:
            self._emit("DEBUG", msg)

    def info(self, msg: str) -> None:
        self._emit("INFO", msg)

    def warning(self, msg: str) -> None:
        self._emit("WARNING", msg)

    def error(self, msg: str) -> None:
        self._emit("ERROR", msg)

    def ndjson(self, pid: int, kind: str, data: Dict[str, Any]) -> None:
        if self.ndjson_path is None or self.ndjson_every_n <= 0:
            return
        if pid % self.ndjson_every_n != 0:
            return
        try:
            rec = {"ts": datetime.datetime.utcnow().isoformat(), "pid": pid, "kind": kind, "data": data}
            with self.ndjson_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        except Exception:
            pass


class _TFStore:
    __slots__ = ("close", "high", "low", "n", "open", "ts", "volume")

    def __init__(self, df: pd.DataFrame) -> None:

        self.ts = pd.to_datetime(df["timestamp"]).to_numpy(dtype="datetime64[ns]", copy=False)
        self.open = df["open"].to_numpy(dtype="float64", copy=False)
        self.high = df["high"].to_numpy(dtype="float64", copy=False)
        self.low = df["low"].to_numpy(dtype="float64", copy=False)
        self.close = df["close"].to_numpy(dtype="float64", copy=False)
        self.volume = df["volume"].to_numpy(dtype="int64", copy=False)
        self.n = int(self.close.shape[0])


@dataclass
class MarketDataConfig:

    mode: str = "training"


    data_directory: str = "data/processed"


    mt5_account: Optional[int] = None
    mt5_password: Optional[str] = None
    mt5_server: Optional[str] = None
    mt5_timeout: int = 60000
    mt5_reconnect_attempts: int = 3
    mt5_reconnect_delay: float = 5.0


    supported_symbols: List[str] = field(default_factory=lambda: ["XAUUSD"])
    allowed_symbols: Optional[List[str]] = field(default_factory=lambda: ["XAUUSD"])


    supported_timeframes: List[str] = field(default_factory=lambda: ["M15", "H1", "H4", "D1"])
    primary_timeframe: str = "M15"


    buffer_size: int = 512
    window_min: int = 60
    window_max: int = 120


    update_frequency: float = 1.0
    live_bars_to_fetch: int = 3000
    live_refresh_bars: int = 600
    min_tick_change_for_recalc: float = 0.0


    live_history_refresh_s: float = 10.0


    mt5_time_offset_refresh_s: float = 300.0


    enable_technical_indicators: bool = True


    include_forming_bar_in_primary_tf: bool = True


    debug: bool = True
    log_every_n: int = 250
    ndjson_every_n: int = 0


    symbol_mapping: Dict[str, str] = field(default_factory=dict)


    allow_runtime_mode_switch: bool = False


@module(**module_args(
    "MarketDataProvider",
    description="Unified market data provider supporting both live MT5 and offline CSV modes.",
    error_handling=True,
    hot_reload=True,
    timeout_ms=5000,
    critical=True,
))
class MarketDataProvider(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:

        cfg_in: Dict[str, Any] = dict(config) if isinstance(config, dict) else {}
        cfg_fields = {f.name for f in fields(MarketDataConfig)}
        self.cfg = MarketDataConfig(**{k: v for k, v in cfg_in.items() if k in cfg_fields})


        self.cfg.supported_symbols = ["XAUUSD"]
        self.cfg.allowed_symbols = ["XAUUSD"]


        if self.cfg.primary_timeframe not in self.cfg.supported_timeframes:
            self.cfg.supported_timeframes = [self.cfg.primary_timeframe] + list(self.cfg.supported_timeframes)

        self._lock = threading.RLock()


        log_dir = Path("logs/external")
        log_dir.mkdir(parents=True, exist_ok=True)
        self.logger: RotatingLogger = RotatingLogger(
            "MarketDataProvider",
            log_path=str(log_dir / "market_data_provider.log")
        )

        self.audit = AuditLogger(
            sink=self.logger,
            debug=bool(self.cfg.debug),
            log_every_n=int(self.cfg.log_every_n),
            ndjson_path=(log_dir / "market_data_provider.ndjson") if self.cfg.ndjson_every_n > 0 else None,
            ndjson_every_n=int(self.cfg.ndjson_every_n),
        )

        mdp = CONTRACTS.get("MarketDataProvider")
        self._contract_provides: Set[str] = set(mdp.provides) if mdp else set()


        self.data_files: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.tfs: Dict[str, Dict[str, _TFStore]] = {}
        self.primary_tf: Dict[str, str] = {}
        self.ptr_primary: Dict[str, int] = {}
        self.ptrs_by_tf: Dict[str, Dict[str, int]] = {}


        self.current_bars: Dict[str, Dict[str, Any]] = {}
        self.price_buffers: Dict[str, Dict[str, deque]] = {}
        self.technical_indicators: Dict[str, Dict[str, float]] = {}


        self.last_tick_prices: Dict[str, Dict[str, float]] = {}
        self.current_forming_bars: Dict[str, Dict[str, Any]] = {}


        self._last_update_ts: float = 0.0
        self._update_count: int = 0
        self._success: int = 0
        self._fail: int = 0
        self._ema_ms: float = 0.0
        self._mt5_connected: bool = False
        self._mt5_connection_failures: int = 0
        self._last_history_refresh_ts: float = 0.0


        self._mt5_server_utc_offset_s: int = 0
        self._mt5_server_utc_offset_last_ts: float = 0.0


        self.current_timestamp: Optional[datetime.datetime] = None
        self.trading_session: str = "closed"
        self.session_type: str = "normal"
        self.active_sessions: List[str] = ["closed"]


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
            }


        try:
            from modules.utils.info_bus import InfoBusManager
            self.smart_bus = InfoBusManager.get_instance()
        except Exception as e:
            self.audit.warning(f"[WARN] InfoBusManager unavailable, using SmartInfoBus fallback: {type(e).__name__}: {e}")
            self.smart_bus = SmartInfoBus()

        super().__init__(config=asdict(self.cfg))

        mode_label = "LIVE (MT5)" if self.cfg.mode == "live" else "TRAINING (CSV)"
        self.logger.info(format_operator_message(
            "[BOOT]", "MARKET_DATA_PROVIDER_INIT",
            details=f"Mode={mode_label}, Symbols={self.cfg.supported_symbols}, TF={self.cfg.supported_timeframes}, Primary={self.cfg.primary_timeframe}, Debug={self.cfg.debug}",
            result="Provider ready",
            context="system_startup",
        ))


    def _compute_mt5_server_offset_s(self, tick_epoch_s: int) -> int:
        try:
            now_epoch = int(time.time())
            raw = int(tick_epoch_s) - now_epoch
            raw = max(-14 * 3600, min(raw, 14 * 3600))
            rounded = int(round(raw / 3600.0) * 3600)
            if abs(rounded) < 300:
                return 0
            return rounded
        except Exception:
            return 0

    def _maybe_update_mt5_time_offset(self, tick_epoch_s: int, *, force: bool = False) -> None:
        now = time.time()
        refresh_s = float(getattr(self.cfg, "mt5_time_offset_refresh_s", 300.0))
        if (not force) and self._mt5_server_utc_offset_last_ts > 0 and (now - self._mt5_server_utc_offset_last_ts) < max(1.0, refresh_s):
            return
        if tick_epoch_s <= 0:
            return

        off = self._compute_mt5_server_offset_s(int(tick_epoch_s))
        if off != self._mt5_server_utc_offset_s:
            self.audit.info(f"[MT5_TIME_OFFSET] updated offset_s={off} (prev={self._mt5_server_utc_offset_s})")
            self._mt5_server_utc_offset_s = off
        self._mt5_server_utc_offset_last_ts = now

    def _maybe_refresh_mt5_time_offset_via_tick(self, symbol: str, *, force: bool = False) -> None:
        if not self._mt5_connected or mt5 is None:
            return
        now = time.time()
        refresh_s = float(getattr(self.cfg, "mt5_time_offset_refresh_s", 300.0))
        if (not force) and self._mt5_server_utc_offset_last_ts > 0 and (now - self._mt5_server_utc_offset_last_ts) < max(1.0, refresh_s):
            return

        mt5_symbol = self._get_mt5_symbol(symbol)
        try:
            tick = mt5.symbol_info_tick(mt5_symbol)
            if tick is None:
                return
            t_epoch = _to_int(getattr(tick, "time", None), 0)
            self._maybe_update_mt5_time_offset(t_epoch, force=True)
        except Exception:
            return

    def _mt5_epoch_to_utc_naive(self, epoch_s: int) -> datetime.datetime:
        adj = int(epoch_s) - int(self._mt5_server_utc_offset_s)
        return datetime.datetime.fromtimestamp(adj, tz=datetime.timezone.utc).replace(tzinfo=None)


    def _initialize(self) -> None:
        try:

            env_mode = (os.environ.get("EXECUTION_MODE", "") or "").strip().lower()
            if env_mode in ("live", "training", "train", "sim", "simulation"):
                self.cfg.mode = "live" if env_mode == "live" else "training"
                self.audit.info(f"[MODE] EXECUTION_MODE override -> {self.cfg.mode.upper()}")


            self._sync_mode_with_bus()

            if self.cfg.mode == "live":
                self._initialize_live_mode()
            else:
                self._initialize_training_mode()

            self._initialize_technical_indicators()
            self._setup_initial_conditions()

            total_bars = sum(store.n for sym_stores in self.tfs.values() for store in sym_stores.values())
            self.audit.info(f"[INIT] OK symbols={len(self.tfs)} total_bars={total_bars} mode={self.cfg.mode}")

        except Exception as e:
            self.audit.error(f"[INIT_FAIL] {type(e).__name__}: {e}")
            raise


    def _sync_mode_with_bus(self) -> None:
        if not self.cfg.allow_runtime_mode_switch:
            return
        try:
            bus_mode = self.smart_bus.get("execution_mode", "MarketDataProvider", default=None)
        except Exception as e:
            self.audit.warning(f"[MODE_SYNC] bus get failed: {type(e).__name__}: {e}")
            return
        if isinstance(bus_mode, str) and bus_mode.lower() in ("live", "training"):
            if self.cfg.mode != bus_mode.lower():
                self.audit.info(f"[MODE_SYNC] switching mode {self.cfg.mode} -> {bus_mode.lower()}")
                self.cfg.mode = bus_mode.lower()


    def _initialize_training_mode(self) -> None:
        self._load_data_files()
        self._materialize_tfstores()
        self._init_pointers_and_buffers()
        self.audit.info("[MODE] Initialized in TRAINING mode (CSV)")


    def _initialize_live_mode(self) -> None:
        if not MT5_AVAILABLE:
            raise RuntimeError("MetaTrader5 package not installed. Install with: pip install MetaTrader5")

        if not self.cfg.symbol_mapping:
            self.cfg.symbol_mapping = self._build_default_symbol_mapping()

        self._connect_mt5()
        self._init_live_structures()
        self._fetch_initial_live_data()
        self.audit.info("[MODE] Initialized in LIVE mode (MT5)")

    def _build_default_symbol_mapping(self) -> Dict[str, str]:
        return {"XAUUSD": "XAUUSD"}

    def _get_mt5_symbol(self, internal_symbol: str) -> str:
        for mt5_sym, int_sym in self.cfg.symbol_mapping.items():
            if int_sym == internal_symbol:
                return mt5_sym
        return internal_symbol


    def _connect_mt5(self) -> bool:
        if self._mt5_connected:
            return True
        if mt5 is None:
            raise RuntimeError("MT5 module missing")

        for attempt in range(1, self.cfg.mt5_reconnect_attempts + 1):
            try:
                self.audit.info(f"[MT5] connect attempt {attempt}/{self.cfg.mt5_reconnect_attempts}")

                try:
                    mt5.shutdown()
                except Exception:
                    pass

                init_kwargs: Dict[str, Any] = {}
                if self.cfg.mt5_account and self.cfg.mt5_password and self.cfg.mt5_server:
                    init_kwargs = {
                        "login": self.cfg.mt5_account,
                        "password": self.cfg.mt5_password,
                        "server": self.cfg.mt5_server,
                        "timeout": self.cfg.mt5_timeout,
                    }

                ok = mt5.initialize(**init_kwargs) if init_kwargs else mt5.initialize()
                if not ok:
                    raise ConnectionError(f"MT5 initialize failed: {mt5.last_error()}")

                acc = mt5.account_info()
                if acc is None:
                    raise ConnectionError("MT5 connected but account_info() is None")

                mt5_symbol = self._get_mt5_symbol("XAUUSD")
                if not mt5.symbol_select(mt5_symbol, True):
                    raise ConnectionError(f"MT5 symbol_select failed for {mt5_symbol}")

                self._mt5_connected = True
                self._mt5_connection_failures = 0


                self._maybe_refresh_mt5_time_offset_via_tick("XAUUSD", force=True)

                self.audit.info(f"[MT5] connected login={getattr(acc,'login',None)} balance={getattr(acc,'balance',None)}")
                return True

            except Exception as e:
                self._mt5_connection_failures += 1
                self.audit.error(f"[MT5_CONNECT_FAIL] {type(e).__name__}: {e}")
                if attempt < self.cfg.mt5_reconnect_attempts:
                    time.sleep(float(self.cfg.mt5_reconnect_delay))

        self._mt5_connected = False
        raise ConnectionError(f"Failed to connect MT5 after {self.cfg.mt5_reconnect_attempts} attempts")

    def _disconnect_mt5(self) -> None:
        if MT5_AVAILABLE and self._mt5_connected and mt5 is not None:
            try:
                mt5.shutdown()
            except Exception as e:
                self.audit.warning(f"[MT5] shutdown error: {type(e).__name__}: {e}")
            self._mt5_connected = False
            self.audit.info("[MT5] disconnected")


    def _init_live_structures(self) -> None:
        sym = "XAUUSD"
        self.tfs[sym] = {}
        self.ptrs_by_tf[sym] = {}
        self.primary_tf[sym] = self.cfg.primary_timeframe
        self.ptr_primary[sym] = 0

        self.price_buffers[sym] = {
            "close": deque(maxlen=self.cfg.buffer_size),
            "high": deque(maxlen=self.cfg.buffer_size),
            "low": deque(maxlen=self.cfg.buffer_size),
            "volume": deque(maxlen=self.cfg.buffer_size),
        }

    def _fetch_initial_live_data(self) -> None:
        sym = "XAUUSD"
        for tf in self.cfg.supported_timeframes:
            df = self._fetch_mt5_data(sym, tf, int(self.cfg.live_bars_to_fetch))
            if df is not None and not df.empty:
                self.tfs[sym][tf] = _TFStore(df)
                self.ptrs_by_tf[sym][tf] = self.tfs[sym][tf].n - 1
                self.audit.info(f"[LIVE_SEED] {sym}/{tf} bars={len(df)}")

        ptf = self.primary_tf.get(sym, self.cfg.primary_timeframe)
        store = self.tfs.get(sym, {}).get(ptf)
        if store and store.n > 0:
            self.ptr_primary[sym] = store.n - 1

        self._seed_buffers_from_store(sym)
        self._update_current_bar_from_store(sym)


        self._last_history_refresh_ts = time.time()

        self._last_update_ts = time.time()

    def _fetch_mt5_data(self, symbol: str, timeframe: str, count: int) -> Optional[pd.DataFrame]:
        if not self._mt5_connected:
            self._connect_mt5()
        if mt5 is None:
            return None


        self._maybe_refresh_mt5_time_offset_via_tick(symbol)

        mt5_symbol = self._get_mt5_symbol(symbol)
        tf_constant = self._tf_map.get(timeframe)
        if tf_constant is None:
            self.audit.warning(f"[INVALID_TF] {timeframe}")
            return None

        try:
            rates = mt5.copy_rates_from_pos(mt5_symbol, tf_constant, 0, int(count))
            if rates is None or len(rates) == 0:
                self.audit.warning(f"[NO_DATA] {mt5_symbol}/{timeframe} err={mt5.last_error()}")
                return None

            df = pd.DataFrame(rates)


            off = int(self._mt5_server_utc_offset_s)

            df["timestamp"] = pd.to_datetime(df["time"].astype("int64") - off, unit="s", utc=True).dt.tz_localize(None)

            df = df.rename(columns={"tick_volume": "volume"})
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")


            if len(df) >= 2:
                df = df.iloc[:-1]

            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
            df.dropna(subset=["timestamp", "close"], inplace=True)

            return None if df.empty else df

        except Exception as e:
            self.audit.error(f"[FETCH_MT5_FAIL] {symbol}/{timeframe} {type(e).__name__}: {e}")
            return None

    def _refresh_live_data(self, symbol: str) -> bool:
        if self.cfg.mode != "live":
            return False
        if not self._mt5_connected:
            self._connect_mt5()


        min_s = float(getattr(self.cfg, "live_history_refresh_s", 10.0))
        now = time.time()
        if self._last_history_refresh_ts > 0 and (now - self._last_history_refresh_ts) < max(0.0, min_s):
            return False

        updated = False


        bars_to_fetch = int(
            max(
                int(self.cfg.live_bars_to_fetch),
                self.cfg.buffer_size + self.cfg.window_max + 50,
            )
        )

        for tf in self.cfg.supported_timeframes:
            df = self._fetch_mt5_data(symbol, tf, bars_to_fetch)
            if df is not None and not df.empty:
                self.tfs[symbol][tf] = _TFStore(df)
                self.ptrs_by_tf[symbol][tf] = self.tfs[symbol][tf].n - 1
                updated = True

        if updated:
            self._last_history_refresh_ts = now

            ptf = self.primary_tf.get(symbol, self.cfg.primary_timeframe)
            store = self.tfs.get(symbol, {}).get(ptf)
            if store and store.n > 0:
                self.ptr_primary[symbol] = store.n - 1


            self._seed_buffers_from_store(symbol)
            self._update_current_bar_from_store(symbol)

        return updated


    def _fetch_current_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        if not self._mt5_connected:
            self._connect_mt5()
        if mt5 is None:
            return None

        mt5_symbol = self._get_mt5_symbol(symbol)

        try:
            tick = mt5.symbol_info_tick(mt5_symbol)
            if tick is None:
                self.audit.warning(f"[TICK_NONE] {symbol}")
                return None

            bid = _to_float(getattr(tick, "bid", None), 0.0)
            ask = _to_float(getattr(tick, "ask", None), 0.0)

            last_raw = getattr(tick, "last", None)
            last_raw_f = _to_float(last_raw, 0.0)

            mid = (bid + ask) / 2.0 if (bid > 0.0 and ask > 0.0 and bid < ask) else 0.0
            last = last_raw_f if (_is_num(last_raw_f) and last_raw_f > 0.0) else mid

            t_epoch = _to_int(getattr(tick, "time", None), 0)
            if t_epoch > 0:

                self._maybe_update_mt5_time_offset(t_epoch)
                time_dt = self._mt5_epoch_to_utc_naive(t_epoch)
            else:
                time_dt = _now_utc_naive()

            vol = _to_int(getattr(tick, "volume", 0), 0)
            vol_real = getattr(tick, "volume_real", None)
            if vol_real is not None:
                try:
                    vol = max(vol, int(float(vol_real)))
                except Exception:
                    pass

            if not (_is_num(bid) and _is_num(ask) and _is_num(last)):
                self.audit.warning(f"[TICK_NONFINITE] {symbol} bid={bid} ask={ask} last_raw={last_raw_f} last={last}")
                return None
            if bid <= 0.0 or ask <= 0.0:
                self.audit.warning(f"[TICK_NONPOS_BIDASK] {symbol} bid={bid} ask={ask} last_raw={last_raw_f} last={last}")
                return None
            if bid >= ask:
                self.audit.warning(f"[TICK_BAD_SPREAD] {symbol} bid={bid} ask={ask} last_raw={last_raw_f} last={last}")
                return None
            if last <= 0.0:
                self.audit.warning(f"[TICK_NONPOS_LAST] {symbol} bid={bid} ask={ask} last_raw={last_raw_f} mid={mid} last={last}")
                return None

            return {"bid": bid, "ask": ask, "last": last, "time": time_dt, "volume": vol}

        except Exception as e:
            self.audit.error(f"[FETCH_TICK_FAIL] {symbol} {type(e).__name__}: {e}")
            return None

    def _timeframe_to_seconds(self, tf: str) -> int:
        tf_u = tf.upper()
        if tf_u.startswith("M"):
            n = _to_int(tf_u[1:], 1)
            return max(1, n) * 60
        if tf_u.startswith("H"):
            n = _to_int(tf_u[1:], 1)
            return max(1, n) * 3600
        if tf_u == "D1":
            return 24 * 3600
        return 60

    def _floor_time_to_tf(self, dt: datetime.datetime, tf: str) -> datetime.datetime:
        secs = int(self._timeframe_to_seconds(tf))
        epoch = int(dt.replace(tzinfo=datetime.timezone.utc).timestamp())
        floored = epoch - (epoch % max(1, secs))
        return datetime.datetime.fromtimestamp(floored, tz=datetime.timezone.utc).replace(tzinfo=None)

    def _update_forming_bar(self, symbol: str) -> bool:
        if self.cfg.mode != "live":
            return False

        tick = self._fetch_current_tick(symbol)
        if not tick:
            return False

        if self.cfg.debug:
            self.audit.debug(
                f"[TICK] {symbol} t={_iso_ts(tick.get('time'))} bid={tick.get('bid')} ask={tick.get('ask')} last={tick.get('last')} vol={tick.get('volume')} offset_s={self._mt5_server_utc_offset_s}"
            )

        ptf = self.primary_tf.get(symbol, self.cfg.primary_timeframe)
        store = self.tfs.get(symbol, {}).get(ptf)
        if store is None or store.n <= 0:
            self.audit.warning(f"[FORMING_NO_STORE] {symbol}/{ptf}")
            return False

        threshold = float(self.cfg.min_tick_change_for_recalc)
        last_tick = self.last_tick_prices.get(symbol, {})

        mid = (_to_float(tick["bid"]) + _to_float(tick["ask"])) / 2.0

        if last_tick:
            bid_change = abs(_to_float(tick["bid"]) - _to_float(last_tick.get("bid"), _to_float(tick["bid"])))
            ask_change = abs(_to_float(tick["ask"]) - _to_float(last_tick.get("ask"), _to_float(tick["ask"])))
            mid_change = abs(mid - _to_float(last_tick.get("mid"), mid))


            if threshold <= 0.0:
                meaningful = (bid_change > 0.0) or (ask_change > 0.0) or (mid_change > 0.0)
            else:
                meaningful = (bid_change >= threshold) or (ask_change >= threshold)
        else:
            meaningful = True

        self.last_tick_prices[symbol] = {
            "bid": _to_float(tick["bid"]),
            "ask": _to_float(tick["ask"]),
            "last": _to_float(tick["last"]),
            "mid": float(mid),
            "last_update_ts": float(time.time()),
            "mt5_time_offset_s": int(self._mt5_server_utc_offset_s),
        }

        idx = self.ptr_primary.get(symbol, store.n - 1)
        idx = max(0, min(int(idx), store.n - 1))
        last_closed = self._bar_from_store(store, idx)

        bar_open = self._floor_time_to_tf(cast(datetime.datetime, tick["time"]), ptf)

        existing = self.current_forming_bars.get(symbol)
        if existing and isinstance(existing.get("timestamp"), datetime.datetime):
            if self._floor_time_to_tf(cast(datetime.datetime, existing["timestamp"]), ptf) != bar_open:
                existing = None

        if existing:
            fb = existing
            fb["high"] = max(_to_float(fb.get("high"), mid), mid)
            fb["low"] = min(_to_float(fb.get("low"), mid), mid)
            fb["close"] = mid
            fb["bid"] = _to_float(tick["bid"])
            fb["ask"] = _to_float(tick["ask"])
            fb["volume"] = _to_int(fb.get("volume"), 0) + _to_int(tick.get("volume"), 0)
        else:
            self.current_forming_bars[symbol] = {
                "timestamp": bar_open,
                "open": _to_float(last_closed.get("close"), mid),
                "high": max(_to_float(last_closed.get("close"), mid), mid),
                "low": min(_to_float(last_closed.get("close"), mid), mid),
                "close": mid,
                "volume": _to_int(tick.get("volume"), 0),
                "bid": _to_float(tick["bid"]),
                "ask": _to_float(tick["ask"]),
            }

        self.current_bars[symbol] = dict(self.current_forming_bars[symbol])


        if bool(getattr(self.cfg, "include_forming_bar_in_primary_tf", True)):
            pb = self.price_buffers.get(symbol)
            if pb and len(pb["close"]) > 0:
                fb2 = self.current_forming_bars[symbol]
                pb["close"][-1] = _to_float(fb2.get("close"))
                pb["high"][-1] = _to_float(fb2.get("high"))
                pb["low"][-1] = _to_float(fb2.get("low"))
                pb["volume"][-1] = _to_int(fb2.get("volume"), 0)

            if self.cfg.enable_technical_indicators:
                self._update_technical_indicators(symbol)

        return bool(meaningful)


    def _load_data_files(self) -> None:
        data_dir = self.cfg.data_directory
        if not os.path.exists(data_dir):
            self.audit.warning(f"[DATA_DIR_MISSING] {data_dir}")
            return

        sym = "XAUUSD"
        per_tf: Dict[str, pd.DataFrame] = {}

        for tf in self.cfg.supported_timeframes:
            path = self._find_file_for(sym, tf, data_dir)
            if not path:
                self.audit.warning(f"[CSV_MISSING] {sym}/{tf} expected file not found")
                continue
            try:
                df = pd.read_csv(path)

                if "timestamp" not in df.columns and "time" in df.columns:
                    df = df.rename(columns={"time": "timestamp"})
                if "timestamp" not in df.columns:
                    self.audit.warning(f"[CSV_BAD] {sym}/{tf} missing timestamp: {os.path.basename(path)}")
                    continue

                need_cols = {"open", "high", "low", "close"}
                if not need_cols.issubset(df.columns):
                    if "close" in df.columns:
                        df["open"] = df["close"] if "open" not in df.columns else df["open"]
                        df["high"] = df["close"] if "high" not in df.columns else df["high"]
                        df["low"] = df["close"] if "low" not in df.columns else df["low"]
                        self.audit.info(f"[CSV_FILL_OHLC] {sym}/{tf} filled O/H/L from close")
                    else:
                        self.audit.warning(f"[CSV_NO_CLOSE] {sym}/{tf} skipping (no close)")
                        continue

                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

                for col in ["open", "high", "low", "close"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                if "volume" in df.columns:
                    vol_series = pd.to_numeric(df["volume"], errors="coerce")
                else:
                    vol_series = pd.Series([0] * len(df), index=df.index, dtype="float64")
                df["volume"] = vol_series.fillna(0).astype("int64")

                df.dropna(subset=["timestamp", "close"], inplace=True)
                df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

                if df.empty:
                    self.audit.warning(f"[CSV_EMPTY] {sym}/{tf} empty after cleaning: {os.path.basename(path)}")
                    continue

                per_tf[tf] = df[["timestamp", "open", "high", "low", "close", "volume"]]
                self.audit.info(f"[CSV_OK] {sym}/{tf} bars={len(df)} file={os.path.basename(path)}")

            except Exception as e:
                self.audit.error(f"[CSV_FAIL] {sym}/{tf} {type(e).__name__}: {e}")

        if per_tf:
            self.data_files[sym] = per_tf
        else:
            self.audit.warning("[CSV_SUMMARY] no valid CSVs loaded")

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
            if not candidates:
                return None
            candidates.sort(key=lambda p: ("_features" not in os.path.basename(p), os.path.basename(p)))
            return candidates[0]
        except Exception as e:
            self.audit.error(f"[FIND_FILE_FAIL] {type(e).__name__}: {e}")
            return None

    def _materialize_tfstores(self) -> None:
        sym = "XAUUSD"
        per_tf = self.data_files.get(sym, {})
        self.tfs[sym] = {}
        for tf, df in per_tf.items():
            self.tfs[sym][tf] = _TFStore(df)

    def _init_pointers_and_buffers(self) -> None:
        sym = "XAUUSD"
        available_tfs = list(self.tfs.get(sym, {}).keys())
        if not available_tfs:
            return

        self.primary_tf[sym] = self.cfg.primary_timeframe if self.cfg.primary_timeframe in available_tfs else available_tfs[0]
        ptf_store = self.tfs[sym][self.primary_tf[sym]]

        init_idx = min(max(0, int(self.cfg.window_min)), max(0, ptf_store.n - 1))
        self.ptr_primary[sym] = init_idx

        self.ptrs_by_tf[sym] = {}
        for tf in available_tfs:
            self.ptrs_by_tf[sym][tf] = init_idx

        self.price_buffers[sym] = {
            "close": deque(maxlen=self.cfg.buffer_size),
            "high": deque(maxlen=self.cfg.buffer_size),
            "low": deque(maxlen=self.cfg.buffer_size),
            "volume": deque(maxlen=self.cfg.buffer_size),
        }

        self._seed_buffers_from_store(sym)


    def _initialize_technical_indicators(self) -> None:
        self.technical_indicators["XAUUSD"] = {
            "sma_20": 0.0, "sma_50": 0.0,
            "rsi": 0.0, "atr": 0.0,
            "bollinger_upper": 0.0, "bollinger_lower": 0.0, "bollinger_middle": 0.0,
            "macd": 0.0, "macd_signal": 0.0,
            "stochastic": 0.0,
        }

    def _seed_buffers_from_store(self, symbol: str) -> None:
        ptf = self.primary_tf.get(symbol, self.cfg.primary_timeframe)
        store = self.tfs.get(symbol, {}).get(ptf)
        pb = self.price_buffers.get(symbol)
        if store is None or pb is None or store.n <= 0:
            self.audit.warning(f"[SEED_SKIP] store/pb missing for {symbol}/{ptf}")
            return

        end_idx = self.ptr_primary.get(symbol, store.n - 1)
        end_idx = max(0, min(int(end_idx), store.n - 1))

        win = min(int(self.cfg.buffer_size), end_idx + 1, store.n)
        start = max(0, end_idx - (win - 1))
        sl = slice(start, end_idx + 1)

        pb["close"].clear()
        pb["high"].clear()
        pb["low"].clear()
        pb["volume"].clear()

        pb["close"].extend([float(x) for x in store.close[sl]])
        pb["high"].extend([float(x) for x in store.high[sl]])
        pb["low"].extend([float(x) for x in store.low[sl]])
        pb["volume"].extend([int(x) for x in store.volume[sl]])

    def _ema(self, data: List[float], period: int) -> float:
        if not data:
            return 0.0
        if len(data) < period:
            return float(np.mean(data))
        alpha = 2.0 / (period + 1)
        ema = float(data[-period])
        for price in data[-period + 1:]:
            ema = alpha * float(price) + (1 - alpha) * ema
        return ema

    def _update_technical_indicators(self, symbol: str) -> None:
        pb = self.price_buffers.get(symbol)
        if not pb:
            return

        closes = list(pb["close"])
        highs = list(pb["high"])
        lows = list(pb["low"])
        ind = self.technical_indicators.setdefault(symbol, {})

        if len(closes) >= 20:
            ind["sma_20"] = float(np.mean(closes[-20:]))
        if len(closes) >= 50:
            ind["sma_50"] = float(np.mean(closes[-50:]))

        if len(closes) >= 15:
            deltas = np.diff(closes[-15:])
            gains = np.where(deltas > 0, deltas, 0.0)
            losses = np.where(deltas < 0, -deltas, 0.0)
            alpha = 1.0 / 14.0
            avg_gain = float(gains[0]) if len(gains) > 0 else 0.0
            avg_loss = float(losses[0]) if len(losses) > 0 else 0.0
            for i in range(1, len(gains)):
                avg_gain = alpha * float(gains[i]) + (1 - alpha) * avg_gain
                avg_loss = alpha * float(losses[i]) + (1 - alpha) * avg_loss
            if avg_loss <= 1e-12:
                rsi = 100.0 if avg_gain > 0 else 50.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100.0 - (100.0 / (1.0 + rs))
            ind["rsi"] = float(np.clip(rsi, 0.0, 100.0))

        if len(closes) >= 15 and len(highs) >= 15 and len(lows) >= 15:
            trs: List[float] = []
            for i in range(1, 15):
                pc = closes[-(i + 1)]
                hi = highs[-i]
                lo = lows[-i]
                tr = max(hi - lo, abs(hi - pc), abs(lo - pc))
                trs.append(float(tr))
            ind["atr"] = float(np.mean(trs)) if trs else 0.0

        if len(closes) >= 20:
            sma20 = float(np.mean(closes[-20:]))
            std20 = float(np.std(closes[-20:]))
            ind["bollinger_middle"] = sma20
            ind["bollinger_upper"] = sma20 + 2.0 * std20
            ind["bollinger_lower"] = sma20 - 2.0 * std20

        if len(closes) >= 26:
            ema12 = self._ema(closes, 12)
            ema26 = self._ema(closes, 26)
            macd_line = float(ema12 - ema26)
            ind["macd"] = macd_line
            ind["macd_signal"] = float(macd_line * 0.8)

        if len(closes) >= 14 and len(highs) >= 14 and len(lows) >= 14:
            highest_high = max(highs[-14:])
            lowest_low = min(lows[-14:])
            current_close = closes[-1]
            rng = highest_high - lowest_low
            stoch_k = 50.0 if rng <= 1e-12 else ((current_close - lowest_low) / rng) * 100.0
            ind["stochastic"] = float(np.clip(stoch_k, 0.0, 100.0))


    def _setup_initial_conditions(self) -> None:
        self.current_timestamp = _now_utc_naive()

        if self.cfg.mode == "live":

            self._update_current_bar_from_store("XAUUSD")
            self._update_session_labels()
            return


        self._advance_symbol_data("XAUUSD")


    def _advance_symbol_data(self, symbol: str) -> bool:
        with self._lock:
            if self.cfg.mode == "live":
                return self._advance_symbol_data_live(symbol)
            return self._advance_symbol_data_training(symbol)

    def _advance_symbol_data_live(self, symbol: str) -> bool:

        refreshed = self._refresh_live_data(symbol)


        changed = self._update_current_bar_from_store(symbol)
        return bool(refreshed or changed)

    def _advance_symbol_data_training(self, symbol: str) -> bool:
        if symbol not in self.tfs or not self.tfs[symbol]:
            return False

        ptf = self.primary_tf[symbol]
        store = self.tfs[symbol].get(ptf)
        if store is None or store.n == 0:
            return False

        idx = int(self.ptr_primary[symbol]) + 1
        if idx >= store.n:
            idx = min(max(0, int(self.cfg.window_min)), max(0, store.n - 1))
            self.audit.debug(f"[DATA_WRAP] {symbol} -> idx={idx}")

        self.ptr_primary[symbol] = idx
        ts_np = store.ts[idx]

        for tf, tstore in self.tfs[symbol].items():
            pos = int(np.searchsorted(tstore.ts, ts_np, side="right") - 1)
            pos = max(0, min(pos, tstore.n - 1))
            self.ptrs_by_tf[symbol][tf] = pos

        return self._update_current_bar_from_store(symbol)

    def _bar_from_store(self, store: _TFStore, i: int) -> Dict[str, Any]:
        ts_py = pd.Timestamp(store.ts[i]).to_pydatetime()
        return {
            "timestamp": ts_py,
            "open": float(store.open[i]),
            "high": float(store.high[i]),
            "low": float(store.low[i]),
            "close": float(store.close[i]),
            "volume": int(store.volume[i]),
            "bid": None,
            "ask": None,
        }

    def _window_from_store(
        self, store: _TFStore, end_idx: int
    ) -> Tuple[List[float], List[float], List[float], List[float], List[int]]:
        win = min(int(self.cfg.window_max), end_idx + 1, store.n)
        start = max(0, end_idx - (win - 1))
        sl = slice(start, end_idx + 1)
        return (
            store.open[sl].astype(np.float64).tolist(),
            store.high[sl].astype(np.float64).tolist(),
            store.low[sl].astype(np.float64).tolist(),
            store.close[sl].astype(np.float64).tolist(),
            store.volume[sl].astype(np.int64).tolist(),
        )

    def _update_current_bar_from_store(self, symbol: str) -> bool:
        ptf = self.primary_tf.get(symbol, self.cfg.primary_timeframe)
        store = self.tfs.get(symbol, {}).get(ptf)
        if store is None or store.n <= 0:
            return False

        idx = self.ptr_primary.get(symbol, 0)
        idx = max(0, min(int(idx), store.n - 1))
        cur = self._bar_from_store(store, idx)


        prev = self.current_bars.get(symbol)
        prev_ts = prev.get("timestamp") if isinstance(prev, dict) else None
        prev_close = _to_float(prev.get("close"), 0.0) if isinstance(prev, dict) else 0.0

        cur_ts = cur.get("timestamp")
        cur_close = _to_float(cur.get("close"), 0.0)

        changed = (prev_ts != cur_ts) or (abs(cur_close - prev_close) > 1e-12)


        self.current_bars[symbol] = cur

        pb = self.price_buffers.get(symbol)
        if pb:
            if len(pb["close"]) == 0:
                self._seed_buffers_from_store(symbol)
            else:
                if self.cfg.mode == "live":

                    pb["close"][-1] = float(cur["close"])
                    pb["high"][-1] = float(cur["high"])
                    pb["low"][-1] = float(cur["low"])
                    pb["volume"][-1] = int(cur["volume"])
                else:

                    pb["close"].append(float(cur["close"]))
                    pb["high"].append(float(cur["high"]))
                    pb["low"].append(float(cur["low"]))
                    pb["volume"].append(int(cur["volume"]))

        if self.cfg.enable_technical_indicators:
            self._update_technical_indicators(symbol)

        return bool(changed)


    def _is_market_hours(self) -> bool:
        if self.cfg.mode != "live":
            return True
        now = _now_utc_naive()
        wd = now.weekday()
        if wd >= 5:
            return False
        if wd == 4 and now.hour >= 21:
            return False
        return True

    def _update_session_labels(self) -> None:
        ref_time = _now_utc_naive()


        if self.cfg.mode != "live" and self.current_bars:
            ts_list: List[datetime.datetime] = []
            for bar in self.current_bars.values():
                ts = bar.get("timestamp") if isinstance(bar, dict) else None
                if isinstance(ts, datetime.datetime):
                    ts_list.append(ts)
            if ts_list:
                ref_time = max(ts_list)

        hour = ref_time.hour
        active: List[str] = []
        if 8 <= hour < 16:
            active.append("london")
        if 13 <= hour < 21:
            active.append("new_york")
        if 21 <= hour or hour < 6:
            active.append("sydney")
        if 0 <= hour < 8:
            active.append("tokyo")
        self.active_sessions = active if active else ["closed"]

        if "london" in active and "new_york" in active:
            self.trading_session = "london_newyork_overlap"
        elif "london" in active:
            self.trading_session = "london"
        elif "new_york" in active:
            self.trading_session = "new_york"
        elif "sydney" in active and "tokyo" in active:
            self.trading_session = "asia_overlap"
        elif "tokyo" in active:
            self.trading_session = "tokyo"
        elif "sydney" in active:
            self.trading_session = "sydney"
        else:
            self.trading_session = "closed"

        if 13 <= hour < 16:
            self.session_type = "london_ny_overlap"
        elif 9 <= hour < 17:
            self.session_type = "main"
        elif 17 <= hour < 21:
            self.session_type = "late"
        else:
            self.session_type = "overnight"

        self.current_timestamp = ref_time


    async def calculate_confidence(self, action: Optional[Dict[str, Any]] = None, **inputs: Any) -> float:
        try:
            bar = self.current_bars.get("XAUUSD")
            if not isinstance(bar, dict):
                return 0.0

            age_s = (time.time() - self._last_update_ts) if self._last_update_ts > 0 else 9999.0
            freshness = max(0.0, 1.0 - age_s / 60.0)

            hi = _to_float(bar.get("high"), 0.0)
            lo = _to_float(bar.get("low"), 0.0)
            cl = _to_float(bar.get("close"), 0.0)
            quality = 1.0
            if hi < lo or cl <= 0:
                quality = 0.2

            return float(np.clip(0.6 * freshness + 0.4 * quality, 0.0, 1.0))
        except Exception:
            return 0.0

    async def propose_action(self, **inputs: Any) -> Dict[str, Any]:
        return {
            "update_data": True,
            "symbols_to_update": ["XAUUSD"],
            "maintenance_required": (self._update_count > 0 and self._update_count % 2000 == 0),
            "maintenance_type": "buffer_cleanup" if (self._update_count > 0 and self._update_count % 2000 == 0) else None,
            "data_quality": await self.calculate_confidence(),
        }


    async def process(self, **inputs: Any) -> Dict[str, Any]:
        t0 = time.time()
        pid = self._update_count + 1

        self.audit.set_debug(bool(self.cfg.debug))

        if self.cfg.debug:
            self.audit.debug(f"[CYCLE_START] pid={pid} mode={self.cfg.mode} inputs={_safe_json(self._sanitize_inputs(inputs))}")
        else:
            if pid % self.audit.log_every_n == 0:
                self.audit.info(f"[CYCLE] pid={pid} mode={self.cfg.mode} keys={list(inputs.keys())}")

        errors: Dict[str, str] = {}

        try:
            should_refresh = True
            if self.cfg.mode == "live" and self.cfg.update_frequency > 0 and self._last_update_ts > 0:
                if (time.time() - self._last_update_ts) < float(self.cfg.update_frequency):
                    should_refresh = False

            sym = "XAUUSD"
            any_price_changed = False


            if self.cfg.mode == "live":
                if not self._mt5_connected:
                    try:
                        self._connect_mt5()
                    except Exception as e:
                        errors["connection"] = str(e)


            if should_refresh:
                try:
                    ok = self._advance_symbol_data(sym)
                    any_price_changed = any_price_changed or bool(ok)
                    self._last_update_ts = time.time()
                except Exception as e:
                    errors["advance"] = str(e)
                    self.audit.error(f"[ADVANCE_FAIL] {type(e).__name__}: {e}")
            else:

                try:
                    ok2 = self._update_current_bar_from_store(sym)
                    any_price_changed = any_price_changed or bool(ok2)
                except Exception as e:
                    errors["update_current"] = str(e)
                    self.audit.error(f"[UPDATE_BAR_FAIL] {type(e).__name__}: {e}")


            if self.cfg.mode == "live":
                try:
                    if self._update_forming_bar(sym):
                        any_price_changed = True
                except Exception as e:
                    errors["forming_bar"] = str(e)
                    self.audit.error(f"[FORMING_BAR_FAIL] {type(e).__name__}: {e}")


            with self._lock:
                self._update_count += 1
                dt_ms = (time.time() - t0) * 1000.0
                self._ema_ms = dt_ms if self._ema_ms <= 0.0 else (0.9 * self._ema_ms + 0.1 * dt_ms)

            self._update_session_labels()

            snapshot = self._build_snapshot()
            snapshot["universe"] = ["XAUUSD"]
            snapshot["watched_instruments"] = ["XAUUSD"]

            ms_since_last = (time.time() - self._last_update_ts) * 1000.0 if self._last_update_ts else None
            market_open = self._is_market_hours()

            wallclock_iso = _iso_ts(_now_utc_naive())

            snapshot["provider_status"] = {
                "mode": self.cfg.mode,
                "update_count": int(self._update_count),
                "last_update_ts": float(self._last_update_ts),
                "last_update_iso": wallclock_iso,
                "ms_since_last": ms_since_last,
                "symbol_errors": errors,
                "fail_count": int(self._fail),
                "success_count": int(self._success),
                "mt5_connected": self._mt5_connected if self.cfg.mode == "live" else None,
                "market_open": market_open,
                "price_changed": bool(any_price_changed) if self.cfg.mode == "live" else True,
                "ema_ms": float(self._ema_ms),
                "mt5_time_offset_s": int(self._mt5_server_utc_offset_s) if self.cfg.mode == "live" else None,
            }
            if self.last_tick_prices:
                snapshot["tick_prices"] = dict(self.last_tick_prices)


            snapshot["module_insights"] = {
                "provider": "MarketDataProvider",
                "mode": self.cfg.mode,
                "symbols": ["XAUUSD"],
                "timeframes": list(self.cfg.supported_timeframes),
                "primary_timeframe": self.primary_tf.get("XAUUSD", self.cfg.primary_timeframe),
                "update_count": int(self._update_count),
                "market_open": bool(market_open),
                "ms_since_last": ms_since_last,
                "ema_ms": float(self._ema_ms),
                "mt5_connected": self._mt5_connected if self.cfg.mode == "live" else None,
                "symbol_errors": errors,
                "mt5_time_offset_s": int(self._mt5_server_utc_offset_s) if self.cfg.mode == "live" else None,
                "last_update_iso": wallclock_iso,
            }

            self._publish_snapshot(snapshot, pid)

            conf = await self.calculate_confidence()
            if self.cfg.debug:
                self.audit.debug(f"[CYCLE_END] pid={pid} conf={conf:.3f} output={_safe_json(self._sanitize_outputs(snapshot))}")
            else:
                if pid % self.audit.log_every_n == 0:
                    md = snapshot.get("market_data", {}).get("XAUUSD", {})
                    self.audit.info(
                        f"[SUMMARY] pid={pid} close={_to_float(md.get('close'),0.0):.5f} "
                        f"session={snapshot.get('trading_session')} conf={conf:.3f}"
                    )

            self.audit.ndjson(pid, "snapshot_meta", {
                "tick": int(self._update_count),
                "mode": self.cfg.mode,
                "ema_ms": round(float(self._ema_ms), 3),
                "errors": errors,
                "mt5_time_offset_s": int(self._mt5_server_utc_offset_s) if self.cfg.mode == "live" else None,
            })

            self._success += 1
            return snapshot

        except Exception as e:
            self._fail += 1
            self.audit.error(f"[PROCESS_FAIL] pid={pid} {type(e).__name__}: {e}")
            empty = self._empty_snapshot(error=str(e))
            empty["universe"] = ["XAUUSD"]
            empty["watched_instruments"] = ["XAUUSD"]
            empty["provider_status"] = {
                "mode": self.cfg.mode,
                "update_count": int(self._update_count),
                "last_error": str(e),
                "fail_count": int(self._fail),
                "success_count": int(self._success),
                "mt5_connected": self._mt5_connected if self.cfg.mode == "live" else None,
                "mt5_time_offset_s": int(self._mt5_server_utc_offset_s) if self.cfg.mode == "live" else None,
            }
            return empty


    def _build_snapshot(self) -> Dict[str, Any]:
        sym = "XAUUSD"

        multi_tf: Dict[str, Dict[str, Any]] = {}
        if sym in self.tfs:
            multi_tf[sym] = {}
            for tf, store in self.tfs[sym].items():
                end_idx = self.ptrs_by_tf.get(sym, {}).get(tf, store.n - 1)
                end_idx = max(0, min(int(end_idx), store.n - 1))
                o, h, l, c, v = self._window_from_store(store, end_idx)
                cur = self._bar_from_store(store, end_idx)

                if (
                    self.cfg.mode == "live"
                    and tf == self.primary_tf.get(sym, self.cfg.primary_timeframe)
                    and bool(getattr(self.cfg, "include_forming_bar_in_primary_tf", True))
                ):
                    fb = self.current_forming_bars.get(sym)
                    if isinstance(fb, dict) and len(c) > 0:
                        c = list(c); h = list(h); l = list(l); v = list(v)
                        c[-1] = _to_float(fb.get("close"), c[-1])
                        h[-1] = max(_to_float(h[-1]), _to_float(fb.get("high"), h[-1]))
                        l[-1] = min(_to_float(l[-1]), _to_float(fb.get("low"), l[-1]))
                        v[-1] = _to_int(fb.get("volume"), v[-1])
                        cur = dict(fb)

                cur = self._decorate_intrabar(sym, tf, cur)
                rec = {
                    "open": o, "high": h, "low": l, "close": c, "volume": v,
                    "current_bar": {k: (_iso_ts(cur[k]) if k == "timestamp" else cur[k]) for k in cur.keys()},
                    "timeframe": tf,
                    "bars_available": int(store.n),
                }
                multi_tf[sym][tf] = rec

        market_data: Dict[str, Dict[str, Any]] = {}
        bar = self.current_bars.get(sym)
        if isinstance(bar, dict):
            tf_for_sym = self.primary_tf.get(sym, self.cfg.primary_timeframe)
            decorated = self._decorate_intrabar(sym, tf_for_sym, dict(bar))
            market_data[sym] = self._bar_with_iso(decorated)

        price_data: Dict[str, Any] = {}
        ohlcv_data: Dict[str, Any] = {}
        if sym in market_data:
            md = market_data[sym]
            price_data[sym] = {
                "last": _to_float(md.get("close"), 0.0),
                "close": _to_float(md.get("close"), 0.0),
                "open": _to_float(md.get("open"), 0.0),
                "high": _to_float(md.get("high"), 0.0),
                "low": _to_float(md.get("low"), 0.0),
            }
            ohlcv_data[sym] = {
                "open": _to_float(md.get("open"), 0.0),
                "high": _to_float(md.get("high"), 0.0),
                "low": _to_float(md.get("low"), 0.0),
                "close": _to_float(md.get("close"), 0.0),
                "volume": _to_int(md.get("volume"), 0),
            }

        bid_ask_data: Dict[str, Any] = {}
        spread: Optional[float] = None
        if sym in market_data:
            md = market_data[sym]
            bid = md.get("bid")
            ask = md.get("ask")
            if isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and bid > 0 and ask > 0:
                spread = float(ask - bid)
            if spread is None:
                spread = self._calculate_spread(md)
            bid_ask_data[sym] = {"bid": bid, "ask": ask, "spread": spread}

        atr = _to_float(self.technical_indicators.get(sym, {}).get("atr"), 0.0)
        last = _to_float(market_data.get(sym, {}).get("close"), 0.0)
        vol = float(atr / max(last, 1e-9)) if last > 0 else 0.0

        vol_data = {sym: {"atr": atr, "volatility": vol}}
        vol_level_by_instrument = {sym: ("high" if vol > 0.02 else ("medium" if vol > 0.01 else "low"))}
        vol_by_instrument = {sym: vol}
        vol_level = vol_level_by_instrument[sym]


        last_vol = _to_int(ohlcv_data.get(sym, {}).get("volume"), 0) if isinstance(ohlcv_data.get(sym), dict) else 0
        volume_level = "high" if last_vol >= 3000 else ("medium" if last_vol >= 800 else "low")
        volume_data = {sym: {"volume": int(last_vol), "volume_level": volume_level}}

        ts_candidates: List[datetime.datetime] = []
        for v_ in self.current_bars.values():
            if isinstance(v_, dict):
                ts = v_.get("timestamp")
                if isinstance(ts, datetime.datetime):
                    ts_candidates.append(ts)
        self.current_timestamp = max(ts_candidates) if ts_candidates else _now_utc_naive()
        ts_iso = _iso_ts(self.current_timestamp) or _iso_ts(_now_utc_naive())

        indicators_map = {sym: {k: float(v) for k, v in self.technical_indicators.get(sym, {}).items()}}

        ms_since_last = (time.time() - float(self._last_update_ts)) * 1000.0 if self._last_update_ts else None
        market_open = self._is_market_hours()

        liq_score, liq_details = self._compute_market_liquidity(
            sym,
            spread=spread if isinstance(spread, (int, float)) else None,
            volume=int(last_vol),
            volatility=float(vol),
            market_open=bool(market_open),
            ms_since_last=ms_since_last,
        )


        liquidity_data = {
            sym: {
                "market_liquidity": float(liq_score),
                "liquidity_level": liq_details.get("level"),
                "spread": liq_details.get("spread"),
                "relative_spread": liq_details.get("relative_spread"),
            }
        }

        snapshot: Dict[str, Any] = {
            "bid_ask_data": bid_ask_data,
            "historical_prices": multi_tf,
            "indicators": indicators_map,
            "market_data": market_data,
            "multi_timeframe_data": multi_tf,
            "ohlcv_data": ohlcv_data,
            "price_data": price_data,
            "prices": {sym: _to_float(price_data.get(sym, {}).get("last"), 0.0)} if sym in price_data else {},
            "session_type": self.session_type,
            "step_idx": int(self._update_count),
            "symbols": [sym],
            "technical_indicators": indicators_map,
            "timestamp": ts_iso,
            "trading_session": self.trading_session,


            "volatility": float(vol),
            "volatility_data": vol_data,
            "volatility_level": vol_level,
            "volatility_level_by_instrument": vol_level_by_instrument,
            "volatility_by_instrument": vol_by_instrument,
            "volume_data": volume_data,
            "market_liquidity": float(liq_score),
            "liquidity_data": liquidity_data,


            "market_liquidity_details": liq_details,

            "market_data_latest": {
                sym: {
                    "open": _to_float(market_data.get(sym, {}).get("open"), 0.0),
                    "high": _to_float(market_data.get(sym, {}).get("high"), 0.0),
                    "low": _to_float(market_data.get(sym, {}).get("low"), 0.0),
                    "close": _to_float(market_data.get(sym, {}).get("close"), 0.0),
                    "volume": _to_float(market_data.get(sym, {}).get("volume"), 0.0),
                    "timestamp": market_data.get(sym, {}).get("timestamp", ts_iso),
                }
            },
            "atr_values": {sym: atr},
            "session_info": {
                "hour": self.current_timestamp.hour if self.current_timestamp else 12,
                "minute": self.current_timestamp.minute if self.current_timestamp else 0,
                "weekday": self.current_timestamp.weekday() if self.current_timestamp else 0,
                "session_type": self.session_type,
                "trading_session": self.trading_session,
                "timestamp": ts_iso,
            },
        }


        alias_payload = self._build_alias_map(snapshot)
        snapshot.update(alias_payload)

        return snapshot

    def _build_alias_map(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        mtd = snapshot.get("multi_timeframe_data", {}) or {}
        sym = "XAUUSD"
        per_tf = mtd.get(sym, {}) if isinstance(mtd, dict) else {}

        for tf in self.cfg.supported_timeframes:
            alias_key = self._alias_key(sym, tf)
            if self._contract_provides and alias_key not in self._contract_provides:
                continue
            cur_bar: Dict[str, Any] = {}
            if isinstance(per_tf, dict):
                tf_payload = per_tf.get(tf, {})
                if isinstance(tf_payload, dict):
                    cur_bar = tf_payload.get("current_bar", {}) or {}
            out[alias_key] = cur_bar if isinstance(cur_bar, dict) else {}
        return out

    def _alias_key(self, symbol: str, timeframe: str) -> str:
        sym_clean = symbol.replace("/", "_")
        return f"market_data_{sym_clean}_{timeframe}"

    def _bar_with_iso(self, bar: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(bar)
        out["timestamp"] = _iso_ts(out.get("timestamp"))
        return out

    def _decorate_intrabar(self, symbol: str, timeframe: str, bar: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(bar)
        try:
            tf_secs = float(self._timeframe_to_seconds(timeframe))
            ts = out.get("timestamp")
            if isinstance(ts, datetime.datetime):
                ts_dt = ts
            elif isinstance(ts, str) and ts:
                try:
                    ts_dt = datetime.datetime.fromisoformat(ts)
                except Exception:
                    ts_dt = _now_utc_naive()
            else:
                ts_dt = _now_utc_naive()


            bar_state = "closed"
            if self.cfg.mode == "live" and self._is_market_hours():
                if timeframe == self.primary_tf.get(symbol, self.cfg.primary_timeframe):
                    fb = self.current_forming_bars.get(symbol)
                    fb_ts = fb.get("timestamp") if isinstance(fb, dict) else None
                    if isinstance(fb_ts, datetime.datetime):
                        try:
                            fb_open = self._floor_time_to_tf(fb_ts, timeframe)
                            bar_open = self._floor_time_to_tf(ts_dt, timeframe)
                            if bar_open == fb_open:
                                bar_state = "forming"
                        except Exception:

                            bar_state = "closed"


            if self.cfg.mode != "live":
                progress = 1.0
            elif bar_state == "closed":
                progress = 1.0
            else:
                now = _now_utc_naive()
                elapsed = max(0.0, (now - ts_dt).total_seconds())
                progress = float(np.clip(elapsed / max(tf_secs, 1.0), 0.0, 1.0))

            low = _to_float(out.get("low"), 0.0)
            high = _to_float(out.get("high"), 0.0)
            close = _to_float(out.get("close"), 0.0)
            open_ = _to_float(out.get("open"), 0.0)

            rng = max(high - low, 1e-9)
            out["progress_in_bar"] = progress
            out["position_in_range"] = float(np.clip((close - low) / rng, 0.0, 1.0))
            out["body_relative"] = float(np.clip(abs(close - open_) / rng, 0.0, 1.0))
            out["bar_state"] = bar_state
        except Exception as e:
            self.audit.warning(f"[INTRABAR_DECOR_FAIL] {type(e).__name__}: {e}")
        return out

    def _calculate_spread(self, bar: Dict[str, Any]) -> Optional[float]:
        bid = bar.get("bid")
        ask = bar.get("ask")
        if isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and bid > 0 and ask > 0:
            return float(ask - bid)
        close = _to_float(bar.get("close"), 0.0)
        if close <= 0:
            return None
        return close * 0.0003 if close > 1000 else close * 0.0001

    def _compute_market_liquidity(
        self,
        symbol: str,
        *,
        spread: Optional[float],
        volume: int,
        volatility: float,
        market_open: bool,
        ms_since_last: Optional[float],
    ) -> Tuple[float, Dict[str, Any]]:
        md = self.current_bars.get(symbol, {}) if isinstance(self.current_bars.get(symbol), dict) else {}
        price = _to_float(md.get("close"), 0.0)
        price = price if price > 0 else 2000.0

        spr = float(spread) if isinstance(spread, (int, float)) and math.isfinite(float(spread)) else None
        rel_spread = (spr / price) if (spr is not None and price > 0) else None

        if rel_spread is None:
            spread_score = 0.5
        else:
            spread_score = float(np.clip(1.0 - (rel_spread / 0.00040), 0.0, 1.0))

        vol_i = max(0, int(volume))
        volume_score = float(np.clip(np.log1p(vol_i) / np.log1p(5000), 0.0, 1.0))

        volat = float(volatility) if math.isfinite(float(volatility)) else 0.0
        vol_penalty = float(np.clip((max(0.0, volat - 0.02) / 0.04), 0.0, 1.0))
        vol_score = 1.0 - 0.5 * vol_penalty

        base = 0.55 * spread_score + 0.35 * volume_score + 0.10 * vol_score

        if not market_open:
            base *= 0.10

        if ms_since_last is not None and isinstance(ms_since_last, (int, float)) and math.isfinite(float(ms_since_last)):
            age_s = float(ms_since_last) / 1000.0
            stale = float(np.clip((age_s - 10.0) / 50.0, 0.0, 1.0))
            base *= (1.0 - 0.7 * stale)

        score = float(np.clip(base, 0.0, 1.0))
        level = "high" if score >= 0.70 else ("medium" if score >= 0.40 else "low")

        details = {
            "symbol": symbol,
            "score": score,
            "level": level,
            "spread": spr,
            "relative_spread": rel_spread,
            "volume": vol_i,
            "volatility": volat,
            "market_open": bool(market_open),
            "ms_since_last": float(ms_since_last) if ms_since_last is not None else None,
            "components": {
                "spread_score": spread_score,
                "volume_score": volume_score,
                "vol_score": vol_score,
            },
        }
        return score, details

    def _empty_snapshot(self, error: Optional[str] = None) -> Dict[str, Any]:
        now = _iso_ts(_now_utc_naive()) or _iso_ts(_now_utc_naive())
        sym = "XAUUSD"
        indicators_map = {sym: dict(self.technical_indicators.get(sym, {}))}

        empty_aliases = {self._alias_key(sym, tf): {} for tf in self.cfg.supported_timeframes}

        return {
            "bid_ask_data": {},
            "historical_prices": {},
            "indicators": {},
            "market_data": {},
            "market_liquidity": 0.0,
            "module_insights": {
                "provider": "MarketDataProvider",
                "symbols": [sym],
                "timeframes": list(self.cfg.supported_timeframes),
                "update_count": int(self._update_count),
                "last_error": error,
                "mt5_time_offset_s": int(self._mt5_server_utc_offset_s) if self.cfg.mode == "live" else None,
            },
            "multi_timeframe_data": {},
            "ohlcv_data": {},
            "price_data": {},
            "prices": {},
            "session_type": self.session_type,
            "step_idx": int(self._update_count),
            "symbols": [sym],
            "technical_indicators": indicators_map,
            "timestamp": now,
            "trading_session": self.trading_session,

            "volatility": 0.0,
            "volatility_data": {},
            "volatility_level": "low",
            "volume_data": {sym: {"volume": 0, "volume_level": "low"}},
            "liquidity_data": {sym: {"market_liquidity": 0.0, "liquidity_level": "low"}},

            "volatility_level_by_instrument": {sym: "low"},
            "volatility_by_instrument": {sym: 0.0},

            "market_data_latest": {},
            "atr_values": {sym: 0.0},
            "session_info": {
                "hour": 12,
                "minute": 0,
                "weekday": 0,
                "session_type": self.session_type,
                "trading_session": self.trading_session,
                "timestamp": now,
            },

            **empty_aliases,
        }


    def _bus_set(self, key: str, value: Any, *, thesis: str, confidence: float, pid: int) -> None:
        try:
            self.smart_bus.set(key, value, module="MarketDataProvider", thesis=thesis, confidence=float(confidence))
        except Exception as e:
            self.audit.error(f"[BUS_SET_FAIL] pid={pid} key={key} {type(e).__name__}: {e}")

    def _publish_snapshot(self, snapshot: Dict[str, Any], pid: int) -> None:
        core_keys = [
            "bid_ask_data", "historical_prices", "indicators",
            "market_data", "market_liquidity",
            "module_insights",
            "multi_timeframe_data", "ohlcv_data", "price_data", "prices",
            "session_type", "step_idx", "symbols",
            "technical_indicators", "timestamp", "trading_session",
            "volatility", "volatility_data", "volatility_level",
            "volume_data", "liquidity_data",
            "volatility_level_by_instrument", "volatility_by_instrument",
            "market_data_latest", "atr_values", "session_info",
            "universe", "watched_instruments",
            "provider_status", "tick_prices",
        ]
        for key in core_keys:
            if key in snapshot:
                self._bus_set(key, snapshot[key], thesis=f"Market data update {key}", confidence=0.8, pid=pid)

        alias_keys = [k for k in snapshot.keys() if k.startswith("market_data_XAUUSD_")]
        for k in alias_keys:
            if not self._contract_provides or k in self._contract_provides:
                self._bus_set(k, snapshot[k], thesis=f"Alias publish {k}", confidence=0.8, pid=pid)


    def _sanitize_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in inputs.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                out[k] = v
            elif isinstance(v, dict):
                out[k] = {"_type": "dict", "keys": list(v.keys())[:50]}
            elif isinstance(v, (list, tuple)):
                out[k] = {"_type": "list", "len": len(v)}
            else:
                out[k] = {"_type": type(v).__name__}
        return out

    def _sanitize_outputs(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        md = snapshot.get("market_data", {}).get("XAUUSD", {}) if isinstance(snapshot.get("market_data"), dict) else {}
        ps = snapshot.get("provider_status", {}) if isinstance(snapshot.get("provider_status"), dict) else {}
        out = {
            "step_idx": snapshot.get("step_idx"),
            "timestamp": snapshot.get("timestamp"),
            "mode": ps.get("mode"),
            "mt5_time_offset_s": ps.get("mt5_time_offset_s"),
            "last_update_iso": ps.get("last_update_iso"),
            "xauusd": {
                "open": _to_float(md.get("open"), 0.0),
                "high": _to_float(md.get("high"), 0.0),
                "low": _to_float(md.get("low"), 0.0),
                "close": _to_float(md.get("close"), 0.0),
                "bar_state": md.get("bar_state"),
            },
            "errors": ps.get("symbol_errors"),
        }
        return out


    def shutdown(self) -> None:
        if self.cfg.mode == "live":
            self._disconnect_mt5()
        self.audit.info(f"[SHUTDOWN] total={self._update_count} ok={self._success} fail={self._fail}")
        self.audit.info("[SHUTDOWN] MarketDataProvider shut down")
