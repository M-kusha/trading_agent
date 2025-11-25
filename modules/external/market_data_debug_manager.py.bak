# modules/external/debug/market_data_debug_manager.py
"""
Market Data Debug Manager
Human-centric, bracketed-line logger for MarketDataProvider, mirroring RewardDebugManager style.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from contextlib import contextmanager
from collections import deque, defaultdict
import threading
import time
import numpy as np


@dataclass
class _Levels:
    TRACE: int = 0
    DEBUG: int = 1
    INFO: int = 2
    WARNING: int = 3
    ERROR: int = 4


class MarketDataDebugManager:

    def __init__(
        self,
        enabled: bool = True,
        level: str = "INFO",
        logger: Optional[Any] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self.enabled: bool = enabled
        self._levels = {
            "TRACE": _Levels.TRACE,
            "DEBUG": _Levels.DEBUG,
            "INFO": _Levels.INFO,
            "WARNING": _Levels.WARNING,
            "ERROR": _Levels.ERROR,
        }
        self.set_level(level)

        self.logger = logger
        self.session_id = session_id or ""
        self._lock = threading.RLock()

        # Process index state (used for [p#N] bracket)
        self._current_proc_idx: Optional[int] = None

        # Rate limit registry: (level, context) -> last_ts
        self._last_log_time: Dict[Tuple[str, str], float] = {}

        # Rolling timings by operation
        self.operation_timings: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Simple counters
        self._csv_loaded_files: int = 0
        self._csv_loaded_bars: int = 0
        self._total_processes: int = 0
        self._successful_processes: int = 0
        self._failed_processes: int = 0

    # ───────────────────────────── Runtime controls ─────────────────────────────

    def enable(self) -> None:
        with self._lock:
            self.enabled = True

    def disable(self) -> None:
        with self._lock:
            self.enabled = False

    def set_level(self, level: str) -> None:
        lvl = self._levels.get(level.upper(), _Levels.INFO)
        self.current_level: int = lvl
        self.level_name: str = level.upper()

    # ───────────────────────────── Timers & Profilers ───────────────────────────

    @contextmanager
    def time_block(self, operation: str):
        start_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            end_ns = time.perf_counter_ns()
            ms = (end_ns - start_ns) / 1_000_000.0
            with self._lock:
                self.operation_timings[operation].append(ms)

    def profiled(self, operation: str):
        def decorator(fn):
            def wrapper(*args, **kwargs):
                start_ns = time.perf_counter_ns()
                try:
                    return fn(*args, **kwargs)
                finally:
                    end_ns = time.perf_counter_ns()
                    ms = (end_ns - start_ns) / 1_000_000.0
                    with self._lock:
                        self.operation_timings[operation].append(ms)
            return wrapper
        return decorator

    # ───────────────────────────── CSV loading ──────────────────────────────────

    def log_csv_loading_start(self) -> None:
        self._log("DEBUG", "Loading CSVs…", "INIT")

    def log_csv_ok(self, symbol: str, timeframe: str, bars: int, filename: str) -> None:
        self._csv_loaded_files += 1
        self._csv_loaded_bars += int(max(0, bars))
        self._log(
            "INFO",
            f"{symbol}/{timeframe}: {int(bars)} bars from {filename}",
            "CSV_OK"
        )

    def log_csv_warn(self, symbol: str, timeframe: str, msg: str) -> None:
        self._log("WARNING", f"{symbol}/{timeframe}: {msg}", "CSV_WARN")

    def log_init_ok(self, symbols_count: int, total_bars: int) -> None:
        self._log("INFO", f"Loaded {int(symbols_count)} symbols, ~{int(total_bars):,} clean bars.", "SUMMARY")
        self._log("INFO", "MarketDataProvider ready.", "INIT_OK")

    # ───────────────────────────── Process lifecycle ────────────────────────────

    def log_process_start(self, proc_index: int) -> None:
        with self._lock:
            self._current_proc_idx = int(proc_index)
            self._total_processes += 1
        line = "─" * 60
        self._log("DEBUG", line, "PROCESS_START")
        self._log("DEBUG", f"Starting process p#{self._current_proc_idx}", "PROCESS_START")

    def log_input_inspection(self, summary: Dict[str, Any]) -> None:
        # Normalize a few fields for readability
        keys = summary.get("keys_provided")
        keys_brief = f"[{len(keys)} items]" if isinstance(keys, list) else "n/a"
        step_idx = summary.get("step_idx", "not_provided")
        has_actions = 1 if summary.get("has_actions") else 0
        has_reward_inputs = 1 if summary.get("has_reward_inputs") else 0
        msg = f"Input summary: {{keys_provided: {keys_brief}, step_idx: {step_idx}, has_actions: {has_actions}, has_reward_inputs: {has_reward_inputs}}}"
        self._log("DEBUG", msg, "INPUT_INSPECTION")

    def log_advance(
        self,
        symbol: str,
        bar: Optional[Dict[str, Any]],
        rate_limit_sec: float = 1.5
    ) -> None:
        """
        Log a lightweight advance note. Rate-limited to avoid spam.
        """
        if not self._should_log("TRACE"):
            return
        if not bar:
            self._log("TRACE", f"{symbol}: advance skipped (no bar)", "ADVANCE", rate_limit_sec=rate_limit_sec)
            return
        try:
            close_raw = bar.get("close")
            close_v = float(close_raw) if close_raw is not None else None
            ts = bar.get("timestamp")
            ts_str = str(ts) if ts is not None else "n/a"
            self._log("TRACE", f"{symbol}: close={close_v} @ {ts_str}", "ADVANCE", rate_limit_sec=rate_limit_sec)
        except Exception:
            self._log("TRACE", f"{symbol}: advance (unprintable bar)", "ADVANCE", rate_limit_sec=rate_limit_sec)

    def log_alias_publish(self, keys: List[str]) -> None:
        n = len(keys)
        if n == 0:
            return
        # keep it trace to avoid extra noise under HF calls
        self._log("TRACE", f"Published {n} alias keys", "ALIAS_PUBLISH", rate_limit_sec=1.0)

    def log_bus_publish_ok(self, key: str) -> None:
        self._log("TRACE", f"Published {key} to InfoBus", "BUS_PUBLISH", rate_limit_sec=0.5)

    def log_bus_publish_fail(self, key: str, err: Exception) -> None:
        self._log("WARNING", f"Failed to publish {key} to InfoBus: {type(err).__name__}: {str(err)[:140]}", "BUS_PUBLISH")

    def log_snapshot_summary(self, snapshot: Dict[str, Any]) -> None:
        """
        Provide a compact human summary about the just-built snapshot.
        """
        try:
            sym_list = snapshot.get("symbols") or snapshot.get("universe") or []
            mtd = snapshot.get("multi_timeframe_data", {}) or {}
            aliases = [k for k in snapshot.keys() if k.startswith("market_data_")]
            vol_level = snapshot.get("volatility_level", "unknown")
            ts = snapshot.get("timestamp", "n/a")

            # Count distinct (symbol, tf) windows present
            tf_pairs = 0
            for _sym, per_tf in mtd.items():
                try:
                    tf_pairs += len(per_tf or {})
                except Exception:
                    pass

            msg = (
                f"Snapshot: symbols={len(sym_list)}, tf_windows={tf_pairs}, aliases={len(aliases)}, "
                f"vol={vol_level}, ts={ts}"
            )
            self._log("DEBUG", msg, "SNAPSHOT_SUMMARY", rate_limit_sec=0.5)
        except Exception as e:
            self._log("WARNING", f"Snapshot summary error: {type(e).__name__}: {str(e)[:140]}", "SNAPSHOT_SUMMARY")

    def log_process_complete(self, ok: bool, duration_ms: float) -> None:
        with self._lock:
            if ok:
                self._successful_processes += 1
            else:
                self._failed_processes += 1
        status = "OK" if ok else "FAIL"
        self._log("INFO", f"Process {self._proc_label()} completed in {duration_ms:.2f} ms [{status}]", "PROCESS_END")
        self._log("DEBUG", "─" * 60, "PROCESS_END")

    # ───────────────────────────── Monitoring / Health ─────────────────────────

    def log_monitoring_cycle(self) -> None:
        if not self._should_log("TRACE"):
            return
        self._log("TRACE", "Monitoring cycle executed", "MONITORING")

    def log_health(self, details: Dict[str, Any]) -> None:
        """
        Log compact health line (intended to be posted every few seconds max).
        Example payload:
            {"status":"HEALTHY","mode":"training","quality":0.45,"win_rate":0.136,"circuit_breaker":"CLOSED"}
        """
        status = str(details.get("status", "unknown")).upper()
        mode = str(details.get("mode", details.get("current_mode", "unknown")))
        cb = str(details.get("circuit_breaker", "unknown"))
        try:
            quality = float(details.get("quality", details.get("reward_quality", 0.0)))
        except Exception:
            quality = 0.0
        try:
            win_rate = float(details.get("win_rate", 0.0))
        except Exception:
            win_rate = 0.0

        self._log("INFO", f"Health Status: {status}", "HEALTH")
        self._log("DEBUG", f"  • Circuit Breaker: {cb}", "HEALTH_DETAIL")
        self._log("DEBUG", f"  • Mode: {mode}", "HEALTH_DETAIL")
        self._log("DEBUG", f"  • Quality: {quality:.3f}", "HEALTH_DETAIL")
        self._log("DEBUG", f"  • Win Rate: {win_rate:.1%}", "HEALTH_DETAIL")

    # ───────────────────────────── Errors ──────────────────────────────────────

    def log_error(self, context: str, error: Exception) -> None:
        self._log("ERROR", f"{type(error).__name__}: {str(error)[:200]}", context)

    # ───────────────────────────── Stats / Report ──────────────────────────────

    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            success_rate = self._successful_processes / max(1, self._total_processes)
            timing_summaries = {
                name: self._summarize_timings(list(samples))
                for name, samples in self.operation_timings.items()
                if samples
            }
            return {
                "enabled": self.enabled,
                "level": self.level_name,
                "csv_files": self._csv_loaded_files,
                "csv_bars": self._csv_loaded_bars,
                "processes": self._total_processes,
                "success_rate": success_rate,
                "timing_summaries": timing_summaries,
            }

    # ───────────────────────────── Internals ───────────────────────────────────

    def _should_log(self, level: str) -> bool:
        if not self.enabled:
            return False
        req = self._levels.get(level, _Levels.ERROR + 1)
        return req >= self.current_level

    def _proc_label(self) -> str:
        return f"p#{self._current_proc_idx}" if self._current_proc_idx is not None else "p#?"

    def _log(
        self,
        level: str,
        message: str,
        context: str = "",
        rate_limit_sec: Optional[float] = None,
    ) -> None:
        if not self._should_log(level):
            return

        if rate_limit_sec is not None:
            key = (level, context)
            now = time.time()
            last = self._last_log_time.get(key, 0.0)
            if now - last < rate_limit_sec:
                return
            self._last_log_time[key] = now

        # Format to match: [LOG] [DEBUG  ] 14:40:43.981 [p#8] [CONTEXT             ] message
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        level_padded = f"{level:<7}"
        ctx_padded = f"{context:<20}" if context else ""
        pid = f"[{self._proc_label()}] " if self._current_proc_idx is not None else ""

        pretty = f"[LOG] [{level_padded}] {ts} {pid}"
        pretty += f"[{ctx_padded}] " if ctx_padded else ""
        pretty += f"{message}"

        # Emit via host logger if available; otherwise print
        try:
            if self.logger is None:
                print(pretty)
            else:
                if level == "ERROR":
                    self.logger.error(pretty)
                elif level == "WARNING":
                    self.logger.warning(pretty)
                elif level in ("INFO", "DEBUG", "TRACE"):
                    # Many loggers don’t have .trace; map TRACE to .debug/.info routes.
                    # Prefer .debug to keep noisy traces out of INFO in prod.
                    if hasattr(self.logger, "debug") and level in ("TRACE", "DEBUG"):
                        self.logger.debug(pretty)
                    else:
                        self.logger.info(pretty)
                else:
                    self.logger.info(pretty)
        except Exception:
            # Ultimate fallback
            print(pretty)

    @staticmethod
    def _summarize_timings(samples: List[float]) -> Dict[str, Any]:
        if not samples:
            return {"n": 0, "mean": 0.0, "p95": 0.0, "max": 0.0}
        arr = np.asarray(samples, dtype=np.float64)
        return {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "p95": float(np.percentile(arr, 95)),
            "max": float(arr.max()),
        }
