

from __future__ import annotations

import json
import os
import threading
import time
import traceback
import tracemalloc
from collections import Counter, defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from modules.utils.audit_utils import RotatingLogger


@dataclass
class ExecutorDebugConfig:
    enabled: bool = True
    level: str = "TRACE"
    keep_history: int = 200
    file_logging: bool = True
    jsonl_dir: str = "logs/executor/debug"
    redact_sizes: bool = False
    mem_profile: bool = False
    bus_module_name: str = "Executor"
    max_preview_items: int = 5
    report_key_text: str = "executor_debug_report"

    expected_bus_keys: Optional[Dict[str, Dict[str, Any]]] = None


_DEFAULT_EXPECTED_BUS_KEYS: Dict[str, Dict[str, Any]] = {

    "order_queue": {"source": "SignalRouter", "required": False},
    "position_decision_EURUSD": {"source": "DecisionEngine", "required": False},
    "position_decision_XAUUSD": {"source": "DecisionEngine", "required": False},


    "current_positions": {"source": "PositionManager", "required": False},
    "portfolio_metrics": {"source": "Account/Env", "required": True},
    "trading_result": {"source": "Env", "required": False},


    "environment_config": {"source": "Environment", "required": False},
    "env_mode": {"source": "Environment", "required": False},
    "live_adapter_status": {"source": "Executor", "required": False},


    "performance_data": {"source": "PerformanceTracker", "required": False},
}


class ExecutorDebugManager:


    def __init__(self, bus: Any, config: Optional[Dict[str, Any]] = None, logger: Optional[RotatingLogger] = None):
        cfg = ExecutorDebugConfig(**(config or {}))
        self.cfg: ExecutorDebugConfig = cfg
        self.bus = bus
        self.logger = logger or RotatingLogger("ExecutorDebugger", log_path="logs/executor/debugger.log")


        self.levels: Dict[str, int] = {
            "TRACE": 0, "DEBUG": 1, "FULL": 1, "LIGHT": 2, "INFO": 2, "WARNING": 3, "ERROR": 4, "OFF": 99
        }

        self.current_level: int = self.levels.get(str(self.cfg.level).upper(), 1)


        self._history: deque = deque(maxlen=self.cfg.keep_history)
        self._metrics: Counter = Counter()
        self._last_errors: deque[str] = deque(maxlen=50)
        self._stage_t0: Dict[str, float] = {}
        self._stage_dt: Dict[str, float] = {}
        self._op_timings: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.RLock()


        self.expected_keys: Dict[str, Dict[str, Any]] = self.cfg.expected_bus_keys or dict(_DEFAULT_EXPECTED_BUS_KEYS)
        self.key_miss_counts: Dict[str, int] = defaultdict(int)
        self.key_error_counts: Dict[str, int] = defaultdict(int)


        self._mem_enabled = bool(self.cfg.mem_profile)
        if self._mem_enabled and not tracemalloc.is_tracing():
            try:
                tracemalloc.start()
            except Exception:
                self._mem_enabled = False


        if self.cfg.file_logging:
            try:
                os.makedirs(self.cfg.jsonl_dir, exist_ok=True)
            except Exception:
                pass


        self._last_log_time: Dict[Tuple[str, str], float] = {}


    def enable(self) -> None:
        self.cfg.enabled = True

    def disable(self) -> None:
        self.cfg.enabled = False

    def set_level(self, level: str) -> None:
        self.cfg.level = level.upper()
        self.current_level = self.levels.get(self.cfg.level, 1)

    def enable_memory_tracking(self) -> None:
        with self._lock:
            if not tracemalloc.is_tracing():
                try:
                    tracemalloc.start()
                except Exception:
                    return
            self._mem_enabled = True

    def disable_memory_tracking(self) -> None:
        with self._lock:
            self._mem_enabled = False


    def begin(self, name: str) -> None:
        if not self._should_log("DEBUG"):
            return
        self._stage_t0[name] = time.perf_counter()

    def end(self, name: str) -> None:
        if not self._should_log("DEBUG"):
            return
        t0 = self._stage_t0.pop(name, None)
        if t0 is None:
            return
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self._stage_dt[name] = dt_ms
        self._op_timings[name].append(dt_ms)

    @contextmanager
    def stage(self, name: str):
        self.begin(name)
        try:
            yield
        finally:
            self.end(name)

    @contextmanager
    def memory_block(self, name: str):
        if not self._mem_enabled or not self._should_log("TRACE"):
            yield
            return
        snap_before = tracemalloc.take_snapshot()
        try:
            yield
        finally:
            snap_after = tracemalloc.take_snapshot()
            stats = snap_after.compare_to(snap_before, "lineno")
            delta_bytes = float(sum(s.size_diff for s in stats))

            key = f"{name}#mem_kb"
            self._op_timings[key].append(delta_bytes / 1024.0)


    def record_error(self, msg: str) -> None:
        if not self.cfg.enabled:
            return
        with self._lock:
            self._last_errors.append(msg)

    def log_exception(self, context: str, exc: Exception, inputs: Optional[Dict[str, Any]] = None) -> None:
        tb = traceback.format_exc()
        self.record_error(f"{context}: {exc}")
        self._log("ERROR", f"ERROR in {context}: {exc}", "EXCEPTION")
        if self._should_log("DEBUG"):
            self._log("DEBUG", f"Traceback:\n{tb}", "TRACEBACK")
        if inputs and self._should_log("TRACE"):
            try:
                j = json.dumps(inputs)[:2000]
            except Exception:
                j = str(inputs)[:2000]
            self._log("TRACE", f"Inputs at error: {j}", "ERROR_INPUTS")


    def log_initialization_state(self) -> None:
        if not self._should_log("INFO"):
            return
        self._log("INFO", "Executor debug initialized - probing SmartInfoBus...", "INIT")
        for key, info in self.expected_keys.items():
            try:
                v = self._bus_get(key)
                if v is not None:
                    self._log("DEBUG", f"  ✓ {key}: available", "BUS")
                else:
                    self._log("WARNING", f"  ✗ {key}: not available (expected from {info.get('source','?')})", "BUS")
            except Exception as e:
                self._log("ERROR", f"  ✗ {key}: error reading ({e})", "BUS")


    def publish(
        self,
        *,
        step: int,
        mode: str,
        queue_count: int,
        decisions_count: int,
        accepted: List[Dict[str, Any]],
        rejected: List[Dict[str, Any]],
        fills: List[Dict[str, Any]],
        positions_after: Dict[str, Any],
        balance_before: float,
        equity_before: float,
        balance_after: float,
        equity_after: float,
        realized_step: float,
        unreal_after: float,
        step_pnl: float,
        reason: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.cfg.enabled or not self.bus or self.current_level >= self.levels["OFF"]:
            return


        rej_hist = Counter([str(r.get("reason", "unknown")) for r in (rejected or [])])
        by_inst_notional: Dict[str, float] = defaultdict(float)
        for f in (fills or []):
            try:
                inst = str(f.get("instrument") or f.get("symbol") or "UNK")
                notional = float(f.get("notional_eur", f.get("notional", 0.0)) or 0.0)
                by_inst_notional[inst] += abs(notional)
            except Exception:
                pass


        issues: List[str] = []
        if accepted and not fills:
            issues.append("accepted_but_no_fills")
        if float(step_pnl or 0.0) > 0 and float(equity_after or 0.0) < float(equity_before or 0.0):
            issues.append("pnl_positive_but_equity_down")
        if float(step_pnl or 0.0) < 0 and float(equity_after or 0.0) > float(equity_before or 0.0):
            issues.append("pnl_negative_but_equity_up")

        if any(float(a.get("lots", a.get("units", 0.0)) or 0.0) <= 0.0 for a in (accepted or [])):
            issues.append("non_positive_size_in_accepted")

        if self._has_duplicate_instruments(accepted):
            issues.append("duplicate_intents_same_instrument")


        with self._lock:
            timings = dict(self._stage_dt)
            self._stage_dt.clear()


        preview_n = int(self.cfg.max_preview_items)
        if self.cfg.redact_sizes:
            accepted_prev = [self._redact(a) for a in (accepted or [])][-preview_n:]
            rejected_prev = [self._redact(r) for r in (rejected or [])][-preview_n:]
            fills_prev = [self._redact(f) for f in (fills or [])][-preview_n:]
        else:
            accepted_prev = (accepted or [])[-preview_n:]
            rejected_prev = (rejected or [])[-preview_n:]
            fills_prev = (fills or [])[-preview_n:]


        env_mode = self._bus_get("env_mode") or self._infer_mode_from_config()
        live_status = self._bus_get("live_adapter_status")


        node = {
            "header": {
                "ts": time.time(),
                "step": int(step),
                "mode": str(mode),
                "env_mode": env_mode or "unknown",
                "reason": reason or "ok",
            },
            "timings_ms": timings,
            "io": {
                "order_queue_in": int(queue_count),
                "position_decisions_in": int(decisions_count),
                "accepted": len(accepted or []),
                "rejected": len(rejected or []),
                "reject_hist": dict(rej_hist),
            },
            "fills": {
                "count": len(fills or []),
                "notional_total": float(sum(abs(float(f.get("notional_eur", f.get("notional", 0.0)) or 0.0)) for f in (fills or []))),
                "by_instrument_notional": dict(by_inst_notional),
                "preview": fills_prev,
            },
            "pnl": {
                "balance_before": float(balance_before or 0.0),
                "equity_before": float(equity_before or 0.0),
                "realized_step": float(realized_step or 0.0),
                "unreal_after": float(unreal_after or 0.0),
                "balance_after": float(balance_after or 0.0),
                "equity_after": float(equity_after or 0.0),
                "step_pnl": float(step_pnl or 0.0),
            },
            "positions_after_keys": sorted(list((positions_after or {}).keys()))[:100],
            "issues": issues,
            "last_errors": list(self._last_errors),
        }
        if live_status:
            node["live_adapter_status"] = live_status
        if extra:
            node["extra"] = extra


        with self._lock:
            self._metrics.update({
                "steps": 1,
                "accepted": len(accepted or []),
                "rejected": len(rejected or []),
                "fills": len(fills or []),
            })
            accepted_total = int(self._metrics["accepted"])
            rejected_total = int(self._metrics["rejected"])
            fills_total = int(self._metrics["fills"])
            steps_total = int(self._metrics["steps"])

        accept_rate = self._safe_div(accepted_total, max(1, accepted_total + rejected_total))
        fill_rate = self._safe_div(fills_total, max(1, accepted_total))


        self._bus_set("executor_debug", node, "Executor debug snapshot")


        hist_item = {
            "step": node["header"]["step"],
            "mode": node["header"]["mode"],
            "env_mode": node["header"]["env_mode"],
            "accepted": node["io"]["accepted"],
            "rejected": node["io"]["rejected"],
            "fills": node["fills"]["count"],
            "step_pnl": node["pnl"]["step_pnl"],
            "issues": node["issues"],
            "timings_ms": node["timings_ms"],
        }
        with self._lock:
            self._history.append(hist_item)
            self._bus_set("executor_debug_history", list(self._history), "Executor debug ring")


        metrics_payload = {
            "steps": steps_total,
            "accepted": accepted_total,
            "rejected": rejected_total,
            "fills": fills_total,
            "accept_rate": float(accept_rate),
            "fill_rate": float(fill_rate),
            "last_step": int(step),
        }
        self._bus_set("executor_debug_metrics", metrics_payload, "Executor rolling metrics")


        if self._should_log("DEBUG"):
            self._bus_set(self.cfg.report_key_text, self._make_report_line(node, metrics_payload), "Executor text report")


        try:

            if self._should_log("DEBUG"):
                self.logger.debug(
                    f"[EXECDBG] step={step} mode={mode}/{env_mode or 'n/a'} ok={len(accepted or [])} "
                    f"rej={len(rejected or [])} fills={len(fills or [])} dPnL={float(step_pnl or 0.0):.2f} "
                    f"issues={len(issues)}"
                )

            if self.cfg.file_logging:
                path = os.path.join(self.cfg.jsonl_dir, "steps.jsonl")
                with open(path, "a", encoding="utf-8") as fp:
                    fp.write(json.dumps(node, ensure_ascii=False) + "\n")
        except Exception:

            pass


    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            op_summaries = {
                k: self._summarize(self._op_timings.get(k, []))
                for k in sorted(self._op_timings.keys())
                if self._op_timings.get(k)
            }
            return {
                "enabled": self.cfg.enabled,
                "level": self.cfg.level,
                "history_len": len(self._history),
                "metrics": dict(self._metrics),
                "accept_rate": self._safe_div(self._metrics["accepted"], max(1, self._metrics["accepted"] + self._metrics["rejected"])),
                "fill_rate": self._safe_div(self._metrics["fills"], max(1, self._metrics["accepted"])),
                "key_miss_counts": dict(self.key_miss_counts),
                "key_error_counts": dict(self.key_error_counts),
                "timing_summaries": op_summaries,
            }

    def get_report(self) -> str:
        s = self.get_statistics()
        lines: List[str] = []
        lines.append("\n" + "═" * 60)
        lines.append("EXECUTOR DEBUG REPORT")
        lines.append("═" * 60)
        lines.append(f"Level: {s['level']} | Enabled: {s['enabled']}")
        m = s.get("metrics", {})
        lines.append(f"Steps: {int(m.get('steps', 0))} | Accepted: {int(m.get('accepted', 0))} | "
                     f"Rejected: {int(m.get('rejected', 0))} | Fills: {int(m.get('fills', 0))}")
        lines.append(f"Accept Rate: {float(s.get('accept_rate', 0.0)):.2%} | Fill Rate: {float(s.get('fill_rate', 0.0)):.2%}")

        km = s.get("key_miss_counts", {}) or {}
        if km:
            top_miss = sorted(km.items(), key=lambda kv: kv[1], reverse=True)[:5]
            lines.append("Most Missed Bus Keys: " + ", ".join(f"{k}({v})" for k, v in top_miss))

        ts = s.get("timing_summaries", {}) or {}
        if ts:
            lines.append("Timing Summaries (ms):")
            top_ops = sorted(ts.items(), key=lambda kv: kv[1]["mean"], reverse=True)[:6]
            for name, d in top_ops:
                lines.append(f"  • {name}: mean={d['mean']:.2f}, p95={d['p95']:.2f}, max={d['max']:.2f}, n={d['n']}")
        lines.append("═" * 60)
        return "\n".join(lines)

    def log_shutdown(self) -> None:
        if not self._should_log("INFO"):
            return
        report = self.get_report()
        self._log("INFO", "Executor SHUTDOWN - final statistics below", "SHUTDOWN")
        for ln in report.splitlines():
            self._log("INFO", ln, "REPORT")


    def _should_log(self, level: str) -> bool:
        if not self.cfg.enabled:
            return False
        return self.levels.get(level, 5) >= self.current_level

    def _log(self, level: str, message: str, context: str = "", rate_limit_sec: Optional[float] = None) -> None:
        if not self._should_log(level):
            return
        if rate_limit_sec is not None:
            key = (level, context)
            now = time.time()
            last = self._last_log_time.get(key, 0.0)
            if now - last < rate_limit_sec:
                return
            self._last_log_time[key] = now
        ts = time.strftime("%H:%M:%S")
        msg = f"[{level:7}] {ts} [{context or 'EXECDBG':>12}] {message}"
        try:
            if level in ("ERROR",):
                self.logger.error(msg)
            elif level in ("WARNING",):
                self.logger.warning(msg)
            else:
                self.logger.info(msg)
        except Exception:

            print(msg)

    def _bus_get(self, key: str) -> Any:
        if not self.bus:
            return None

        try:
            return self.bus.get(key, self.cfg.bus_module_name)
        except TypeError:
            try:
                return self.bus.get(key)
            except Exception:
                return None
        except Exception:
            return None

    def _bus_set(self, key: str, value: Any, thesis: str = "") -> None:
        if not self.bus:
            return
        try:
            self.bus.set(key, value, module=self.cfg.bus_module_name, thesis=thesis or key)
        except TypeError:
            try:
                self.bus.set(key, value)
            except Exception:
                pass
        except Exception:
            pass

    @staticmethod
    def _safe_div(a: float, b: float) -> float:
        try:
            b = float(b)
            return float(a) / b if b else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _redact(d: Dict[str, Any]) -> Dict[str, Any]:
        z = dict(d or {})
        for k in ("units", "size_eur", "notional_eur", "notional", "lots"):
            if k in z:
                z[k] = "•••"
        return z

    @staticmethod
    def _has_duplicate_instruments(intents: List[Dict[str, Any]]) -> bool:
        try:
            instruments = [str(i.get("instrument") or i.get("symbol") or "") for i in intents or []]
            cnt = Counter([x for x in instruments if x])
            return any(v > 1 for v in cnt.values())
        except Exception:
            return False

    def _infer_mode_from_config(self) -> Optional[str]:
        try:
            cfg = self._bus_get("environment_config")
            if isinstance(cfg, dict):

                if "mode" in cfg:
                    return str(cfg["mode"])
                if cfg.get("live_enabled"):
                    return "live"
        except Exception:
            pass
        return None

    @staticmethod
    def _summarize(samples: List[float]) -> Dict[str, Any]:
        if not samples:
            return {"n": 0, "mean": 0.0, "p95": 0.0, "max": 0.0}
        arr = np.asarray(samples, dtype=np.float64)
        return {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "p95": float(np.percentile(arr, 95)),
            "max": float(arr.max()),
        }

    def _make_report_line(self, node: Dict[str, Any], metrics: Dict[str, Any]) -> str:
        pnl = node.get("pnl", {})
        io = node.get("io", {})
        fills = node.get("fills", {})
        issues = node.get("issues", []) or []
        parts = [
            f"Step {node['header'].get('step','?')} [{node['header'].get('mode','?')}/{node['header'].get('env_mode','?')}]",
            f"accepted={io.get('accepted',0)} rejected={io.get('rejected',0)} fills={fills.get('count',0)}",
            f"dPnL={float(pnl.get('step_pnl',0.0)):.2f} eq:{float(pnl.get('equity_before',0.0)):.2f}→{float(pnl.get('equity_after',0.0)):.2f}",
            f"rates: accept={float(metrics.get('accept_rate',0.0)):.1%} fill={float(metrics.get('fill_rate',0.0)):.1%}",
        ]
        if issues:
            parts.append(f"issues={','.join(issues)}")
        return " | ".join(parts)
