# ─────────────────────────────────────────────────────────────
# File: modules/market/debug/visualizer.py
# Debug visualization for market module — Production Upgrade
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np


class DebugVisualizer:
    """
    Debug visualization for market module data flow and state.
    Thread-safe, robust to missing/partial inputs, and optimized for
    sustained, real-time updates.
    """

    # Default metric names we track if present in input payloads
    DEFAULT_METRICS: Tuple[str, ...] = (
        "regime_strength",
        "liquidity_score",
        "theme_confidence",
        "risk_scaling_factor",
        # from regime_matrix: {'regime_accuracy': {'value': ...}}
        "overall_accuracy",  # visualizer maps this from regime_accuracy.value when present
    )

    def __init__(
        self,
        max_history: int = 100,
        *,
        compute_ema: bool = True,
        ema_alpha: float = 0.2,
        downsample_to: Optional[int] = None,   # if set, downsample series to N points for ASCII
        anomaly_z_threshold: float = 2.5,
    ):
        self.max_history = max(10, int(max_history))
        self.compute_ema = bool(compute_ema)
        self.ema_alpha = float(ema_alpha)
        self.downsample_to = downsample_to if (downsample_to is None or downsample_to >= 10) else 10
        self.anomaly_z_threshold = float(anomaly_z_threshold)

        # Thread-safety
        self._lock = threading.RLock()

        # Data flow tracking
        self._data_flow: Deque[Dict[str, Any]] = deque(maxlen=self.max_history)

        # Component states: comp -> deque of {'status': str, 'execution_time': float, 'ts': float}
        self._component_states: Dict[str, Deque[Dict[str, Any]]] = {}

        # Metrics: metric_name -> deque of floats
        self._metrics_history: Dict[str, Deque[float]] = {}
        # Optional EMA store: metric_name -> last_ema_value
        self._ema_last: Dict[str, float] = {}

        # Per-metric anomalies: metric_name -> deque[bool]
        self._metric_anomalies: Dict[str, Deque[bool]] = {}

        # Current snapshot (last update payload)
        self._current_snapshot: Dict[str, Any] = {}

        # Registered metrics (can be extended dynamically)
        self._registered_metrics: Dict[str, bool] = {m: True for m in self.DEFAULT_METRICS}

    # --------------------------- Public API ---------------------------

    def register_metric(self, name: str):
        """Register an additional metric to track if present in updates."""
        if not name:
            return
        with self._lock:
            self._registered_metrics[name] = True
            if name not in self._metrics_history:
                self._metrics_history[name] = deque(maxlen=self.max_history)
            if name not in self._metric_anomalies:
                self._metric_anomalies[name] = deque(maxlen=self.max_history)

    def update(self, data: Dict[str, Any]):
        """Update visualization with new data (thread-safe)."""
        ts = _extract_timestamp(data)

        flow_entry = {
            "timestamp": ts,
            "components": _safe_get(data, ["_metadata", "components_executed"], default=[]),
            "success": bool(_safe_get(data, ["_metadata", "success"], default=False)),
        }

        with self._lock:
            # Track data flow
            self._data_flow.append(flow_entry)

            # Update component states (robust to structure)
            comp_health = data.get("component_health") or {}
            if isinstance(comp_health, dict):
                for comp, health in comp_health.items():
                    if comp not in self._component_states:
                        self._component_states[comp] = deque(maxlen=self.max_history)
                    status = str(health.get("status", "")).upper() or "UNKNOWN"
                    exec_ms = _to_float(health.get("execution_time_ms", health.get("execution_time", 0)))
                    self._component_states[comp].append({
                        "status": status,
                        "execution_time": exec_ms,
                        "ts": ts,
                    })

            # Track metrics (top-level or nested)
            self._track_metrics(data)

            # Store snapshot
            self._current_snapshot = data

    def get_data_flow_summary(self) -> Dict[str, Any]:
        """Get summary of data flow."""
        with self._lock:
            if not self._data_flow:
                return {}
            total = len(self._data_flow)
            success_count = sum(1 for f in self._data_flow if f.get("success"))
            return {
                "total_executions": total,
                "success_rate": success_count / max(total, 1),
                "component_frequency": self._get_component_frequency_locked(),
                "recent_flows": list(self._data_flow)[-10:],
            }

    def get_component_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for each component."""
        with self._lock:
            performance: Dict[str, Dict[str, Any]] = {}
            for comp, states in self._component_states.items():
                if not states:
                    continue
                success_count = sum(1 for s in states if _is_success(s.get("status")))
                exec_times = [float(s.get("execution_time", 0) or 0) for s in states if (s.get("execution_time", 0) or 0) > 0]
                performance[comp] = {
                    "success_rate": success_count / max(len(states), 1),
                    "avg_execution_time": float(np.mean(exec_times)) if exec_times else 0.0,
                    "max_execution_time": float(np.max(exec_times)) if exec_times else 0.0,
                    "min_execution_time": float(np.min(exec_times)) if exec_times else 0.0,
                    "total_executions": len(states),
                }
            return performance

    def get_metrics_trends(self) -> Dict[str, Dict[str, Any]]:
        """Get trends, stats, and anomaly flags for tracked metrics."""
        out: Dict[str, Dict[str, Any]] = {}
        with self._lock:
            for metric, hist in self._metrics_history.items():
                if len(hist) < 1:
                    continue
                values = np.asarray(hist, dtype=float)
                current = float(values[-1])
                mean = float(np.mean(values))
                std = float(np.std(values))
                mn = float(np.min(values))
                mx = float(np.max(values))
                trend = _trend(values)

                # z-score & anomaly flag on latest
                z = 0.0 if std == 0.0 else (current - mean) / std
                is_anom = abs(z) >= self.anomaly_z_threshold

                out[metric] = {
                    "current": current,
                    "mean": mean,
                    "std": std,
                    "min": mn,
                    "max": mx,
                    "trend": trend,
                    "history_length": int(values.size),
                    "z_score": float(z),
                    "anomaly": bool(is_anom),
                }
        return out

    def get_ascii_chart(self, metric: str, width: int = 50, height: int = 10) -> str:
        """Generate ASCII chart for a metric (robust & bounded)."""
        with self._lock:
            if metric not in self._metrics_history:
                return f"No data for metric: {metric}"
            values = list(self._metrics_history[metric])

        if len(values) < 2:
            return f"Insufficient data for metric: {metric}"

        # Downsample for display if requested
        if self.downsample_to and len(values) > self.downsample_to:
            values = _downsample(values, self.downsample_to)

        # Normalize to chart area
        min_val = float(min(values))
        max_val = float(max(values))
        range_val = max(max_val - min_val, 1e-12)

        # Header
        chart = []
        chart.append(f"Metric: {metric}")
        chart.append(f"Range: [{min_val:.4f}, {max_val:.4f}]")
        chart.append("─" * width)

        # Build grid by thresholds from top to bottom
        for row in range(height, 0, -1):
            threshold = min_val + (row / height) * range_val
            line_chars = []
            # map values to width points aligned from the end (latest right)
            window = values[-width:] if len(values) > width else values
            pad_left = width - len(window)
            if pad_left > 0:
                line_chars.extend([" "] * pad_left)

            for v in window:
                line_chars.append("█" if v >= threshold else " ")
            chart.append("".join(line_chars))

        chart.append("─" * width)
        chart.append(f"Latest: {values[-1]:.4f}")
        return "\n".join(chart)

    def export_debug_data(self, filepath: str):
        """Export debug data for offline analysis (JSON)."""
        with self._lock:
            payload = {
                "data_flow": list(self._data_flow),
                "component_states": {comp: list(states) for comp, states in self._component_states.items()},
                "metrics_history": {metric: list(history) for metric, history in self._metrics_history.items()},
                "performance": self.get_component_performance(),
                "trends": self.get_metrics_trends(),
                "snapshot_time": time.time(),
            }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

    def get_debug_summary(self) -> str:
        """Human-readable, compact summary for dashboards/logs."""
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("MARKET MODULE DEBUG SUMMARY")
        lines.append("=" * 60)

        flow_summary = self.get_data_flow_summary()
        if flow_summary:
            lines.append("\nData Flow:")
            lines.append(f"  Total executions: {flow_summary.get('total_executions', 0)}")
            sr = flow_summary.get("success_rate", 0.0) or 0.0
            lines.append(f"  Success rate: {sr:.1%}")

        lines.append("\nComponent Performance:")
        comp_perf = self.get_component_performance()
        for comp, perf in comp_perf.items():
            lines.append(f"  {comp}:")
            lines.append(f"    Success rate: {perf['success_rate']:.1%}")
            lines.append(f"    Avg time: {perf['avg_execution_time']:.1f}ms")

        lines.append("\nMetrics Trends:")
        trends = self.get_metrics_trends()
        for metric, t in trends.items():
            lines.append(f"  {metric}:")
            lines.append(f"    Current: {t['current']:.4f} | Mean: {t['mean']:.4f} | Std: {t['std']:.4f}")
            lines.append(f"    Trend: {t['trend']} | Anomaly: {t['anomaly']} (z={t['z_score']:.2f})")

        lines.append("=" * 60)
        return "\n".join(lines)

    # ----------------------- Internal helpers -----------------------

    def _track_metrics(self, data: Dict[str, Any]):
        """Track metrics in a robust, schema-tolerant way."""
        # 1) Ensure default/registered metrics exist
        for m in self._registered_metrics.keys():
            # gather value from data (flat or nested known patterns)
            val = _extract_metric_value(data, m)
            if val is None and m == "overall_accuracy":
                # back-compat: often at data['regime_accuracy']['value']
                val = _safe_get(data, ["regime_accuracy", "value"])
            if val is None:
                continue
            fval = _to_float(val)
            if not math.isfinite(fval):
                continue

            if m not in self._metrics_history:
                self._metrics_history[m] = deque(maxlen=self.max_history)
                self._metric_anomalies[m] = deque(maxlen=self.max_history)

            # Optional EMA
            if self.compute_ema:
                prev = self._ema_last.get(m)
                ema = fval if prev is None else (self.ema_alpha * fval + (1.0 - self.ema_alpha) * prev)
                self._ema_last[m] = float(ema)
                store_val = float(ema)
            else:
                store_val = fval

            self._metrics_history[m].append(store_val)
            # anomaly flag filled in get_metrics_trends (on-demand)

    def _get_component_frequency_locked(self) -> Dict[str, int]:
        """Get frequency of component execution (caller holds lock)."""
        frequency: Dict[str, int] = {}
        for flow in self._data_flow:
            for comp in flow.get("components", []) or []:
                frequency[comp] = frequency.get(comp, 0) + 1
        return frequency


# ----------------------- Module-level helpers -----------------------

def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _extract_timestamp(data: Dict[str, Any]) -> float:
    # Try metadata timestamp (ISO or epoch), else now
    ts_raw = _safe_get(data, ["_metadata", "timestamp"])
    if ts_raw is None:
        return time.time()
    # Accept iso string or epoch
    if isinstance(ts_raw, (int, float)):
        return float(ts_raw)
    if isinstance(ts_raw, str):
        try:
            # Fast ISO parse fallback: strip non-digit for epoch-ish strings
            # If it's ISO, we won't parse to datetime (avoid dependency); just return now for simplicity
            # (We only display as float epoch in this visualizer.)
            return time.time()
        except Exception:
            return time.time()
    return time.time()


def _safe_get(d: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur = d
    try:
        for p in path:
            if not isinstance(cur, dict) or p not in cur:
                return default
            cur = cur[p]
        return cur
    except Exception:
        return default


def _is_success(status: Optional[str]) -> bool:
    if not status:
        return False
    s = str(status).upper()
    # Support both enum-like and strings
    return s in ("SUCCESS", "OK", "DONE", "COMPLETED")


def _trend(values: np.ndarray) -> str:
    # Use simple slope on last N points
    n = values.size
    if n < 3:
        return "stable"
    span = min(10, n)
    y = values[-span:]
    x = np.arange(y.size, dtype=float)
    if float(np.std(y)) == 0.0:
        return "stable"
    try:
        slope = float(np.polyfit(x, y, 1)[0])
    except Exception:
        return "stable"
    if slope > 1e-3:
        return "increasing"
    if slope < -1e-3:
        return "decreasing"
    return "stable"


def _downsample(values: List[float], target: int) -> List[float]:
    """Uniformly downsample list to target length (>= 10)."""
    n = len(values)
    if target >= n or target <= 0:
        return values
    idx = np.linspace(0, n - 1, target).astype(int)
    return [values[i] for i in idx]


def _extract_metric_value(data: Dict[str, Any], name: str) -> Optional[float]:
    """
    Try to extract metric by name from common places:
    - top-level key: data[name]
    - direct nested: data.get('metrics', {}).get(name)
    - shared_context-like nesting if provided
    """
    # exact top-level
    if name in data:
        return _to_float(data[name])

    # 'metrics' bag
    metrics = data.get("metrics")
    if isinstance(metrics, dict) and name in metrics:
        return _to_float(metrics[name])

    # shared_context typical flattening
    sc = data.get("shared_context")
    if isinstance(sc, dict) and name in sc:
        return _to_float(sc[name])

    return None
