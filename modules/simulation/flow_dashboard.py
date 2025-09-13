# # File: modules/monitoring/flow_dashboard.py
# from __future__ import annotations

# import os
# import json
# import threading
# import time
# from collections import defaultdict, deque
# from dataclasses import dataclass
# from typing import Any, Dict, List, Optional, Set, Tuple

# from modules.core.module_base import module, BaseModule
# from modules.utils.info_bus import InfoBusManager, SmartInfoBus
# from modules.utils.audit_utils import format_operator_message

# # ─────────────────────────────────────────────────────────────
# # FlowDashboard (v1.3.1)
# # - Same outputs/features as 1.3.0, but with SAFE LAZY START:
# #   • Snapshot thread starts AFTER boot (timer), not in _initialize()
# #   • No heavy InfoBus reads in _initialize()
# #   • All bus reads are best-effort and exception-safe
# #   • First snapshot is deferred (configurable), avoiding startup locks
# # - Still writes graph.json + flow_dashboard.log autonomously
# # ─────────────────────────────────────────────────────────────

# @dataclass
# class _Meters:
#     reads: deque
#     writes: deque

# def _now() -> float:
#     return time.time()

# def _abs_path(p: str) -> str:
#     try:
#         return os.path.abspath(p)
#     except Exception:
#         return p

# @module(
#     name="FlowDashboard",
#     provides=[
#         "module_flow_graph",
#         "module_health_map",
#         "flow_io_map",
#         "module_theses",
#         "flow_debug_extract",
#         "flow_alerts",
#         "flow_dashboard_state",
#         "flow_dashboard_log",
#         "flow_mermaid",
#         "flow_graphviz_dot",
#     ],
#     requires=[],
#     version="1.3.1",
#     category="monitoring",
#     description="Real-time module data-flow dashboard with meters, health, and debugging extracts. Writes graph.json + log file autonomously (lazy start).",
#     is_voting_member=False,
#     explainable=False,
#     timeout_ms=300,
#     priority=0,
#     min_confidence=0.0,
#     max_retries=0,
#     critical=False,
#     dependencies=[],
#     thesis_required=False,
#     health_monitoring=True,
#     performance_tracking=True,
#     error_handling=True,
# )
# class FlowDashboard(BaseModule):

#     # ───── lifecycle ─────
#     def _initialize(self):
#         # Config knobs
#         self.window_seconds: float = float(self.config.get("window_seconds", 15.0))
#         self.latency_warn_ms: float = float(self.config.get("latency_warn_ms", 300.0))
#         self.latency_crit_ms: float = float(self.config.get("latency_crit_ms", 700.0))
#         self.health_red: float = float(self.config.get("health_red", 0.30))
#         self.health_yellow: float = float(self.config.get("health_yellow", 0.70))
#         self.output_dir: str = _abs_path(str(self.config.get("output_dir", "logs/flow_dashboard")))
#         self.include_mermaid: bool = bool(self.config.get("include_mermaid", True))
#         self.include_dot: bool = bool(self.config.get("include_dot", True))
#         self.write_file_log: bool = bool(self.config.get("write_file_log", True))

#         # Autonomous snapshotting (LAZY)
#         self.auto_snapshot: bool = bool(self.config.get("auto_snapshot", True))
#         self.snapshot_interval_s: float = float(self.config.get("snapshot_interval_s", 3.0))   # periodic floor
#         self.snapshot_debounce_s: float = float(self.config.get("snapshot_debounce_s", 0.5))   # after activity
#         self.events_tail_count: int = int(self.config.get("events_tail_count", 120))
#         self.lazy_startup_s: float = float(self.config.get("lazy_startup_s", 2.0))             # <-- key change

#         # Core deps
#         self.bus: SmartInfoBus = InfoBusManager.get_instance()

#         # Ensure dirs
#         try:
#             for d in ("logs", "logs/modules", "logs/orchestrator", "logs/infobus", self.output_dir):
#                 os.makedirs(d, exist_ok=True)
#         except Exception:
#             pass

#         # Rolling meters
#         self._meters_lock = threading.RLock()
#         self._meters: Dict[str, _Meters] = defaultdict(
#             lambda: _Meters(reads=deque(maxlen=5000), writes=deque(maxlen=5000))
#         )

#         # Observed module names
#         self._known_modules_lock = threading.RLock()
#         self._known_modules: Set[str] = set()

#         # Snapshot loop signals
#         self._dirty_lock = threading.RLock()
#         self._dirty: bool = False   # set to True only after events or timer fires
#         self._last_activity_ts: float = _now()
#         self._stop_evt = threading.Event()
#         self._snap_thread: Optional[threading.Thread] = None

#         # Subscribe to bus events (lightweight; no heavy bus calls here)
#         try:
#             self.bus.subscribe("event_logged", self._on_any_event)
#             self.bus.subscribe("data_updated", self._on_data_updated)
#             self.bus.subscribe("data_get", self._on_data_get)
#             self.bus.subscribe("data_get_blocked", self._on_data_get)
#             self.bus.subscribe("performance_alert", self._on_any_event)
#             self.bus.subscribe("quality_warning", self._on_any_event)
#             self.bus.subscribe("module_disabled", self._on_any_event)
#             self.bus.subscribe("module_enabled", self._on_any_event)
#         except Exception as e:
#             self.logger.warning(f"FlowDashboard subscriptions failed (running without live meters): {e}")

#         # LAZY START: do not snapshot here; schedule bootstrap after boot settles
#         if self.auto_snapshot:
#             timer = threading.Timer(self.lazy_startup_s, self._bootstrap_async)
#             timer.daemon = True
#             timer.start()

#         self.logger.info(
#             f"[OK] FlowDashboard ready (auto_snapshot={self.auto_snapshot}, lazy_startup_s={self.lazy_startup_s})"
#         )

#     # runs after lazy_startup_s
#     def _bootstrap_async(self):
#         try:
#             # mark dirty so first snapshot happens even if idle
#             self._touch()
#             # fire one snapshot (best-effort)
#             try:
#                 self._write_snapshot()
#             except Exception as e:
#                 self.logger.warning(f"FlowDashboard initial snapshot skipped: {e}")

#             # start background snapshotter
#             self._snap_thread = threading.Thread(target=self._snapshot_loop, name="FlowDashboardSnapshot", daemon=True)
#             self._snap_thread.start()
#             self.logger.info("FlowDashboard snapshot loop started")
#         except Exception as e:
#             self.logger.warning(f"FlowDashboard bootstrap failed: {e}")

#     # ───── Event handlers (metering/observability) ─────
#     def _touch(self):
#         with self._dirty_lock:
#             self._dirty = True
#             self._last_activity_ts = _now()

#     def _on_any_event(self, evt: Dict[str, Any]):
#         try:
#             m = str(evt.get("module") or evt.get("source_module") or "")
#             if m:
#                 with self._known_modules_lock:
#                     self._known_modules.add(m)
#         except Exception:
#             pass
#         self._touch()

#     def _on_data_get(self, evt: Dict[str, Any]):
#         try:
#             mod = str(evt.get("module") or "unknown")
#             with self._meters_lock:
#                 self._meters[mod].reads.append(_now())
#         except Exception:
#             pass
#         self._touch()

#     def _on_data_updated(self, evt: Dict[str, Any]):
#         try:
#             mod = str(evt.get("module") or "unknown")
#             with self._meters_lock:
#                 self._meters[mod].writes.append(_now())
#         except Exception:
#             pass
#         self._touch()

#     # ───── Helpers ─────
#     def _rps(self, stamps: deque, window: float) -> float:
#         if not stamps:
#             return 0.0
#         now = _now()
#         while stamps and (now - stamps[0]) > window:
#             stamps.popleft()
#         return len(stamps) / max(window, 1e-6)

#     def _mk_status(self, enabled: bool, cb_state: str, health: float, p95_ms: float, failures: int) -> Tuple[str, str]:
#         if not enabled or cb_state == "OPEN" or health < self.health_red or p95_ms >= self.latency_crit_ms:
#             reason = []
#             if not enabled or cb_state == "OPEN":
#                 reason.append("circuit_breaker_open")
#             if health < self.health_red:
#                 reason.append(f"low_health({health:.2f})")
#             if p95_ms >= self.latency_crit_ms:
#                 reason.append(f"high_latency_p95({p95_ms:.0f}ms)")
#             if failures > 0:
#                 reason.append(f"recent_failures({failures})")
#             return "RED", ", ".join(reason) or "critical"
#         if health < self.health_yellow or p95_ms >= self.latency_warn_ms or failures > 0:
#             reason = []
#             if health < self.health_yellow:
#                 reason.append(f"degraded_health({health:.2f})")
#             if p95_ms >= self.latency_warn_ms:
#                 reason.append(f"elevated_latency_p95({p95_ms:.0f}ms)")
#             if failures > 0:
#                 reason.append(f"recent_failures({failures})")
#             return "YELLOW", ", ".join(reason) or "degraded"
#         return "GREEN", "ok"

#     def _collect_io_maps(self) -> Tuple[
#         Dict[str, List[str]], Dict[str, List[str]], Set[str], Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, List[str]]
#     ]:
#         """Return (provides_map, consumes_map, known_modules, providers_snapshot, consumers_snapshot, graph)"""
#         provides_map: Dict[str, List[str]] = defaultdict(list)
#         consumes_map: Dict[str, List[str]] = defaultdict(list)
#         known_modules: Set[str] = set()
#         providers_snapshot: Dict[str, Set[str]] = {}
#         consumers_snapshot: Dict[str, Set[str]] = {}
#         graph: Dict[str, List[str]] = {}

#         try:
#             # Preferred API
#             if hasattr(self.bus, "get_dependency_graph"):
#                 g = self.bus.get_dependency_graph() or {}
#                 # normalize to {str: List[str]}
#                 graph = {str(k): sorted(list(map(str, v or []))) for k, v in g.items()}
#             # collect modules
#             known_modules.update(graph.keys())
#             for cs in graph.values():
#                 known_modules.update(cs)

#             # Guarded peeks (optional)
#             p_raw = getattr(self.bus, "_providers", {})  # { key: set(modules) }
#             c_raw = getattr(self.bus, "_consumers", {})  # { key: set(modules) }
#             if isinstance(p_raw, dict):
#                 providers_snapshot = {str(k): set(map(str, (v or []))) for k, v in p_raw.items()}
#             if isinstance(c_raw, dict):
#                 consumers_snapshot = {str(k): set(map(str, (v or []))) for k, v in c_raw.items()}

#             # derive IO maps + derive graph if needed
#             for key, provs in providers_snapshot.items():
#                 for m in provs:
#                     provides_map[m].append(key)
#                     known_modules.add(m)
#             for key, cons in consumers_snapshot.items():
#                 for m in cons:
#                     consumes_map[m].append(key)
#                     known_modules.add(m)

#             if not graph and providers_snapshot and consumers_snapshot:
#                 graph = {}
#                 for key, provs in providers_snapshot.items():
#                     consumers = consumers_snapshot.get(key, set())
#                     for p in provs:
#                         graph.setdefault(p, [])
#                         graph[p].extend(list(consumers))
#                 for k, v in list(graph.items()):
#                     graph[k] = sorted(list(set(v)))

#             with self._known_modules_lock:
#                 known_modules.update(self._known_modules)
#         except Exception:
#             pass

#         # Deduplicate + sort
#         for m in list(provides_map.keys()):
#             provides_map[m] = sorted(list(dict.fromkeys(provides_map[m])))
#         for m in list(consumes_map.keys()):
#             consumes_map[m] = sorted(list(dict.fromkeys(consumes_map[m])))
#         return provides_map, consumes_map, known_modules, providers_snapshot, consumers_snapshot, graph

#     def _collect_theses(self, provides_map: Dict[str, List[str]]) -> Dict[str, List[Dict[str, Any]]]:
#         out: Dict[str, List[Dict[str, Any]]] = {}
#         for module_name, keys in provides_map.items():
#             arr: List[Dict[str, Any]] = []
#             for key in keys:
#                 try:
#                     dv = self.bus.get_with_metadata(key, "FlowDashboard")
#                     if not dv or not getattr(dv, "thesis", None):
#                         continue
#                     arr.append({
#                         "key": key,
#                         "thesis": dv.thesis,
#                         "confidence": float(dv.confidence),
#                         "version": int(dv.version),
#                         "ts": float(dv.timestamp),
#                         "age_seconds": float(dv.age_seconds()),
#                         "source_module": dv.source_module,
#                     })
#                 except Exception:
#                     continue
#             if arr:
#                 out[module_name] = arr
#         return out

#     def _events_tail(self, n: int) -> List[Dict[str, Any]]:
#         items: List[Dict[str, Any]] = []
#         try:
#             ev = getattr(self.bus, "_event_log", None)
#             if ev and hasattr(ev, "__iter__"):
#                 tail = list(ev)[-max(5, min(1500, n)):]
#                 for e in tail:
#                     if not isinstance(e, dict):
#                         continue
#                     items.append({
#                         "type": e.get("type"),
#                         "key": e.get("key"),
#                         "module": e.get("module") or e.get("source_module"),
#                         "reason": e.get("reason"),
#                         "version": e.get("version"),
#                         "timestamp": e.get("timestamp"),
#                     })
#         except Exception:
#             pass
#         return items

#     def _find_cycles(self, graph: Dict[str, List[str]]) -> List[List[str]]:
#         nodes = set(graph.keys())
#         for lst in graph.values():
#             nodes.update(lst)
#         visited: Set[str] = set()
#         stack: Set[str] = set()
#         path: List[str] = []
#         cycles: List[List[str]] = []

#         def dfs(u: str):
#             if len(cycles) >= 10:
#                 return
#             visited.add(u)
#             stack.add(u)
#             path.append(u)
#             for v in graph.get(u, []):
#                 if v not in visited:
#                     dfs(v)
#                 elif v in stack:
#                     try:
#                         i = path.index(v)
#                         cycles.append(path[i:] + [v])
#                     except ValueError:
#                         pass
#             stack.remove(u)
#             path.pop()

#         for n in nodes:
#             if n not in visited:
#                 dfs(n)
#                 if len(cycles) >= 10:
#                     break
#         return cycles

#     # ───── snapshot builder ─────
#     def _build_snapshot(self) -> Tuple[Dict[str, Any], List[str], str, str]:
#         # 1) dependency graph (provider -> consumers)
#         try:
#             graph = self.bus.get_dependency_graph() or {}
#         except Exception:
#             graph = {}
#         nodes_set: Set[str] = set(graph.keys())
#         for cs in graph.values():
#             try:
#                 nodes_set.update(cs)
#             except Exception:
#                 pass
#         with self._known_modules_lock:
#             nodes_set.update(self._known_modules)
#         nodes = sorted(nodes_set)

#         # 2) performance + health
#         try:
#             perf = self.bus.get_performance_metrics() or {}
#         except Exception:
#             perf = {}
#         mod_latency = perf.get("module_latencies", {}) or {}

#         provides_map, consumes_map, known_modules, p_snap, c_snap, graph2 = self._collect_io_maps()
#         if not graph:
#             graph = graph2  # fallback if direct call failed
#         module_theses = self._collect_theses(provides_map)

#         # 3) per-module health/status + meters
#         health_map: Dict[str, Dict[str, Any]] = {}
#         alerts: List[Dict[str, Any]] = []

#         for m in sorted(known_modules):
#             try:
#                 h = self.bus.get_module_health(m) or {}
#             except Exception:
#                 h = {}
#             enabled = bool(h.get("enabled", True))
#             cb_state = str(h.get("circuit_breaker_state", "CLOSED"))
#             health_score = float(h.get("health_score", 1.0))
#             failures = int(h.get("failures", 0))

#             lat = mod_latency.get(m, {}) or {}
#             p95 = float(lat.get("p95_ms", h.get("avg_latency_ms", 0.0)))
#             avg = float(lat.get("avg_ms", h.get("avg_latency_ms", 0.0)))
#             count = int(lat.get("count", h.get("total_executions", 0)))

#             with self._meters_lock:
#                 reads_rps = self._rps(self._meters[m].reads, self.window_seconds)
#                 writes_rps = self._rps(self._meters[m].writes, self.window_seconds)

#             status, reason = self._mk_status(enabled, cb_state, health_score, p95, failures)
#             if status != "GREEN":
#                 alerts.append({
#                     "module": m, "level": status, "reason": reason,
#                     "p95_ms": p95, "avg_ms": avg, "failures": failures,
#                 })

#             health_map[m] = {
#                 "status": status,
#                 "reason": reason,
#                 "enabled": enabled,
#                 "breaker": cb_state,
#                 "health_score": health_score,
#                 "latency_ms": {"avg": avg, "p95": p95},
#                 "executions": count,
#                 "meters": {"rps_reads": reads_rps, "rps_writes": writes_rps},
#                 "provides_keys": provides_map.get(m, []),
#                 "consumes_keys": consumes_map.get(m, []),
#             }

#         # 4) edges + nodes for graph
#         edges: List[Dict[str, str]] = []
#         try:
#             for provider, consumers in (graph or {}).items():
#                 for c in consumers:
#                     edges.append({"from": provider, "to": c})
#         except Exception:
#             pass

#         nodes_out: List[Dict[str, Any]] = []
#         for m in nodes:
#             hm = health_map.get(m, {})
#             nodes_out.append({
#                 "id": m,
#                 "status": hm.get("status", "GREEN"),
#                 "label": m,
#                 "breaker": hm.get("breaker", "CLOSED"),
#                 "health": hm.get("health_score", 1.0),
#                 "rps_reads": hm.get("meters", {}).get("rps_reads", 0.0),
#                 "rps_writes": hm.get("meters", {}).get("rps_writes", 0.0),
#             })
#         flow_graph = {"nodes": nodes_out, "edges": edges}

#         # 5) human-readable log
#         def dot(status: str) -> str:
#             return {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get(status, "⚪")

#         reds = sum(1 for v in health_map.values() if v["status"] == "RED")
#         yellows = sum(1 for v in health_map.values() if v["status"] == "YELLOW")
#         greens = sum(1 for v in health_map.values() if v["status"] == "GREEN")

#         log_lines: List[str] = []
#         log_lines.append("FLOW DASHBOARD")
#         log_lines.append("==============")
#         log_lines.append(f"window: {self.window_seconds:.0f}s | modules: {len(nodes)} | edges: {len(edges)}")
#         log_lines.append(f"status: {greens}🟢  {yellows}🟡  {reds}🔴")
#         log_lines.append("")
#         log_lines.append("🟢=OK  🟡=Degraded  🔴=Issue")
#         log_lines.append("")

#         sort_key = lambda item: (
#             {"RED": 0, "YELLOW": 1, "GREEN": 2}.get(item[1]["status"], 3),
#             -float(item[1]["latency_ms"]["p95"])
#         )
#         for m, hm in sorted(health_map.items(), key=sort_key):
#             log_lines.append(
#                 f"{dot(hm['status'])} {m:<28} | p95 {hm['latency_ms']['p95']:>5.0f} ms "
#                 f"| rps R:{hm['meters']['rps_reads']:.2f} W:{hm['meters']['rps_writes']:.2f} "
#                 f"| brk {hm['breaker']:<8} | health {hm['health_score']:.2f} | {hm['reason']}"
#             )

#         # 6) Mermaid / DOT (optional)
#         mermaid_str = ""
#         if self.include_mermaid:
#             try:
#                 lines = ["graph LR"]
#                 for e in edges:
#                     lines.append(f"    {e['from']} --> {e['to']}")
#                 for n in nodes_out:
#                     c = {"GREEN": "#2ECC71", "YELLOW": "#F4D03F", "RED": "#E74C3C"}.get(n["status"], "#BDC3C7")
#                     lines.append(f'    style {n["id"]} fill:{c},stroke:#333,stroke-width:1px')
#                 mermaid_str = "\n".join(lines)
#             except Exception:
#                 mermaid_str = ""

#         dot_str = ""
#         if self.include_dot:
#             try:
#                 lines = ["digraph Flow {", '  rankdir=LR;']
#                 for n in nodes_out:
#                     color = {"GREEN": "green", "YELLOW": "gold", "RED": "red"}.get(n["status"], "lightgray")
#                     lbl = f'{n["id"]}\\nRPS R:{n["rps_reads"]:.2f} W:{n["rps_writes"]:.2f}'
#                     lines.append(f'  "{n["id"]}" [style=filled, fillcolor="{color}", label="{lbl}"];')
#                 for e in edges:
#                     lines.append(f'  "{e["from"]}" -> "{e["to"]}";')
#                 lines.append("}")
#                 dot_str = "\n".join(lines)
#             except Exception:
#                 dot_str = ""

#         # 7) deep debug extract
#         events_tail = self._events_tail(self.events_tail_count)
#         top_latency = sorted(
#             [(m, (mod_latency.get(m, {}) or {}).get("p95_ms", 0.0)) for m in health_map.keys()],
#             key=lambda x: x[1],
#             reverse=True
#         )[:10]
#         try:
#             cache_stats = self.bus.get_cache_stats()
#         except Exception:
#             cache_stats = {}

#         # I/O anomalies
#         missing_providers: Dict[str, List[str]] = {}
#         for m, keys in (consumes_map or {}).items():
#             miss = [k for k in keys if not (p_snap or {}).get(k)]
#             if miss:
#                 missing_providers[m] = miss

#         unused_providers: Dict[str, List[str]] = {}
#         for m, keys in (provides_map or {}).items():
#             unused = [k for k in keys if not (c_snap or {}).get(k)]
#             if unused:
#                 unused_providers[m] = unused

#         cycles = self._find_cycles(graph or {})

#         debug_extract = {
#             "events_tail": events_tail,
#             "pending_requests": int((perf or {}).get("pending_requests", 0)),
#             "disabled_modules": list((perf or {}).get("disabled_modules", [])),
#             "cache_stats": cache_stats,
#             "top_latency_p95": [{"module": m, "p95_ms": float(p)} for m, p in top_latency],
#             "io_anomalies": {
#                 "missing_providers": missing_providers,
#                 "unused_providers": unused_providers,
#                 "cycles": cycles,
#             },
#         }

#         # 8) Summary state
#         state = {
#             "modules": len(nodes),
#             "edges": len(edges),
#             "status_counts": {"green": greens, "yellow": yellows, "red": reds},
#             "generated_at": time.time(),
#             "window_seconds": self.window_seconds,
#             "file_log_path": f"{self.output_dir}/flow_dashboard.log" if self.write_file_log else None,
#         }

#         packaged = {
#             "graph": flow_graph,
#             "health": health_map,
#             "alerts": alerts,
#             "state": state,
#             "io_map": {"provides": provides_map, "consumes": consumes_map},
#             "module_theses": module_theses,
#             "debug": debug_extract,
#         }
#         return packaged, log_lines, mermaid_str, dot_str

#     def _write_snapshot(self) -> Dict[str, Any]:
#         packaged, log_lines, mermaid_str, dot_str = self._build_snapshot()

#         # Persist JSON
#         try:
#             os.makedirs(self.output_dir, exist_ok=True)
#             with open(f"{self.output_dir}/graph.json", "w", encoding="utf-8") as f:
#                 json.dump(packaged, f, indent=2)
#         except Exception as e:
#             self.logger.warning(f"FlowDashboard JSON write failed: {e}")

#         # Append human-readable file log
#         if self.write_file_log:
#             try:
#                 with open(f"{self.output_dir}/flow_dashboard.log", "a", encoding="utf-8") as f:
#                     f.write("\n".join(log_lines) + "\n\n")
#             except Exception as e:
#                 self.logger.warning(f"FlowDashboard log write failed: {e}")

#         # Operator banner (visible in orchestrator logs)
#         try:
#             health = packaged.get("health", {})
#             reds = sum(1 for v in health.values() if v.get("status") == "RED")
#             yellows = sum(1 for v in health.values() if v.get("status") == "YELLOW")
#             greens = sum(1 for v in health.values() if v.get("status") == "GREEN")
#             self.logger.info(
#                 format_operator_message(
#                     "[FLOW]", "DASHBOARD UPDATE",
#                     details=f"{greens}🟢 {yellows}🟡 {reds}🔴 (modules={packaged.get('state',{}).get('modules','?')}, edges={packaged.get('state',{}).get('edges','?')})",
#                     context="flow_dashboard"
#                 )
#             )
#         except Exception:
#             pass

#         return packaged

#     def _snapshot_loop(self):
#         next_periodic = _now()
#         while not self._stop_evt.wait(0.25):
#             now = _now()
#             should_debounced = False
#             with self._dirty_lock:
#                 if self._dirty and (now - self._last_activity_ts) >= self.snapshot_debounce_s:
#                     should_debounced = True
#                     self._dirty = False

#             if should_debounced or now >= next_periodic:
#                 try:
#                     self._write_snapshot()
#                 except Exception as e:
#                     self.logger.warning(f"FlowDashboard snapshot failed: {e}")
#                 next_periodic = now + self.snapshot_interval_s

#     # ───── main orchestrated step (optional) ─────
#     async def process(self, **inputs) -> Dict[str, Any]:
#         # Build snapshot and also write files (keeps parity with autonomous path)
#         t0 = time.perf_counter()
#         packaged, log_lines, mermaid_str, dot_str = self._build_snapshot()
#         try:
#             self._write_snapshot()
#         except Exception:
#             pass

#         thesis = (
#             f"FlowDashboard snapshot: "
#             f"{packaged.get('state',{}).get('status_counts',{}).get('green',0)} green, "
#             f"{packaged.get('state',{}).get('status_counts',{}).get('yellow',0)} yellow, "
#             f"{packaged.get('state',{}).get('status_counts',{}).get('red',0)} red modules"
#         )

#         out = {
#             "module_flow_graph": packaged["graph"],
#             "module_health_map": packaged["health"],
#             "flow_io_map": {m: {"provides": packaged["io_map"]["provides"].get(m, []),
#                                 "consumes": packaged["io_map"]["consumes"].get(m, [])}
#                             for m in sorted(set(list(packaged["io_map"]["provides"].keys()) +
#                                                 list(packaged["io_map"]["consumes"].keys())))},
#             "module_theses": packaged["module_theses"],
#             "flow_debug_extract": packaged["debug"],
#             "flow_alerts": packaged["alerts"],
#             "flow_dashboard_state": packaged["state"],
#             "flow_dashboard_log": "\n".join(log_lines),
#             "_confidence": 0.95,
#             "_thesis": thesis,
#         }
#         if self.include_mermaid:
#             out["flow_mermaid"] = mermaid_str
#         if self.include_dot:
#             out["flow_graphviz_dot"] = dot_str

#         dur_ms = (time.perf_counter() - t0) * 1000.0
#         self.record_execution(dur_ms, True)
#         return out

#     # ───── teardown (best effort) ─────
#     def __del__(self):
#         try:
#             self._stop_evt.set()
#         except Exception:
#             pass
