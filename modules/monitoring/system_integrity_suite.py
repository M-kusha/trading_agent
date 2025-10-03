from __future__ import annotations

import ast
import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import (
    Any,
    Callable,
    Deque,
    DefaultDict,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    runtime_checkable,
)
from collections import defaultdict, deque

# Optional libs (visualization)
try:
    import networkx as _nx  # type: ignore
except Exception:
    _nx = None  # type: ignore
try:
    import matplotlib.pyplot as _plt  # type: ignore
except Exception:
    _plt = None  # type: ignore


# ─────────────────────────────────────────────────────────────
# Light fallbacks for logger and InfoBus (override by wiring)
# ─────────────────────────────────────────────────────────────

@runtime_checkable
class LoggerProto(Protocol):
    def info(self, *args: Any, **kwargs: Any) -> None: ...
    def warning(self, *args: Any, **kwargs: Any) -> None: ...
    def error(self, *args: Any, **kwargs: Any) -> None: ...
    # Optional in some codebases
    def debug(self, *args: Any, **kwargs: Any) -> None: ...  # type: ignore[override]

def _fmt_op(icon: str, message: str, **ctx: Any) -> str:
    if not ctx:
        return f"{icon} {message}"
    pairs = " ".join(f"{k}={v}" for k, v in ctx.items() if v is not None)
    return f"{icon} {message} :: {pairs}"

class _FallbackLogger:
    def __init__(self, name: str = "SystemIntegrity", **_: Any) -> None:
        self._name = name
    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        print(f"[INFO] {self._name} | {msg}")
    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        print(f"[WARN] {self._name} | {msg}")
    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        print(f"[ERROR] {self._name} | {msg}")
    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        print(f"[DEBUG] {self._name} | {msg}")

# Prefer project logger if available
try:
    from modules.utils.audit_utils import RotatingLogger as _RealLogger, format_operator_message as _real_fmt  # type: ignore
    def _fmt(icon: str, message: str, **ctx: Any) -> str:
        try:
            return _real_fmt(icon, message, **ctx)
        except TypeError:
            # Different signature in some repos
            return _fmt_op(icon, message, **ctx)
    LoggerClass: Callable[..., LoggerProto] = _RealLogger  # type: ignore[assignment]
except Exception:
    LoggerClass = _FallbackLogger  # type: ignore[assignment]
    _fmt = _fmt_op

# InfoBus adapter
class _NullBus:
    def subscribe(self, *_: Any, **__: Any) -> None: ...
    def unsubscribe(self, *_: Any, **__: Any) -> None: ...
    def get_data_freshness_report(self) -> Dict[str, Dict[str, Any]]: return {}
    def get_providers(self, *_: Any, **__: Any) -> List[str]: return []
    # Optional internal snapshots used by auditor (if your bus exposes them)
    _providers: Dict[str, Set[str]] = {}
    _consumers: Dict[str, Set[str]] = {}
    def set(self, *_: Any, **__: Any) -> None: ...

def _get_bus() -> Any:
    try:
        from modules.utils.info_bus import InfoBusManager  # type: ignore
        inst = getattr(InfoBusManager, "get_instance", None)
        return inst() if callable(inst) else _NullBus()
    except Exception:
        return _NullBus()

# Optional orchestrator (for module registry + metadata)
def _get_orchestrator() -> Any:
    try:
        from modules.core.module_system import ModuleOrchestrator  # type: ignore
        get = getattr(ModuleOrchestrator, "get_instance", None)
        return get() if callable(get) else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# Config & Models
# ─────────────────────────────────────────────────────────────

@dataclass
class SuiteConfig:
    # Heartbeat & lifecycle thresholds
    heartbeat_interval_seconds: float = 5.0
    stale_age_warn_seconds: float = 60.0
    stale_age_critical_seconds: float = 180.0  # escalate after this
    resolution_warn_seconds: float = 15.0  # SLA before first FRESH on watchlist keys
    # Logging / Debug
    operator_mode: bool = True
    plain_english: bool = True
    debug: bool = True                              # <── master debug switch
    debug_ring_size: int = 500                       # recent events ring buffer
    log_on_every_event: bool = True                 # chatty trace if True
    # Discovery roots
    modules_root: str = "modules"
    # Visualization defaults
    viz_output_path: str = "dependency_graph.png"
    # Validation
    critical_single_writer_suffixes: Tuple[str, ...] = ("_vote",)
    critical_single_writer_keys: Tuple[str, ...] = (
        "market_regime", "training_metrics", "performance_metrics",
        "risk_data", "sequence_quality", "trade_vote"
    )

@dataclass
class KeyLifecycle:
    status: str = "UNKNOWN"                # UNKNOWN | MISSING | STALE | FRESH
    first_seen_ts: float = 0.0
    last_event_ts: float = 0.0
    resolved_ts: float = 0.0
    miss_count: int = 0
    blocked_count: int = 0
    set_count: int = 0
    flap_count: int = 0
    provider_changes: int = 0
    last_provider: Optional[str] = None
    last_version: Optional[int] = None
    consumers: Set[str] = field(default_factory=set)
    providers: Set[str] = field(default_factory=set)

    def mttr(self) -> float:
        if self.first_seen_ts and self.resolved_ts and self.resolved_ts >= self.first_seen_ts:
            return self.resolved_ts - self.first_seen_ts
        return 0.0

@dataclass
class ValidationIssue:
    module: str
    issue_type: str
    severity: str  # 'error' | 'warning' | 'info'
    message: str
    suggestion: Optional[str] = None

@dataclass
class ValidationReport:
    total_modules: int
    validated_modules: int
    issues: List[ValidationIssue] = field(default_factory=list)
    duplicate_writers: Dict[str, List[str]] = field(default_factory=dict)
    missing_writers: Dict[str, List[str]] = field(default_factory=dict)
    cycles: List[List[str]] = field(default_factory=list)
    integration_score: float = 100.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_modules": self.total_modules,
            "validated_modules": self.validated_modules,
            "issues": [asdict(i) for i in self.issues],
            "duplicate_writers": self.duplicate_writers,
            "missing_writers": self.missing_writers,
            "cycles": self.cycles,
            "integration_score": self.integration_score,
        }

# ─────────────────────────────────────────────────────────────
# Graph helpers (fallbacks if networkx missing)
# ─────────────────────────────────────────────────────────────

class _Graph:
    """Tiny directed-graph helper with cycle detection, isolates, SCC groups."""
    def __init__(self) -> None:
        self.adj: Dict[str, Set[str]] = defaultdict(set)
        self.nodes: Set[str] = set()

    def add_node(self, n: str) -> None:
        self.nodes.add(n)

    def add_edge(self, u: str, v: str) -> None:
        self.nodes.add(u); self.nodes.add(v)
        self.adj[u].add(v)

    def isolates(self) -> List[str]:
        incoming: DefaultDict[str, int] = defaultdict(int)
        for u, vs in self.adj.items():
            for v in vs: incoming[v] += 1
        return [n for n in self.nodes if not self.adj.get(n) and incoming[n] == 0]

    def simple_cycles(self) -> List[List[str]]:
        cycles: List[List[str]] = []
        visited: Set[str] = set()
        stack: List[str] = []
        in_stack: Set[str] = set()

        def dfs(u: str) -> None:
            visited.add(u)
            stack.append(u); in_stack.add(u)
            for v in self.adj.get(u, set()):
                if v not in visited:
                    dfs(v)
                elif v in in_stack:
                    i = stack.index(v)
                    cyc = stack[i:] + [v]
                    if cyc not in cycles:
                        cycles.append(cyc)
            stack.pop(); in_stack.discard(u)

        for n in list(self.nodes):
            if n not in visited:
                dfs(n)
        return cycles

    def strongly_connected_components(self) -> List[Set[str]]:
        # Tarjan
        index = 0
        indices: Dict[str, int] = {}
        low: Dict[str, int] = {}
        stack: List[str] = []
        on_stack: Set[str] = set()
        sccs: List[Set[str]] = []

        def strongconnect(v: str) -> None:
            nonlocal index
            indices[v] = index; low[v] = index; index += 1
            stack.append(v); on_stack.add(v)
            for w in self.adj.get(v, set()):
                if w not in indices:
                    strongconnect(w)
                    low[v] = min(low[v], low[w])
                elif w in on_stack:
                    low[v] = min(low[v], indices[w])
            if low[v] == indices[v]:
                comp: Set[str] = set()
                while True:
                    w = stack.pop(); on_stack.remove(w)
                    comp.add(w)
                    if w == v: break
                sccs.append(comp)

        for v in self.nodes:
            if v not in indices:
                strongconnect(v)
        return sccs


# ─────────────────────────────────────────────────────────────
# SystemIntegritySuite (unified)
# ─────────────────────────────────────────────────────────────

class SystemIntegritySuite:
    """
    One-stop monitoring, validation, and (optional) visualization.

    Key capabilities:
      - Live lifecycle tracking (MISSING/STALE/FRESH), MTTR, flaps, provider changes
      - Heartbeat summaries + watchlist SLA alerts
      - Single-pass dependency audit (orphans, duplicates, danglers, stale)
      - Module graph validation (duplicate/missing writers, cycles)
      - Visualization (if networkx/matplotlib available)
      - Enhanced debug telemetry (ring buffer, stale root-cause, module hotspots)

    Integrations expected (overridable on __init__):
      • bus: InfoBus-like object with .subscribe/.unsubscribe/.get_data_freshness_report()
             and best-effort _providers/_consumers snapshots.
      • logger: RotatingLogger-like with info/warning/error(/debug).
      • orchestrator: optional; if present, used for module list + metadata.
    """

    # ── Construction ──────────────────────────────────────────
    def __init__(
        self,
        *,
        config: Optional[SuiteConfig] = None,
        bus: Any = None,
        logger: Optional[LoggerProto] = None,
        orchestrator: Any = None,
    ) -> None:
        # Allow env overrides for quick toggling
        env_debug = os.getenv("SIS_DEBUG")
        cfg = config or SuiteConfig()
        if env_debug is not None:
            cfg.debug = env_debug.lower() in {"1", "true", "yes", "on"}
        self.cfg = cfg

        self.bus = bus or _get_bus()
        self.logger = logger or LoggerClass(
            name="SystemIntegrity",
            log_path="logs/integrity/system.log",
            operator_mode=True,
            plain_english=True,
        )  # type: ignore[arg-type]
        self.orchestrator = orchestrator or _get_orchestrator()

        # Lifecycle store
        self._lock = threading.RLock()
        self._lifecycle: Dict[str, KeyLifecycle] = {}
        self._recent_resolved: Deque[Tuple[float, str, float]] = deque(maxlen=200)
        self._recent_flaps: Deque[Tuple[float, str, int]] = deque(maxlen=200)
        self._provider_changes: Deque[Tuple[float, str, str, str]] = deque(maxlen=200)

        # Debug telemetry
        self._events: Deque[Dict[str, Any]] = deque(maxlen=self.cfg.debug_ring_size)
        self._stale_index: Dict[str, Dict[str, Any]] = {}
        self._miss_index: Dict[str, Dict[str, Any]] = {}
        self._module_stats: DefaultDict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Live taps & heartbeat
        self._live_attached = False
        self._monitor_stop = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._heartbeat_interval = self.cfg.heartbeat_interval_seconds

        # Defaults for taps (safe even if attach_live_taps not called yet)
        self._show_values = False
        self._max_preview = 160
        self._problems_only = True
        self._ignore_re: Optional[re.Pattern] = re.compile(r"^(log_|module_events/|log_metrics_)")

        # Watchlist (customize as needed)
        self.watchlist: Set[str] = {
            "environment_config", "execution_mode", "time_risk_analysis",
            "pending_orders", "account_state", "market_state", "market_context"
        }

        self._d("Debug enabled")  # initial debug note
        self._i("🧭", "SystemIntegritySuite initialized")

    # ── Internal logging helpers ──────────────────────────────
    def _d(self, message: str, **ctx: Any) -> None:
        if self.cfg.debug:
            try:
                if hasattr(self.logger, "debug"):
                    self.logger.debug(_fmt("🔎", message, **ctx))  # type: ignore[attr-defined]
                else:
                    self.logger.info(_fmt("🔎", message, **ctx))
            except Exception:
                pass

    def _i(self, icon: str, message: str, **ctx: Any) -> None:
        try:
            self.logger.info(_fmt(icon, message, **ctx))
        except Exception:
            pass

    def _w(self, icon: str, message: str, **ctx: Any) -> None:
        try:
            self.logger.warning(_fmt(icon, message, **ctx))
        except Exception:
            pass

    def _e(self, icon: str, message: str, **ctx: Any) -> None:
        try:
            self.logger.error(_fmt(icon, message, **ctx))
        except Exception:
            pass

    def _record_event(self, etype: str, **fields: Any) -> None:
        if not (self.cfg.debug or self.cfg.log_on_every_event):
            return
        evt = {"ts": time.time(), "type": etype, **{k: v for k, v in fields.items() if v is not None}}
        with self._lock:
            self._events.append(evt)

    # ── Public debug toggles ──────────────────────────────────
    def enable_debug(self) -> None:
        self.cfg.debug = True
        self._d("Debug toggled ON by runtime")

    def disable_debug(self) -> None:
        self._d("Debug toggled OFF by runtime")
        self.cfg.debug = False

    # ── Lifecycle monitor: live taps ──────────────────────────
    def attach_live_taps(
        self,
        *,
        show_values: bool = False,
        max_preview: int = 160,
        problems_only: bool = True,
        ignore_pattern: Optional[str] = r"^(log_|module_events/|log_metrics_)",
    ) -> None:
        if self._live_attached:
            return
        self._show_values = show_values
        self._max_preview = max_preview
        self._problems_only = problems_only
        self._ignore_re = re.compile(ignore_pattern) if ignore_pattern else None

        # Subscribe to core InfoBus channels (best-effort)
        try:
            self.bus.subscribe("data_miss", self._on_miss)
            self.bus.subscribe("data_get_blocked", self._on_get_blocked)
            self.bus.subscribe("data_get", self._on_get_ok)
            self.bus.subscribe("data_updated", self._on_set)
            self.bus.subscribe("module_disabled", self._on_module_disabled)
            self.bus.subscribe("module_enabled", self._on_module_enabled)
            self._live_attached = True
            self._i("🔌", "Live taps attached", show_values=show_values, problems_only=problems_only)
        except Exception as e:
            self._e("💥", f"Failed to attach live taps: {e}")

    def detach_live_taps(self) -> None:
        if not self._live_attached:
            return
        for event, handler in [
            ("data_miss", self._on_miss),
            ("data_get_blocked", self._on_get_blocked),
            ("data_get", self._on_get_ok),
            ("data_updated", self._on_set),
            ("module_disabled", self._on_module_disabled),
            ("module_enabled", self._on_module_enabled),
        ]:
            try:
                self.bus.unsubscribe(event, handler)
            except Exception:
                pass
        self._live_attached = False
        self._i("🧹", "Live taps detached")

    # ── Heartbeat ─────────────────────────────────────────────
    def start_heartbeat(self, interval_s: float = 30.0) -> None:
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        self._heartbeat_interval = max(1.0, float(interval_s))
        self._monitor_stop.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, name="IntegrityHeartbeat", daemon=True)
        self._monitor_thread.start()
        self._i("⏱️", "Heartbeat loop started", interval=f"{self._heartbeat_interval:.1f}s")

    def stop_heartbeat(self) -> None:
        self._monitor_stop.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        self._i("⏹️", "Heartbeat loop stopped")

    def _monitor_loop(self) -> None:
        interval = self._heartbeat_interval
        while not self._monitor_stop.is_set():
            try:
                self._heartbeat_once()
            except Exception as e:
                self._e("💥", f"[HEARTBEAT] Error: {e}")
            self._monitor_stop.wait(interval)

    def _heartbeat_once(self) -> None:
        now = time.time()
        freshness, providers_map, consumers_map = self._snapshot_bus()
        # Map provider/consumer membership
        with self._lock:
            for k, provs in providers_map.items():
                self._lifecycle.setdefault(k, KeyLifecycle()).providers |= set(provs)
            for k, cons in consumers_map.items():
                self._lifecycle.setdefault(k, KeyLifecycle()).consumers |= set(cons)

        # Fresh / Stale
        for key, meta in freshness.items():
            age = float(meta.get("age_seconds", 0.0) or 0.0)
            provider = meta.get("source") or meta.get("source_module") or meta.get("provider")
            version = meta.get("version")
            if age <= self.cfg.stale_age_warn_seconds:
                self._mark_fresh(key, provider, version, now)
            else:
                self._mark_stale(key, provider, version, now)
                # Detailed root-cause trace for stale detected by heartbeat
                consumers = sorted(list(consumers_map.get(key, [])))
                self._note_stale(
                    key=key,
                    cause="heartbeat_age",
                    provider=provider,
                    version=version,
                    age=age,
                    consumers=consumers,
                    requester=None,
                )

        # Missing (consumed but no provider and not in freshness)
        for key, consumers in consumers_map.items():
            if key not in providers_map or not providers_map[key]:
                if key not in freshness:
                    self._mark_missing(key, "/".join(sorted(consumers)) if consumers else None, now)
                    self._note_missing(key=key, requester_hint=None, consumers=sorted(consumers))

        # Summary
        snap = self._lifecycle_snapshot()
        self._i("💓", "Heartbeat",
                 missing=len(snap["unresolved"]),
                 stale=len(snap["stale"]),
                 fresh=len(snap["fresh"]))

        # SLA for watchlist
        overdue: List[Tuple[str, float]] = []
        with self._lock:
            for key in self.watchlist:
                lc = self._lifecycle.get(key)
                if lc and lc.status != "FRESH" and lc.first_seen_ts:
                    age = now - lc.first_seen_ts
                    if age >= self.cfg.resolution_warn_seconds:
                        overdue.append((key, age))
        for key, age in overdue:
            self._w("⏰", "Watchlist SLA breach", key=key, age=f"{age:.1f}s",
                    threshold=f"{self.cfg.resolution_warn_seconds:.1f}s")

        # Escalate critically stale items
        criticals = self._collect_critically_stale(now)
        for entry in criticals[:10]:
            self._w("🐢", "Critically stale key",
                    key=entry["key"], age=f"{entry['age']:.1f}s",
                    provider=entry.get("provider") or "unknown",
                    consumers=",".join(entry.get("consumers", [])) or "-",
                    last_requesters=",".join(list(entry.get("last_requesters", []))[:4]) or "-")

    # ── One-shot audit & validation ───────────────────────────
    def validate_and_audit(self, *, title: str = "System Audit", export_path: Optional[str] = None) -> Dict[str, Any]:
        """
        1) Dependency audit (orphans/dups/danglers/stale + lifecycle snapshot)
        2) Module graph validation (duplicate or missing writers, cycles)
        """
        audit = self._scan_dependencies(title=title)
        validation = self._validate_modules()
        result = {
            "audit": audit,
            "validation": validation.to_dict()
        }
        if export_path:
            try:
                Path(export_path).parent.mkdir(parents=True, exist_ok=True)
                with open(export_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)
                self._i("📄", "Audit+Validation exported", path=export_path)
            except Exception as e:
                self._e("💥", f"Failed to export report: {e}")
        return result

    # ── Visualization (optional) ──────────────────────────────
    def visualize(self, output_path: Optional[str] = None) -> None:
        """
        Generate a dependency graph of providers→consumers using networkx/matplotlib if available.
        """
        output_path = output_path or self.cfg.viz_output_path
        _, providers_map, consumers_map = self._snapshot_bus()

        # Build graph
        if _nx and _plt:
            G = _nx.DiGraph()
            for key, providers in providers_map.items():
                for p in providers:
                    # draw edge: provider module -> each consumer module for that key
                    for c in consumers_map.get(key, []):
                        G.add_edge(p, c)  # module-level dependency via shared key

            _plt.figure(figsize=(12, 8))
            pos = _nx.spring_layout(G, k=1)  # type: ignore[arg-type]
            _nx.draw_networkx_nodes(G, pos, node_size=1000)
            _nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20)
            _nx.draw_networkx_labels(G, pos, font_size=9)
            _plt.title("Module Dependency Graph")
            _plt.axis('off')
            _plt.savefig(output_path, dpi=300, bbox_inches='tight')
            _plt.close()
            self._i("🗺️", "Visualization generated", path=output_path, nodes=len(G.nodes), edges=len(G.edges))
        else:
            self._w("🖼️", "Visualization skipped (networkx/matplotlib not available)")

    # ── Internal: dependency audit ────────────────────────────
    def _scan_dependencies(self, *, title: str) -> Dict[str, Any]:
        t0 = time.time()
        freshness, providers_map, consumers_map = self._snapshot_bus()

        # Orphans: consumed but no provider
        orphans: List[Tuple[str, List[str]]] = []
        for key, consumers in consumers_map.items():
            if not providers_map.get(key):
                orphans.append((key, sorted(consumers)))

        # Duplicates: >1 provider
        dups: List[Tuple[str, List[str]]] = [
            (k, sorted(list(providers)))
            for k, providers in providers_map.items()
            if len(providers) > 1
        ]

        # Danglers: provided but no consumers
        danglers: List[Tuple[str, List[str]]] = []
        for key, provs in providers_map.items():
            if not consumers_map.get(key):
                danglers.append((key, sorted(list(provs))))

        # Stale keys
        stale: List[Tuple[str, Dict[str, Any]]] = []
        for key, meta in freshness.items():
            try:
                if float(meta.get("age_seconds", 0.0)) > self.cfg.stale_age_warn_seconds:
                    stale.append((key, meta))
            except Exception:
                pass

        # Lifecycle slice
        lifecycle = self._lifecycle_snapshot()

        elapsed_ms = int((time.time() - t0) * 1000)
        results = {
            "title": title,
            "summary": {
                "providers_total": sum(len(v) for v in providers_map.values()),
                "consumers_total": sum(len(v) for v in consumers_map.values()),
                "active_keys": len(freshness),
                "stale_threshold_seconds": self.cfg.stale_age_warn_seconds,
                "elapsed_ms": elapsed_ms,
            },
            "orphans": sorted(orphans, key=lambda x: x[0]),
            "duplicate_providers": sorted(dups, key=lambda x: x[0]),
            "dangling_providers": sorted(danglers, key=lambda x: x[0]),
            "stale": sorted(stale, key=lambda x: x[0]),
            "lifecycle": lifecycle,
            "generated_at": time.time(),
        }

        # Log concise summary
        self._i("📋", "Dependency audit",
                orphans=len(orphans), dups=len(dups),
                danglers=len(danglers), stale=len(stale),
                elapsed=f"{elapsed_ms}ms")
        return results

    # ── Internal: module validation ───────────────────────────
    def _validate_modules(self) -> ValidationReport:
        """
        Validates module dependency graph for:
          - duplicate writers of critical keys
          - missing writers for required keys
          - potential cycles between modules
        Uses orchestrator if available; otherwise best-effort static scan.
        """
        modules: Set[str] = set()
        provides: DefaultDict[str, Set[str]] = defaultdict(set)  # key -> provider modules
        requires: DefaultDict[str, Set[str]] = defaultdict(set)  # key -> consumer modules

        # Preferred path: orchestrator metadata (if your project exposes it)
        if self.orchestrator and getattr(self.orchestrator, "metadata", None):
            for mname, meta in self.orchestrator.metadata.items():  # type: ignore[attr-defined]
                modules.add(mname)
                for k in getattr(meta, "provides", []) or []:
                    provides[k].add(mname)
                for k in getattr(meta, "requires", []) or []:
                    requires[k].add(mname)
        else:
            # Fallback: derive from InfoBus provider/consumer snapshots or reconstruct
            _, providers_map, consumers_map = self._snapshot_bus()
            for key, provs in providers_map.items():
                for p in provs:
                    modules.add(p)
                    provides[key].add(p)
            for key, cons in consumers_map.items():
                for c in cons:
                    modules.add(c)
                    requires[key].add(c)

            # As a last resort, do a lightweight AST scan to discover module class names
            modules |= self._discover_module_class_names(Path(self.cfg.modules_root))

        report = ValidationReport(total_modules=len(modules), validated_modules=len(modules))

        # Duplicate writers (critical keys only)
        duplicate: Dict[str, List[str]] = {}
        for key, provs in provides.items():
            if len(provs) > 1 and (key in self.cfg.critical_single_writer_keys or key.endswith(self.cfg.critical_single_writer_suffixes)):
                duplicate[key] = sorted(provs)
        report.duplicate_writers = duplicate
        for k, mods in duplicate.items():
            report.issues.append(ValidationIssue(
                module="Graph",
                issue_type="duplicate_writer",
                severity="warning",
                message=f"Key '{k}' has multiple providers: {mods}",
                suggestion="Enforce single-writer policy or namespace outputs."
            ))

        # Missing writers
        missing: Dict[str, List[str]] = {}
        for key, consumers in requires.items():
            if not provides.get(key):
                missing[key] = sorted(consumers)
        report.missing_writers = missing
        for k, mods in missing.items():
            report.issues.append(ValidationIssue(
                module="Graph",
                issue_type="missing_writer",
                severity="error",
                message=f"Key '{k}' has no providers (needed by {mods})",
                suggestion="Add a provider module or remove/soften requirement."
            ))

        # Module-level graph: edges from provider module -> consumer module if any shared key
        if _nx:
            G = _nx.DiGraph()
            for key, provs in provides.items():
                for p in provs:
                    for c in requires.get(key, set()):
                        G.add_edge(p, c)
            cycles = [list(c) for c in _nx.simple_cycles(G)]  # type: ignore
        else:
            G = _Graph()
            for m in modules:
                G.add_node(m)
            for key, provs in provides.items():
                for p in provs:
                    for c in requires.get(key, set()):
                        G.add_edge(p, c)
            cycles = G.simple_cycles()

        report.cycles = [c for c in cycles if len(c) > 2 or (len(c) == 2 and c[0] != c[1])]
        for cyc in report.cycles[:5]:
            report.issues.append(ValidationIssue(
                module="Graph",
                issue_type="cycle",
                severity="warning",
                message=f"Potential dependency cycle: {' → '.join(cyc)}",
                suggestion="Break cycles via decoupling or bus-first handoffs."
            ))

        # Score (simple penalties)
        score = 100.0
        for i in report.issues:
            if i.severity == "error":
                score -= 4.0
            elif i.severity == "warning":
                score -= 1.5
            else:
                score -= 0.5
        score -= 1.5 * len(report.duplicate_writers)
        score -= 0.5 * min(5, len(report.cycles))
        report.integration_score = max(0.0, min(100.0, score))

        self._i("🧪", "Validation",
                total=report.total_modules,
                dup_writers=len(report.duplicate_writers),
                missing=len(report.missing_writers),
                cycles=len(report.cycles),
                score=f"{report.integration_score:.1f}%")
        # Publish a tiny summary on the bus if available
        try:
            self.bus.set("validation/summary", {
                "score": report.integration_score,
                "modules": report.total_modules,
                "duplicate_writers": len(report.duplicate_writers),
                "missing_writers": len(report.missing_writers),
                "cycles": len(report.cycles),
            }, module="SystemIntegritySuite", thesis="Validation summary")
        except Exception:
            pass

        return report

    # ── Helpers: lifecycle events ─────────────────────────────
    def _should_ignore(self, key: Any) -> bool:
        try:
            s = str(key or "")
        except Exception:
            return False
        return bool(self._ignore_re and self._ignore_re.search(s))

    def _preview(self, txt: Any) -> str:
        try:
            s = repr(txt)
        except Exception:
            s = str(txt)
        return s if len(s) <= self._max_preview else s[: self._max_preview] + "…"

    def _on_miss(self, evt: Dict[str, Any]) -> None:
        key = evt.get("key")
        if self._should_ignore(key): return
        now = time.time()
        requester = evt.get("module")
        self._mark_missing(str(key), requester, now)
        self._note_missing(key=str(key), requester_hint=requester, consumers=None)
        if requester:
            self._module_stats[requester]["miss"] += 1
        self._w("❌", "BUS MISS", key=key, requester=requester)

    def _on_get_ok(self, evt: Dict[str, Any]) -> None:
        key = evt.get("key")
        if self._should_ignore(key): return
        now = time.time()
        provider = evt.get("source_module") or evt.get("provider")
        version = evt.get("version")
        requester = evt.get("module")
        age = float(evt.get("age_seconds", 0.0) or 0.0)
        self._mark_fresh(str(key), provider, version, now)
        if provider:
            self._module_stats[provider]["set_served"] += 1
        if requester:
            self._module_stats[requester]["get_ok"] += 1
        self._record_event("get_ok", key=key, provider=provider, requester=requester, version=version, age=age)
        if not self._problems_only and (self.cfg.debug or self.cfg.log_on_every_event):
            self._i("📦", "BUS GET", key=key, provider=provider,
                    age=f"{age:.2f}s",
                    preview=self._preview(evt.get("preview")) if self._show_values else "hidden")

    def _on_get_blocked(self, evt: Dict[str, Any]) -> None:
        key = evt.get("key")
        if self._should_ignore(key): return
        now = time.time()
        provider = evt.get("source_module") or evt.get("provider")
        version = evt.get("version")
        requester = evt.get("module")
        reason = evt.get("reason")
        age = float(evt.get("age_seconds", 0.0) or 0.0)
        self._mark_stale(str(key), provider, version, now)
        self._note_stale(
            key=str(key),
            cause="get_blocked",
            provider=provider,
            version=version,
            age=age,
            consumers=None,
            requester=requester,
            reason=reason,
        )
        if requester:
            self._module_stats[requester]["get_blocked"] += 1
        if provider:
            self._module_stats[provider]["served_stale"] += 1
        self._w("⛔", "BUS GET BLOCKED", key=key, requester=requester,
                provider=provider, reason=reason, age=f"{age:.2f}s")

    def _on_set(self, evt: Dict[str, Any]) -> None:
        key = evt.get("key")
        if self._should_ignore(key): return
        now = time.time()
        provider = evt.get("module")
        version = evt.get("version")
        self._mark_fresh(str(key), provider, version, now)
        if provider:
            self._module_stats[provider]["set"] += 1
        self._record_event("set", key=key, provider=provider, version=version)
        if not self._problems_only and (self.cfg.debug or self.cfg.log_on_every_event):
            self._i("📝", "BUS SET", key=key, provider=provider, version=version)

    def _on_module_disabled(self, evt: Dict[str, Any]) -> None:
        try:
            cb = evt.get("circuit_breaker_state", {}) or {}
            reason = evt.get("reason") or cb.get("open_reason")
            last_error = cb.get("last_error") or evt.get("error")
            module = evt.get("module")
            if module:
                self._module_stats[module]["disabled"] += 1
            self._e("🚫",
                "MODULE DISABLED",
                module=module,
                failures=evt.get("failures"),
                consecutive=evt.get("consecutive_failures"),
                failure_rate=f"{float(evt.get('failure_rate', 0.0)):.1%}" if evt.get('failure_rate') is not None else None,
                reason=reason,
                last_error=(str(last_error)[:160] if last_error else None),
            )
        except Exception:
            # Fallback to minimal message
            self._e("🚫", "MODULE DISABLED", module=evt.get("module"),
                    failures=evt.get("failures"), consecutive=evt.get("consecutive_failures"))

    def _on_module_enabled(self, evt: Dict[str, Any]) -> None:
        module = evt.get("module")
        if module:
            self._module_stats[module]["enabled"] += 1
        self._i("✅", "MODULE ENABLED", module=module)

    # ── Helpers: lifecycle transitions ────────────────────────
    def _get_lc(self, key: str) -> KeyLifecycle:
        with self._lock:
            return self._lifecycle.setdefault(key, KeyLifecycle())

    def _transition(self, key: str, new_status: str, now: float, provider: Optional[str], version: Any) -> None:
        with self._lock:
            lc = self._lifecycle.setdefault(key, KeyLifecycle())
            old = lc.status
            lc.last_event_ts = now

            # Provider change tracking
            if provider and provider != lc.last_provider and lc.last_provider is not None:
                lc.provider_changes += 1
                self._provider_changes.append((now, key, lc.last_provider, provider))
                self._i("🔁", "Provider changed", key=key, old=lc.last_provider, new=provider)
            if provider:
                lc.last_provider = provider
            try:
                lc.last_version = int(version) if version is not None else lc.last_version
            except Exception:
                pass

            if old != new_status:
                if old in ("MISSING", "STALE") and new_status == "FRESH":
                    if not lc.first_seen_ts:
                        lc.first_seen_ts = now
                    lc.resolved_ts = now
                    lc.flap_count += 1
                    self._recent_resolved.append((now, key, lc.mttr()))
                elif old == "FRESH" and new_status in ("MISSING", "STALE"):
                    lc.flap_count += 1
                    self._recent_flaps.append((now, key, lc.flap_count))
                if new_status in ("MISSING", "STALE") and (old == "FRESH" or lc.first_seen_ts == 0.0):
                    lc.first_seen_ts = now
                    lc.resolved_ts = 0.0

            lc.status = new_status

    def _mark_missing(self, key: str, requester_hint: Optional[str], now: float) -> None:
        lc = self._get_lc(key)
        lc.miss_count += 1
        self._transition(key, "MISSING", now, None, None)
        self._record_event("miss", key=key, requester=requester_hint)
        if lc.miss_count == 1:
            self._w("🕵️", "Tracking missing key", key=key, requester=requester_hint or "unknown")

    def _mark_stale(self, key: str, provider: Optional[str], version: Any, now: float) -> None:
        lc = self._get_lc(key)
        lc.blocked_count += 1
        self._transition(key, "STALE", now, provider, version)
        self._record_event("stale", key=key, provider=provider, version=version)

    def _mark_fresh(self, key: str, provider: Optional[str], version: Any, now: float) -> None:
        lc = self._get_lc(key)
        lc.set_count += 1
        self._transition(key, "FRESH", now, provider, version)
        self._record_event("fresh", key=key, provider=provider, version=version)

    def _lifecycle_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            unresolved, stale, fresh = [], [], []
            for key, lc in self._lifecycle.items():
                row = {
                    "status": lc.status,
                    "miss": lc.miss_count,
                    "blocked": lc.blocked_count,
                    "set": lc.set_count,
                    "flaps": lc.flap_count,
                    "provider_changes": lc.provider_changes,
                    "mttr_s": round(lc.mttr(), 3),
                    "last_provider": lc.last_provider,
                    "last_version": lc.last_version,
                    "consumers": sorted(list(lc.consumers)) if lc.consumers else [],
                    "providers": sorted(list(lc.providers)) if lc.providers else [],
                }
                if lc.status == "FRESH":
                    fresh.append((key, row))
                elif lc.status == "STALE":
                    stale.append((key, row))
                elif lc.status == "MISSING":
                    unresolved.append((key, row))
            return {
                "unresolved": sorted(unresolved, key=lambda x: x[0]),
                "stale": sorted(stale, key=lambda x: x[0]),
                "fresh": sorted(fresh, key=lambda x: x[0]),
                "stats": {
                    "tracked_keys": len(self._lifecycle),
                    "recent_resolved": list(self._recent_resolved),
                    "recent_flaps": list(self._recent_flaps),
                    "provider_changes": list(self._provider_changes),
                },
                "watchlist": sorted(list(self.watchlist)),
            }

    # ── Helpers: data snapshots & discovery ───────────────────
    def _snapshot_bus(self) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Set[str]], Dict[str, Set[str]]]:
        """
        Return (freshness, providers_map, consumers_map) with multiple fallbacks:
          1) Native bus maps (_providers/_consumers) if present & non-empty
          2) A bus method called get_provider_consumer_snapshot() if your bus exposes one
          3) Reconstruction via orchestrator.metadata + bus.get_providers(key)
          4) Final empty maps (no crash)
        """
        freshness: Dict[str, Dict[str, Any]] = {}
        providers_map: Dict[str, Set[str]] = {}
        consumers_map: Dict[str, Set[str]] = {}

        # Freshness (best-effort)
        try:
            fr = self.bus.get_data_freshness_report()
            if isinstance(fr, dict):
                freshness = fr
        except Exception:
            freshness = {}

        # 1) Native bus snapshots (_providers/_consumers)
        try:
            _p = getattr(self.bus, "_providers", {}) or {}
            _c = getattr(self.bus, "_consumers", {}) or {}
            if _p or _c:
                providers_map = {k: set(v) for k, v in _p.items()}
                consumers_map = {k: set(v) for k, v in _c.items()}
                return freshness, providers_map, consumers_map
        except Exception:
            pass

        # 2) Custom bus API (optional): get_provider_consumer_snapshot()
        try:
            if hasattr(self.bus, "get_provider_consumer_snapshot"):
                snap = self.bus.get_provider_consumer_snapshot()  # expected: {"providers": {k:[mods]}, "consumers": {k:[mods]}}
                if isinstance(snap, dict):
                    p = snap.get("providers") or {}
                    c = snap.get("consumers") or {}
                    if isinstance(p, dict) or isinstance(c, dict):
                        providers_map = {k: set(v) for k, v in (p or {}).items()}
                        consumers_map = {k: set(v) for k, v in (c or {}).items()}
                        return freshness, providers_map, consumers_map
        except Exception:
            pass

        # 3) Reconstruct from orchestrator metadata + bus.get_providers()
        try:
            pm, cm = self._reconstruct_maps_from_orchestrator()
            if pm or cm:
                return freshness, pm, cm
        except Exception:
            pass

        # 4) Nothing available – return empty maps (suite stays quiet but alive)
        return freshness, {}, {}

    def _reconstruct_maps_from_orchestrator(self) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
        """
        Build providers/consumers maps using orchestrator.metadata and bus.get_providers(key).
        This lets the suite work even when the bus doesn't expose internal maps.
        """
        providers_map: Dict[str, Set[str]] = defaultdict(set)
        consumers_map: Dict[str, Set[str]] = defaultdict(set)

        # Prefer orchestrator if available
        meta = getattr(self.orchestrator, "metadata", None)
        if not meta or not hasattr(self.bus, "get_providers"):
            return {}, {}

        for module_name, m in meta.items():
            reqs = (getattr(m, "requires", []) or [])
            provs = (getattr(m, "provides", []) or [])

            # Consumers: this module requires these keys
            for key in reqs:
                consumers_map[key].add(module_name)

                # Providers: ask the bus who writes this key (if any)
                try:
                    writers = self.bus.get_providers(key) or []
                    for w in writers:
                        providers_map[key].add(str(w))
                except Exception:
                    # If the bus can't answer yet, we still have valid consumers; providers may remain empty for now.
                    pass

            # Providers (declared): even if bus can't resolve, at least register declared writers
            for key in provs:
                providers_map[key].add(module_name)

        # Cast defaultdicts to plain dicts for safety
        return dict(providers_map), dict(consumers_map)

    def _discover_module_class_names(self, root: Path) -> Set[str]:
        names: Set[str] = set()
        if not root.exists():
            return names
        for py in root.rglob("*.py"):
            if py.name.startswith("_") or py.name == "__init__.py": continue
            try:
                src = py.read_text(encoding="utf-8")
                tree = ast.parse(src)
            except Exception:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Heuristic: classes that declare process()/get_state()/set_state()
                    has_process = False; has_support = False
                    for b in node.body:
                        if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if b.name in {"process", "step"}:
                                has_process = True
                            if b.name in {"get_state", "set_state"}:
                                has_support = True
                    if has_process and has_support:
                        names.add(node.name)
        return names

    # ── Diagnostics: quick debug snapshot ─────────────────────
    def debug_dump(self) -> Dict[str, Any]:
        """One-shot, human-readable snapshot counts; prints to logger and returns the data."""
        fr, pm, cm = self._snapshot_bus()
        
        # Top hotspots (modules causing/experiencing issues)
        miss_hot = sorted(((m, s.get("miss", 0)) for m, s in self._module_stats.items()), key=lambda x: x[1], reverse=True)[:5]
        block_hot = sorted(((m, s.get("get_blocked", 0)) for m, s in self._module_stats.items()), key=lambda x: x[1], reverse=True)[:5]
        stale_served = sorted(((m, s.get("served_stale", 0)) for m, s in self._module_stats.items()), key=lambda x: x[1], reverse=True)[:5]
        
        data: Dict[str, Any] = {
            "freshness_keys": len(fr),
            "provider_keys": len(pm),
            "consumer_keys": len(cm),
            "providers_total": sum(len(v) for v in pm.values()),
            "consumers_total": sum(len(v) for v in cm.values()),
            "tracked_lifecycle": len(self._lifecycle),
            "events_buffer_size": len(self._events),
            "hotspots": {
                "miss_requesters": [h for h in miss_hot if h[1] > 0],
                "blocked_requesters": [h for h in block_hot if h[1] > 0],
                "stale_providers": [h for h in stale_served if h[1] > 0],
            }
        }

        self._i("🔎", "Debug dump", **data)
        return data

    # ── Health monitoring interface (for HealthMonitor integration) ──
    def get_health_status(self) -> Dict[str, Any]:
        """
        Returns health status compatible with HealthMonitor.
        Reports on suite's own operational health and system integrity metrics.
        """
        try:
            # Get lifecycle snapshot
            lifecycle = self._lifecycle_snapshot()
            fr, pm, cm = self._snapshot_bus()

            # Count issues
            unresolved_count = len(lifecycle.get('unresolved', []))
            stale_count = len(lifecycle.get('stale', []))
            fresh_count = len(lifecycle.get('fresh', []))
            total_tracked = len(self._lifecycle)

            # Determine health status
            status = 'OK'
            is_healthy = True
            issues = []

            # Check for critical issues
            if unresolved_count > 10:
                status = 'DEGRADED'
                is_healthy = False
                issues.append(f"{unresolved_count} unresolved data keys")

            if stale_count > 5:
                if status == 'OK':
                    status = 'WARNING'
                issues.append(f"{stale_count} stale data keys")

            # Check if monitoring is active
            if self._monitor_thread and not self._monitor_thread.is_alive():
                status = 'DEGRADED'
                is_healthy = False
                issues.append("Heartbeat monitor not running")

            # Check if taps are attached
            if not self._live_attached:
                issues.append("Live taps not attached (reduced observability)")

            # Check for watchlist violations
            watchlist_violations = 0
            with self._lock:
                for key in self.watchlist:
                    lc = self._lifecycle.get(key)
                    if lc and lc.status != "FRESH":
                        watchlist_violations += 1

            if watchlist_violations > 0:
                status = 'WARNING'
                issues.append(f"{watchlist_violations}/{len(self.watchlist)} watchlist keys not fresh")

            # Top offenders summary
            hotspots = self.debug_dump().get("hotspots", {}) if self.cfg.debug else None

            return {
                'status': status,
                'module': 'SystemIntegritySuite',
                'version': '1.1',
                'is_healthy': is_healthy,
                'tracked_keys': total_tracked,
                'unresolved_keys': unresolved_count,
                'stale_keys': stale_count,
                'fresh_keys': fresh_count,
                'watchlist_violations': watchlist_violations,
                'heartbeat_active': bool(self._monitor_thread and self._monitor_thread.is_alive()),
                'taps_attached': self._live_attached,
                'provider_keys': len(pm),
                'consumer_keys': len(cm),
                'issues': issues if issues else None,
                'performance': {
                    'total_providers': sum(len(v) for v in pm.values()),
                    'total_consumers': sum(len(v) for v in cm.values()),
                    'lifecycle_tracked': total_tracked,
                    'recent_flaps': len(self._recent_flaps),
                    'provider_changes': len(self._provider_changes)
                },
                'hotspots': hotspots
            }
        except Exception as e:
            return {
                'status': 'ERROR',
                'module': 'SystemIntegritySuite',
                'version': '1.1',
                'is_healthy': False,
                'error': str(e),
                'last_error': str(e)
            }

    # ── Enhanced debug analytics (stale & miss RCA) ───────────
    def _note_stale(
        self,
        *,
        key: str,
        cause: str,                      # 'heartbeat_age' | 'get_blocked'
        provider: Optional[str],
        version: Any,
        age: Optional[float],
        consumers: Optional[List[str]],
        requester: Optional[str],
        reason: Optional[str] = None,
    ) -> None:
        now = time.time()
        with self._lock:
            rec = self._stale_index.setdefault(key, {
                "key": key,
                "first_ts": now,
                "last_ts": now,
                "count": 0,
                "provider": provider,
                "last_version": version,
                "last_age": age,
                "causes": set(),
                "consumers": set(consumers or []),
                "last_requesters": set(),
                "reasons": set(),
            })
            rec["last_ts"] = now
            rec["count"] += 1
            rec["provider"] = provider or rec.get("provider")
            rec["last_version"] = version if version is not None else rec.get("last_version")
            if age is not None:
                rec["last_age"] = age
            if consumers:
                rec["consumers"].update(consumers)
            if requester:
                rec["last_requesters"].add(requester)
            if reason:
                rec["reasons"].add(str(reason))
            rec["causes"].add(cause)

        self._record_event("stale_trace", key=key, cause=cause, provider=provider,
                           requester=requester, age=age, reason=reason)

        # Optional chatty debug line
        if self.cfg.debug:
            self._d("Stale trace",
                    key=key, cause=cause, provider=provider or "unknown",
                    requester=requester or "-", age=f"{age:.2f}" if age is not None else "NA",
                    reasons="|".join(rec["reasons"]) if rec.get("reasons") else "-")

    def _note_missing(self, *, key: str, requester_hint: Optional[str], consumers: Optional[List[str]]) -> None:
        now = time.time()
        with self._lock:
            rec = self._miss_index.setdefault(key, {
                "key": key,
                "first_ts": now,
                "last_ts": now,
                "count": 0,
                "requesters": set(),
                "consumers": set(consumers or []),
            })
            rec["last_ts"] = now
            rec["count"] += 1
            if requester_hint:
                rec["requesters"].add(requester_hint)
            if consumers:
                rec["consumers"].update(consumers)
        self._record_event("miss_trace", key=key, requester=requester_hint)

    def get_stale_report(self, *, top_n: Optional[int] = None) -> Dict[str, Any]:
        with self._lock:
            items = []
            now = time.time()
            for key, rec in self._stale_index.items():
                age = float(rec.get("last_age") or 0.0)
                items.append({
                    "key": key,
                    "events": rec["count"],
                    "provider": rec.get("provider"),
                    "last_version": rec.get("last_version"),
                    "last_age_s": age,
                    "first_seen_ago_s": now - float(rec["first_ts"]),
                    "last_seen_ago_s": now - float(rec["last_ts"]),
                    "causes": sorted(list(rec.get("causes", []))),
                    "consumers": sorted(list(rec.get("consumers", []))),
                    "last_requesters": sorted(list(rec.get("last_requesters", []))),
                    "reasons": sorted(list(rec.get("reasons", []))),
                    "critical": age >= self.cfg.stale_age_critical_seconds if age else False,
                })
            items.sort(key=lambda x: (x["critical"], x["events"], x["last_age_s"]), reverse=True)
            if top_n is not None:
                items = items[:top_n]

            # Group by provider and by requester (hotspots)
            by_provider: DefaultDict[str, int] = defaultdict(int)
            by_requester: DefaultDict[str, int] = defaultdict(int)
            for it in items:
                prov = it.get("provider") or "unknown"
                by_provider[prov] += 1
                for r in it.get("last_requesters", []) or []:
                    by_requester[r] += 1

            summary = {
                "total_stale_keys": len(self._stale_index),
                "critically_stale_keys": len([1 for rec in items if rec.get("critical")]),
                "top_providers": sorted(by_provider.items(), key=lambda x: x[1], reverse=True)[:10],
                "top_requesters": sorted(by_requester.items(), key=lambda x: x[1], reverse=True)[:10],
                "items": items,
            }
            return summary

    def export_debug(self, path: str) -> None:
        """Export events, stale RCA, misses, and module stats for offline forensics."""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            out = {
                "generated_at": time.time(),
                "events": list(self._events),
                "stale_report": self.get_stale_report(),
                "miss_index": {
                    k: {
                        "count": v["count"],
                        "first_ts": v["first_ts"],
                        "last_ts": v["last_ts"],
                        "requesters": sorted(list(v.get("requesters", []))),
                        "consumers": sorted(list(v.get("consumers", []))),
                    }
                    for k, v in self._miss_index.items()
                },
                "module_stats": {m: dict(stats) for m, stats in self._module_stats.items()},
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
            self._i("🧾", "Debug export written", path=path,
                    events=len(out["events"]), stale=len(out["stale_report"]["items"]))
        except Exception as e:
            self._e("💥", f"Failed to export debug: {e}")

    def _collect_critically_stale(self, now: float) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        with self._lock:
            for key, rec in self._stale_index.items():
                age = float(rec.get("last_age") or 0.0)
                if age and age >= self.cfg.stale_age_critical_seconds:
                    out.append({
                        "key": key,
                        "age": age,
                        "provider": rec.get("provider"),
                        "consumers": sorted(list(rec.get("consumers", []))),
                        "last_requesters": sorted(list(rec.get("last_requesters", []))),
                    })
        out.sort(key=lambda x: x["age"], reverse=True)
        return out


# ─────────────────────────────────────────────────────────────
# CLI demo (optional): python monitoring_unified.py
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Try to force a real bus if available in this process
    try:
        from modules.utils.info_bus import InfoBusManager  # type: ignore
        bus = InfoBusManager.get_instance()
    except Exception:
        bus = _get_bus()

    suite = SystemIntegritySuite(bus=bus)
    # Example: toggle debug quickly via env SIS_DEBUG=1
    suite.attach_live_taps(ignore_pattern=None, problems_only=False, show_values=False)
    suite.start_heartbeat(interval_s=suite.cfg.heartbeat_interval_seconds)
    try:
        time.sleep(1.0)  # let providers register if system is booting
        out = suite.validate_and_audit(export_path="logs/integrity/audit_validation.json")
        print("AUDIT SUMMARY:", json.dumps(out["audit"]["summary"], indent=2))
        print("VALIDATION:", json.dumps(out["validation"], indent=2))
        print("DEBUG:", suite.debug_dump())
        # Export richer debug bundle
        suite.export_debug("logs/integrity/debug_bundle.json")
        suite.visualize()  # requires networkx + matplotlib
        # Optional: print top stale root causes
        if suite.cfg.debug:
            print("STALE REPORT:", json.dumps(suite.get_stale_report(top_n=20), indent=2))
    finally:
        time.sleep(1.0)
        suite.stop_heartbeat()
