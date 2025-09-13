# modules/monitoring/dependency_inspector.py
from __future__ import annotations

import re
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Tuple, DefaultDict, Set, Deque
from collections import defaultdict, deque

from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class KeyLifecycle:
    """
    Per-key lifecycle tracker.
    status: UNKNOWN | MISSING | STALE | FRESH
    """
    status: str = "UNKNOWN"
    first_seen_ts: float = 0.0          # when we first observed an issue (MISSING/STALE)
    last_event_ts: float = 0.0          # last touched time (any event)
    resolved_ts: float = 0.0            # when we transitioned to FRESH from a non-fresh state
    miss_count: int = 0                 # live 'data_miss' events
    blocked_count: int = 0              # live 'data_get_blocked' events
    set_count: int = 0                  # live 'data_updated' events
    flap_count: int = 0                 # number of MISSING/STALE <-> FRESH transitions
    provider_changes: int = 0           # number of times provider changed
    last_provider: Optional[str] = None # last provider module seen on set/freshness
    last_version: Optional[int] = None  # last version id from freshness/set event
    last_source: Optional[str] = None   # last 'source' from freshness
    consumers: Set[str] = field(default_factory=set)
    providers: Set[str] = field(default_factory=set)

    def age(self, now: float) -> float:
        return max(0.0, now - (self.first_seen_ts or now))

    def mttr(self) -> float:
        if self.first_seen_ts and self.resolved_ts and self.resolved_ts >= self.first_seen_ts:
            return self.resolved_ts - self.first_seen_ts
        return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Main Inspector
# ──────────────────────────────────────────────────────────────────────────────

class DependencyInspector:
    """
    DependencyInspector
    -------------------
    Live bus taps (optional) + on-demand, single-pass dependency audit + stateful
    lifecycle monitoring with heartbeats.

    Use cases:
      - attach_live_taps(): subscribe to bus events during a focused debugging session.
      - start_monitoring(): periodic heartbeats summarizing unresolved/resolved state.
      - scan_once(): one-shot audit with grouped findings.
      - emit_report(): legacy wrapper for scan_once() (noise-free snapshot).

    Features added:
      ✓ Lifecycle states (MISSING→STALE→FRESH) per key
      ✓ MTTR computation when a key resolves
      ✓ Flap detection (toggle between FRESH and non-fresh)
      ✓ Provider change alerts (single-writer drift visibility)
      ✓ Critical watchlist with SLA breach warnings
      ✓ Heartbeat summaries: still-missing, newly resolved, flapping
      ✓ Works even with problems_only=True (uses events internally without logging noise)
    """

    # ---------- Lifecycle ----------

    def __init__(
        self,
        bus=None,
        log: Optional[RotatingLogger] = None,
        issue_log: Optional[RotatingLogger] = None,
        stale_age_warn_seconds: float = 60.0,
    ):
        self.bus = bus or InfoBusManager.get_instance()

        self.logger = log or RotatingLogger(
            name="DependencyInspector",
            log_path="logs/monitoring/dependency_inspector.log",
            max_lines=20000,
            operator_mode=True,
            plain_english=True,
        )

        # Issues-only logger to keep the main log clean/compact
        self.issue_logger = issue_log or RotatingLogger(
            name="DependencyInspectorIssues",
            log_path="logs/monitoring/dependency_issues.log",
            max_lines=20000,
            operator_mode=True,
            plain_english=True,
        )

        # Thresholds & noisy-key filters
        self.stale_age_warn_seconds = float(stale_age_warn_seconds)
        self._live_attached = False
        self.show_values = True
        self.max_preview = 160
        self.problems_only = True
        self.ignore_key_re: Optional[re.Pattern] = re.compile(r"^(log_|module_events/|log_metrics_)")

        # Stateful lifecycle store
        self._lock = threading.RLock()
        self._lifecycle: Dict[str, KeyLifecycle] = {}
        self._recent_resolved: Deque[Tuple[float, str, float]] = deque(maxlen=200)  # (ts, key, mttr)
        self._recent_flaps: Deque[Tuple[float, str, int]] = deque(maxlen=200)       # (ts, key, flap_count)
        self._provider_changes: Deque[Tuple[float, str, str, str]] = deque(maxlen=200)  # (ts, key, old, new)

        # Heartbeat / watchlist
        self.resolution_warn_seconds: float = 15.0     # SLA for watchlist keys to get first FRESH
        self.heartbeat_interval_seconds: float = 5.0   # report cadence
        self.watchlist: Set[str] = set([
            "environment_config", "execution_mode", "env_mode",
            "time_risk_analysis", "pending_orders", "account_state",
            "market_state", "market_context", "performance_data",
        ])

        # Monitor thread
        self._monitor_stop = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None

    # ---------- Configuration helpers ----------

    def set_thresholds(
        self,
        *,
        stale_age_warn_seconds: Optional[float] = None,
        resolution_warn_seconds: Optional[float] = None,
        heartbeat_interval_seconds: Optional[float] = None,
    ) -> None:
        with self._lock:
            if stale_age_warn_seconds is not None:
                self.stale_age_warn_seconds = float(stale_age_warn_seconds)
            if resolution_warn_seconds is not None:
                self.resolution_warn_seconds = float(resolution_warn_seconds)
            if heartbeat_interval_seconds is not None:
                self.heartbeat_interval_seconds = float(heartbeat_interval_seconds)

    def set_ignore_pattern(self, pattern: Optional[str]) -> None:
        with self._lock:
            if pattern is None:
                self.ignore_key_re = None
            else:
                try:
                    self.ignore_key_re = re.compile(pattern)
                except Exception:
                    # keep old pattern on error
                    pass

    def set_watchlist(self, keys: List[str]) -> None:
        with self._lock:
            self.watchlist = set(keys or [])

    def add_watch(self, key: str) -> None:
        with self._lock:
            self.watchlist.add(key)

    def remove_watch(self, key: str) -> None:
        with self._lock:
            self.watchlist.discard(key)

    # ---------- Monitor loop ----------

    def start_monitoring(self) -> None:
        """Start periodic heartbeats that summarize live lifecycle state."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return

        self._monitor_stop.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, name="DependencyInspectorLoop", daemon=True)
        self._monitor_thread.start()
        self.logger.info("[OK] DependencyInspector heartbeat loop started")

    def stop_monitoring(self) -> None:
        """Stop the heartbeat loop."""
        self._monitor_stop.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        self.logger.info("[OK] DependencyInspector heartbeat loop stopped")

    def _monitor_loop(self) -> None:
        interval = max(1.0, self.heartbeat_interval_seconds)
        while not self._monitor_stop.is_set():
            try:
                self._heartbeat_once()
            except Exception as e:
                self.logger.error(f"[DependencyInspector] Heartbeat error: {e}")
            self._monitor_stop.wait(interval)

    def _heartbeat_once(self) -> None:
        """Pull current freshness, upgrade lifecycle states, and emit a concise summary."""
        now = time.time()
        freshness: Dict[str, Dict[str, Any]] = {}
        providers_map: Dict[str, Set[str]] = {}
        consumers_map: Dict[str, Set[str]] = {}

        try:
            freshness = self.bus.get_data_freshness_report() or {}
        except Exception:
            pass

        # The InfoBus exposes _providers/_consumers in this codebase; read-only snapshot:
        try:
            providers_map = {k: set(v) for k, v in getattr(self.bus, "_providers", {}).items()}
        except Exception:
            providers_map = {}
        try:
            consumers_map = {k: set(v) for k, v in getattr(self.bus, "_consumers", {}).items()}
        except Exception:
            consumers_map = {}

        # 1) Refresh lifecycle from provider/consumer maps (book-keeping only)
        with self._lock:
            for key, provs in providers_map.items():
                lc = self._lifecycle.setdefault(key, KeyLifecycle())
                lc.providers |= set(provs)
            for key, cons in consumers_map.items():
                lc = self._lifecycle.setdefault(key, KeyLifecycle())
                lc.consumers |= set(cons)

        # 2) Derive state from freshness ages
        for key, meta in (freshness.items() if isinstance(freshness, dict) else []):
            try:
                age = float(meta.get("age_seconds", 0.0))
                provider = meta.get("source") or meta.get("source_module") or meta.get("provider")
                version = meta.get("version")
            except Exception:
                age = 0.0
                provider = None
                version = None

            if age <= self.stale_age_warn_seconds:
                self._mark_fresh(key, provider, version, now)
            else:
                self._mark_stale(key, provider, version, now)

        # 3) Detect orphan keys: consumed but no provider (mark MISSING)
        for key, consumers in consumers_map.items():
            provs = providers_map.get(key)
            if not provs or len(provs) == 0:
                # only mark missing if we don't already see a fresh/stale from freshness
                if key not in freshness:
                    self._mark_missing(key, requester_hint="/".join(sorted(consumers)) if consumers else None, now=now)

        # 4) Heartbeat summary
        self._emit_heartbeat(now)

        # 5) SLA alerts for watchlist
        self._emit_watchlist_sla(now)

    # ---------- Live taps (optional) ----------

    def attach_live_taps(
        self,
        show_values: bool = True,
        max_preview: int = 160,
        problems_only: bool = True,
        ignore_pattern: Optional[str] = None,
    ):
        """
        Attach event listeners. Safe to call multiple times; attaches only once.

        Args:
          show_values: include preview payloads for GET/blocked logs (when enabled).
          max_preview: truncate previews to this many chars.
          problems_only: if True, log only issues (misses, blocked, module enable/disable).
                         Success events still update lifecycle state but won't be logged.
          ignore_pattern: regex for keys to ignore in live logs.
        """
        if self._live_attached:
            return

        self.show_values = show_values
        self.max_preview = max_preview
        self.problems_only = problems_only

        if ignore_pattern is not None:
            try:
                self.ignore_key_re = re.compile(ignore_pattern)
            except Exception:
                # Keep previous pattern on regex error
                pass

        # Always attach these (we want signals)
        self.bus.subscribe("data_miss", self._on_miss)
        self.bus.subscribe("data_get_blocked", self._on_get_blocked)
        self.bus.subscribe("module_disabled", self._on_module_disabled)
        self.bus.subscribe("module_enabled", self._on_module_enabled)

        # Attach success channels ALWAYS for internal state, but gate their logging by problems_only.
        self.bus.subscribe("data_get", self._on_get_ok)
        self.bus.subscribe("data_updated", self._on_set)

        self._live_attached = True
        self.logger.info("[OK] DependencyInspector live taps attached (problems_only=%s)" % self.problems_only)

    def detach_live_taps(self):
        """Detach event listeners to avoid per-step logs during long training runs."""
        if not self._live_attached:
            return
        for event, handler in [
            ("data_miss", self._on_miss),
            ("data_get", self._on_get_ok),
            ("data_get_blocked", self._on_get_blocked),
            ("data_updated", self._on_set),
            ("module_disabled", self._on_module_disabled),
            ("module_enabled", self._on_module_enabled),
        ]:
            try:
                self.bus.unsubscribe(event, handler)  # if supported by InfoBus
            except Exception:
                pass
        self._live_attached = False
        self.logger.info("[OK] DependencyInspector live taps detached")

    # ---------- Live event handlers ----------

    def _should_ignore(self, key: Any) -> bool:
        try:
            s = str(key or "")
        except Exception:
            return False
        return bool(self.ignore_key_re and self.ignore_key_re.search(s))

    def _fmt_preview(self, txt: Any) -> str:
        try:
            s = repr(txt)
        except Exception:
            s = str(txt)
        return s if len(s) <= self.max_preview else s[: self.max_preview] + "…"

    def _on_miss(self, evt: Dict[str, Any]):
        key = evt.get("key")
        if self._should_ignore(key):
            return

        requester = evt.get("module")
        providers = evt.get("providers") or []
        now = time.time()
        self._mark_missing(str(key), requester_hint=requester, now=now)

        # Log (issue)
        self.issue_logger.info(
            format_operator_message(
                icon="❌",
                message="BUS MISS",
                key=key,
                requester=requester,
                providers=(providers or "∅"),
            )
        )

    def _on_get_ok(self, evt: Dict[str, Any]):
        # Always update lifecycle, but suppress noise when problems_only
        key = evt.get("key")
        if self._should_ignore(key):
            return

        now = time.time()
        provider = evt.get("source_module") or evt.get("provider")
        version = evt.get("version")
        self._mark_fresh(str(key), provider, version, now)

        if self.problems_only:
            return

        if not self.show_values:
            evt = {k: v for k, v in evt.items() if k != "preview"}

        self.logger.info(
            format_operator_message(
                icon="📦",
                message="BUS GET",
                key=key,
                requester=evt.get("module"),
                provider=provider,
                version=version,
                confidence=f"{evt.get('confidence', 0):.2f}",
                age=f"{evt.get('age_seconds', 0):.2f}s",
                preview=self._fmt_preview(evt.get("preview")) if self.show_values else "hidden",
            )
        )

    def _on_get_blocked(self, evt: Dict[str, Any]):
        key = evt.get("key")
        if self._should_ignore(key):
            return
        now = time.time()
        provider = evt.get("source_module") or evt.get("provider")
        version = evt.get("version")
        self._mark_stale(str(key), provider, version, now)

        self.logger.warning(
            format_operator_message(
                icon="⛔",
                message="BUS GET BLOCKED",
                key=key,
                requester=evt.get("module"),
                provider=provider,
                reason=evt.get("reason"),
                confidence=f"{evt.get('confidence', 0):.2f}",
                age=f"{evt.get('age_seconds', 0):.2f}s",
                preview=self._fmt_preview(evt.get("preview")) if self.show_values else "hidden",
            )
        )

    def _on_set(self, evt: Dict[str, Any]):
        key = evt.get("key")
        if self._should_ignore(key):
            return
        now = time.time()
        provider = evt.get("module")
        version = evt.get("version")
        self._mark_fresh(str(key), provider, version, now)  # treat sets as fresh by default

        # Log only if not problems_only
        if not self.problems_only:
            self.logger.info(
                format_operator_message(
                    icon="📝",
                    message="BUS SET",
                    key=key,
                    provider=provider,
                    version=version,
                    conf=f"{evt.get('confidence', 0):.2f}",
                    thesis="yes" if evt.get("has_thesis") else "no",
                )
            )

    def _on_module_disabled(self, evt: Dict[str, Any]):
        self.logger.error(
            format_operator_message(
                icon="🚫",
                message="MODULE DISABLED",
                module=evt.get("module"),
                failures=evt.get("failures"),
                consecutive_failures=evt.get("consecutive_failures"),
                failure_rate=f"{(evt.get('failure_rate') or 0) * 100:.1f}%",
            )
        )

    def _on_module_enabled(self, evt: Dict[str, Any]):
        self.logger.info(
            format_operator_message(
                icon="[OK]",
                message="MODULE ENABLED",
                module=evt.get("module"),
            )
        )

    # ---------- Lifecycle state transitions ----------

    def _get_lc(self, key: str) -> KeyLifecycle:
        with self._lock:
            return self._lifecycle.setdefault(key, KeyLifecycle())

    def _transition(self, key: str, new_status: str, now: float, provider: Optional[str] = None, version: Any = None):
        with self._lock:
            lc = self._lifecycle.setdefault(key, KeyLifecycle())
            old_status = lc.status
            lc.last_event_ts = now

            # Provider-change tracking
            if provider and provider != lc.last_provider and lc.last_provider is not None:
                lc.provider_changes += 1
                self._provider_changes.append((now, key, lc.last_provider, provider))
                self.issue_logger.info(
                    format_operator_message(
                        icon="🔁",
                        message="PROVIDER CHANGED",
                        key=key,
                        old_provider=lc.last_provider,
                        new_provider=provider,
                        version=version,
                    )
                )
            if provider:
                lc.last_provider = provider
            try:
                lc.last_version = int(version) if version is not None else lc.last_version
            except Exception:
                lc.last_version = lc.last_version

            # Status transitions & metrics
            if old_status != new_status:
                if old_status in ("MISSING", "STALE") and new_status == "FRESH":
                    # resolution
                    if not lc.first_seen_ts:
                        lc.first_seen_ts = now
                    lc.resolved_ts = now
                    lc.flap_count += 1
                    mttr = lc.mttr()
                    self._recent_resolved.append((now, key, mttr))
                    self.logger.info(
                        format_operator_message(
                            icon="✅",
                            message="KEY RESOLVED",
                            key=key,
                            mttr=f"{mttr:.2f}s",
                            provider=lc.last_provider,
                            version=lc.last_version,
                        )
                    )
                elif old_status == "FRESH" and new_status in ("MISSING", "STALE"):
                    lc.flap_count += 1
                    self._recent_flaps.append((now, key, lc.flap_count))
                    self.issue_logger.warning(
                        format_operator_message(
                            icon="↕️",
                            message="KEY FLAPPED",
                            key=key,
                            from_status=old_status,
                            to_status=new_status,
                            flaps=lc.flap_count,
                        )
                    )
                # open a new "incident" window for MTTR if entering a non-fresh state
                if new_status in ("MISSING", "STALE"):
                    if lc.status == "FRESH" or lc.first_seen_ts == 0.0:
                        lc.first_seen_ts = now
                        lc.resolved_ts = 0.0

            lc.status = new_status

    def _mark_missing(self, key: str, requester_hint: Optional[str], now: float):
        lc = self._get_lc(key)
        lc.miss_count += 1
        self._transition(key, "MISSING", now, provider=None, version=None)
        # Emit a one-time "tracking" note when first seen missing
        if lc.miss_count == 1:
            self.issue_logger.warning(
                format_operator_message(
                    icon="🕵️",
                    message="TRACKING MISSING KEY",
                    key=key,
                    requester=requester_hint or "unknown",
                )
            )

    def _mark_stale(self, key: str, provider: Optional[str], version: Any, now: float):
        lc = self._get_lc(key)
        lc.blocked_count += 1
        self._transition(key, "STALE", now, provider=provider, version=version)

    def _mark_fresh(self, key: str, provider: Optional[str], version: Any, now: float):
        lc = self._get_lc(key)
        lc.set_count += 1
        self._transition(key, "FRESH", now, provider=provider, version=version)

    # ---------- One-shot, grouped audit ----------

    def scan_once(self, title: str = "Dependency Audit", sample_n: int = 10) -> Dict[str, Any]:
        """
        Single-pass dependency audit with grouped findings and current lifecycle state.
        Returns a dict of findings (also logged).
        """
        t0 = time.time()

        # Snapshots (read-only)
        try:
            providers_map: Dict[str, Set[str]] = dict((k, set(v)) for k, v in getattr(self.bus, "_providers", {}).items())
        except Exception:
            providers_map = {}
        try:
            consumers_map: Dict[str, Set[str]] = dict((k, set(v)) for k, v in getattr(self.bus, "_consumers", {}).items())
        except Exception:
            consumers_map = {}

        try:
            freshness: Dict[str, Dict[str, Any]] = self.bus.get_data_freshness_report() or {}
        except Exception:
            freshness = {}

        try:
            perf: Dict[str, Any] = self.bus.get_performance_metrics() or {}
        except Exception:
            perf = {}

        # ---- Group A: Orphans (consumed but no provider) ----
        orphans: List[Tuple[str, List[str]]] = []
        for key, consumers in consumers_map.items():
            if not self.bus.get_providers(key):
                orphans.append((key, sorted(consumers)))

        # ---- Group B: Duplicate Providers (>1) ----
        dups: List[Tuple[str, List[str]]] = [
            (k, sorted(list(providers)))
            for k, providers in providers_map.items()
            if len(providers) > 1
        ]

        # ---- Group C: Dangling Providers (provided but unused) ----
        danglers: List[Tuple[str, List[str]]] = []
        for key, provs in providers_map.items():
            if key not in consumers_map or not consumers_map[key]:
                danglers.append((key, sorted(list(provs))))

        # ---- Group D: Stale Data (> threshold) ----
        stale: List[Tuple[str, Dict[str, Any]]] = []
        for key, meta in freshness.items():
            try:
                if float(meta.get("age_seconds", 0.0)) > self.stale_age_warn_seconds:
                    stale.append((key, meta))
            except Exception:
                pass

        # ---- Group E: Alias Groups (normalize symbol variants) ----
        alias_groups = self._compute_alias_groups(
            keys=set(consumers_map.keys()) | set(providers_map.keys())
        )

        # ---- Lifecycle snapshot.
        lifecycle_snapshot = self._lifecycle_snapshot()

        # ---- Compose results ----
        results: Dict[str, Any] = {
            "title": title,
            "summary": {
                "providers_total": sum(len(v) for v in providers_map.values()),
                "consumers_total": sum(len(v) for v in consumers_map.values()),
                "active_keys": len(freshness),
                "stale_threshold_seconds": self.stale_age_warn_seconds,
                "cache_hit_rate": float(perf.get("cache_hit_rate", 0.0)),
                "cache_hits": int(perf.get("cache_hits", 0)),
                "total_requests": int(perf.get("total_requests", 0)),
                "disabled_modules": list(perf.get("disabled_modules", [])),
            },
            "orphans": orphans,
            "duplicate_providers": dups,
            "dangling_providers": danglers,
            "stale": stale,
            "alias_groups": alias_groups,
            "lifecycle": lifecycle_snapshot,
            "sample_fresh": self._sample_freshness(freshness, n=sample_n),
            "generated_at": time.time(),
            "elapsed_ms": int((time.time() - t0) * 1000),
        }

        # ---- Log the grouped report ----
        self._log_section_header(f"DEPENDENCY AUDIT: {title}")
        self._log_summary(results["summary"])

        if orphans:
            self._log_group_orphans(orphans)
        if dups:
            self._log_group_dups(dups)
        if danglers:
            self._log_group_danglers(danglers)

        if alias_groups:
            self._log_group_alias(alias_groups, orphans)

        if stale:
            self._log_group_stale(stale)

        # Lifecycle slice
        self._log_lifecycle(lifecycle_snapshot)

        self._log_sample_fresh(results["sample_fresh"])
        self._log_action_list(orphans, dups, danglers, alias_groups, stale)

        self._log_section_footer("END DEPENDENCY AUDIT")
        return results

    # ---------- Legacy wrapper ----------

    def emit_report(self, title: str = "Dependency report"):
        """
        Backward-compatibility wrapper.
        Uses scan_once() with sample_n=0 to eliminate value dump noise.
        """
        return self.scan_once(title=title, sample_n=0)

    # ---------- Helpers ----------

    @staticmethod
    def _sample_freshness(fresh: Dict[str, Dict[str, Any]], n: int = 10) -> List[Tuple[str, Dict[str, Any]]]:
        items = list(fresh.items())
        return items[: max(0, n)]

    def _compute_alias_groups(self, keys: set) -> Dict[str, Dict[str, List[str]]]:
        """
        Group keys that are likely the same instrument under different spellings.
        Watches prefixes: price_, indicators_, signal_.
        Returns:
            { "price_": {"XAUUSD": ["price_XAU/USD", "price_XAUUSD", ...], ...}, ... }
        """
        prefixes = ("price_", "indicators_", "signal_")
        groups: Dict[str, Dict[str, List[str]]] = {p: defaultdict(list) for p in prefixes}

        for key in keys:
            for p in prefixes:
                if key.startswith(p):
                    sym = key[len(p):]
                    norm = re.sub(r"[/_]", "", sym).upper()
                    groups[p][norm].append(key)

        # Remove entries with only one alias
        finalized: Dict[str, Dict[str, List[str]]] = {}
        for p, d in groups.items():
            cluster = {norm: sorted(aliases) for norm, aliases in d.items() if len(aliases) > 1}
            if cluster:
                finalized[p] = cluster
        return finalized

    def _lifecycle_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            unresolved = []
            fresh = []
            stale = []
            stats = {
                "tracked_keys": len(self._lifecycle),
                "recent_resolved": list(self._recent_resolved),
                "recent_flaps": list(self._recent_flaps),
                "provider_changes": list(self._provider_changes),
            }
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
                "stats": stats,
                "watchlist": sorted(list(self.watchlist)),
            }

    # ---------- Pretty logging ----------

    def _log_section_header(self, title: str):
        self.logger.info("═" * 78)
        self.logger.info(title)
        self.logger.info("─" * 78)

    def _log_section_footer(self, title: str):
        self.logger.info("─" * 78)
        self.logger.info(title)
        self.logger.info("═" * 78)

    def _log_summary(self, s: Dict[str, Any]):
        msg = (
            f"Providers: {s['providers_total']}  |  Consumers: {s['consumers_total']}  |  "
            f"Active keys: {s['active_keys']}  |  Stale>{self.stale_age_warn_seconds:.0f}s\n"
            f"[PERF] Cache hit rate: {s['cache_hit_rate']*100:.1f}% "
            f"({s['cache_hits']}/{s['total_requests']})  "
            f"Disabled modules: {s['disabled_modules']}"
        )
        self.logger.info(msg)

    def _log_group_orphans(self, orphans: List[Tuple[str, List[str]]]):
        self.issue_logger.warning("[ORPHANS] Keys consumed but not provided:")
        for key, consumers in sorted(orphans, key=lambda x: x[0]):
            self.issue_logger.warning(f"  - {key}  ←  consumers={consumers}")

    def _log_group_dups(self, dups: List[Tuple[str, List[str]]]):
        self.issue_logger.warning("[DUPLICATE PROVIDERS] Keys with multiple providers:")
        for key, providers in sorted(dups, key=lambda x: x[0]):
            self.issue_logger.warning(f"  - {key}  ←  providers={providers}")

    def _log_group_danglers(self, danglers: List[Tuple[str, List[str]]]):
        self.issue_logger.info("[DANGLING PROVIDERS] Provided but not consumed:")
        for key, providers in sorted(danglers, key=lambda x: x[0]):
            self.issue_logger.info(f"  - {key}  ←  providers={providers}")

    def _log_group_stale(self, stale: List[Tuple[str, Dict[str, Any]]]):
        self.issue_logger.warning(f"[STALE DATA] Older than {self.stale_age_warn_seconds:.0f}s:")
        for key, meta in sorted(stale, key=lambda x: x[0]):
            try:
                self.issue_logger.warning(
                    f"  - {key}: v{meta.get('version')} from {meta.get('source')} "
                    f"conf={meta.get('confidence', 0):.2f} age={meta.get('age_seconds', 0):.1f}s "
                    f"hash={meta.get('validation_hash')}"
                )
            except Exception:
                self.issue_logger.warning(f"  - {key}: {meta}")

    def _log_group_alias(
        self,
        alias_groups: Dict[str, Dict[str, List[str]]],
        orphans: List[Tuple[str, List[str]]],
    ):
        """Highlight alias clusters and which of them are orphans (no providers)."""
        if not alias_groups:
            return
        orphan_keys = {k for k, _ in orphans}
        self.issue_logger.info("[ALIAS GROUPS] Symbol variants likely referring to the same instrument:")
        for prefix, clusters in alias_groups.items():
            self.issue_logger.info(f"  {prefix}")
            for norm, aliases in sorted(clusters.items(), key=lambda kv: kv[0]):
                marks = [("❌ " + a) if a in orphan_keys else ("✓ " + a) for a in aliases]
                self.issue_logger.info(f"    - {norm}: {marks}")

    def _log_sample_fresh(self, sample: List[Tuple[str, Dict[str, Any]]]):
        if not sample:
            return
        self.logger.info("[SAMPLE LIVE VALUES]")
        for key, meta in sample:
            try:
                self.logger.info(
                    f"  {key}: v{meta.get('version')} from {meta.get('source')} "
                    f"conf={meta.get('confidence', 0):.2f} age={meta.get('age_seconds', 0):.1f}s "
                    f"hash={meta.get('validation_hash')}"
                )
            except Exception:
                self.logger.info(f"  {key}: {meta}")

    def _log_lifecycle(self, snapshot: Dict[str, Any]) -> None:
        unresolved = snapshot.get("unresolved", [])
        stale = snapshot.get("stale", [])
        fresh = snapshot.get("fresh", [])
        stats = snapshot.get("stats", {})
        self.logger.info(f"[LIFECYCLE] Tracked={stats.get('tracked_keys', 0)}  "
                         f"Fresh={len(fresh)}  Stale={len(stale)}  Missing={len(unresolved)}")
        if unresolved:
            self.issue_logger.warning("[LIFECYCLE] Currently missing:")
            for key, row in unresolved[:50]:
                self.issue_logger.warning(f"  - {key} (miss={row['miss']}, blocked={row['blocked']}, "
                                          f"flaps={row['flaps']}, mttr={row['mttr_s']}s)")
        if stale:
            self.issue_logger.warning("[LIFECYCLE] Currently stale:")
            for key, row in stale[:50]:
                self.issue_logger.warning(f"  - {key} (blocked={row['blocked']}, flaps={row['flaps']})")

    def _log_action_list(
        self,
        orphans: List[Tuple[str, List[str]]],
        dups: List[Tuple[str, List[str]]],
        danglers: List[Tuple[str, List[str]]],
        alias_groups: Dict[str, Dict[str, List[str]]],
        stale: List[Tuple[str, Dict[str, Any]]],
    ):
        """
        Action checklist without guessing ownership:
          - For each orphan: either add a provider for the key or remove/soften consumers.
          - For each duplicate: pick a single writer or namespace secondary writers.
          - For alias clusters: standardize on ONE canonical spelling.
          - For stale data: refresh providers or lower TTLs.
        """
        self.logger.info("─" * 78)
        self.logger.info("ACTION LIST")
        if orphans:
            self.logger.info("  • Resolve ORPHANS (no provider):")
            for key, consumers in sorted(orphans, key=lambda x: x[0]):
                self.logger.info(f"    - {key}: add provider OR adjust consumers={consumers}")
        if dups:
            self.logger.info("  • Resolve DUPLICATE PROVIDERS (single-writer policy):")
            for key, providers in sorted(dups, key=lambda x: x[0]):
                self.logger.info(f"    - {key}: pick one of providers={providers} or namespace extras")
        if alias_groups:
            self.logger.info("  • Consolidate ALIAS GROUPS (reduce / _ / concat variants):")
            for prefix, clusters in alias_groups.items():
                for norm, variants in sorted(clusters.items(), key=lambda kv: kv[0]):
                    self.logger.info(f"    - {prefix}{norm}: choose ONE spelling from {variants}")
        if stale:
            self.logger.info(f"  • Refresh STALE DATA (> {self.stale_age_warn_seconds:.0f}s):")
            for key, _ in sorted(stale, key=lambda x: x[0]):
                self.logger.info(f"    - {key}: investigate provider refresh / TTL")
        if not (orphans or dups or alias_groups or stale or danglers):
            self.logger.info("  • No issues detected.")

    # ---------- Heartbeat summaries & SLA checks ----------

    def _emit_heartbeat(self, now: float) -> None:
        snap = self._lifecycle_snapshot()
        unresolved = [k for k, _ in snap.get("unresolved", [])]
        stale = [k for k, _ in snap.get("stale", [])]
        fresh = [k for k, _ in snap.get("fresh", [])]

        # Recently resolved
        recent_resolved = []
        with self._lock:
            # copy & filter last N seconds (2 * heartbeat interval for visibility)
            window = max(5.0, 2 * self.heartbeat_interval_seconds)
            rs = [it for it in list(self._recent_resolved) if now - it[0] <= window]
            recent_resolved = [(k, round(mttr, 2)) for _, k, mttr in rs]

        msg = (
            f"[HEARTBEAT] Missing={len(unresolved)}  Stale={len(stale)}  Fresh={len(fresh)}  "
            f"Resolved_recent={len(recent_resolved)}"
        )
        self.logger.info(msg)

        if recent_resolved:
            items = ", ".join([f"{k}(MTTR={mttr}s)" for k, mttr in recent_resolved[:10]])
            self.logger.info(f"[HEARTBEAT] Resolved: {items}")

    def _emit_watchlist_sla(self, now: float) -> None:
        overdue: List[Tuple[str, float]] = []
        with self._lock:
            for key in self.watchlist:
                lc = self._lifecycle.get(key)
                if not lc:
                    continue
                if lc.status != "FRESH" and lc.first_seen_ts:
                    age = now - lc.first_seen_ts
                    if age >= self.resolution_warn_seconds:
                        overdue.append((key, age))
        if overdue:
            for key, age in overdue:
                self.issue_logger.warning(
                    format_operator_message(
                        icon="⏰",
                        message="WATCHLIST RESOLUTION SLA BREACH",
                        key=key,
                        age=f"{age:.1f}s",
                        threshold=f"{self.resolution_warn_seconds:.1f}s",
                    )
                )
