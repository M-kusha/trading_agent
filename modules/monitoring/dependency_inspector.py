# modules/monitoring/dependency_inspector.py
from __future__ import annotations

import re
import time
from collections import defaultdict
from typing import Any, Optional, Dict, List, Tuple, DefaultDict

from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message


class DependencyInspector:
    """
    DependencyInspector
    -------------------
    Live bus taps (optional) + on-demand, single-pass dependency audit.

    Use cases:
      - attach_live_taps(): subscribe to bus events during a short debugging session.
      - detach_live_taps(): stop live event spam (recommended for training).
      - scan_once(): one-shot, grouped audit that pinpoints ROOT CAUSES:
           * Orphans (consumed but no provider)
           * Duplicate Providers (>1 writer)
           * Dangling Providers (provided but unused)
           * Stale Data (older than threshold)
           * Alias Groups (e.g., price_/indicators_/signal_ storms for the same symbol)
      - emit_report(): legacy wrapper that calls scan_once() with low-noise defaults.

    Notes:
      * This inspector does NOT guess ownership. It only reports what exists.
      * It reads the bus's provider/consumer maps and freshness report at a point in time.
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

        self.stale_age_warn_seconds = stale_age_warn_seconds
        self._live_attached = False
        self.show_values = True
        self.max_preview = 160

        # Live-tap controls (noise filters)
        # problems_only=True => do NOT log normal BUS GET/SET (success) events.
        self.problems_only = True
        # Ignore spammy keys (e.g., logs, module events) for ALL live handlers
        self.ignore_key_re: Optional[re.Pattern] = re.compile(r"^(log_|module_events/|log_metrics_)")

    # ---------- Live taps (optional) ----------

    def attach_live_taps(
        self,
        show_values: bool = True,
        max_preview: int = 160,
        problems_only: bool = True,
        ignore_pattern: Optional[str] = None,
    ):
        """
        Attach event listeners. Prefer scan_once() for a single audit.
        Safe to call multiple times; attaches only once.

        Args:
          show_values: include preview payloads for GET/blocked logs (when enabled).
          max_preview: truncate previews to this many chars.
          problems_only: if True, only log issues (misses, blocked, enable/disable).
                         Suppresses normal BUS GET/SET noise.
          ignore_pattern: regex for keys to ignore in live logs (default hides log_*, module_events/*, log_metrics_*).
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

        # Always attach problem signals
        self.bus.subscribe("data_miss", self._on_miss)
        self.bus.subscribe("data_get_blocked", self._on_get_blocked)
        self.bus.subscribe("module_disabled", self._on_module_disabled)
        self.bus.subscribe("module_enabled", self._on_module_enabled)

        # Only attach success/noise channels when not in problems-only mode
        if not self.problems_only:
            self.bus.subscribe("data_get", self._on_get_ok)
            self.bus.subscribe("data_updated", self._on_set)

        self._live_attached = True
        self.logger.info("[OK] DependencyInspector live taps attached (problems_only=%s)" % self.problems_only)

    def detach_live_taps(self):
        """Detach event listeners to avoid per-step logs during training."""
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
        providers = evt.get("providers") or []
        self.logger.info(
            format_operator_message(
                icon="❌",
                message="BUS MISS",
                key=key,
                requester=evt.get("module"),
                providers=(providers or "∅"),
            )
        )

    def _on_get_ok(self, evt: Dict[str, Any]):
        # Suppress success GETs in problems-only mode
        if self.problems_only:
            return
        key = evt.get("key")
        if self._should_ignore(key):
            return
        if not self.show_values:
            evt = {k: v for k, v in evt.items() if k != "preview"}
        self.logger.info(
            format_operator_message(
                icon="📦",
                message="BUS GET",
                key=key,
                requester=evt.get("module"),
                provider=evt.get("source_module"),
                version=evt.get("version"),
                confidence=f"{evt.get('confidence', 0):.2f}",
                age=f"{evt.get('age_seconds', 0):.2f}s",
                preview=self._fmt_preview(evt.get("preview")) if self.show_values else "hidden",
            )
        )

    def _on_get_blocked(self, evt: Dict[str, Any]):
        key = evt.get("key")
        if self._should_ignore(key):
            return
        self.logger.warning(
            format_operator_message(
                icon="⛔",
                message="BUS GET BLOCKED",
                key=key,
                requester=evt.get("module"),
                provider=evt.get("source_module"),
                reason=evt.get("reason"),
                confidence=f"{evt.get('confidence', 0):.2f}",
                age=f"{evt.get('age_seconds', 0):.2f}s",
                preview=self._fmt_preview(evt.get("preview")) if self.show_values else "hidden",
            )
        )

    def _on_set(self, evt: Dict[str, Any]):
        # Suppress success SETs in problems-only mode
        if self.problems_only:
            return
        key = evt.get("key")
        if self._should_ignore(key):
            return
        self.logger.info(
            format_operator_message(
                icon="📝",
                message="BUS SET",
                key=key,
                provider=evt.get("module"),
                version=evt.get("version"),
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

    # ---------- One-shot, grouped audit ----------

    def scan_once(self, title: str = "Dependency Audit", sample_n: int = 10) -> Dict[str, Any]:
        """
        Single-pass dependency audit with grouped findings.
        - No subscriptions (no per-step spam).
        - Pure snapshot: reads bus internals once.
        Returns a dict of findings (also logged nicely).
        """
        t0 = time.time()

        # Snapshots (read-only)
        providers_map: Dict[str, set] = dict((k, set(v)) for k, v in self.bus._providers.items())
        consumers_map: Dict[str, set] = dict((k, set(v)) for k, v in self.bus._consumers.items())
        freshness: Dict[str, Dict[str, Any]] = self.bus.get_data_freshness_report()
        perf: Dict[str, Any] = self.bus.get_performance_metrics()

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
                # ignore malformed meta
                pass

        # ---- Group E: Alias Groups (normalize symbol variants) ----
        alias_groups = self._compute_alias_groups(
            keys=set(consumers_map.keys()) | set(providers_map.keys())
        )

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
            # High-signal mapping for the single-writer keys
            "critical_keys": {
                k: {
                    "providers": sorted(list(self.bus.get_providers(k))),
                    "consumers": sorted(list(self.bus.get_consumers(k))),
                }
                for k in (
                    "market_regime",
                    "training_metrics",
                    "performance_metrics",
                    "risk_data",
                    "sequence_quality",
                    "trade_vote",
                )
            },
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

        self._log_sample_fresh(results["sample_fresh"])

        # Actionable checklist (no guesses—just what to look at)
        self._log_action_list(orphans, dups, danglers, alias_groups, stale)

        self._log_section_footer("END DEPENDENCY AUDIT")
        return results

    # ---------- Legacy wrapper ----------

    def emit_report(self, title: str = "Dependency report"):
        """
        Kept for backward-compatibility with the old interface.
        Uses scan_once() but with sample_n=0 to eliminate value dump noise.
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
        We DO NOT claim these are wrong—only report clusters to help you consolidate.
        Currently watches prefixes: price_, indicators_, signal_.
        Returns:
            {
              "price_":  {"XAUUSD": ["price_XAU/USD", "price_XAUUSD", "price_XAU_USD"], ...},
              "signal_": {"EURUSD": ["signal_EUR/USD", "signal_EUR_USD"], ...},
              ...
            }
        """
        prefixes = ("price_", "indicators_", "signal_")
        groups: Dict[str, Dict[str, List[str]]] = {p: defaultdict(list) for p in prefixes}

        for key in keys:
            for p in prefixes:
                if key.startswith(p):
                    sym = key[len(p):]
                    norm = re.sub(r"[/_]", "", sym).upper()
                    groups[p][norm].append(key)

        # Remove entries with only one alias (not really a group)
        finalized: Dict[str, Dict[str, List[str]]] = {}
        for p, d in groups.items():
            cluster = {norm: sorted(aliases) for norm, aliases in d.items() if len(aliases) > 1}
            if cluster:
                finalized[p] = cluster
        return finalized

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
          - For alias clusters: standardize on ONE canonical spelling to reduce mismatch risk.
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
