from __future__ import annotations

import ast
import re
import time
import sys
import os
from collections import defaultdict
from typing import Any, Optional, Dict, List, Tuple, DefaultDict, Set

# Add the project root to Python path so we can import modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Go up one level from tools/ to project root
sys.path.insert(0, PROJECT_ROOT)

from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
import json

# ---------- Paths ----------
ROOT = PROJECT_ROOT  # Use the corrected project root
MODULES_DIR = os.path.join(ROOT, 'modules')

# ---------- Noise controls ----------
INFRA_WHITELIST: Set[str] = {
    'modules/utils/info_bus.py',
    'modules/core/mixins.py',
    'modules/core/module_system.py',
    'modules/utils/audit_utils.py',
    'modules/core/error_pinpointer.py',
}

PLACEHOLDER_KEY_RE = re.compile(r"^\{.*\}$")  # e.g. "{key}"
IGNORE_KEYS: Set[str] = set()  # add global ignores here


def is_placeholder(k: str) -> bool:
    return bool(PLACEHOLDER_KEY_RE.match(k))


def sanitize_key(k: str) -> str:
    if not isinstance(k, str):
        return ""
    k = k.strip().strip('"').strip("'")
    if not k or k.startswith('#'):
        return ""
    return k


def sanitize_key_list(items: List[str]) -> List[str]:
    out: List[str] = []
    for x in items or []:
        k = sanitize_key(x)
        if not k:
            continue
        if is_placeholder(k):
            continue
        if k in IGNORE_KEYS:
            continue
        out.append(k)
    seen = set()
    uniq = []
    for k in out:
        if k not in seen:
            uniq.append(k)
            seen.add(k)
    return uniq


def read_text(path: str) -> str:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        return ""


# ---------- Discovery of @module classes ----------
DECORATOR_RE = re.compile(r"@module\((.*?)\)\s*class\s+(\w+)", re.S)

def find_modules() -> Dict[str, Dict[str, Any]]:
    """
    Discover classes decorated with @module(...).
    Regex-based but robust for typical usage.
    """
    result: Dict[str, Dict[str, Any]] = {}

    def extract_list(args: str, key: str) -> List[str]:
        m2 = re.search(key + r"\s*=\s*\[(.*?)\]", args, flags=re.S)
        if not m2:
            return []
        raw = m2.group(1)
        parts = [p.strip() for p in re.split(r",\s*", raw) if p.strip()]
        return sanitize_key_list(parts)

    for dirpath, _, filenames in os.walk(MODULES_DIR):
        for fn in filenames:
            if not fn.endswith('.py'):
                continue
            fp = os.path.join(dirpath, fn)
            txt = read_text(fp)
            if not txt:
                continue
            for m in DECORATOR_RE.finditer(txt):
                args, cls = m.group(1), m.group(2)
                provides = extract_list(args, 'provides')
                requires = extract_list(args, 'requires')
                rel = os.path.relpath(fp, ROOT).replace('\\', '/')
                result[cls] = {'file': rel, 'provides': provides, 'requires': requires}
    return result


# ---------- AST-scoped bus access ----------
class _BusAccessVisitor(ast.NodeVisitor):
    """
    Collect smart_bus.get/set per enclosing class.
    Anything outside decorated classes goes to 'module_level'.
    """
    def __init__(self, interested_classes: Set[str]):
        self.interested = set(interested_classes)
        self.current_class: Optional[str] = None
        self.by_class: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: {'writes': [], 'reads': []})
        self.module_level: Dict[str, List[Dict[str, Any]]] = {'writes': [], 'reads': []}

    # --- helpers ---
    @staticmethod
    def _contains_smart_bus_base(node: ast.AST) -> bool:
        """
        True if node looks like *.smart_bus or smart_bus.
        Handles:
          - self.smart_bus.set(...)
          - smart_bus.set(...)
          - foo.bar.smart_bus.set(...)
        """
        cur = node
        while isinstance(cur, ast.Attribute):
            if cur.attr == 'smart_bus':
                return True
            cur = cur.value
        if isinstance(cur, ast.Name):
            return cur.id == 'smart_bus'
        return False

    @staticmethod
    def _extract_literal_key(call: ast.Call) -> Optional[str]:
        if not call.args:
            return None
        a0 = call.args[0]
        if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
            return sanitize_key(a0.value)
        return None

    def _record(self, kind: str, key: Optional[str], lineno: int):
        if not key or is_placeholder(key) or key in IGNORE_KEYS:
            return
        entry = {'line': int(lineno), 'key': key}
        if self.current_class and self.current_class in self.interested:
            self.by_class[self.current_class][kind].append(entry)
        else:
            self.module_level[kind].append(entry)

    # --- visitors ---
    def visit_ClassDef(self, node: ast.ClassDef):
        prev = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = prev

    def visit_Call(self, node: ast.Call):
        try:
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in ('set', 'get'):
                if self._contains_smart_bus_base(func.value):
                    key = self._extract_literal_key(node)
                    kind = 'writes' if func.attr == 'set' else 'reads'
                    self._record(kind, key, getattr(node, 'lineno', -1))
        except Exception:
            pass
        self.generic_visit(node)


def find_bus_accesses_scoped(modules: Dict[str, Dict[str, Any]]) -> Tuple[
    Dict[str, Dict[str, List[Dict[str, Any]]]],  # by_class: {ClassName: {'writes':[], 'reads':[]}}
    Dict[str, Dict[str, List[Dict[str, Any]]]]   # by_file_extra: {relpath: {'writes':[], 'reads':[]}} (outside classes)
]:
    """
    Parse every file once, attribute gets/sets to the enclosing class (if decorated),
    otherwise collect them under a per-file pseudo-node.
    """
    # map file -> set of decorated classes we care about in that file
    classes_by_file: Dict[str, Set[str]] = defaultdict(set)
    for cls, meta in modules.items():
        classes_by_file[meta['file']].add(cls)

    by_class: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    by_file_extra: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for dirpath, _, filenames in os.walk(MODULES_DIR):
        for fn in filenames:
            if not fn.endswith('.py'):
                continue
            fp = os.path.join(dirpath, fn)
            rel = os.path.relpath(fp, ROOT).replace('\\', '/')
            txt = read_text(fp)
            if not txt:
                continue

            try:
                tree = ast.parse(txt, filename=rel)
            except Exception:
                continue

            visitor = _BusAccessVisitor(classes_by_file.get(rel, set()))
            visitor.visit(tree)

            # Merge class-scoped results
            for cls, rw in visitor.by_class.items():
                # store with key '<ClassName>' (unique across file)
                if cls not in by_class:
                    by_class[cls] = {'writes': [], 'reads': []}
                by_class[cls]['writes'].extend(rw['writes'])
                by_class[cls]['reads'].extend(rw['reads'])

            # Module-level / non-decorated classes
            if visitor.module_level['writes'] or visitor.module_level['reads']:
                if rel not in by_file_extra:
                    by_file_extra[rel] = {'writes': [], 'reads': []}
                by_file_extra[rel]['writes'].extend(visitor.module_level['writes'])
                by_file_extra[rel]['reads'].extend(visitor.module_level['reads'])

    # Sort by line for determinism
    for cls in by_class:
        by_class[cls]['writes'].sort(key=lambda x: x['line'])
        by_class[cls]['reads'].sort(key=lambda x: x['line'])
    for rel in by_file_extra:
        by_file_extra[rel]['writes'].sort(key=lambda x: x['line'])
        by_file_extra[rel]['reads'].sort(key=lambda x: x['line'])

    return by_class, by_file_extra


# ---------- Graph build ----------
def build_graph() -> Dict[str, Any]:
    modules = find_modules()
    class_access, extra_access = find_bus_accesses_scoped(modules)

    graph: Dict[str, Any] = {}

    # Decorated modules (class-scoped get/set)
    for cls, meta in modules.items():
        rel = meta['file']
        acc = class_access.get(cls, {'reads': [], 'writes': []})
        graph[cls] = {
            'file': rel,
            'provides': meta['provides'],
            'requires': meta['requires'],
            'reads': sorted({sanitize_key(r['key']) for r in acc.get('reads', []) if sanitize_key(r['key'])}),
            'writes': sorted({sanitize_key(w['key']) for w in acc.get('writes', []) if sanitize_key(w['key'])}),
            'infra': rel in INFRA_WHITELIST,
        }

    # Non-decorated / module-level bus accesses remain visible as pseudo-modules
    for rel, rw in sorted(extra_access.items()):
        name = os.path.basename(rel)[:-3]
        graph[name] = {
            'file': rel,
            'provides': [],
            'requires': [],
            'reads': sorted({sanitize_key(r['key']) for r in rw.get('reads', []) if sanitize_key(r['key'])}),
            'writes': sorted({sanitize_key(w['key']) for w in rw.get('writes', []) if sanitize_key(w['key'])}),
            'infra': rel in INFRA_WHITELIST,
        }

    return graph


# ---------- Analysis ----------
def compute_mismatches(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for name, info in graph.items():
        if info.get('infra'):
            continue

        prov = set(info.get('provides', []) or [])
        req = set(info.get('requires', []) or [])
        wr = set(info.get('writes', []) or [])
        rd = set(info.get('reads', []) or [])

        prov = {k for k in prov if not is_placeholder(k) and k not in IGNORE_KEYS}
        req  = {k for k in req  if not is_placeholder(k) and k not in IGNORE_KEYS}
        wr   = {k for k in wr   if not is_placeholder(k) and k not in IGNORE_KEYS}
        rd   = {k for k in rd   if not is_placeholder(k) and k not in IGNORE_KEYS}

        for k in sorted(wr - prov):
            out.append({'module': name, 'key': k, 'type': 'UndefinedWrite'})
        for k in sorted(prov - wr):
            out.append({'module': name, 'key': k, 'type': 'DeclaredButNotWritten'})
        for k in sorted(rd - req):
            out.append({'module': name, 'key': k, 'type': 'ReadsMissingRequire'})
        for k in sorted(req - rd):
            out.append({'module': name, 'key': k, 'type': 'RequiresButNeverRead'})
    return out


def summarize_mismatches(mismatches: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {
        'UndefinedWrite': 0,
        'DeclaredButNotWritten': 0,
        'ReadsMissingRequire': 0,
        'RequiresButNeverRead': 0,
    }
    for m in mismatches:
        t = m.get('type')
        counts[t] = counts.get(t, 0) + 1
    counts['TOTAL'] = len(mismatches)
    return counts


def collect_critical_writers_from_graph(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Keep using the graph (now class-scoped) to find writers of critical keys.
    """
    critical = {'market_regime', 'training_metrics', 'performance_metrics', 'risk_data', 'sequence_quality', 'trade_vote'}
    out: List[Dict[str, Any]] = []
    for name, meta in graph.items():
        for k in meta.get('writes', []):
            if k in critical:
                out.append({'module': name, 'key': k, 'file': meta.get('file')})
    out.sort(key=lambda x: (x['key'], x['module']))
    return out


# ---------- One-shot inspector (runtime) ----------
class DependencyInspector:
    """
    Live taps (optional) + on-demand, single-pass dependency audit.
    """

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

    # ---------- Live taps ----------
    def attach_live_taps(self, show_values: bool = True, max_preview: int = 160):
        if self._live_attached:
            return
        self.show_values = show_values
        self.max_preview = max_preview

        self.bus.subscribe("data_miss", self._on_miss)
        self.bus.subscribe("data_get", self._on_get_ok)
        self.bus.subscribe("data_get_blocked", self._on_get_blocked)
        self.bus.subscribe("data_updated", self._on_set)
        self.bus.subscribe("module_disabled", self._on_module_disabled)
        self.bus.subscribe("module_enabled", self._on_module_enabled)

        self._live_attached = True
        self.logger.info("[OK] DependencyInspector live taps attached")

    def detach_live_taps(self):
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
                self.bus.unsubscribe(event, handler)
            except Exception:
                pass
        self._live_attached = False
        self.logger.info("[OK] DependencyInspector live taps detached")

    # ---------- Live event handlers ----------
    def _fmt_preview(self, txt: Any) -> str:
        try:
            s = repr(txt)
        except Exception:
            s = str(txt)
        return s if len(s) <= self.max_preview else s[: self.max_preview] + "…"

    def _on_miss(self, evt: Dict[str, Any]):
        providers = evt.get("providers") or []
        self.logger.info(
            format_operator_message(
                icon="❌",
                message="BUS MISS",
                key=evt.get("key"),
                requester=evt.get("module"),
                providers=(providers or "∅"),
            )
        )

    def _on_get_ok(self, evt: Dict[str, Any]):
        payload = dict(evt)
        if not self.show_values:
            payload.pop("preview", None)
        self.logger.info(
            format_operator_message(
                icon="📦",
                message="BUS GET",
                key=evt.get("key"),
                requester=evt.get("module"),
                provider=evt.get("source_module"),
                version=evt.get("version"),
                confidence=f"{evt.get('confidence', 0):.2f}",
                age=f"{evt.get('age_seconds', 0):.2f}s",
                preview=self._fmt_preview(evt.get("preview")) if self.show_values else "hidden",
            )
        )

    def _on_get_blocked(self, evt: Dict[str, Any]):
        self.logger.warning(
            format_operator_message(
                icon="⛔",
                message="BUS GET BLOCKED",
                key=evt.get("key"),
                requester=evt.get("module"),
                provider=evt.get("source_module"),
                reason=evt.get("reason"),
                confidence=f"{evt.get('confidence', 0):.2f}",
                age=f"{evt.get('age_seconds', 0):.2f}s",
                preview=self._fmt_preview(evt.get("preview")) if self.show_values else "hidden",
            )
        )

    def _on_set(self, evt: Dict[str, Any]):
        self.logger.info(
            format_operator_message(
                icon="📝",
                message="BUS SET",
                key=evt.get("key"),
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
        t0 = time.time()

        providers_map: Dict[str, set] = dict((k, set(v)) for k, v in self.bus._providers.items())
        consumers_map: Dict[str, set] = dict((k, set(v)) for k, v in self.bus._consumers.items())
        freshness: Dict[str, Dict[str, Any]] = self.bus.get_data_freshness_report()
        perf: Dict[str, Any] = self.bus.get_performance_metrics()

        # Orphans
        orphans: List[Tuple[str, List[str]]] = []
        for key, consumers in consumers_map.items():
            if not self.bus.get_providers(key):
                orphans.append((key, sorted(consumers)))

        # Duplicate providers
        dups: List[Tuple[str, List[str]]] = [
            (k, sorted(list(providers)))
            for k, providers in providers_map.items()
            if len(providers) > 1
        ]

        # Dangling providers (provided but unused)
        danglers: List[Tuple[str, List[str]]] = []
        for key, provs in providers_map.items():
            if key not in consumers_map or not consumers_map[key]:
                danglers.append((key, sorted(list(provs))))

        # Stale
        stale: List[Tuple[str, Dict[str, Any]]] = []
        for key, meta in freshness.items():
            try:
                if float(meta.get("age_seconds", 0.0)) > self.stale_age_warn_seconds:
                    stale.append((key, meta))
            except Exception:
                pass

        # Alias groups
        alias_groups = self._compute_alias_groups(
            keys=set(consumers_map.keys()) | set(providers_map.keys())
        )

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
        self._log_action_list(orphans, dups, danglers, alias_groups, stale)
        self._log_section_footer("END DEPENDENCY AUDIT")
        return results

    # ---------- Legacy wrapper ----------
    def emit_report(self, title: str = "Dependency report"):
        return self.scan_once(title=title, sample_n=10)

    # ---------- Helpers ----------
    @staticmethod
    def _sample_freshness(fresh: Dict[str, Dict[str, Any]], n: int = 10) -> List[Tuple[str, Dict[str, Any]]]:
        items = list(fresh.items())
        return items[: max(0, n)]

    def _compute_alias_groups(self, keys: set) -> Dict[str, Dict[str, List[str]]]:
        prefixes = ("price_", "indicators_", "signal_")
        groups: Dict[str, Dict[str, List[str]]] = {p: defaultdict(list) for p in prefixes}
        for key in keys:
            for p in prefixes:
                if key.startswith(p):
                    sym = key[len(p):]
                    norm = re.sub(r"[/_]", "", sym).upper()
                    groups[p][norm].append(key)
        finalized: Dict[str, Dict[str, List[str]]] = {}
        for p, d in groups.items():
            cluster = {norm: sorted(aliases) for norm, aliases in d.items() if len(aliases) > 1}
            if cluster:
                finalized[p] = cluster
        return finalized

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


# ---------- CLI-style runner (optional) ----------
def main():
    graph = build_graph()
    mismatches = compute_mismatches(graph)
    summary = summarize_mismatches(mismatches)
    crit = collect_critical_writers_from_graph(graph)

    out_dir = os.path.join(ROOT, 'audit')
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, 'topology.json'), 'w', encoding='utf-8') as f:
        json.dump(graph, f, indent=2, sort_keys=True)

    with open(os.path.join(out_dir, 'critical_writers.json'), 'w', encoding='utf-8') as f:
        json.dump(crit, f, indent=2)

    with open(os.path.join(out_dir, 'mismatches.json'), 'w', encoding='utf-8') as f:
        json.dump(mismatches, f, indent=2)

    with open(os.path.join(out_dir, 'mismatch_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print('GRAPH_OK')
    print(json.dumps({
        'critical_writer_count': len(crit),
        'mismatch_count': len(mismatches),
        'by_type': summary
    }, indent=2))


if __name__ == '__main__':
    main()
