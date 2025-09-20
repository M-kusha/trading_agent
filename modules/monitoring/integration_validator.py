# ─────────────────────────────────────────────────────────────
# File: modules/monitoring/integration_validator.py
# SmartInfoBus Integration Validator (v3.3 “Autopilot-PylanceClean”)
# - Auto-discovery of @module classes (AST + import)
# - Contract cross-check against modules/contracts.CONTRACTS
# - Provider/consumer graph, duplicate/missing writers, cycle scan
# - Robust typing: no name shadowing; Protocol-based adapters
# - Safe InfoBus access wrapper; no static attr diagnostics
# - Auto-run on system start via INTEGRATION_VALIDATE_ON_BOOT
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import sys
import ast
import json
import inspect
import importlib
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, Set, Tuple, Protocol, runtime_checkable,
    Callable, Deque, DefaultDict, cast
)
from collections import defaultdict, deque

# Establish project root (…/trading_agent)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Optional YAML for config checks (degrades gracefully)
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


# ─────────────────────────────────────────────────────────────
# Protocols & Adapters (avoid Pylance name shadowing)
# ─────────────────────────────────────────────────────────────

@runtime_checkable
class ExplainerProto(Protocol):
    def explain(self, x: Any) -> str: ...


@runtime_checkable
class LoggerProto(Protocol):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...  # allow flexible construction
    def info(self, *args: Any, **kwargs: Any) -> None: ...
    def warning(self, *args: Any, **kwargs: Any) -> None: ...
    def error(self, *args: Any, **kwargs: Any) -> None: ...


FormatOperator = Callable[[str, str, str, str], str]


# Import external utilities under stable aliases (no shadowing)
try:
    from modules.utils.system_utilities import EnglishExplainer as _RealExplainer  # type: ignore
    ExplainerClass: type[ExplainerProto] = cast(type[ExplainerProto], _RealExplainer)
except Exception:  # pragma: no cover
    class _FallbackExplainer:
        def explain(self, x: Any) -> str:
            return str(x)
    ExplainerClass = _FallbackExplainer  # type: ignore[misc]


try:
    from modules.utils.audit_utils import (  # type: ignore
        RotatingLogger as _RealLogger,
        format_operator_message as _real_format_operator_message
    )
    LoggerClass: type[LoggerProto] = cast(type[LoggerProto], _RealLogger)

    # Wrap to normalize signature: (tag, title, details, context) -> str
    def format_operator(tag: str, title: str, details: str = "", context: str = "") -> str:
        try:
            # Preferred signature if available
            return cast(FormatOperator, _real_format_operator_message)(tag, title, details, context)  # type: ignore[arg-type]
        except TypeError:
            # Some builds use (icon, message, **ctx)
            return _real_format_operator_message("ℹ️", f"{tag} {title} :: {details} [{context}]")  # type: ignore[call-arg]
except Exception:  # pragma: no cover
    class _FallbackLogger:
        def __init__(self, *args: Any, **kwargs: Any) -> None: ...
        def info(self, *args: Any, **kwargs: Any) -> None: print(*args)
        def warning(self, *args: Any, **kwargs: Any) -> None: print(*args)
        def error(self, *args: Any, **kwargs: Any) -> None: print(*args)

    LoggerClass = _FallbackLogger  # type: ignore[misc]

    def format_operator(tag: str, title: str, details: str = "", context: str = "") -> str:
        return f"{tag} {title} :: {details} [{context}]"


# InfoBus wrapper to avoid attribute diagnostics
class _NullBus:
    def set(self, *_: Any, **__: Any) -> None: ...
    def get_providers(self, *_: Any) -> Set[str]:
        return set()


def _get_bus() -> Any:
    try:
        from modules.utils.info_bus import InfoBusManager as _IBM  # type: ignore
    except Exception:
        return _NullBus()
    get_inst = getattr(_IBM, "get_instance", None)
    if callable(get_inst):
        try:
            return get_inst()
        except Exception:
            return _NullBus()
    return _NullBus()


# Optional configuration manager (best-effort)
try:
    from modules.core.configuration_manager import ConfigurationManager  # type: ignore
except Exception:  # pragma: no cover
    ConfigurationManager = None  # type: ignore


# Contracts registry (normalize to dict[str, dict[str, Any]])
try:
    from modules.contracts import ModuleContract, CONTRACTS  # type: ignore
except Exception:  # pragma: no cover
    ModuleContract, CONTRACTS = None, {}  # type: ignore


def _normalize_contracts_map() -> Dict[str, Dict[str, Any]]:
    """Convert CONTRACTS (dataclass or dict entries) to a plain dict for typing sanity."""
    norm: Dict[str, Dict[str, Any]] = {}
    if not CONTRACTS:
        return norm
    for name, obj in CONTRACTS.items():  # type: ignore[attr-defined]
        try:
            if ModuleContract is not None and isinstance(obj, ModuleContract):  # dataclass shape
                norm[name] = {
                    "file": cast(str, getattr(obj, "file", "")),
                    "provides": list(getattr(obj, "provides", []) or []),
                    "requires": list(getattr(obj, "requires", []) or []),
                    "meta": dict(getattr(obj, "meta", {}) or {}),
                }
            else:
                d = cast(Dict[str, Any], obj)
                norm[name] = {
                    "file": d.get("file", "") or "",
                    "provides": list(d.get("provides", []) or []),
                    "requires": list(d.get("requires", []) or []),
                    "meta": dict(d.get("meta", {}) or {}),
                }
        except Exception:
            # Skip malformed entries safely
            continue
    return norm


# ─────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────

@dataclass
class ValidationIssue:
    module: str
    issue_type: str
    severity: str  # 'error' | 'warning' | 'info'
    message: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationReport:
    total_modules: int
    validated_modules: int
    issues: List[ValidationIssue] = field(default_factory=list)
    missing_decorators: List[str] = field(default_factory=list)
    missing_thesis: List[str] = field(default_factory=list)
    legacy_modules: List[str] = field(default_factory=list)
    config_issues: List[str] = field(default_factory=list)
    bad_categories: List[str] = field(default_factory=list)
    ghost_contracts: List[str] = field(default_factory=list)
    unregistered_modules: List[str] = field(default_factory=list)
    duplicate_writers: Dict[str, List[str]] = field(default_factory=dict)
    missing_writers: Dict[str, List[str]] = field(default_factory=dict)
    cycles: List[List[str]] = field(default_factory=list)
    manifest: Dict[str, Any] = field(default_factory=dict)      # filled only in debug
    integration_score: float = 100.0

    def to_plain_english(self) -> str:
        explainer = ExplainerClass()
        lines: List[str] = [
            "SMARTINFOBUS INTEGRATION VALIDATION REPORT",
            "=" * 50,
            f"\nOverall Integration Score: {self.integration_score:.1f}%",
            f"Modules Checked: {self.validated_modules}/{self.total_modules}",
            ""
        ]
        if self.integration_score >= 90:
            lines.append("[OK] Excellent integration — system is well connected")
        elif self.integration_score >= 70:
            lines.append("[WARN] Good integration — some improvements recommended")
        else:
            lines.append("[RED] Poor integration — significant remediation required")

        def dump_section(title: str, items: List[str], prefix: str = "  • ", cap: int = 15) -> None:
            if not items:
                return
            lines.extend([f"\n{title}", "-" * max(8, len(title))])
            for x in items[:cap]:
                lines.append(f"{prefix}{x}")
            if len(items) > cap:
                lines.append(f"{prefix}… and {len(items)-cap} more")

        if self.issues:
            errs = [i for i in self.issues if i.severity == "error"]
            warns = [i for i in self.issues if i.severity == "warning"]
            infos = [i for i in self.issues if i.severity == "info"]

            if errs:
                lines.append(f"\n[ERRORS] ({len(errs)})")
                for i in errs[:10]:
                    lines.append(f"  • {i.module}: {i.message}")
                if len(errs) > 10:
                    lines.append(f"  • … and {len(errs)-10} more")
            if warns:
                lines.append(f"\n[WARNINGS] ({len(warns)})")
                for i in warns[:12]:
                    lines.append(f"  • {i.module}: {i.message}")
                if len(warns) > 12:
                    lines.append(f"  • … and {len(warns)-12} more")
            if infos:
                lines.append(f"\n[INFO] ({len(infos)})")
                for i in infos[:8]:
                    lines.append(f"  • {i.module}: {i.message}")

        dump_section("[FAIL] Modules missing @module", self.missing_decorators)
        dump_section("[WARN] Invalid/unknown categories", self.bad_categories)
        dump_section("[LOG] Explainable modules missing thesis", self.missing_thesis)
        dump_section("[WARN] Ghost contracts (file missing)", self.ghost_contracts)
        if self.duplicate_writers:
            lines.extend(["\n[DUPLICATE WRITERS] Detected", "-" * 30])
            for k, mods in sorted(self.duplicate_writers.items()):
                lines.append(f"  • {k}: {sorted(mods)}")
        if self.missing_writers:
            lines.extend(["\n[MISSING WRITERS] Required keys with no providers", "-" * 45])
            for k, consumers in sorted(self.missing_writers.items()):
                lines.append(f"  • {k}: needed by {sorted(consumers)}")
        if self.cycles:
            lines.extend(["\n[CYCLES] potential dependency cycles", "-" * 35])
            for c in self.cycles[:5]:
                lines.append("  • " + " → ".join(c))

        lines += ["\nRECOMMENDATIONS", "-" * 20]
        if self.missing_decorators:
            lines.append("1) Add @module(provides=[…], requires=[…], category=…) to all modules.")
        if self.bad_categories:
            lines.append("2) Normalize categories to folder-derived canonical set.")
        if self.missing_thesis:
            lines.append("3) Implement thesis via `explain_decision()` or include `_thesis` in process().")
        if self.ghost_contracts:
            lines.append("4) Remove/repair contract entries pointing to deleted files.")
        if self.duplicate_writers:
            lines.append("5) Enforce single-writer for critical keys or namespace outputs.")
        if self.missing_writers:
            lines.append("6) Ensure required inputs have at least one provider before runtime.")
        if self.cycles:
            lines.append("7) Break cycles via bus-first handoff or decouple reads/writes.")

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Validator
# ─────────────────────────────────────────────────────────────

class IntegrationValidator:
    """
    Validates SmartInfoBus integration, contracts, and system patterns.

    Pylance-clean by:
      • Protocol-based adapters (ExplainerProto, LoggerProto, FormatOperator)
      • No attribute access on unknown static members (InfoBus wrapper)
      • CONTRACTS normalized to dict[str, dict[str, Any]]
    """

    CRITICAL_SINGLE_WRITER: Set[str] = {
        "market_regime", "training_metrics", "performance_metrics",
        "risk_data", "sequence_quality", "trade_vote"
    }

    CONFIG_PATHS: List[str] = [
        "config/system_config.yaml",
        "config/risk_policy.yaml",
        "config/explainability_standards.yaml",
        "config/module_registry.yaml",
    ]

    def __init__(self, orchestrator: Optional[Any] = None, debug: Optional[bool] = None) -> None:
        self.debug: bool = bool(int(os.getenv("INTEGRATION_VALIDATOR_DEBUG", "1" if debug else "0"))) if debug is None else debug
        self.orchestrator = orchestrator
        self.smart_bus: Any = _get_bus()
        self.explainer: ExplainerProto = ExplainerClass()

        self.modules_root: Path = PROJECT_ROOT / "modules"
        self.module_roots: List[Path] = self._derive_module_roots()

        # LoggerClass is typed as type[LoggerProto] (Protocol with no __init__ signature),
        # but actual implementation RotatingLogger supports rich kwargs. Cast & ignore for Pylance.
        # Instantiate real RotatingLogger with explicit keyword args (LoggerProto allows flexible init)
        self.logger: LoggerProto = LoggerClass(
            name="IntegrationValidator",
            log_path="logs/validation/integration.log",
            max_lines=15000,
            operator_mode=True,
            plain_english=True,
            info_bus_aware=True,
        )

        # Discovery state
        self.discovered_modules: Dict[str, Dict[str, Any]] = {}
        self.module_files: Dict[str, Path] = {}

        # Contracts (fully normalized dict)
        self.contracts: Dict[str, Dict[str, Any]] = _normalize_contracts_map()

    # Public API ------------------------------------------------

    def validate_system(self, export_path: Optional[str] = None) -> ValidationReport:
        self.logger.info(format_operator("[SEARCH]", "Starting validation", context="validation"))

        self._discover_modules_recursive()
        report = ValidationReport(total_modules=len(self.discovered_modules), validated_modules=0)

        for class_name, info in sorted(self.discovered_modules.items()):
            self._validate_module(class_name, info, report)
            report.validated_modules += 1

        self._crosscheck_contracts(report)
        self._provider_consumer_analysis(report)
        self._validate_configurations(report)
        self._check_configuration_manager(report)

        report.integration_score = self._calculate_integration_score(report)
        if self.debug:
            report.manifest = self._build_debug_manifest()

        self._publish_report_to_bus(report)
        if export_path:
            self._export_report(report, export_path)

        self.logger.info(format_operator("[OK]", f"Validation complete (Score {report.integration_score:.1f}%)", context="validation"))
        return report

    # Roots & categories ---------------------------------------

    def _derive_module_roots(self) -> List[Path]:
        roots: List[Path] = []
        if self.modules_root.exists():
            for child in self.modules_root.iterdir():
                if child.is_dir() and child.name not in {"__pycache__", "tests", "test"}:
                    roots.append(child)
        special = self.modules_root / "market_1"
        if special.exists():
            roots.append(special)
        uniq: List[Path] = []
        seen: Set[str] = set()
        for r in sorted(roots, key=lambda p: (len(p.parts), p.as_posix())):
            k = r.as_posix()
            if k not in seen:
                seen.add(k)
                uniq.append(r)
        return uniq

    def _derive_categories_from_paths(self) -> Set[str]:
        cats: Set[str] = set()
        for root in self.module_roots:
            cats.add(root.name if root.name != "market_1" else "market")
        # keep minimal known extras
        for k in ("executor", "external", "trading_modes"):
            cats.add(k)
        return cats or {"other"}

    # Discovery -------------------------------------------------

    def _discover_modules_recursive(self) -> None:
        for root in self.module_roots:
            for py_file in root.rglob("*.py"):
                if py_file.name.startswith("_") or py_file.name == "__init__.py":
                    continue
                if any(seg in {"tests", "test", "migrations"} for seg in py_file.parts):
                    continue
                self._scan_file_for_modules(py_file, root)

    def _scan_file_for_modules(self, file_path: Path, root: Path) -> None:
        module_path = f"{root.as_posix().replace('/', '.')}.{file_path.relative_to(root).with_suffix('').as_posix().replace('/', '.')}"
        try:
            src = file_path.read_text(encoding="utf-8")
            tree = ast.parse(src)
        except Exception as e:
            self.logger.error(f"[PARSE] Failed to parse {file_path}: {e}")
            return

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and self._looks_like_module_class(node):
                class_name = node.name
                has_decorator, decorator_category = self._has_module_decorator(node)
                legacy = self._is_legacy_module_source(src)
                self.discovered_modules[class_name] = {
                    "module_path": module_path,
                    "file_path": file_path,
                    "ast_node": node,
                    "has_decorator": has_decorator,
                    "decorator_category": decorator_category,
                    "is_legacy": legacy
                }
                self.module_files[class_name] = file_path

    def _looks_like_module_class(self, node: ast.ClassDef) -> bool:
        if self._has_module_decorator(node)[0]:
            return True
        has_process = False
        has_aux = False
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if item.name in {"process", "step", "_step_impl"}:
                    has_process = True
                if item.name in {"get_state", "set_state", "validate_inputs", "explain_decision"}:
                    has_aux = True
        return has_process and has_aux

    def _has_module_decorator(self, node: ast.ClassDef) -> Tuple[bool, Optional[str]]:
        for dec in node.decorator_list:
            target: Optional[str] = None
            if isinstance(dec, ast.Name):
                target = dec.id
            elif isinstance(dec, ast.Attribute):
                target = dec.attr
            elif isinstance(dec, ast.Call):
                if isinstance(dec.func, ast.Name):
                    target = dec.func.id
                elif isinstance(dec.func, ast.Attribute):
                    target = dec.func.attr
            if target == "module":
                if isinstance(dec, ast.Call):
                    for kw in dec.keywords or []:
                        if kw.arg == "category":
                            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                                return True, kw.value.value
                return True, None
        return False, None

    def _is_legacy_module_source(self, src: str) -> bool:
        legacy_patterns = ("info_bus.get(", "info_bus[", "InfoBusExtractor.", "InfoBusUpdater.", "create_info_bus(")
        return any(p in src for p in legacy_patterns)

    # Per-module checks ----------------------------------------

    def _validate_module(self, module_name: str, info: Dict[str, Any], report: ValidationReport) -> None:
        if not info["has_decorator"]:
            report.missing_decorators.append(module_name)
            report.issues.append(ValidationIssue(
                module=module_name, issue_type="missing_decorator", severity="error",
                message="Module lacks @module decorator",
                file_path=str(info["file_path"]),
                suggestion="Add @module(provides=[...], requires=[...], category=...)"
            ))

        # Import class
        try:
            mod = importlib.import_module(info["module_path"])
            cls = getattr(mod, module_name, None)
        except Exception as e:
            report.issues.append(ValidationIssue(
                module=module_name, issue_type="import_error", severity="error",
                message=f"Failed to import: {e}", file_path=str(info["file_path"])
            ))
            return

        if cls is None:
            report.issues.append(ValidationIssue(
                module=module_name, issue_type="class_missing", severity="error",
                message="Class not found in module", file_path=str(info["file_path"])
            ))
            return

        metadata = getattr(cls, "__module_metadata__", None)
        if info["has_decorator"] and metadata is None:
            report.issues.append(ValidationIssue(
                module=module_name, issue_type="decorator_failed", severity="error",
                message="@module decorator didn't set metadata",
                suggestion="Check decorator wiring/imports"
            ))

        # Category (decorator > path inference)
        category = getattr(metadata, "category", None) if metadata is not None else None
        if not category:
            category = self._infer_category(module_name, str(info["file_path"]))
        valid_cats = self._derive_categories_from_paths()
        if category not in valid_cats:
            report.bad_categories.append(f"{module_name} → '{category}'")
            report.issues.append(ValidationIssue(
                module=module_name, issue_type="bad_category", severity="warning",
                message=f"Unknown category '{category}'",
                suggestion=f"Use one of: {', '.join(sorted(valid_cats))}"
            ))

        # Provides / requires
        provides = set(getattr(metadata, "provides", []) or []) if metadata is not None else set()
        requires = set(getattr(metadata, "requires", []) or []) if metadata is not None else set()
        explainable = bool(getattr(metadata, "explainable", False)) if metadata is not None else False

        if not provides:
            report.issues.append(ValidationIssue(
                module=module_name, issue_type="no_outputs", severity="warning",
                message="Module doesn't provide any outputs",
                suggestion="Add provides=['output_key'] to @module"
            ))

        # Explainability / thesis
        if explainable and not self._check_thesis_generation(cls):
            report.missing_thesis.append(module_name)
            report.issues.append(ValidationIssue(
                module=module_name, issue_type="missing_thesis", severity="warning",
                message="Explainable module doesn't generate thesis",
                suggestion="Implement thesis via explain_decision() or include '_thesis' in process() output"
            ))

        # Core methods + signature
        self._validate_core_methods(cls, module_name, report)

        # Legacy source markers
        if info["is_legacy"]:
            report.legacy_modules.append(module_name)
            report.issues.append(ValidationIssue(
                module=module_name, issue_type="legacy_pattern", severity="warning",
                message="Module uses legacy InfoBus patterns",
                file_path=str(info["file_path"]),
                suggestion="Migrate to SmartInfoBus.get/set"
            ))

        # Persist resolved
        info["resolved_category"] = category
        info["resolved_provides"] = sorted(provides)
        info["resolved_requires"] = sorted(requires)

    def _infer_category(self, class_name: str, file_path: str) -> str:
        p = Path(file_path)
        try:
            idx = p.parts.index("modules")
            if idx + 1 < len(p.parts):
                part = p.parts[idx + 1]
                return "market" if part == "market_1" else part
        except Exception:
            pass
        if class_name.startswith(("PPO", "Meta")):
            return "meta"
        if class_name.startswith(("Risk", "PortfolioRisk")):
            return "risk"
        if class_name.startswith(("Feature", "AdvancedFeature", "MultiScale")):
            return "features"
        if class_name.startswith(("Position",)):
            return "position"
        if class_name.startswith(("Reward",)):
            return "reward"
        return "other"

    def _check_thesis_generation(self, cls: Any) -> bool:
        if hasattr(cls, "explain_decision"):
            return True
        if hasattr(cls, "process"):
            try:
                src = inspect.getsource(cls.process)
                return ("_thesis" in src) or ("thesis" in src)
            except Exception:
                return False
        return False

    def _validate_core_methods(self, cls: Any, module_name: str, report: ValidationReport) -> None:
        required = ["process", "get_state", "set_state", "validate_inputs"]
        for m in required:
            if not hasattr(cls, m):
                report.issues.append(ValidationIssue(
                    module=module_name, issue_type="missing_method", severity="warning",
                    message=f"Missing {m}() method",
                    suggestion=f"Implement {m}() or inherit from BaseModule"
                ))
        if hasattr(cls, "process"):
            sig = inspect.signature(cls.process)
            has_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
            if not has_kwargs:
                report.issues.append(ValidationIssue(
                    module=module_name, issue_type="invalid_signature", severity="error",
                    message="process() should accept **inputs",
                    suggestion="Change signature to: async def process(self, **inputs)"
                ))

    # Contracts -------------------------------------------------

    def _crosscheck_contracts(self, report: ValidationReport) -> None:
        if not self.contracts:
            return

        discovered = set(self.discovered_modules.keys())
        registered = set(self.contracts.keys())

        # Ghosts
        for name in sorted(registered):
            file_rel = self.contracts[name].get("file", "")
            abs_path = (self.modules_root / file_rel).resolve() if file_rel else None
            if not file_rel or not (abs_path and abs_path.exists()):
                report.ghost_contracts.append(name)
                report.issues.append(ValidationIssue(
                    module=name, issue_type="ghost_contract", severity="warning",
                    message=f"Contract file not found: {file_rel or '[empty]'}",
                    suggestion="Remove or correct this entry in modules/contracts.py"
                ))

        # Unregistered
        unregistered = sorted(discovered - registered)
        if unregistered:
            report.unregistered_modules = unregistered
            report.issues.append(ValidationIssue(
                module="Contracts", issue_type="unregistered_modules", severity="info",
                message=f"{len(unregistered)} discovered modules not in contracts",
                suggestion="Optional: register in modules/contracts.py if you want strict auditing"
            ))

        # Mismatches
        for name in sorted(registered & discovered):
            info = self.discovered_modules[name]
            meta_provides = set(info.get("resolved_provides") or [])
            meta_requires = set(info.get("resolved_requires") or [])
            cobj: Dict[str, Any] = self.contracts[name]

            c_provides = set(cobj.get("provides") or [])
            c_requires = set(cobj.get("requires") or [])
            c_category = (cobj.get("meta") or {}).get("category", None)

            if c_provides and not c_provides.issubset(meta_provides):
                report.issues.append(ValidationIssue(
                    module=name, issue_type="contract_mismatch", severity="warning",
                    message=f"Decorator provides {sorted(meta_provides)} vs Contract {sorted(c_provides)}",
                    suggestion="Align @module(provides=...) with contract or update contract"
                ))
            if c_requires and not c_requires.issubset(meta_requires):
                report.issues.append(ValidationIssue(
                    module=name, issue_type="contract_mismatch", severity="warning",
                    message=f"Decorator requires {sorted(meta_requires)} vs Contract {sorted(c_requires)}",
                    suggestion="Align @module(requires=...) with contract or update contract"
                ))

            dcat = info.get("resolved_category")
            if c_category and dcat and (c_category != dcat):
                report.issues.append(ValidationIssue(
                    module=name, issue_type="category_mismatch", severity="warning",
                    message=f"Category differs (decorator/path='{dcat}' vs contract='{c_category}')",
                    suggestion="Normalize category in decorator or contract meta"
                ))

    # Topology --------------------------------------------------

    def _provider_consumer_analysis(self, report: ValidationReport) -> None:
        providers: DefaultDict[str, Set[str]] = defaultdict(set)
        consumers: DefaultDict[str, Set[str]] = defaultdict(set)

        for mname, info in self.discovered_modules.items():
            for k in info.get("resolved_provides") or []:
                providers[k].add(mname)
            for k in info.get("resolved_requires") or []:
                consumers[k].add(mname)

        duplicate: Dict[str, List[str]] = {}
        for key in sorted(providers.keys()):
            provs = sorted(providers[key])
            if len(provs) > 1 and (key in self.CRITICAL_SINGLE_WRITER or key.endswith("_vote")):
                duplicate[key] = provs

        missing: Dict[str, List[str]] = {}
        for key in sorted(consumers.keys()):
            if key not in providers or not providers[key]:
                missing[key] = sorted(consumers[key])

        report.duplicate_writers = duplicate
        report.missing_writers = missing

        # Build requires→providers graph for cycle scan
        graph: Dict[str, Set[str]] = defaultdict(set)
        for consumer, req_keys in ((m, set(info.get("resolved_requires") or [])) for m, info in self.discovered_modules.items()):
            for key in req_keys:
                for prov in providers.get(key, []):
                    graph[consumer].add(prov)

        report.cycles = self._find_cycles(graph)

    def _find_cycles(self, graph: Dict[str, Set[str]]) -> List[List[str]]:
        cycles: List[List[str]] = []
        visited: Set[str] = set()
        stack: List[str] = []

        def dfs(node: str) -> None:
            visited.add(node)
            stack.append(node)
            for nxt in graph.get(node, []):
                if nxt not in visited:
                    dfs(nxt)
                elif nxt in stack:
                    i = stack.index(nxt)
                    cyc = stack[i:] + [nxt]
                    text = "->".join(cyc)
                    if all("->".join(c) != text for c in cycles):
                        cycles.append(cyc)
            stack.pop()

        for n in list(graph.keys()):
            if n not in visited:
                dfs(n)
        return cycles

    # Config checks --------------------------------------------

    def _validate_configurations(self, report: ValidationReport) -> None:
        if yaml is None:
            report.config_issues.append("PyYAML not installed; config checks skipped")
            report.issues.append(ValidationIssue(
                module="Configuration", issue_type="dependency_missing", severity="info",
                message="PyYAML not available; skipping YAML config validation"
            ))
            return

        for path in self.CONFIG_PATHS:
            p = PROJECT_ROOT / path
            if not p.exists():
                report.config_issues.append(f"Missing config: {path}")
                report.issues.append(ValidationIssue(
                    module="Configuration", issue_type="missing_config", severity="warning",
                    message=f"Configuration file not found: {path}",
                    suggestion="Create configuration file from template"
                ))
                continue
            try:
                cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            except Exception as e:
                report.config_issues.append(f"Invalid YAML in {path}")
                report.issues.append(ValidationIssue(
                    module="Configuration", issue_type="invalid_yaml", severity="error",
                    message=f"Invalid YAML in {path}: {e}", suggestion="Fix YAML syntax errors"
                ))
                continue

            if not isinstance(cfg, dict):
                report.config_issues.append(f"Invalid configuration format in {path}: expected dict")
                continue

            s = p.name
            if "system_config" in s:
                for sec in ("system", "execution", "monitoring"):
                    if sec not in cfg:
                        report.issues.append(ValidationIssue(
                            module="Configuration", issue_type="incomplete_config", severity="warning",
                            message=f"Missing '{sec}' in system_config.yaml",
                            suggestion=f"Add '{sec}' section"
                        ))
            elif "risk_policy" in s:
                for sec in ("limits", "controls", "escalation"):
                    if sec not in cfg:
                        report.issues.append(ValidationIssue(
                            module="Configuration", issue_type="incomplete_config", severity="warning",
                            message=f"Missing '{sec}' in risk_policy.yaml",
                            suggestion=f"Add '{sec}' section"
                        ))
            elif "explainability_standards" in s:
                if "thesis_requirements" not in cfg:
                    report.issues.append(ValidationIssue(
                        module="Configuration", issue_type="missing_standards", severity="warning",
                        message="No thesis requirements in explainability standards",
                        suggestion="Define thesis requirements for consistency"
                    ))
            elif "module_registry" in s:
                modules = (cfg or {}).get("modules", {})
                if not modules:
                    report.issues.append(ValidationIssue(
                        module="Configuration", issue_type="empty_registry", severity="info",
                        message="module_registry.yaml has no modules",
                        suggestion="Optional: register active modules with category/provides/requires"
                    ))
                else:
                    registered = set(modules.keys())
                    discovered = set(self.discovered_modules.keys())
                    unregistered = discovered - registered
                    ghosts = registered - discovered
                    if unregistered:
                        report.issues.append(ValidationIssue(
                            module="Configuration", issue_type="unregistered_modules", severity="info",
                            message=f"Modules not in registry (sample): {', '.join(sorted(list(unregistered))[:6])}",
                            suggestion="Add all modules to module_registry.yaml"
                        ))
                    if ghosts:
                        report.issues.append(ValidationIssue(
                            module="Configuration", issue_type="ghost_modules", severity="warning",
                            message=f"Registry has non-existent modules: {', '.join(sorted(list(ghosts))[:6])}",
                            suggestion="Remove deleted modules from registry"
                        ))

    def _check_configuration_manager(self, report: ValidationReport) -> None:
        if not ConfigurationManager:
            return
        try:
            cm = ConfigurationManager.get_instance()  # type: ignore[attr-defined]
            syscfg = cm.get_system_config() or {}
            if not syscfg:
                report.config_issues.append("ConfigurationManager returned empty system config")
        except Exception:
            # CM may not be initialized during cold boot
            pass

    # Scoring & publishing -------------------------------------

    def _calculate_integration_score(self, report: ValidationReport) -> float:
        score = 100.0
        penalty_map = {"error": 4.0, "warning": 1.5, "info": 0.5}
        for i in report.issues:
            score -= penalty_map.get(i.severity, 1.0)
        score -= 1.0 * len(report.ghost_contracts)
        score -= 1.5 * len(report.duplicate_writers)
        score -= 1.0 * len([k for k in report.missing_writers if k in self.CRITICAL_SINGLE_WRITER])
        score -= 0.5 * len(report.cycles[:5])
        return max(0.0, min(100.0, score))

    def _publish_report_to_bus(self, report: ValidationReport) -> None:
        try:
            self.smart_bus.set(
                "validation/summary",
                {
                    "score": report.integration_score,
                    "checked": report.validated_modules,
                    "total": report.total_modules,
                    "errors": len([i for i in report.issues if i.severity == "error"]),
                    "warnings": len([i for i in report.issues if i.severity == "warning"]),
                },
                module="IntegrationValidator",
                thesis="SmartInfoBus integration validation summary"
            )
            issues_payload = [vars(i) for i in report.issues][-200:]
            self.smart_bus.set(
                "validation/issues",
                issues_payload,
                module="IntegrationValidator",
                thesis="Recent integration issues"
            )
        except Exception:
            pass

    def _export_report(self, report: ValidationReport, export_path: str) -> None:
        try:
            ep = Path(export_path)
            ep.parent.mkdir(parents=True, exist_ok=True)
            with ep.open("w", encoding="utf-8") as f:
                json.dump({
                    "report": {
                        "score": report.integration_score,
                        "total_modules": report.total_modules,
                        "validated_modules": report.validated_modules,
                        "issues": [vars(i) for i in report.issues],
                        "missing_decorators": report.missing_decorators,
                        "missing_thesis": report.missing_thesis,
                        "legacy_modules": report.legacy_modules,
                        "config_issues": report.config_issues,
                        "bad_categories": report.bad_categories,
                        "ghost_contracts": report.ghost_contracts,
                        "unregistered_modules": report.unregistered_modules,
                        "duplicate_writers": report.duplicate_writers,
                        "missing_writers": report.missing_writers,
                        "cycles": report.cycles,
                        "manifest": report.manifest if self.debug else {}
                    },
                    "plain_english": report.to_plain_english()
                }, f, indent=2)
            self.logger.info(format_operator("📄", "Validation exported", details=export_path, context="export"))
        except Exception as e:
            self.logger.error(f"[EXPORT] Failed to export validation report: {e}")

    # Debug manifest -------------------------------------------

    def _build_debug_manifest(self) -> Dict[str, Any]:
        manifest: Dict[str, Any] = {}
        for name, info in sorted(self.discovered_modules.items()):
            manifest[name] = {
                "file": str(info.get("file_path")),
                "category": info.get("resolved_category"),
                "provides": info.get("resolved_provides"),
                "requires": info.get("resolved_requires"),
                "has_decorator": bool(info.get("has_decorator")),
                "legacy_source": bool(info.get("is_legacy")),
                "module_path": info.get("module_path"),
            }
        return manifest

    # ─────────────────────────────────────────────────────────────
    # Convenience helpers used by SystemUtilities
    # ─────────────────────────────────────────────────────────────

    def generate_migration_guide(self) -> str:
        """Produce a plain-english migration guide based on current validation report."""
        report = self.validate_system()
        lines: List[str] = [
            "MIGRATION GUIDE",
            "=" * 50,
            f"Integration Score: {report.integration_score:.1f}%",
            "",
            "PRIORITIZED ACTIONS:",
            "- Add missing @module decorators (highest impact)",
            "- Normalize categories to folder-derived set",
            "- Remove ghost contract entries or restore files",
            "- Ensure each critical key has a single writer",
            "- Register active modules in module_registry.yaml",
            "",
        ]

        if report.missing_decorators:
            lines += ["Missing @module:"] + [f"  • {m}" for m in report.missing_decorators[:20]]
            if len(report.missing_decorators) > 20:
                lines.append(f"  • … and {len(report.missing_decorators)-20} more")

        if report.legacy_modules:
            lines += ["\nLegacy source patterns detected (migrate to SmartInfoBus):"] + [
                f"  • {m}" for m in report.legacy_modules[:20]
            ]

        if report.bad_categories:
            lines += ["\nUnknown categories:"] + [f"  • {c}" for c in report.bad_categories[:20]]

        if report.ghost_contracts:
            lines += ["\nGhost contracts (file missing):"] + [f"  • {g}" for g in report.ghost_contracts[:20]]

        if report.duplicate_writers:
            lines.append("\nDuplicate writers (critical keys should be single-writer):")
            for k, mods in sorted(report.duplicate_writers.items()):
                lines.append(f"  • {k}: {sorted(mods)}")

        if report.missing_writers:
            lines.append("\nMissing writers (required keys with no providers):")
            for k, consumers in sorted(report.missing_writers.items()):
                lines.append(f"  • {k}: needed by {sorted(consumers)}")

        if report.cycles:
            lines.append("\nPotential dependency cycles:")
            for cyc in report.cycles[:5]:
                lines.append("  • " + " → ".join(cyc))

        lines += [
            "\nNEXT STEPS:",
            "1) Update decorators and contracts to match provides/requires",
            "2) Register all active modules in module_registry.yaml",
            "3) Split duplicate writers or namespace their outputs",
            "4) Add validators for critical keys and enforce schemas",
            "5) Re-run the validator and ensure score ≥ 90%",
        ]

        return "\n".join(lines)

    def fix_common_issues(self, dry_run: bool = True) -> List[str]:
        """Return a list of recommended fixes derived from the current report.

        This method does not modify files. If automation is desired later,
        we can wire optional editors behind `dry_run=False`.
        """
        report = self.validate_system()
        actions: List[str] = []

        for m in report.missing_decorators:
            actions.append(f"Add @module decorator to {m}")
        for c in report.bad_categories:
            actions.append(f"Normalize category: {c}")
        for g in report.ghost_contracts:
            actions.append(f"Remove or fix ghost contract: {g}")
        for k, mods in sorted(report.duplicate_writers.items()):
            actions.append(f"Resolve duplicate writers for '{k}': {sorted(mods)}")
        for k, consumers in sorted(report.missing_writers.items()):
            actions.append(f"Add provider for '{k}' (needed by {sorted(consumers)})")
        if not actions:
            actions.append("No common issues detected — system looks good")

        # Placeholder for potential future automation branch
        if not dry_run:
            # Currently we do not auto-edit; keep behavior explicit and safe
            pass

        return actions


# ─────────────────────────────────────────────────────────────
# Boot-time auto-run (no manual execution required)
# ─────────────────────────────────────────────────────────────

def _run_validation_on_boot() -> None:
    """
    Runs once at process start (import time) if INTEGRATION_VALIDATE_ON_BOOT != '0'.
    Uses a daemon thread with a slight delay to let imports settle.
    """
    def _task() -> None:
        try:
            validator = IntegrationValidator(debug=bool(int(os.getenv("INTEGRATION_VALIDATOR_DEBUG", "0"))))
            export = os.getenv("INTEGRATION_VALIDATOR_EXPORT", "")
            validator.validate_system(export_path=export or None)
        except Exception:
            # Never break boot
            pass

    # default ON unless explicitly disabled
    if os.getenv("INTEGRATION_VALIDATE_ON_BOOT", "1") != "0":
        t = threading.Timer(0.25, _task)  # light delay to avoid import races
        t.daemon = True
        t.start()


# Trigger at import (safe & non-blocking)
_run_validation_on_boot()
