# ─────────────────────────────────────────────────────────────
# File: modules/utils/integration_validator.py
# [ROCKET] SmartInfoBus Integration Validator (v2.3)
# - contract-aware, config-aware, bus-publishing, full category coverage
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import sys
from pathlib import Path
# Add project root to Python path to ensure modules can be imported
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import importlib
import inspect
import ast
from typing import Dict, List, Any, Optional, TYPE_CHECKING, Tuple, Set
from dataclasses import dataclass, field
import json
import yaml

# Optional registries/managers (graceful fallbacks)
try:
    from modules.core.configuration_manager import ConfigurationManager  # type: ignore
except Exception:
    ConfigurationManager = None  # type: ignore

try:
    from modules.core.contracts_registry import ContractsRegistry  # type: ignore
except Exception:
    ContractsRegistry = None  # type: ignore

from modules.utils.info_bus import InfoBusManager
from modules.utils.system_utilities import EnglishExplainer
from modules.utils.audit_utils import RotatingLogger, format_operator_message

if TYPE_CHECKING:
    from modules.core.module_system import ModuleOrchestrator


# ─────────────────────────────────────────────────────────────
# Canonical categories (merge of your logger categories + module dirs)
# ─────────────────────────────────────────────────────────────
ALL_CATEGORIES: Set[str] = {
    # from rotating logger categories
    "agents","memory","risk","features","meta","models","environment","position",
    "reward","utils","auditing","analysis","trading",
    # existing module trees
    "market","strategy","voting","monitoring","core","orchestration","other"
}

# Heuristics for mapping file path or class prefix → category
CATEGORY_HINTS: List[Tuple[str, str]] = [
    ("modules/agents", "agents"),
    ("modules/memory", "memory"),
    ("modules/risk", "risk"),
    ("modules/features", "features"),
    ("modules/meta", "meta"),
    ("modules/models", "models"),
    ("modules/environment", "environment"),
    ("modules/position", "position"),
    ("modules/reward", "reward"),
    ("modules/utils", "utils"),
    ("modules/auditing", "auditing"),
    ("modules/analysis", "analysis"),
    ("modules/trading", "trading"),
    ("modules/market", "market"),
    ("modules/strategy", "strategy"),
    ("modules/voting", "voting"),
    ("modules/monitoring", "monitoring"),
    ("modules/core", "core"),
]

CLASS_PREFIX_HINTS: List[Tuple[str, str]] = [
    ("PPO", "agents"), ("Meta", "meta"),
    ("Memory", "memory"), ("PortfolioRisk", "risk"), ("Risk", "risk"),
    ("Feature", "features"), ("Model", "models"), ("Env", "environment"),
    ("Position", "position"), ("Reward", "reward"),
    ("Audit", "auditing"), ("Analyze", "analysis"), ("Trade", "trading"),
]


# ─────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────

@dataclass
class ValidationIssue:
    module: str
    issue_type: str
    severity: str  # 'error', 'warning', 'info'
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
    integration_score: float = 100.0

    def to_plain_english(self) -> str:
        explainer = EnglishExplainer()
        lines = [
            "SMARTINFOBUS INTEGRATION VALIDATION REPORT",
            "=" * 50,
            f"\nOverall Integration Score: {self.integration_score:.1f}%",
            f"Modules Checked: {self.validated_modules}/{self.total_modules}",
            ""
        ]
        if self.integration_score >= 90:
            lines.append("[OK] Excellent integration - System is well connected")
        elif self.integration_score >= 70:
            lines.append("[YELLOW] Good integration - Some improvements needed")
        else:
            lines.append("[RED] Poor integration - Significant work required")

        if self.issues:
            lines += ["\nISSUES FOUND:", "-" * 30]
            errors = [i for i in self.issues if i.severity == 'error']
            warnings = [i for i in self.issues if i.severity == 'warning']
            if errors:
                lines.append(f"\n[RED] Errors ({len(errors)}):")
                for issue in errors[:10]:
                    lines.append(f"  • {issue.module}: {issue.message}")
                if len(errors) > 10:
                    lines.append(f"  ... and {len(errors) - 10} more errors")
            if warnings:
                lines.append(f"\n[YELLOW] Warnings ({len(warnings)}):")
                for issue in warnings[:10]:
                    lines.append(f"  • {issue.module}: {issue.message}")

        if self.missing_decorators:
            lines += ["\n[FAIL] MODULES MISSING @module DECORATOR:", "-" * 40]
            for m in self.missing_decorators[:10]:
                lines.append(f"  • {m}")
            lines.append("\n  → Add @module(provides=[...], requires=[...], category=...)")

        if self.bad_categories:
            lines += ["\n[WARN] INVALID/UNKNOWN CATEGORIES:", "-" * 35]
            for m in self.bad_categories[:10]:
                lines.append(f"  • {m}")
            lines.append(f"\n  → Valid categories: {', '.join(sorted(ALL_CATEGORIES))}")

        if self.missing_thesis:
            lines += ["\n[LOG] MODULES NOT GENERATING THESIS:", "-" * 35]
            for m in self.missing_thesis[:10]:
                lines.append(f"  • {m}")
            lines.append("\n  → Explainable modules must provide thesis in outputs")

        if self.legacy_modules:
            lines += ["\n[RELOAD] LEGACY MODULES NEEDING MIGRATION:", "-" * 40]
            for m in self.legacy_modules[:10]:
                lines.append(f"  • {m}")
            lines.append("\n  → Replace legacy InfoBus patterns with SmartInfoBus.get/set")

        lines += ["\n\nRECOMMENDATIONS:", "-" * 20]
        if self.missing_decorators: lines.append("1. Add @module decorator to all module classes")
        if self.bad_categories:      lines.append("2. Normalize categories to the canonical list")
        if self.missing_thesis:      lines.append("3. Implement thesis generation in explain_decision()/process()")
        if self.legacy_modules:      lines.append("4. Migrate any legacy InfoBus usage to SmartInfoBus")
        if self.config_issues:       lines.append("5. Fix configuration file issues (YAML correctness & required sections)")
        if self.integration_score >= 90: lines.append("6. Continue monitoring integration health (CI gate)")

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Validator
# ─────────────────────────────────────────────────────────────

class IntegrationValidator:
    """
    Validates SmartInfoBus integration across the entire system:
      • decorator & metadata correctness (provides/requires/category/explainable)
      • thesis/explainability checks
      • legacy InfoBus patterns
      • process() signature (must accept **inputs; async allowed)
      • config sanity (system/risk/explainability)
      • contract-awareness (optional ContractsRegistry)
      • publishes results to SmartInfoBus (validation/*)
    """

    def __init__(self, orchestrator: Optional[ModuleOrchestrator] = None):
        self.orchestrator = orchestrator
        self.smart_bus = InfoBusManager.get_instance()
        self.explainer = EnglishExplainer()

        # Recursing discovery roots (you can add more safely)
        self.module_roots = [
            "modules/auditing", "modules/market", "modules/memory", "modules/strategy",
            "modules/risk", "modules/voting", "modules/monitoring", "modules/core",
            "modules/agents", "modules/features", "modules/meta", "modules/models",
            "modules/environment", "modules/position", "modules/reward", "modules/trading",
            "modules/analysis", "modules/utils"
        ]

        self.config_paths = [
            "config/system_config.yaml",
            "config/risk_policy.yaml",
            "config/explainability_standards.yaml",
            "config/module_registry.yaml"
        ]

        self.logger = RotatingLogger(
            name="IntegrationValidator",
            log_path="logs/validation/integration.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True,
            info_bus_aware=True
        )

        self.discovered_modules: Dict[str, Dict[str, Any]] = {}
        self.module_files: Dict[str, Path] = {}

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────
    def validate_system(self) -> ValidationReport:
        self.logger.info(format_operator_message(
            "[SEARCH]", "STARTING VALIDATION",
            details="SmartInfoBus integration check", context="validation"
        ))

        self._discover_modules_recursive()

        report = ValidationReport(total_modules=len(self.discovered_modules), validated_modules=0)

        for class_name, info in self.discovered_modules.items():
            self._validate_module(class_name, info, report)
            report.validated_modules += 1

        self._validate_configurations(report)
        self._check_system_patterns(report)
        report.integration_score = self._calculate_integration_score(report)

        # publish summary to the bus (best-effort)
        try:
            self.smart_bus.set(
                "validation/summary",
                {
                    'score': report.integration_score,
                    'checked': report.validated_modules,
                    'total': report.total_modules,
                    'errors': len([i for i in report.issues if i.severity == 'error']),
                    'warnings': len([i for i in report.issues if i.severity == 'warning']),
                },
                module="IntegrationValidator",
                thesis="SmartInfoBus integration validation summary"
            )
            self.smart_bus.set(
                "validation/issues",
                [vars(i) for i in report.issues][-200:],  # trim
                module="IntegrationValidator",
                thesis="Recent integration issues"
            )
        except Exception:
            pass

        self.logger.info(format_operator_message(
            "[OK]", "VALIDATION COMPLETE",
            details=f"Score: {report.integration_score:.1f}%", context="validation"
        ))
        return report

    def validate_and_export(self, export_path: str) -> ValidationReport:
        report = self.validate_system()
        try:
            Path(export_path).parent.mkdir(parents=True, exist_ok=True)
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump({
                    'report': {
                        'score': report.integration_score,
                        'total_modules': report.total_modules,
                        'validated_modules': report.validated_modules,
                        'issues': [vars(i) for i in report.issues],
                        'missing_decorators': report.missing_decorators,
                        'missing_thesis': report.missing_thesis,
                        'legacy_modules': report.legacy_modules,
                        'config_issues': report.config_issues,
                        'bad_categories': report.bad_categories
                    },
                    'plain_english': report.to_plain_english()
                }, f, indent=2)
            self.logger.info(format_operator_message("📄", "Validation exported", details=export_path, context="export"))
        except Exception as e:
            self.logger.error(f"Failed to export validation report: {e}")
        return report

    # ─────────────────────────────────────────────────────────
    # Discovery
    # ─────────────────────────────────────────────────────────
    def _discover_modules_recursive(self):
        for root in self.module_roots:
            r = Path(root)
            if not r.exists():
                continue
            for py_file in r.rglob("*.py"):
                if py_file.name.startswith("_") or py_file.name == "__init__.py":
                    continue
                # skip tests/migrations/etc.
                if any(seg in {"tests", "test", "migrations"} for seg in py_file.parts):
                    continue
                self._scan_file_for_modules(py_file, root)

    def _scan_file_for_modules(self, py_file: Path, root: str):
        full_module = f"{root.replace('/', '.')}.{py_file.relative_to(root).with_suffix('').as_posix().replace('/', '.')}"
        try:
            src = py_file.read_text(encoding="utf-8")
            tree = ast.parse(src)
        except Exception as e:
            self.logger.error(f"Failed to parse {py_file}: {e}")
            return

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and self._is_module_class(node):
                class_name = node.name
                has_decorator, category_arg = self._has_module_decorator(node)
                legacy = self._is_legacy_module(src)
                self.discovered_modules[class_name] = {
                    'module_path': full_module,
                    'file_path': py_file,
                    'ast_node': node,
                    'has_decorator': has_decorator,
                    'decorator_category': category_arg,
                    'is_legacy': legacy
                }
                self.module_files[class_name] = py_file

    def _is_module_class(self, node: ast.ClassDef) -> bool:
        # Exclude abstract base classes, neural network components, and base classes
        exclude_patterns = {
            'Base', 'Abstract', 'Network', 'Model', 'LSTM', 'RNN', 'CNN', 'MLP', 
            'Agent', 'Optimizer', 'Layer', 'FeatureFusion', 'ExpertBase'
        }
        
        # Check if class name contains any exclusion patterns
        if any(pattern in node.name for pattern in exclude_patterns):
            return False
            
        # Check if this is a PyTorch nn.Module (neural network component)
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == 'Module' and not base.id in {'Module', 'BaseModule'}:
                # This is likely a PyTorch nn.Module, not our Module
                return False
            if isinstance(base, ast.Attribute) and hasattr(base, 'attr') and base.attr == 'Module':
                # Check if it's from torch.nn
                if isinstance(base.value, ast.Name) and base.value.id == 'nn':
                    return False
        
        # Check for ABC metaclass (abstract base classes)
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == 'abstractmethod':
                return False
            if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name) and decorator.func.id == 'abstractmethod':
                return False
        
        # base class check - only consider classes that explicitly inherit from Module/BaseModule
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id in {'Module', 'BaseModule'}:
                return True
            if isinstance(base, ast.Attribute) and base.attr in {'Module', 'BaseModule'}:
                return True
                
        # For classes without explicit inheritance, use more conservative heuristics
        # Only consider classes that have both process and other module-like methods
        has_process = False
        has_module_methods = False
        
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if item.name in {'process', 'step', '_step_impl'}:
                    has_process = True
                if item.name in {'get_state', 'set_state', 'validate_inputs', 'explain_decision'}:
                    has_module_methods = True
                    
        return has_process and has_module_methods

    def _has_module_decorator(self, node: ast.ClassDef) -> Tuple[bool, Optional[str]]:
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name) and dec.id == 'module':
                return True, None
            if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name) and dec.func.id == 'module':
                # extract category kwarg if present
                for kw in dec.keywords or []:
                    if kw.arg == 'category':
                        if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                            return True, kw.value.value
                return True, None
        return False, None

    def _is_legacy_module(self, source: str) -> bool:
        legacy_patterns = (
            'info_bus.get(', 'info_bus[', 'InfoBusExtractor.', 'InfoBusUpdater.', 'create_info_bus('
        )
        return any(p in source for p in legacy_patterns)

    # ─────────────────────────────────────────────────────────
    # Per-module validation
    # ─────────────────────────────────────────────────────────
    def _validate_module(self, module_name: str, info: Dict[str, Any], report: ValidationReport):
        if not info['has_decorator']:
            report.missing_decorators.append(module_name)
            report.issues.append(ValidationIssue(
                module=module_name, issue_type='missing_decorator', severity='error',
                message="Module lacks @module decorator",
                file_path=str(info['file_path']),
                suggestion="Add @module(provides=[...], requires=[...], category=...)"
            ))

        # import class
        try:
            mod = importlib.import_module(info['module_path'])
            cls = getattr(mod, module_name, None)
        except Exception as e:
            report.issues.append(ValidationIssue(
                module=module_name, issue_type='import_error', severity='error',
                message=f"Failed to import: {e}", file_path=str(info['file_path'])
            ))
            return

        if not cls:
            report.issues.append(ValidationIssue(
                module=module_name, issue_type='class_missing', severity='error',
                message="Class not found in module", file_path=str(info['file_path'])
            ))
            return

        # metadata/decorator expectations
        metadata = getattr(cls, '__module_metadata__', None)
        if info['has_decorator'] and metadata is None:
            report.issues.append(ValidationIssue(
                module=module_name, issue_type='decorator_failed', severity='error',
                message="@module decorator didn't set metadata",
                suggestion="Check decorator syntax and imports"
            ))

        # category check (from decorator or inferred)
        cat = (getattr(metadata, 'category', None) if metadata else None) or info.get('decorator_category')
        if not cat:
            # infer from path/prefix
            cat = self._infer_category(module_name, str(info['file_path']))
        if cat not in ALL_CATEGORIES:
            report.bad_categories.append(f"{module_name} → '{cat}'")
            report.issues.append(ValidationIssue(
                module=module_name, issue_type='bad_category', severity='warning',
                message=f"Unknown category '{cat}'",
                suggestion=f"Use one of: {', '.join(sorted(ALL_CATEGORIES))}"
            ))

        # provides/requires
        if metadata:
            if not getattr(metadata, 'provides', None):
                report.issues.append(ValidationIssue(
                    module=module_name, issue_type='no_outputs', severity='warning',
                    message="Module doesn't provide any outputs",
                    suggestion="Add provides=['output_key'] to @module"
                ))
            if not hasattr(metadata, 'requires'):
                report.issues.append(ValidationIssue(
                    module=module_name, issue_type='no_requires', severity='info',
                    message="No `requires` specified", suggestion="Add requires=[...] when applicable"
                ))

        # explainability/thesis
        explainable = bool(getattr(metadata, 'explainable', False)) if metadata else False
        if explainable and not self._check_thesis_generation(cls):
            report.missing_thesis.append(module_name)
            report.issues.append(ValidationIssue(
                module=module_name, issue_type='missing_thesis', severity='warning',
                message="Explainable module doesn't generate thesis",
                suggestion="Implement thesis via explain_decision() or include '_thesis' in process() output"
            ))

        # process signature (must accept **inputs; async allowed)
        self._validate_methods(cls, module_name, report)

        # legacy usage flag
        if info['is_legacy']:
            report.legacy_modules.append(module_name)
            report.issues.append(ValidationIssue(
                module=module_name, issue_type='legacy_pattern', severity='warning',
                message="Module uses legacy InfoBus patterns",
                file_path=str(info['file_path']),
                suggestion="Migrate to SmartInfoBus.get/set"
            ))

        # optional: contracts registry cross-check + single-writer policy surfacing
        if ContractsRegistry:
            try:
                contract = ContractsRegistry.get(module_name)  # type: ignore
                if contract:
                    # verify provides/requires intersection
                    m_provides = set(getattr(metadata, 'provides', []) or [])
                    c_provides = set(contract.get('provides', []) or [])
                    if c_provides and not c_provides.issubset(m_provides):
                        report.issues.append(ValidationIssue(
                            module=module_name, issue_type='contract_mismatch', severity='warning',
                            message=f"Decorator provides {sorted(m_provides)} vs Contract {sorted(c_provides)}",
                            suggestion="Align @module(provides=...) with contract"
                        ))

                    # Single-writer policy: flag duplicates among critical keys
                    critical = {"market_regime", "training_metrics", "performance_metrics", "risk_data", "sequence_quality", "trade_vote"}
                    for key in (m_provides & critical):
                        try:
                            providers = self.smart_bus.get_providers(key)
                            if len(providers) > 1:
                                report.issues.append(ValidationIssue(
                                    module=module_name,
                                    issue_type='duplicate_writer',
                                    severity='warning',
                                    message=f"Key '{key}' has multiple providers: {sorted(list(providers))}",
                                    suggestion="Ensure single-writer or namespace keys; rely on committee where applicable"
                                ))
                        except Exception:
                            pass
            except Exception:
                pass

    def _infer_category(self, class_name: str, file_path: str) -> str:
        p = file_path.replace("\\", "/")
        for hint, cat in CATEGORY_HINTS:
            if hint in p: return cat
        for pref, cat in CLASS_PREFIX_HINTS:
            if class_name.startswith(pref): return cat
        return "other"

    def _check_thesis_generation(self, cls) -> bool:
        if hasattr(cls, 'explain_decision'):
            return True
        if hasattr(cls, 'process'):
            try:
                src = inspect.getsource(cls.process)
                return ('_thesis' in src) or ('thesis' in src)
            except Exception:
                return False
        return False

    def _validate_methods(self, cls, module_name: str, report: ValidationReport):
        required = ['process', 'get_state', 'set_state', 'validate_inputs']
        for m in required:
            if not hasattr(cls, m):
                report.issues.append(ValidationIssue(
                    module=module_name, issue_type='missing_method', severity='warning',
                    message=f"Missing {m}() method",
                    suggestion=f"Implement {m}() or inherit from BaseModule"
                ))
        # process signature
        if hasattr(cls, 'process'):
            sig = inspect.signature(cls.process)
            has_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
            if not has_kwargs:
                report.issues.append(ValidationIssue(
                    module=module_name, issue_type='invalid_signature', severity='error',
                    message="process() should accept **inputs",
                    suggestion="Change signature to: async def process(self, **inputs)"
                ))

    # ─────────────────────────────────────────────────────────
    # Config validation
    # ─────────────────────────────────────────────────────────
    def _validate_configurations(self, report: ValidationReport):
        for path in self.config_paths:
            p = Path(path)
            if not p.exists():
                report.config_issues.append(f"Missing config: {path}")
                report.issues.append(ValidationIssue(
                    module="Configuration", issue_type='missing_config', severity='error',
                    message=f"Configuration file not found: {path}",
                    suggestion="Create configuration file from template"
                ))
                continue
            try:
                cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError as e:
                report.config_issues.append(f"Invalid YAML in {path}")
                report.issues.append(ValidationIssue(
                    module="Configuration", issue_type='invalid_yaml', severity='error',
                    message=f"Invalid YAML in {path}: {e}", suggestion="Fix YAML syntax errors"
                ))
                continue

            # Only proceed with validation if cfg is a dictionary
            if not isinstance(cfg, dict):
                report.config_issues.append(f"Invalid configuration format in {path}: expected dict, got {type(cfg).__name__}")
                continue
                
            if 'system_config' in str(p):
                self._validate_system_config(cfg, report)
            elif 'risk_policy' in str(p):
                self._validate_risk_policy(cfg, report)
            elif 'explainability_standards' in str(p):
                self._validate_explainability_standards(cfg, report)
            elif 'module_registry' in str(p):
                self._validate_module_registry(cfg, report)

        # also consult ConfigurationManager if present
        if ConfigurationManager:
            try:
                cm = ConfigurationManager.get_instance()
                syscfg = cm.get_system_config() or {}
                if not syscfg:
                    report.config_issues.append("ConfigurationManager returned empty system config")
            except Exception:
                pass

    def _validate_system_config(self, cfg: Dict, report: ValidationReport):
        required = ['system', 'execution', 'monitoring']
        for sec in required:
            if sec not in cfg:
                report.issues.append(ValidationIssue(
                    module="Configuration", issue_type='incomplete_config', severity='warning',
                    message=f"Missing '{sec}' in system_config.yaml",
                    suggestion=f"Add {sec} section to configuration"
                ))

    def _validate_risk_policy(self, cfg: Dict, report: ValidationReport):
        required = ['limits', 'controls', 'escalation']
        for sec in required:
            if sec not in cfg:
                report.issues.append(ValidationIssue(
                    module="Configuration", issue_type='incomplete_config', severity='warning',
                    message=f"Missing '{sec}' in risk_policy.yaml",
                    suggestion=f"Add {sec} section to configuration"
                ))

    def _validate_module_registry(self, cfg: Dict, report: ValidationReport):
        modules = (cfg or {}).get('modules', {})
        if not modules:
            report.issues.append(ValidationIssue(
                module="Configuration", issue_type='empty_registry', severity='warning',
                message="module_registry.yaml has no modules",
                suggestion="Register all active modules with category/provides/requires"
            ))
            return
        registered = set(modules.keys())
        discovered = set(self.discovered_modules.keys())

        unregistered = discovered - registered
        if unregistered:
            sample = ", ".join(list(unregistered)[:5])
            report.issues.append(ValidationIssue(
                module="Configuration", issue_type='unregistered_modules', severity='warning',
                message=f"Modules not in registry: {sample}",
                suggestion="Add all modules to module_registry.yaml"
            ))

        ghosts = registered - discovered
        if ghosts:
            report.issues.append(ValidationIssue(
                module="Configuration", issue_type='ghost_modules', severity='warning',
                message=f"Registry has non-existent modules: {', '.join(sorted(ghosts))}",
                suggestion="Remove deleted modules from registry"
            ))

    def _validate_explainability_standards(self, cfg: Dict, report: ValidationReport):
        if 'thesis_requirements' not in cfg:
            report.issues.append(ValidationIssue(
                module="Configuration", issue_type='missing_standards', severity='warning',
                message="No thesis requirements in explainability standards",
                suggestion="Define thesis requirements for consistency"
            ))

    # ─────────────────────────────────────────────────────────
    # System-wide patterns & scoring
    # ─────────────────────────────────────────────────────────
    def _check_system_patterns(self, report: ValidationReport):
        smart_bus_usage = 0
        legacy_usage = 0
        for class_name, info in self.discovered_modules.items():
            try:
                content = info['file_path'].read_text(encoding="utf-8")
                if 'SmartInfoBus' in content or 'smart_bus' in content:
                    smart_bus_usage += 1
                if 'InfoBusExtractor' in content or 'InfoBusUpdater' in content:
                    legacy_usage += 1
            except Exception:
                pass

        if smart_bus_usage < max(1, int(len(self.discovered_modules) * 0.5)):
            report.issues.append(ValidationIssue(
                module="System", issue_type='low_adoption', severity='warning',
                message=f"Only {smart_bus_usage}/{len(self.discovered_modules)} modules reference SmartInfoBus",
                suggestion="Accelerate migration to SmartInfoBus"
            ))

        # circular deps (if bus provides such a method)
        try:
            if hasattr(self.smart_bus, 'find_circular_dependencies'):
                circ = self.smart_bus.find_circular_dependencies()
                if circ:
                    report.issues.append(ValidationIssue(
                        module="System", issue_type='circular_dependencies', severity='error',
                        message=f"Found {len(circ)} circular dependencies",
                        suggestion="Refactor to break circular dependencies"
                    ))
                else:
                    # No circular dependencies found
                    self.logger.info("No circular dependencies detected in SmartInfoBus")
            else:
                # Method not available, don't report false positives
                self.logger.warning("SmartInfoBus does not have find_circular_dependencies method")
        except Exception as e:
            # Log the exception but don't add a false positive issue
            self.logger.error(f"Error checking circular dependencies: {e}")

    def _calculate_integration_score(self, report: ValidationReport) -> float:
        score = 100.0
        for issue in report.issues:
            if issue.severity == 'error': score -= 5
            elif issue.severity == 'warning': score -= 2
        score -= len(report.missing_decorators) * 3
        score -= len(report.missing_thesis) * 2
        score -= len(report.legacy_modules) * 2
        score -= len(report.config_issues) * 5
        score -= len(report.bad_categories) * 1
        return max(0.0, score)

    # ─────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────
    def generate_migration_guide(self) -> str:
        # (kept from your original, trimmed where necessary)
        return """
SMARTINFOBUS MIGRATION GUIDE
============================
1) Add @module(provides=[...], requires=[...], category='...') on module classes.
2) Replace direct InfoBus usage with SmartInfoBus.get/set (with thesis/confidence).
3) Implement async def process(self, **inputs) and include '_thesis' for explainable outputs.
4) Provide get_state/set_state/validate_inputs for hot-reload and safety.
5) Register modules in config/module_registry.yaml and keep categories canonical:
   agents, analysis, auditing, core, environment, features, market, memory, meta,
   models, monitoring, position, reward, risk, strategy, trading, utils, voting, orchestration.

 SINGLE-WRITER KEYS POLICY (OBSERVABILITY-ONLY)
 ----------------------------------------------
 The system enforces a single-writer policy via audit hooks for the following keys:
   - market_regime, training_metrics, performance_metrics, risk_data, sequence_quality, trade_vote
 If multiple providers are detected, the pre-set hook logs a duplicate_writer event.
 Modules should avoid overwriting these keys if another provider exists or should namespace outputs.
"""

    def fix_common_issues(self, dry_run: bool = True) -> List[str]:
        fixes: List[str] = []
        for module_name, info in self.discovered_modules.items():
            if not info['has_decorator']:
                hint_cat = self._infer_category(module_name, str(info['file_path']))
                if dry_run:
                    fixes.append(f"Would add @module(provides=[], requires=[], category='{hint_cat}') to {module_name}")
                else:
                    fixes.append(f"Added @module(...) to {module_name} (manual insertion required)")
        return fixes
