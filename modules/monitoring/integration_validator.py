# ─────────────────────────────────────────────────────────────
# File: modules/utils/integration_validator.py
# [ROCKET] Validates SmartInfoBus integration across all modules
# ─────────────────────────────────────────────────────────────

from __future__ import annotations
import importlib
import inspect
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional,  TYPE_CHECKING
from dataclasses import dataclass, field
import yaml
import json

from modules.utils.info_bus import  InfoBusManager
from modules.utils.system_utilities import EnglishExplainer
from modules.utils.audit_utils import RotatingLogger, format_operator_message

if TYPE_CHECKING:
    from modules.core.module_system import ModuleOrchestrator


@dataclass
class ValidationIssue:
    """Single validation issue"""
    module: str
    issue_type: str
    severity: str  # 'error', 'warning', 'info'
    message: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    suggestion: Optional[str] = None


@dataclass 
class ValidationReport:
    """Complete validation report"""
    total_modules: int
    validated_modules: int
    issues: List[ValidationIssue] = field(default_factory=list)
    missing_decorators: List[str] = field(default_factory=list)
    missing_thesis: List[str] = field(default_factory=list)
    legacy_modules: List[str] = field(default_factory=list)
    config_issues: List[str] = field(default_factory=list)
    integration_score: float = 100.0
    
    def to_plain_english(self) -> str:
        """Convert to plain English report"""
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
        
        # Issues summary
        if self.issues:
            lines.extend([
                "\nISSUES FOUND:",
                "-" * 30
            ])
            
            # Group by severity
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
        
        # Specific issues
        if self.missing_decorators:
            lines.extend([
                "\n[FAIL] MODULES MISSING @module DECORATOR:",
                "-" * 40
            ])
            for module in self.missing_decorators[:10]:
                lines.append(f"  • {module}")
            lines.append("\n  → These modules need the @module decorator for auto-discovery")
        
        if self.missing_thesis:
            lines.extend([
                "\n[LOG] MODULES NOT GENERATING THESIS:",
                "-" * 35
            ])
            for module in self.missing_thesis[:10]:
                lines.append(f"  • {module}")
            lines.append("\n  → Explainable modules must provide thesis in outputs")
        
        if self.legacy_modules:
            lines.extend([
                "\n[RELOAD] LEGACY MODULES NEEDING MIGRATION:",
                "-" * 40
            ])
            for module in self.legacy_modules[:10]:
                lines.append(f"  • {module}")
            lines.append("\n  → These still use old InfoBus patterns")
        
        # Recommendations
        lines.extend([
            "\n\nRECOMMENDATIONS:",
            "-" * 20
        ])
        
        if self.missing_decorators:
            lines.append("1. Add @module decorator to all module classes")
        if self.missing_thesis:
            lines.append("2. Implement thesis generation in explain_decision()")
        if self.legacy_modules:
            lines.append("3. Migrate legacy modules to SmartInfoBus patterns")
        if self.config_issues:
            lines.append("4. Fix configuration file issues")
        
        if self.integration_score >= 90:
            lines.append("5. Continue monitoring integration health")
        
        return "\n".join(lines)


class IntegrationValidator:
    """
    Validates SmartInfoBus integration across the entire system.
    Checks for proper decorators, thesis generation, and configuration.
    """
    
    def __init__(self, orchestrator: Optional[ModuleOrchestrator] = None):
        self.orchestrator = orchestrator
        self.smart_bus = InfoBusManager.get_instance()
        self.explainer = EnglishExplainer()
        
        # Module discovery paths
        self.module_paths = [
            "modules/auditing",
            "modules/market", 
            "modules/memory",
            "modules/strategy",
            "modules/risk",
            "modules/voting",
            "modules/monitoring",
            "modules/core"
        ]
        
        # Config paths
        self.config_paths = [
            "config/system_config.yaml",
            "config/risk_policy.yaml",
            "config/explainability_standards.yaml"
        ]
        
        # Setup logging
        self.logger = RotatingLogger(
            name="IntegrationValidator",
            log_path="logs/validation/integration.log",
            max_lines=2000,
            operator_mode=True,
            plain_english=True
        )
        
        # Discovered modules
        self.discovered_modules: Dict[str, Dict[str, Any]] = {}
        self.module_files: Dict[str, Path] = {}
    
    def validate_system(self) -> ValidationReport:
        """Perform complete system validation"""
        self.logger.info(
            format_operator_message(
                "[SEARCH]", "STARTING VALIDATION",
                details="SmartInfoBus integration check",
                context="validation"
            )
        )
        
        # Discover all modules
        self._discover_modules()
        
        # Create report
        report = ValidationReport(
            total_modules=len(self.discovered_modules),
            validated_modules=0
        )
        
        # Validate each module
        for module_name, module_info in self.discovered_modules.items():
            self._validate_module(module_name, module_info, report)
            report.validated_modules += 1
        
        # Validate configurations
        self._validate_configurations(report)
        
        # Check system-wide patterns
        self._check_system_patterns(report)
        
        # Calculate integration score
        report.integration_score = self._calculate_integration_score(report)
        
        self.logger.info(
            format_operator_message(
                "[OK]", "VALIDATION COMPLETE",
                details=f"Score: {report.integration_score:.1f}%",
                context="validation"
            )
        )
        
        return report
    
    def _discover_modules(self):
        """Discover all Python modules in the system"""
        for module_dir in self.module_paths:
            path = Path(module_dir)
            if not path.exists():
                continue
            
            for py_file in path.glob("*.py"):
                if py_file.name.startswith("_") or py_file.name == "__init__.py":
                    continue
                
                module_name = py_file.stem
                full_module = f"{module_dir.replace('/', '.')}.{module_name}"
                
                # Parse AST to check for classes
                try:
                    with open(py_file, 'r') as f:
                        tree = ast.parse(f.read())
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            # Check if it's a module class
                            if self._is_module_class(node):
                                class_name = node.name
                                self.discovered_modules[class_name] = {
                                    'module_path': full_module,
                                    'file_path': py_file,
                                    'ast_node': node,
                                    'has_decorator': self._has_module_decorator(node),
                                    'is_legacy': self._is_legacy_module(node)
                                }
                                self.module_files[class_name] = py_file
                                
                except Exception as e:
                    self.logger.error(f"Failed to parse {py_file}: {e}")
    
    def _is_module_class(self, node: ast.ClassDef) -> bool:
        """Check if AST node is a module class"""
        # Check base classes
        for base in node.bases:
            if isinstance(base, ast.Name):
                if base.id in ['Module', 'BaseModule']:
                    return True
            elif isinstance(base, ast.Attribute):
                if base.attr in ['Module', 'BaseModule']:
                    return True
        
        # Check if it has step or process method
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if item.name in ['step', 'process', '_step_impl']:
                    return True
        
        return False
    
    def _has_module_decorator(self, node: ast.ClassDef) -> bool:
        """Check if class has @module decorator"""
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == 'module':
                return True
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name) and decorator.func.id == 'module':
                    return True
        return False
    
    def _is_legacy_module(self, node: ast.ClassDef) -> bool:
        """Check if module uses legacy patterns"""
        source = ast.unparse(node)
        
        # Check for legacy patterns
        legacy_patterns = [
            'info_bus.get(',  # Direct InfoBus access
            'info_bus[',      # Dictionary-style access
            'InfoBusExtractor.',  # Legacy extractor
            'InfoBusUpdater.',    # Legacy updater
            'create_info_bus('    # Legacy creation
        ]
        
        return any(pattern in source for pattern in legacy_patterns)
    
    def _validate_module(self, module_name: str, module_info: Dict[str, Any], 
                        report: ValidationReport):
        """Validate a single module"""
        # Check for @module decorator
        if not module_info['has_decorator']:
            report.missing_decorators.append(module_name)
            report.issues.append(ValidationIssue(
                module=module_name,
                issue_type='missing_decorator',
                severity='error',
                message="Module lacks @module decorator",
                file_path=str(module_info['file_path']),
                suggestion="Add @module decorator with provides/requires lists"
            ))
        
        # Check for legacy patterns
        if module_info['is_legacy']:
            report.legacy_modules.append(module_name)
            report.issues.append(ValidationIssue(
                module=module_name,
                issue_type='legacy_pattern',
                severity='warning',
                message="Module uses legacy InfoBus patterns",
                file_path=str(module_info['file_path']),
                suggestion="Migrate to SmartInfoBus.get/set methods"
            ))
        
        # Try to import and check runtime properties
        try:
            module = importlib.import_module(module_info['module_path'])
            cls = getattr(module, module_name, None)
            
            if cls:
                # Check for metadata
                if hasattr(cls, '__module_metadata__'):
                    metadata = cls.__module_metadata__
                    
                    # Validate metadata
                    if metadata.explainable:
                        # Check for thesis generation
                        if not self._check_thesis_generation(cls):
                            report.missing_thesis.append(module_name)
                            report.issues.append(ValidationIssue(
                                module=module_name,
                                issue_type='missing_thesis',
                                severity='warning',
                                message="Explainable module doesn't generate thesis",
                                suggestion="Implement thesis in process() output"
                            ))
                    
                    # Check provides/requires
                    if not metadata.provides:
                        report.issues.append(ValidationIssue(
                            module=module_name,
                            issue_type='no_outputs',
                            severity='warning',
                            message="Module doesn't provide any outputs",
                            suggestion="Add provides=['output_key'] to @module"
                        ))
                else:
                    # Has decorator but no metadata?
                    if module_info['has_decorator']:
                        report.issues.append(ValidationIssue(
                            module=module_name,
                            issue_type='decorator_failed',
                            severity='error',
                            message="@module decorator didn't set metadata",
                            suggestion="Check decorator syntax and imports"
                        ))
                
                # Check method signatures
                self._validate_methods(cls, module_name, report)
                
        except Exception as e:
            report.issues.append(ValidationIssue(
                module=module_name,
                issue_type='import_error',
                severity='error',
                message=f"Failed to import module: {str(e)}",
                suggestion="Fix import errors and dependencies"
            ))
    
    def _check_thesis_generation(self, cls) -> bool:
        """Check if module can generate thesis"""
        # Check for explain_decision method
        if hasattr(cls, 'explain_decision'):
            return True
        
        # Check if process method includes _thesis in output
        if hasattr(cls, 'process'):
            # Get source code
            try:
                source = inspect.getsource(cls.process)
                return '_thesis' in source or 'thesis' in source
            except:
                pass
        
        return False
    
    def _validate_methods(self, cls, module_name: str, report: ValidationReport):
        """Validate module methods"""
        # Check for required methods
        required_methods = ['process', 'get_state', 'set_state', 'validate_inputs']
        
        for method_name in required_methods:
            if not hasattr(cls, method_name):
                report.issues.append(ValidationIssue(
                    module=module_name,
                    issue_type='missing_method',
                    severity='warning',
                    message=f"Missing {method_name}() method",
                    suggestion=f"Implement {method_name}() or inherit from BaseModule"
                ))
        
        # Check process method signature
        if hasattr(cls, 'process'):
            sig = inspect.signature(cls.process)
            
            # Should accept **kwargs
            has_kwargs = any(
                p.kind == p.VAR_KEYWORD 
                for p in sig.parameters.values()
            )
            
            if not has_kwargs:
                report.issues.append(ValidationIssue(
                    module=module_name,
                    issue_type='invalid_signature',
                    severity='error',
                    message="process() should accept **inputs",
                    suggestion="Change signature to: async def process(self, **inputs)"
                ))
    
    def _validate_configurations(self, report: ValidationReport):
        """Validate configuration files"""
        for config_path in self.config_paths:
            path = Path(config_path)
            
            if not path.exists():
                report.config_issues.append(f"Missing config: {config_path}")
                report.issues.append(ValidationIssue(
                    module="Configuration",
                    issue_type='missing_config',
                    severity='error',
                    message=f"Configuration file not found: {config_path}",
                    suggestion="Create configuration file from template"
                ))
                continue
            
            # Validate YAML
            try:
                with open(path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Validate structure based on file
                if 'system_config' in config_path:
                    self._validate_system_config(config, report)
                elif 'risk_policy' in config_path:
                    self._validate_risk_policy(config, report)
                elif 'explainability_standards' in config_path:
                    self._validate_explainability_standards(config, report)
                    
            except yaml.YAMLError as e:
                report.config_issues.append(f"Invalid YAML in {config_path}")
                report.issues.append(ValidationIssue(
                    module="Configuration",
                    issue_type='invalid_yaml',
                    severity='error',
                    message=f"Invalid YAML in {config_path}: {str(e)}",
                    suggestion="Fix YAML syntax errors"
                ))
    
    def _validate_system_config(self, config: Dict, report: ValidationReport):
        """Validate system configuration"""
        required_sections = ['system', 'execution', 'monitoring']
        
        for section in required_sections:
            if section not in config:
                report.issues.append(ValidationIssue(
                    module="Configuration",
                    issue_type='incomplete_config',
                    severity='warning',
                    message=f"Missing '{section}' in system_config.yaml",
                    suggestion=f"Add {section} section to configuration"
                ))
    
    def _validate_risk_policy(self, config: Dict, report: ValidationReport):
        """Validate risk policy configuration"""
        required_sections = ['system', 'execution', 'monitoring']
        
        for section in required_sections:
            if section not in config:
                report.issues.append(ValidationIssue(
                    module="Configuration",
                    issue_type='incomplete_config',
                    severity='warning',
                    message=f"Missing '{section}' in risk_policy.yaml",
                    suggestion=f"Add {section} section to configuration"
                ))
    
    def _validate_module_registry(self, config: Dict, report: ValidationReport):
        """Validate module registry against discovered modules"""
        if 'modules' not in config:
            return
        
        registered = set(config['modules'].keys())
        discovered = set(self.discovered_modules.keys())
        
        # Check for unregistered modules
        unregistered = discovered - registered
        if unregistered:
            report.issues.append(ValidationIssue(
                module="Configuration",
                issue_type='unregistered_modules',
                severity='warning',
                message=f"Modules not in registry: {', '.join(list(unregistered)[:5])}",
                suggestion="Add all modules to module_registry.yaml"
            ))
        
        # Check for ghost entries
        ghosts = registered - discovered
        if ghosts:
            report.issues.append(ValidationIssue(
                module="Configuration",
                issue_type='ghost_modules',
                severity='warning',
                message=f"Registry has non-existent modules: {', '.join(ghosts)}",
                suggestion="Remove deleted modules from registry"
            ))
    
    def _validate_explainability_standards(self, config: Dict, report: ValidationReport):
        """Validate explainability standards"""
        if 'thesis_requirements' not in config:
            report.issues.append(ValidationIssue(
                module="Configuration",
                issue_type='missing_standards',
                severity='warning',
                message="No thesis requirements in explainability standards",
                suggestion="Define thesis requirements for consistency"
            ))
    
    def _check_system_patterns(self, report: ValidationReport):
        """Check system-wide integration patterns"""
        # Check if SmartInfoBus is being used
        smart_bus_usage = 0
        legacy_usage = 0
        
        for module_name, module_info in self.discovered_modules.items():
            try:
                with open(module_info['file_path'], 'r') as f:
                    content = f.read()
                
                if 'SmartInfoBus' in content or 'smart_bus' in content:
                    smart_bus_usage += 1
                if 'InfoBusExtractor' in content or 'InfoBusUpdater' in content:
                    legacy_usage += 1
                    
            except:
                pass
        
        # Report on adoption
        if smart_bus_usage < len(self.discovered_modules) * 0.5:
            report.issues.append(ValidationIssue(
                module="System",
                issue_type='low_adoption',
                severity='warning',
                message=f"Only {smart_bus_usage}/{len(self.discovered_modules)} modules use SmartInfoBus",
                suggestion="Accelerate migration to SmartInfoBus"
            ))
        
        # Check for circular dependencies
        circular = self.smart_bus.find_circular_dependencies()
        if circular:
            report.issues.append(ValidationIssue(
                module="System",
                issue_type='circular_dependencies',
                severity='error',
                message=f"Found {len(circular)} circular dependencies",
                suggestion="Refactor to break circular dependencies"
            ))
    
    def _calculate_integration_score(self, report: ValidationReport) -> float:
        """Calculate overall integration score"""
        score = 100.0
        
        # Deduct points for issues
        for issue in report.issues:
            if issue.severity == 'error':
                score -= 5
            elif issue.severity == 'warning':
                score -= 2
        
        # Major deductions
        if report.missing_decorators:
            score -= len(report.missing_decorators) * 3
        if report.missing_thesis:
            score -= len(report.missing_thesis) * 2
        if report.legacy_modules:
            score -= len(report.legacy_modules) * 2
        if report.config_issues:
            score -= len(report.config_issues) * 5
        
        # Ensure score doesn't go below 0
        return max(0, score)
    
    def generate_migration_guide(self) -> str:
        """Generate migration guide for legacy modules"""
        guide = """
SMARTINFOBUS MIGRATION GUIDE
============================

1. ADD MODULE DECORATOR
----------------------
Replace class definition:
```python
# OLD:
class MyModule(Module):
    def __init__(self, config=None):
        super().__init__(config)

# NEW:
@module(
    provides=['output1', 'output2'],
    requires=['input1', 'input2'],
    category='market',
    explainable=True
)
class MyModule(BaseModule):
    def _initialize(self):
        # Module-specific init
        pass
```

2. REPLACE INFOBUS ACCESS
------------------------
Replace direct InfoBus access:
```python
# OLD:
value = info_bus.get('key', default)
info_bus['key'] = value

# NEW:
smart_bus = InfoBusManager.get_instance()
value = smart_bus.get('key', self.__class__.__name__)
smart_bus.set('key', value, self.__class__.__name__, 
              thesis="Explanation of why this value was set")
```

3. IMPLEMENT PROCESS METHOD
--------------------------
Replace step with process:
```python
# OLD:
def step(self, info_bus):
    data = info_bus['some_data']
    result = self.calculate(data)
    info_bus['result'] = result

# NEW:
async def process(self, **inputs) -> Dict[str, Any]:
    data = inputs['some_data']
    result = self.calculate(data)
    
    thesis = self.explain_decision(result, inputs)
    
    return {
        'result': result,
        '_thesis': thesis
    }
```

4. ADD STATE MANAGEMENT
----------------------
For hot-reload support:
```python
def get_state(self) -> Dict[str, Any]:
    return {
        'my_data': self.my_data,
        'history': list(self.history)
    }

def set_state(self, state: Dict[str, Any]):
    self.my_data = state.get('my_data', {})
    self.history = deque(state.get('history', []), maxlen=100)
```

5. IMPLEMENT THESIS GENERATION
-----------------------------
For explainable modules:
```python
def explain_decision(self, decision: Any, context: Dict) -> str:
    return f'''
DECISION EXPLANATION
===================
Action: {decision['action']}
Confidence: {decision['confidence']:.1%}

REASONING:
The decision was made because:
- Factor 1: {context['factor1']}
- Factor 2: {context['factor2']}

This approach was chosen to {decision['goal']}.
'''
```

6. USE SMARTINFOBUS FEATURES
---------------------------
- Versioned data: `smart_bus.get_versioned()`
- Data freshness: `smart_bus.get('key', max_age=60)`
- Confidence requirements: `smart_bus.get('key', min_confidence=0.8)`
- Request missing data: `smart_bus.request_data('key', 'MyModule')`

7. REGISTER WITH ORCHESTRATOR
----------------------------
The @module decorator handles registration automatically.
No need for manual registration in env.py.

8. UPDATE IMPORTS
----------------
```python
from modules.utils.info_bus import InfoBusManager
from modules.core.module_base import BaseModule, module
from modules.utils.english_explainer import EnglishExplainer
```

TESTING YOUR MIGRATION
=====================
1. Run the integration validator:
   ```python
   validator = IntegrationValidator()
   report = validator.validate_system()
   print(report.to_plain_english())
   ```

2. Check module appears in orchestrator:
   ```python
   orchestrator.modules.keys()
   ```

3. Verify data flow:
   ```python
   smart_bus.explain_data_flow('your_output_key')
   ```
"""
        return guide
    
    def fix_common_issues(self, dry_run: bool = True) -> List[str]:
        """Attempt to fix common issues automatically"""
        fixes = []
        
        for module_name, module_info in self.discovered_modules.items():
            if not module_info['has_decorator']:
                if dry_run:
                    fixes.append(f"Would add @module decorator to {module_name}")
                else:
                    # Add decorator (simplified - real implementation would parse AST)
                    fixes.append(f"Added @module decorator to {module_name}")
        
        return fixes