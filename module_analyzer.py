#!/usr/bin/env python3
"""
🔍 Comprehensive Module Analysis Tool
Analyzes all modules, their dependencies, configurations, and interconnections
"""

import os
import ast
import re
import json
import yaml
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque

@dataclass
class ModuleInfo:
    """Complete information about a module"""
    name: str
    file_path: str
    category: str
    version: str = "1.0.0"
    provides: List[str] = field(default_factory=list)
    requires: List[str] = field(default_factory=list)
    config_class: Optional[str] = None
    config_location: Optional[str] = None
    config_fields: Dict[str, Any] = field(default_factory=dict)
    base_classes: List[str] = field(default_factory=list)
    mixins: List[str] = field(default_factory=list)
    has_thesis: bool = False
    has_health_monitoring: bool = False
    has_performance_tracking: bool = False
    has_error_handling: bool = False
    has_voting: bool = False
    imports: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

@dataclass
class ConfigInfo:
    """Information about a configuration class"""
    name: str
    file_path: str
    module_name: str
    fields: Dict[str, Any] = field(default_factory=dict)
    default_values: Dict[str, Any] = field(default_factory=dict)
    field_types: Dict[str, str] = field(default_factory=dict)
    has_validation: bool = False
    validation_methods: List[str] = field(default_factory=list)

@dataclass
class DependencyGraph:
    """Dependency graph analysis"""
    nodes: Set[str] = field(default_factory=set)
    edges: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    reverse_edges: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    circular_dependencies: List[List[str]] = field(default_factory=list)
    missing_providers: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    orphaned_modules: List[str] = field(default_factory=list)

class ModuleAnalyzer:
    """Comprehensive module analyzer"""
    
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.modules: Dict[str, ModuleInfo] = {}
        self.configs: Dict[str, ConfigInfo] = {}
        self.dependency_graph = DependencyGraph()
        self.all_providers: Dict[str, List[str]] = defaultdict(list)
        self.all_consumers: Dict[str, List[str]] = defaultdict(list)
        
    def analyze_all_modules(self) -> Dict[str, Any]:
        """Main analysis method"""
        print("🔍 Starting comprehensive module analysis...")
        
        # Step 1: Discover all module files
        module_files = self._discover_module_files()
        print(f"📁 Found {len(module_files)} module files")
        
        # Step 2: Parse each module
        for file_path in module_files:
            try:
                module_info = self._parse_module_file(file_path)
                if module_info:
                    self.modules[module_info.name] = module_info
                    print(f"✅ Parsed: {module_info.name}")
                else:
                    print(f"⚠️  Skipped: {file_path}")
            except Exception as e:
                print(f"❌ Error parsing {file_path}: {e}")
        
        # Step 3: Analyze configurations
        self._analyze_configurations()
        
        # Step 4: Build dependency graph
        self._build_dependency_graph()
        
        # Step 5: Detect issues
        self._detect_dependency_issues()
        
        # Step 6: Generate comprehensive report
        return self._generate_comprehensive_report()
    
    def _discover_module_files(self) -> List[Path]:
        """Discover all Python files in modules directory"""
        modules_dir = self.workspace_path / "modules"
        python_files = []
        
        if modules_dir.exists():
            for file_path in modules_dir.rglob("*.py"):
                if file_path.name != "__init__.py":
                    python_files.append(file_path)
        
        return python_files
    
    def _parse_module_file(self, file_path: Path) -> Optional[ModuleInfo]:
        """Parse a single module file to extract information"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Look for @module decorator
            module_decorator = self._find_module_decorator(content, tree)
            if not module_decorator:
                return None
            
            # Extract module information
            module_info = ModuleInfo(
                name=module_decorator.get('name', 'Unknown'),
                file_path=str(file_path),
                category=module_decorator.get('category', 'unknown'),
                version=module_decorator.get('version', '1.0.0'),
                provides=module_decorator.get('provides', []),
                requires=module_decorator.get('requires', []),
                has_thesis=module_decorator.get('thesis_required', False),
                has_health_monitoring=module_decorator.get('health_monitoring', False),
                has_performance_tracking=module_decorator.get('performance_tracking', False),
                has_error_handling=module_decorator.get('error_handling', False),
                has_voting=module_decorator.get('voting', False)
            )
            
            # Find configuration class
            config_class = self._find_config_class(tree)
            if config_class:
                module_info.config_class = config_class['name']
                module_info.config_location = str(file_path)
                module_info.config_fields = config_class['fields']
            
            # Find base classes and mixins
            module_class = self._find_module_class(tree)
            if module_class:
                module_info.base_classes = module_class['bases']
                module_info.mixins = [base for base in module_class['bases'] if 'Mixin' in base]
            
            # Extract imports
            module_info.imports = self._extract_imports(tree)
            
            return module_info
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None
    
    def _find_module_decorator(self, content: str, tree: ast.AST) -> Optional[Dict[str, Any]]:
        """Find and parse @module decorator"""
        # Use regex to find @module decorator
        module_pattern = r'@module\s*\(\s*([^)]+)\s*\)'
        match = re.search(module_pattern, content, re.MULTILINE | re.DOTALL)
        
        if not match:
            return None
        
        decorator_content = match.group(1)
        
        # Parse decorator parameters
        params = {}
        
        # Extract name
        name_match = re.search(r'name\s*=\s*["\']([^"\']+)["\']', decorator_content)
        if name_match:
            params['name'] = name_match.group(1)
        
        # Extract version
        version_match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', decorator_content)
        if version_match:
            params['version'] = version_match.group(1)
        
        # Extract category
        category_match = re.search(r'category\s*=\s*["\']([^"\']+)["\']', decorator_content)
        if category_match:
            params['category'] = category_match.group(1)
        
        # Extract provides list
        provides_match = re.search(r'provides\s*=\s*\[([^\]]+)\]', decorator_content)
        if provides_match:
            provides_str = provides_match.group(1)
            provides = [item.strip().strip('"\'') for item in provides_str.split(',') if item.strip()]
            params['provides'] = provides
        
        # Extract requires list
        requires_match = re.search(r'requires\s*=\s*\[([^\]]*)\]', decorator_content)
        if requires_match:
            requires_str = requires_match.group(1)
            if requires_str.strip():
                requires = [item.strip().strip('"\'') for item in requires_str.split(',') if item.strip()]
                params['requires'] = requires
            else:
                params['requires'] = []
        
        # Extract boolean flags
        bool_flags = ['thesis_required', 'health_monitoring', 'performance_tracking', 'error_handling', 'voting']
        for flag in bool_flags:
            flag_match = re.search(f'{flag}\\s*=\\s*(True|False)', decorator_content)
            if flag_match:
                params[flag] = flag_match.group(1) == 'True'
        
        return params
    
    def _find_config_class(self, tree: ast.AST) -> Optional[Dict[str, Any]]:
        """Find configuration dataclass in the module"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if node.name.endswith('Config'):
                    # Check if it has @dataclass decorator
                    has_dataclass = any(
                        isinstance(dec, ast.Name) and dec.id == 'dataclass' or
                        isinstance(dec, ast.Attribute) and dec.attr == 'dataclass'
                        for dec in node.decorator_list
                    )
                    
                    if has_dataclass:
                        fields = {}
                        for item in node.body:
                            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                                field_name = item.target.id
                                field_type = ast.unparse(item.annotation) if hasattr(ast, 'unparse') else 'Any'
                                default_value = ast.unparse(item.value) if item.value else None
                                fields[field_name] = {
                                    'type': field_type,
                                    'default': default_value
                                }
                        
                        return {
                            'name': node.name,
                            'fields': fields
                        }
        return None
    
    def _find_module_class(self, tree: ast.AST) -> Optional[Dict[str, Any]]:
        """Find the main module class"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        bases.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        bases.append(f"{base.value.id}.{base.attr}" if hasattr(base.value, 'id') else base.attr)
                
                if 'BaseModule' in bases:
                    return {
                        'name': node.name,
                        'bases': bases
                    }
        return None
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements"""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}" if module else alias.name)
        return imports
    
    def _analyze_configurations(self):
        """Analyze all configuration classes"""
        print("🔧 Analyzing configurations...")
        
        for module_name, module_info in self.modules.items():
            if module_info.config_class:
                config_info = ConfigInfo(
                    name=module_info.config_class,
                    file_path=module_info.config_location,
                    module_name=module_name,
                    fields=module_info.config_fields
                )
                self.configs[module_info.config_class] = config_info
    
    def _build_dependency_graph(self):
        """Build comprehensive dependency graph"""
        print("🕸️  Building dependency graph...")
        
        # Add all modules as nodes
        for module_name in self.modules.keys():
            self.dependency_graph.nodes.add(module_name)
        
        # Build provider mapping
        for module_name, module_info in self.modules.items():
            for provided in module_info.provides:
                self.all_providers[provided].append(module_name)
        
        # Build consumer mapping and edges
        for module_name, module_info in self.modules.items():
            for required in module_info.requires:
                self.all_consumers[required].append(module_name)
                
                # Find providers for this requirement
                providers = self.all_providers.get(required, [])
                for provider in providers:
                    if provider != module_name:  # No self-loops
                        self.dependency_graph.edges[provider].add(module_name)
                        self.dependency_graph.reverse_edges[module_name].add(provider)
    
    def _detect_dependency_issues(self):
        """Detect circular dependencies and other issues"""
        print("🔍 Detecting dependency issues...")
        
        # Detect circular dependencies using DFS
        visited = set()
        rec_stack = set()
        
        def dfs(node, path):
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                self.dependency_graph.circular_dependencies.append(cycle)
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self.dependency_graph.edges[node]:
                dfs(neighbor, path.copy())
            
            rec_stack.remove(node)
        
        for module in self.dependency_graph.nodes:
            if module not in visited:
                dfs(module, [])
        
        # Detect missing providers
        for module_name, module_info in self.modules.items():
            for required in module_info.requires:
                if required not in self.all_providers:
                    self.dependency_graph.missing_providers[required].append(module_name)
        
        # Detect orphaned modules (no providers or consumers)
        for module_name, module_info in self.modules.items():
            has_consumers = any(module_name in providers for providers in self.all_providers.values())
            has_providers = bool(module_info.requires)
            
            if not has_consumers and not module_info.provides:
                self.dependency_graph.orphaned_modules.append(module_name)
    
    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        print("📊 Generating comprehensive report...")
        
        # Module statistics
        total_modules = len(self.modules)
        categories = defaultdict(int)
        total_provides = 0
        total_requires = 0
        
        for module_info in self.modules.values():
            categories[module_info.category] += 1
            total_provides += len(module_info.provides)
            total_requires += len(module_info.requires)
        
        # Configuration statistics
        total_configs = len(self.configs)
        configs_with_validation = sum(1 for config in self.configs.values() if config.has_validation)
        
        report = {
            "summary": {
                "total_modules": total_modules,
                "total_configurations": total_configs,
                "total_data_providers": len(self.all_providers),
                "total_data_consumers": len(self.all_consumers),
                "categories": dict(categories),
                "average_provides_per_module": total_provides / total_modules if total_modules > 0 else 0,
                "average_requires_per_module": total_requires / total_modules if total_modules > 0 else 0
            },
            "modules": {name: {
                "category": info.category,
                "version": info.version,
                "provides": info.provides,
                "requires": info.requires,
                "config_class": info.config_class,
                "has_thesis": info.has_thesis,
                "has_health_monitoring": info.has_health_monitoring,
                "has_performance_tracking": info.has_performance_tracking,
                "has_error_handling": info.has_error_handling,
                "has_voting": info.has_voting,
                "mixins": info.mixins,
                "file_path": info.file_path
            } for name, info in self.modules.items()},
            "configurations": {name: {
                "module_name": info.module_name,
                "fields": info.fields,
                "file_path": info.file_path,
                "has_validation": info.has_validation
            } for name, info in self.configs.items()},
            "dependency_analysis": {
                "data_providers": dict(self.all_providers),
                "data_consumers": dict(self.all_consumers),
                "circular_dependencies": self.dependency_graph.circular_dependencies,
                "missing_providers": dict(self.dependency_graph.missing_providers),
                "orphaned_modules": self.dependency_graph.orphaned_modules
            },
            "interconnection_matrix": self._generate_interconnection_matrix(),
            "configuration_consolidation_plan": self._generate_config_consolidation_plan()
        }
        
        return report
    
    def _generate_interconnection_matrix(self) -> Dict[str, Any]:
        """Generate module interconnection matrix"""
        matrix = {}
        
        for module_name, module_info in self.modules.items():
            connections = {
                "dependencies": [],  # What this module depends on
                "dependents": [],   # What depends on this module
                "data_flow": {
                    "inputs": module_info.requires,
                    "outputs": module_info.provides
                }
            }
            
            # Find dependencies (modules this module depends on)
            for required in module_info.requires:
                providers = self.all_providers.get(required, [])
                connections["dependencies"].extend(providers)
            
            # Find dependents (modules that depend on this module)
            for provided in module_info.provides:
                consumers = self.all_consumers.get(provided, [])
                connections["dependents"].extend(consumers)
            
            matrix[module_name] = connections
        
        return matrix
    
    def _generate_config_consolidation_plan(self) -> Dict[str, Any]:
        """Generate plan for consolidating configurations"""
        plan = {
            "existing_config_files": [],
            "modules_with_configs": [],
            "consolidation_mapping": {},
            "proposed_structure": {}
        }
        
        # Check existing config files
        config_dir = self.workspace_path / "config"
        if config_dir.exists():
            plan["existing_config_files"] = [str(f) for f in config_dir.glob("*.yaml")]
        
        # Find modules with configurations
        for module_name, module_info in self.modules.items():
            if module_info.config_class:
                plan["modules_with_configs"].append({
                    "module": module_name,
                    "config_class": module_info.config_class,
                    "current_location": module_info.config_location,
                    "category": module_info.category
                })
        
        # Generate consolidation mapping
        category_configs = defaultdict(list)
        for module_name, module_info in self.modules.items():
            if module_info.config_class:
                category_configs[module_info.category].append({
                    "module": module_name,
                    "config_class": module_info.config_class,
                    "fields": module_info.config_fields
                })
        
        # Propose config file structure
        for category, configs in category_configs.items():
            if configs:
                plan["proposed_structure"][f"{category}_config.yaml"] = {
                    "description": f"Configuration for {category} modules",
                    "modules": [c["module"] for c in configs],
                    "sections": {c["module"]: c["fields"] for c in configs}
                }
        
        plan["consolidation_mapping"] = dict(category_configs)
        
        return plan

def main():
    """Main function"""
    workspace_path = r"c:\Users\Kushtrimi\Downloads\trading_agent"
    
    analyzer = ModuleAnalyzer(workspace_path)
    report = analyzer.analyze_all_modules()
    
    # Save comprehensive report
    output_file = Path(workspace_path) / "module_analysis_report.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 Analysis complete! Report saved to: {output_file}")
    
    # Print summary
    summary = report["summary"]
    print(f"\n📊 ANALYSIS SUMMARY:")
    print(f"   📦 Total Modules: {summary['total_modules']}")
    print(f"   ⚙️  Total Configurations: {summary['total_configurations']}")
    print(f"   🔗 Data Providers: {summary['total_data_providers']}")
    print(f"   📥 Data Consumers: {summary['total_data_consumers']}")
    print(f"   📂 Categories: {list(summary['categories'].keys())}")
    
    # Print issues
    issues = report["dependency_analysis"]
    if issues["circular_dependencies"]:
        print(f"\n⚠️  CIRCULAR DEPENDENCIES FOUND: {len(issues['circular_dependencies'])}")
        for i, cycle in enumerate(issues["circular_dependencies"][:3], 1):
            print(f"   {i}. {' → '.join(cycle)}")
    
    if issues["missing_providers"]:
        print(f"\n❌ MISSING PROVIDERS: {len(issues['missing_providers'])}")
        for provider, consumers in list(issues["missing_providers"].items())[:3]:
            print(f"   '{provider}' needed by: {', '.join(consumers)}")
    
    if issues["orphaned_modules"]:
        print(f"\n🏝️  ORPHANED MODULES: {len(issues['orphaned_modules'])}")
        print(f"   {', '.join(issues['orphaned_modules'][:5])}")
    
    print(f"\n📋 Full report available in: {output_file}")

if __name__ == "__main__":
    main()
