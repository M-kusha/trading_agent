# ─────────────────────────────────────────────────────────────
# File: modules/core/module_orchestrator.py
# 🚀 SmartInfoBus Module Orchestrator - Zero-wiring execution engine
# ─────────────────────────────────────────────────────────────

from __future__ import annotations
import asyncio
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Set, Type, Optional, Any, Callable
from collections import defaultdict, deque
import traceback
import time
import yaml
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import numpy as np

from modules.utils.info_bus import SmartInfoBus, InfoBusManager
from modules.utils.english_explainer import EnglishExplainer
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.core.module_base import BaseModule, ModuleMetadata
from modules.core.state_manager import StateManager
from modules.core.error_pinpointer import ErrorPinpointer


class ModuleOrchestrator:
    """
    Central orchestrator for SmartInfoBus modules.
    Handles:
    - Automatic module discovery
    - Dependency resolution
    - Parallel execution stages
    - Error handling and circuit breakers
    - Hot-reload support
    - Performance monitoring
    """
    
    _instance = None
    _registered_classes: Dict[str, Type[BaseModule]] = {}
    
    def __init__(self, smart_bus: Optional[SmartInfoBus] = None):
        """Initialize orchestrator with SmartInfoBus"""
        self.smart_bus = smart_bus or InfoBusManager.get_instance()
        self.explainer = EnglishExplainer()
        
        # Module registry
        self.modules: Dict[str, BaseModule] = {}
        self.metadata: Dict[str, ModuleMetadata] = {}
        self.module_classes: Dict[str, Type[BaseModule]] = {}
        
        # Execution planning
        self.execution_order: List[str] = []
        self.execution_stages: List[List[str]] = []
        self.voting_members: List[str] = []
        self.module_dependencies: Dict[str, Set[str]] = defaultdict(set)
        
        # Performance tracking
        self.execution_history: deque = deque(maxlen=1000)
        self.stage_timings: Dict[str, List[float]] = defaultdict(list)
        
        # Configuration
        self.config = self._load_configuration()
        
        # Error handling
        self.error_pinpointer = ErrorPinpointer(self)
        
        # Threading
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.get('max_parallel_modules', 10)
        )
        
        # Setup logging
        self.logger = RotatingLogger(
            name="ModuleOrchestrator",
            log_path="logs/orchestrator/orchestrator.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True
        )
        
        # State manager for hot-reload
        self.state_manager = StateManager()
        
        # Initialize
        ModuleOrchestrator._instance = self
        self._initialized = False
    
    @classmethod
    def get_instance(cls) -> ModuleOrchestrator:
        """Get orchestrator singleton"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def register_class(cls, module_class: Type[BaseModule]):
        """Called by @module decorator to register classes"""
        class_name = module_class.__name__
        cls._registered_classes[class_name] = module_class
        
        # If orchestrator exists, register immediately
        if cls._instance:
            cls._instance.register_module(class_name, module_class)
    
    def initialize(self):
        """Initialize orchestrator and discover modules"""
        if self._initialized:
            return
        
        self.logger.info(
            format_operator_message(
                "🚀", "INITIALIZING ORCHESTRATOR",
                details="Starting SmartInfoBus module discovery",
                context="startup"
            )
        )
        
        # Load configuration
        self._load_policies()
        
        # Discover all modules
        self.discover_all_modules()
        
        # Build execution plan
        self.build_execution_plan()
        
        # Validate system
        self._validate_system_integrity()
        
        self._initialized = True
        
        self.logger.info(
            format_operator_message(
                "✅", "ORCHESTRATOR READY",
                details=f"{len(self.modules)} modules loaded",
                context="startup"
            )
        )
    
    def discover_all_modules(self):
        """Auto-discover all decorated modules"""
        # First, register any classes that were decorated before orchestrator init
        for name, cls in self._registered_classes.items():
            if name not in self.modules:
                self.register_module(name, cls)
        
        # Then scan module directories for any we missed
        module_dirs = self.config.get('module_paths', [
            'modules/auditing',
            'modules/market',
            'modules/memory', 
            'modules/strategy',
            'modules/risk',
            'modules/voting',
            'modules/monitoring',
            'modules/example'
        ])
        
        for dir_path in module_dirs:
            path = Path(dir_path)
            if not path.exists():
                continue
            
            for py_file in path.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                
                try:
                    # Import to trigger decorators
                    module_name = f"{dir_path.replace('/', '.')}.{py_file.stem}"
                    importlib.import_module(module_name)
                except Exception as e:
                    self.logger.error(f"Failed to import {module_name}: {e}")
        
        # Register any new classes discovered
        for name, cls in self._registered_classes.items():
            if name not in self.modules:
                self.register_module(name, cls)
    
    def register_module(self, name: str, module_class: Type[BaseModule]):
        """Register a module with the orchestrator"""
        try:
            # Get metadata
            if not hasattr(module_class, '__module_metadata__'):
                self.logger.warning(f"Module {name} missing metadata, skipping")
                return
            
            metadata = module_class.__module_metadata__
            
            # Create instance
            instance = module_class()
            
            # Store
            self.modules[name] = instance
            self.metadata[name] = metadata
            self.module_classes[name] = module_class
            
            # Register with SmartInfoBus
            self.smart_bus.register_provider(name, metadata.provides)
            self.smart_bus.register_consumer(name, metadata.requires)
            
            # Track voting members
            if metadata.is_voting_member:
                self.voting_members.append(name)
            
            # Build dependencies
            self._build_module_dependencies(name, metadata)
            
            self.logger.info(
                format_operator_message(
                    "📦", "MODULE REGISTERED",
                    instrument=name,
                    details=f"v{metadata.version} ({metadata.category})",
                    context="discovery"
                )
            )
            
        except Exception as e:
            self.logger.error(f"Failed to register {name}: {e}")
            self.error_pinpointer.analyze_exception(e, name, "register_module")
    
    def _build_module_dependencies(self, module_name: str, metadata: ModuleMetadata):
        """Build dependency graph for a module"""
        # Find which modules provide required data
        for required_key in metadata.requires:
            providers = self.smart_bus.get_providers(required_key)
            for provider in providers:
                if provider != module_name and provider in self.modules:
                    self.module_dependencies[module_name].add(provider)
    
    def build_execution_plan(self):
        """Build optimized execution plan with parallel stages"""
        # Get dependency graph
        dependency_graph = self._build_full_dependency_graph()
        
        # Topological sort for base order
        self.execution_order = self._topological_sort(dependency_graph)
        
        # Build parallel execution stages
        self.execution_stages = self._build_parallel_stages(dependency_graph)
        
        # Log execution plan
        self._log_execution_plan()
    
    def _build_full_dependency_graph(self) -> Dict[str, Set[str]]:
        """Build complete dependency graph"""
        graph = defaultdict(set)
        
        # Add module dependencies
        for module, deps in self.module_dependencies.items():
            graph[module].update(deps)
        
        # Add data dependencies
        for module_name, metadata in self.metadata.items():
            for required_key in metadata.requires:
                providers = self.smart_bus.get_providers(required_key)
                for provider in providers:
                    if provider != module_name and provider in self.modules:
                        graph[module_name].add(provider)
        
        return dict(graph)
    
    def _topological_sort(self, graph: Dict[str, Set[str]]) -> List[str]:
        """Topological sort for dependency order"""
        # Kahn's algorithm
        in_degree = defaultdict(int)
        
        # Calculate in-degrees
        for node in graph:
            for dep in graph[node]:
                in_degree[dep] += 1
        
        # Find nodes with no incoming edges
        queue = deque([
            node for node in self.modules 
            if in_degree[node] == 0
        ])
        
        result = []
        
        while queue:
            node = queue.popleft()
            result.append(node)
            
            # Remove edges from this node
            for neighbor in graph.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check for cycles
        if len(result) != len(self.modules):
            # Handle circular dependencies
            remaining = set(self.modules) - set(result)
            self.logger.warning(
                f"Circular dependencies detected involving: {remaining}"
            )
            # Add remaining modules in priority order
            result.extend(sorted(
                remaining,
                key=lambda m: self.metadata[m].priority,
                reverse=True
            ))
        
        return result
    
    def _build_parallel_stages(self, graph: Dict[str, Set[str]]) -> List[List[str]]:
        """Build stages of modules that can run in parallel"""
        stages = []
        remaining = set(self.modules.keys())
        completed = set()
        
        while remaining:
            # Find modules whose dependencies are satisfied
            stage = []
            for module in remaining:
                deps = graph.get(module, set())
                if deps.issubset(completed):
                    stage.append(module)
            
            if not stage:
                # Circular dependency - take modules with highest priority
                stage = sorted(
                    list(remaining),
                    key=lambda m: self.metadata[m].priority,
                    reverse=True
                )[:5]
                self.logger.warning(
                    f"Forcing stage with potential circular deps: {stage}"
                )
            
            stages.append(stage)
            completed.update(stage)
            remaining -= set(stage)
        
        return stages
    
    def _log_execution_plan(self):
        """Log the execution plan in plain English"""
        plan_description = "EXECUTION PLAN:\n"
        plan_description += "=" * 50 + "\n\n"
        
        for i, stage in enumerate(self.execution_stages, 1):
            plan_description += f"Stage {i} (parallel execution):\n"
            for module in stage:
                metadata = self.metadata[module]
                plan_description += f"  • {module} ({metadata.category})"
                if metadata.is_voting_member:
                    plan_description += " [VOTER]"
                plan_description += "\n"
            plan_description += "\n"
        
        self.logger.info(plan_description)
    
    async def execute_step(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute all modules in dependency order with parallel stages.
        This is the main orchestration method.
        """
        start_time = time.time()
        results = {}
        stage_results = []
        
        # Store market data in SmartInfoBus
        self._store_market_data(market_data)
        
        # Execute each stage
        for stage_idx, stage_modules in enumerate(self.execution_stages):
            stage_start = time.time()
            
            # Execute modules in parallel within stage
            stage_result = await self._execute_stage(
                stage_modules,
                stage_idx,
                results
            )
            
            results.update(stage_result)
            stage_results.append(stage_result)
            
            # Record stage timing
            stage_duration = (time.time() - stage_start) * 1000
            self.stage_timings[f"stage_{stage_idx}"].append(stage_duration)
            
            # Check for stage failures
            if self._check_stage_failures(stage_result):
                self.logger.error(
                    f"Stage {stage_idx} had critical failures, stopping execution"
                )
                break
        
        # Aggregate results
        aggregated = self._aggregate_results(results)
        
        # Record execution
        execution_time = (time.time() - start_time) * 1000
        self._record_execution(execution_time, results, aggregated)
        
        # Generate execution summary
        summary = self._generate_execution_summary(
            execution_time,
            stage_results,
            aggregated
        )
        
        self.logger.info(summary)
        
        return aggregated
    
    def _store_market_data(self, market_data: Dict[str, Any]):
        """Store market data in SmartInfoBus"""
        for key, value in market_data.items():
            self.smart_bus.set(
                key,
                value,
                module="Orchestrator",
                thesis="Market data from environment"
            )
    
    async def _execute_stage(self, module_names: List[str], 
                           stage_idx: int,
                           previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a stage of modules in parallel"""
        tasks = []
        results = {}
        
        for module_name in module_names:
            if not self.smart_bus.is_module_enabled(module_name):
                self.logger.warning(f"Skipping disabled module: {module_name}")
                continue
            
            module = self.modules[module_name]
            metadata = self.metadata[module_name]
            
            # Prepare inputs
            inputs = self._prepare_module_inputs(module_name, metadata)
            
            # Create task
            task = asyncio.create_task(
                self._execute_module_with_timeout(
                    module,
                    module_name,
                    inputs,
                    metadata.timeout_ms / 1000.0
                )
            )
            
            tasks.append((module_name, task))
        
        # Wait for all tasks with individual handling
        for module_name, task in tasks:
            try:
                result = await task
                if result:
                    results[module_name] = result
            except Exception as e:
                self.logger.error(f"Module {module_name} failed: {e}")
                self.error_pinpointer.analyze_exception(e, module_name, "execute")
                results[module_name] = {'error': str(e)}
        
        return results
    
    def _prepare_module_inputs(self, module_name: str, 
                             metadata: ModuleMetadata) -> Dict[str, Any]:
        """Prepare inputs for a module from SmartInfoBus"""
        inputs = {}
        
        for required_key in metadata.requires:
            # Get data with metadata
            data = self.smart_bus.get_with_metadata(required_key, module_name)
            
            if data:
                # Check freshness if needed
                if data.age_seconds() > 60:  # 1 minute default
                    self.logger.warning(
                        f"Stale data for {module_name}: "
                        f"{required_key} is {data.age_seconds():.1f}s old"
                    )
                
                inputs[required_key] = data.value
            else:
                # Request data if not available
                self.smart_bus.request_data(required_key, module_name)
                inputs[required_key] = None
        
        return inputs
    
    async def _execute_module_with_timeout(self, module: BaseModule,
                                         module_name: str,
                                         inputs: Dict[str, Any],
                                         timeout: float) -> Optional[Dict[str, Any]]:
        """Execute module with timeout and error handling"""
        try:
            # Start timing
            start_time = time.perf_counter()
            
            # Execute module
            if hasattr(module, 'process'):
                # Async process method
                result = await asyncio.wait_for(
                    module.process(**inputs),
                    timeout=timeout
                )
            else:
                # Legacy step method - run in executor
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        self._execute_legacy_module,
                        module,
                        inputs
                    ),
                    timeout=timeout
                )
            
            # Record timing
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.smart_bus.record_module_timing(module_name, duration_ms)
            
            # Validate output
            if result and isinstance(result, dict):
                self._validate_module_output(module_name, result)
            
            return result
            
        except asyncio.TimeoutError:
            duration_ms = timeout * 1000
            error_msg = f"Timeout after {duration_ms:.0f}ms"
            self.smart_bus.record_module_failure(module_name, error_msg)
            raise
            
        except Exception as e:
            self.smart_bus.record_module_failure(module_name, str(e))
            raise
    
    def _execute_legacy_module(self, module: BaseModule, 
                             inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute legacy module with step method"""
        # Create info_bus structure
        from modules.utils.info_bus import InfoBusManager
        info_bus = InfoBusManager.create_info_bus(None, 0)
        
        # Add inputs to info_bus
        for key, value in inputs.items():
            info_bus[key] = value
        
        # Execute step
        module.step(info_bus)
        
        # Extract outputs based on metadata
        outputs = {}
        metadata = module.__class__.__module_metadata__
        
        for output_key in metadata.provides:
            # Check SmartInfoBus for output
            value = self.smart_bus.get(output_key, module.__class__.__name__)
            if value is not None:
                outputs[output_key] = value
        
        return outputs
    
    def _validate_module_output(self, module_name: str, output: Dict[str, Any]):
        """Validate module output meets requirements"""
        metadata = self.metadata[module_name]
        
        # Check all promised outputs are present
        missing = []
        for output_key in metadata.provides:
            if output_key not in output:
                # Check if it was set directly in SmartInfoBus
                if not self.smart_bus.get(output_key, module_name):
                    missing.append(output_key)
        
        if missing:
            self.logger.warning(
                f"Module {module_name} missing outputs: {missing}"
            )
        
        # Check for thesis if explainable
        if metadata.explainable and '_thesis' not in output:
            # Check SmartInfoBus for thesis
            thesis_key = f'thesis_{module_name}'
            if not self.smart_bus.get(thesis_key, module_name):
                self.logger.warning(
                    f"Explainable module {module_name} did not provide thesis"
                )
    
    def _check_stage_failures(self, stage_result: Dict[str, Any]) -> bool:
        """Check if stage had critical failures"""
        failures = sum(
            1 for result in stage_result.values()
            if isinstance(result, dict) and 'error' in result
        )
        
        total = len(stage_result)
        failure_rate = failures / max(total, 1)
        
        # Critical if more than 50% failed
        return failure_rate > 0.5
    
    def _aggregate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results from all modules"""
        aggregated = {
            'timestamp': time.time(),
            'module_count': len(results),
            'successful_modules': sum(
                1 for r in results.values()
                if isinstance(r, dict) and 'error' not in r
            ),
            'errors': [],
            'votes': {},
            'signals': {},
            'analysis': {}
        }
        
        # Extract specific result types
        for module_name, result in results.items():
            if isinstance(result, dict):
                if 'error' in result:
                    aggregated['errors'].append({
                        'module': module_name,
                        'error': result['error']
                    })
                
                # Extract votes
                if 'vote' in result:
                    aggregated['votes'][module_name] = result['vote']
                
                # Extract trading signals
                if 'trading_signal' in result:
                    aggregated['signals'][module_name] = result['trading_signal']
                
                # General analysis
                for key, value in result.items():
                    if key not in ['error', 'vote', 'trading_signal', '_thesis']:
                        if key not in aggregated['analysis']:
                            aggregated['analysis'][key] = {}
                        aggregated['analysis'][key][module_name] = value
        
        return aggregated
    
    def _record_execution(self, execution_time: float, 
                        results: Dict[str, Any],
                        aggregated: Dict[str, Any]):
        """Record execution for analysis"""
        record = {
            'timestamp': time.time(),
            'execution_time_ms': execution_time,
            'module_count': len(results),
            'success_count': aggregated['successful_modules'],
            'error_count': len(aggregated['errors']),
            'stage_count': len(self.execution_stages)
        }
        
        self.execution_history.append(record)
        
        # Update SmartInfoBus
        self.smart_bus.set(
            'orchestrator_metrics',
            record,
            module='Orchestrator',
            thesis=f"Executed {len(results)} modules in {execution_time:.0f}ms"
        )
    
    def _generate_execution_summary(self, execution_time: float,
                                  stage_results: List[Dict],
                                  aggregated: Dict[str, Any]) -> str:
        """Generate plain English execution summary"""
        summary = f"""
EXECUTION SUMMARY
================
Total Time: {execution_time:.0f}ms
Modules Executed: {aggregated['module_count']}
Successful: {aggregated['successful_modules']}
Failed: {len(aggregated['errors'])}

STAGE BREAKDOWN:
"""
        
        for i, stage_result in enumerate(stage_results):
            stage_time = self.stage_timings[f"stage_{i}"][-1] if f"stage_{i}" in self.stage_timings else 0
            summary += f"  Stage {i+1}: {len(stage_result)} modules in {stage_time:.0f}ms\n"
        
        if aggregated['errors']:
            summary += "\nERRORS:\n"
            for error in aggregated['errors'][:5]:
                summary += f"  • {error['module']}: {error['error']}\n"
        
        if aggregated['votes']:
            summary += f"\nVOTES COLLECTED: {len(aggregated['votes'])}\n"
        
        if aggregated['signals']:
            summary += f"\nTRADING SIGNALS: {len(aggregated['signals'])}\n"
        
        return summary
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load orchestrator configuration"""
        default_config = {
            'max_parallel_modules': 10,
            'default_timeout_ms': 100,
            'circuit_breaker_threshold': 3,
            'module_paths': [
                'modules/auditing',
                'modules/market',
                'modules/memory',
                'modules/strategy',
                'modules/risk',
                'modules/voting',
                'modules/monitoring',
                'modules/example'
            ]
        }
        
        # Try to load from file
        config_path = Path('config/orchestration_policy.yaml')
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    default_config.update(loaded_config.get('execution', {}))
            except Exception as e:
                self.logger.error(f"Failed to load config: {e}")
        
        return default_config
    
    def _load_policies(self):
        """Load additional policy configurations"""
        # Load module-specific policies
        policy_files = [
            'config/risk_policy.yaml',
            'config/explainability_standards.yaml',
            'config/module_registry.yaml'
        ]
        
        for policy_file in policy_files:
            path = Path(policy_file)
            if path.exists():
                try:
                    with open(path, 'r') as f:
                        policy = yaml.safe_load(f)
                        # Apply policies as needed
                        self._apply_policy(policy_file, policy)
                except Exception as e:
                    self.logger.error(f"Failed to load {policy_file}: {e}")
    
    def _apply_policy(self, policy_name: str, policy: Dict[str, Any]):
        """Apply a specific policy configuration"""
        # Implementation depends on policy type
        if 'risk_policy' in policy_name:
            # Apply risk thresholds
            pass
        elif 'explainability_standards' in policy_name:
            # Apply thesis requirements
            pass
        elif 'module_registry' in policy_name:
            # Validate registered modules
            pass
    
    def _validate_system_integrity(self):
        """Validate system configuration and dependencies"""
        issues = []
        
        # Check for missing dependencies
        for module_name, metadata in self.metadata.items():
            for required in metadata.requires:
                providers = self.smart_bus.get_providers(required)
                if not providers:
                    issues.append(
                        f"Module {module_name} requires '{required}' "
                        f"but no provider found"
                    )
        
        # Check for circular dependencies
        circular = self.smart_bus.find_circular_dependencies()
        if circular:
            issues.append(f"Found {len(circular)} circular dependencies")
        
        # Check voting members
        if not self.voting_members:
            issues.append("No voting members registered")
        
        # Log issues
        if issues:
            self.logger.warning(
                "SYSTEM INTEGRITY ISSUES:\n" + "\n".join(issues)
            )
    
    # ─────────────────────────────────────────────────────────────
    # Hot Reload Support
    # ─────────────────────────────────────────────────────────────
    
    def reload_module(self, module_name: str) -> bool:
        """Hot-reload a module preserving state"""
        if module_name not in self.modules:
            self.logger.error(f"Module {module_name} not found")
            return False
        
        try:
            # Save state
            state = self.state_manager.save_module_state(self.modules[module_name])
            
            # Reload
            success = self.state_manager.reload_module(module_name, self)
            
            if success:
                self.logger.info(
                    format_operator_message(
                        "🔄", "MODULE RELOADED",
                        instrument=module_name,
                        context="hot_reload"
                    )
                )
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to reload {module_name}: {e}")
            return False
    
    # ─────────────────────────────────────────────────────────────
    # Analysis and Monitoring
    # ─────────────────────────────────────────────────────────────
    
    def get_module_by_name(self, name: str) -> Optional[BaseModule]:
        """Get module instance by name"""
        return self.modules.get(name)
    
    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get execution performance metrics"""
        if not self.execution_history:
            return {}
        
        recent = list(self.execution_history)[-100:]
        
        return {
            'avg_execution_time_ms': np.mean([r['execution_time_ms'] for r in recent]),
            'max_execution_time_ms': max(r['execution_time_ms'] for r in recent),
            'avg_success_rate': np.mean([
                r['success_count'] / max(r['module_count'], 1) 
                for r in recent
            ]),
            'total_executions': len(self.execution_history),
            'stage_performance': {
                stage: {
                    'avg_ms': np.mean(timings) if timings else 0,
                    'max_ms': max(timings) if timings else 0
                }
                for stage, timings in self.stage_timings.items()
            }
        }
    
    def get_module_status_report(self) -> str:
        """Get plain English status report"""
        metrics = self.get_execution_metrics()
        
        report = f"""
ORCHESTRATOR STATUS REPORT
=========================
Active Modules: {len(self.modules)}
Voting Members: {len(self.voting_members)}
Execution Stages: {len(self.execution_stages)}

PERFORMANCE:
Average Execution: {metrics.get('avg_execution_time_ms', 0):.0f}ms
Success Rate: {metrics.get('avg_success_rate', 0):.1%}
Total Executions: {metrics.get('total_executions', 0)}

MODULE HEALTH:
"""
        
        # Add module health
        for module_name in sorted(self.modules.keys()):
            health = self.smart_bus.get_module_health(module_name)
            status = "✅" if health['enabled'] else "❌"
            report += f"  {status} {module_name}"
            if health['failures'] > 0:
                report += f" ({health['failures']} failures)"
            report += "\n"
        
        return report
    
    def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Shutting down orchestrator")
        
        # Save state
        for module_name, module in self.modules.items():
            if hasattr(module, 'get_state'):
                state = module.get_state()
                # Could persist to disk here
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.logger.info("Orchestrator shutdown complete")