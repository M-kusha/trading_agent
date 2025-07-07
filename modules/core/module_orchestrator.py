# ─────────────────────────────────────────────────────────────
# File: modules/core/module_orchestrator.py
# 🚀 Module Orchestrator - Zero-wiring automatic discovery & execution
# ─────────────────────────────────────────────────────────────

import asyncio
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Set, Type, Optional, Any, Tuple
import traceback
import yaml
from collections import defaultdict, deque
import numpy as np
import time
import logging

from modules.utils.info_bus import SmartInfoBus, InfoBus, InfoBusManager
from modules.utils.audit_utils import format_operator_message, RotatingLogger
from modules.core.core import BaseModule, ModuleMetadata


class ModuleOrchestrator:
    """
    🎯 Central orchestrator for zero-wiring module management.
    Automatically discovers, manages, and executes all decorated modules.
    """
    
    _instance = None
    _registered_classes: Dict[str, Type[BaseModule]] = {}
    
    def __init__(self, info_bus: SmartInfoBus):
        self.info_bus = info_bus
        self.modules: Dict[str, BaseModule] = {}
        self.metadata: Dict[str, ModuleMetadata] = {}
        self.execution_order: List[str] = []
        self.voting_members: List[str] = []
        
        # Execution stages for parallel processing
        self.execution_stages: List[List[str]] = []
        
        # Performance tracking
        self.stage_timings = defaultdict(list)
        self.module_dependencies = defaultdict(set)
        
        # Policy configuration
        self.policy = self._load_orchestration_policy()
        
        # Setup logging
        self.logger = RotatingLogger(
            name="ModuleOrchestrator",
            log_path="logs/orchestrator/orchestrator.log",
            max_lines=2000,
            operator_mode=True,
            plain_english=True
        )
        
        # Auto-discover on init
        self.discover_all_modules()
        
    @classmethod
    def register_class(cls, module_class: Type[BaseModule]):
        """Called by @module decorator to register classes"""
        cls._registered_classes[module_class.__name__] = module_class
        
    def _load_orchestration_policy(self) -> Dict[str, Any]:
        """Load orchestration policy from config"""
        policy_path = Path("config/orchestration_policy.yaml")
        
        if policy_path.exists():
            with open(policy_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default policy
            return {
                'execution': {
                    'parallel_stages': [
                        {'name': 'data_preparation', 'max_parallel': 5},
                        {'name': 'analysis', 'max_parallel': 10},
                        {'name': 'voting', 'max_parallel': 8},
                        {'name': 'risk_check', 'max_parallel': 5}
                    ],
                    'timeouts': {'default_ms': 100},
                    'circuit_breakers': {
                        'failure_threshold': 3,
                        'recovery_time_s': 60
                    }
                },
                'monitoring': {
                    'performance_tracking': True,
                    'latency_alerts_ms': 150,
                    'stale_data_warning_s': 2
                },
                'hot_reload': {
                    'enabled': True,
                    'preserve_state': True
                }
            }
    
    def discover_all_modules(self):
        """Auto-discover all decorated modules in the project"""
        self.logger.info(
            format_operator_message(
                "🔍", "MODULE DISCOVERY",
                details="Scanning for @module decorated classes",
                context="initialization"
            )
        )
        
        # Module directories to scan
        module_dirs = [
            'modules/auditing', 'modules/market', 'modules/memory',
            'modules/strategy', 'modules/risk', 'modules/voting',
            'modules/features', 'modules/position', 'modules/reward',
            'modules/simulation', 'modules/meta', 'modules/trading_modes'
        ]
        
        discovered_count = 0
        
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
                    discovered_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to import {module_name}: {e}")
        
        # Register discovered modules
        for name, cls in self._registered_classes.items():
            self.register_module(name, cls)
            
        # Build execution order
        self.build_execution_order()
        
        # Build execution stages
        self.build_execution_stages()
        
        self.logger.info(
            format_operator_message(
                "✅", "DISCOVERY COMPLETE",
                details=f"Found {len(self.modules)} modules",
                result=f"{len(self.voting_members)} voting members",
                context="initialization"
            )
        )
    
    def register_module(self, name: str, module_class: Type[BaseModule]):
        """Register a module with the orchestrator"""
        try:
            # Get metadata
            metadata = module_class.__module_metadata__
            
            # Create instance
            instance = module_class()
            self.modules[name] = instance
            self.metadata[name] = metadata
            
            # Register with InfoBus
            self.info_bus.register_provider(name, metadata.provides)
            self.info_bus.register_consumer(name, metadata.requires)
            
            # Track dependencies
            for req in metadata.requires:
                providers = self._find_providers(req)
                for provider in providers:
                    self.module_dependencies[name].add(provider)
            
            # Track voting members
            if metadata.is_voting_member:
                self.voting_members.append(name)
                
            self.logger.debug(
                f"✅ Registered: {name} ({metadata.category}) "
                f"v{metadata.version}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to register {name}: {e}")
    
    def _find_providers(self, requirement: str) -> List[str]:
        """Find modules that provide a requirement"""
        providers = []
        for name, metadata in self.metadata.items():
            if requirement in metadata.provides:
                providers.append(name)
        return providers
    
    def build_execution_order(self):
        """Build topological execution order based on dependencies"""
        # Build dependency graph
        graph = defaultdict(set)
        in_degree = defaultdict(int)
        all_modules = set(self.modules.keys())
        
        # Add edges based on dependencies
        for module, deps in self.module_dependencies.items():
            for dep in deps:
                if dep in self.modules:
                    graph[dep].add(module)
                    in_degree[module] += 1
        
        # Add nodes with no dependencies
        for module in all_modules:
            if module not in in_degree:
                in_degree[module] = 0
        
        # Topological sort using Kahn's algorithm
        queue = deque([m for m in all_modules if in_degree[m] == 0])
        order = []
        
        while queue:
            # Sort by priority for deterministic order
            current_batch = sorted(queue, key=lambda m: self.metadata[m].priority, reverse=True)
            queue.clear()
            
            for module in current_batch:
                order.append(module)
                
                # Reduce in-degree for dependent modules
                for dependent in graph[module]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)
        
        # Check for cycles
        if len(order) != len(all_modules):
            missing = all_modules - set(order)
            self.logger.error(f"Circular dependency detected! Missing: {missing}")
            # Add remaining modules anyway
            order.extend(missing)
        
        self.execution_order = order
        
        self.logger.info(
            format_operator_message(
                "📋", "EXECUTION ORDER",
                details=f"{len(order)} modules",
                result=f"First 5: {' → '.join(order[:5])}...",
                context="dependency_analysis"
            )
        )
    
    def build_execution_stages(self):
        """Build parallel execution stages based on policy"""
        stages_config = self.policy['execution']['parallel_stages']
        self.execution_stages = []
        
        remaining_modules = set(self.execution_order)
        
        for stage_config in stages_config:
            stage_name = stage_config['name']
            stage_modules = []
            
            # Find modules matching this stage
            for module in list(remaining_modules):
                metadata = self.metadata[module]
                
                # Match by category or explicit list
                if stage_name == 'voting' and metadata.is_voting_member:
                    stage_modules.append(module)
                    remaining_modules.remove(module)
                elif f"category:{metadata.category}" in stage_config.get('modules', []):
                    stage_modules.append(module)
                    remaining_modules.remove(module)
                elif module in stage_config.get('modules', []):
                    stage_modules.append(module)
                    remaining_modules.remove(module)
                elif metadata.category == stage_name:
                    stage_modules.append(module)
                    remaining_modules.remove(module)
            
            if stage_modules:
                self.execution_stages.append(stage_modules)
                self.logger.debug(f"Stage '{stage_name}': {len(stage_modules)} modules")
        
        # Add remaining modules as final stage
        if remaining_modules:
            self.execution_stages.append(list(remaining_modules))
            self.logger.debug(f"Final stage: {len(remaining_modules)} modules")
    
    async def execute_step(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all modules in optimized stages"""
        step_start = time.perf_counter()
        results = {}
        
        # Store market data in InfoBus
        for key, value in market_data.items():
            self.info_bus.set(key, value, "Environment")
        
        # Execute stages
        for stage_idx, stage_modules in enumerate(self.execution_stages):
            stage_start = time.perf_counter()
            
            # Execute stage in parallel
            stage_results = await self._execute_stage_parallel(
                stage_modules, 
                stage_idx
            )
            
            results.update(stage_results)
            
            # Track stage timing
            stage_time = (time.perf_counter() - stage_start) * 1000
            self.stage_timings[f"stage_{stage_idx}"].append(stage_time)
            
            if stage_time > 200:  # 200ms warning threshold
                self.logger.warning(
                    format_operator_message(
                        "⚠️", "SLOW STAGE",
                        details=f"Stage {stage_idx} took {stage_time:.0f}ms",
                        context="performance"
                    )
                )
        
        # Record total execution time
        total_time = (time.perf_counter() - step_start) * 1000
        self.info_bus.record_module_timing("Orchestrator", total_time)
        
        # Generate execution report
        if total_time > 500:  # 500ms threshold for detailed report
            report = self._generate_execution_report(results, total_time)
            self.logger.info(report)
        
        return results
    
    async def _execute_stage_parallel(self, modules: List[str], 
                                    stage_idx: int) -> Dict[str, Any]:
        """Execute modules in a stage with parallel support"""
        # Get max parallel from policy
        max_parallel = 10  # Default
        for stage_config in self.policy['execution']['parallel_stages']:
            if stage_config.get('index') == stage_idx:
                max_parallel = stage_config.get('max_parallel', 10)
                break
        
        # Create tasks for parallel execution
        tasks = []
        for module_name in modules:
            if not self.info_bus.is_module_enabled(module_name):
                continue
                
            task = asyncio.create_task(
                self._execute_module_async(module_name)
            )
            tasks.append((module_name, task))
        
        # Execute with concurrency limit
        results = {}
        for i in range(0, len(tasks), max_parallel):
            batch = tasks[i:i + max_parallel]
            
            # Wait for batch to complete
            for module_name, task in batch:
                try:
                    result = await task
                    if result:
                        results[module_name] = result
                except Exception as e:
                    self.handle_module_failure(module_name, e)
        
        return results
    
    async def _execute_module_async(self, module_name: str) -> Optional[Dict[str, Any]]:
        """Execute a single module asynchronously"""
        module = self.modules[module_name]
        metadata = self.metadata[module_name]
        
        # Collect inputs based on requirements
        inputs = {}
        for req in metadata.requires:
            value = self.info_bus.get(req, module_name)
            if value is not None:
                inputs[req] = value
        
        # Add info_bus reference
        inputs['info_bus'] = InfoBusManager.get_current()
        
        # Validate inputs
        try:
            module.validate_inputs(inputs)
        except ValueError as e:
            self.logger.warning(
                f"Module {module_name} missing inputs: {e}"
            )
            return None
        
        # Execute with timeout
        timeout = metadata.timeout_ms / 1000.0  # Convert to seconds
        
        try:
            # Call process method if available, otherwise fall back to step
            if hasattr(module, 'process') and inspect.iscoroutinefunction(module.process):
                result = await asyncio.wait_for(
                    module.process(**inputs),
                    timeout=timeout
                )
            else:
                # Synchronous fallback
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        self._execute_module_sync,
                        module_name,
                        inputs
                    ),
                    timeout=timeout
                )
            
            # Store outputs in InfoBus
            if isinstance(result, dict):
                for key, value in result.items():
                    if key in metadata.provides:
                        thesis = result.get(f'{key}_thesis', '')
                        self.info_bus.set(
                            key, value, module_name,
                            thesis=thesis,
                            confidence=result.get('confidence', 0.8)
                        )
            
            return result
            
        except asyncio.TimeoutError:
            self.logger.error(
                format_operator_message(
                    "⏱️", "MODULE TIMEOUT",
                    instrument=module_name,
                    details=f"Exceeded {timeout*1000:.0f}ms",
                    context="execution"
                )
            )
            self.info_bus.record_module_failure(module_name, "Timeout")
            return None
            
        except Exception as e:
            self.handle_module_failure(module_name, e)
            return None
    
    def _execute_module_sync(self, module_name: str, 
                           inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute module synchronously (for legacy modules)"""
        module = self.modules[module_name]
        
        # Legacy modules use step method
        if 'info_bus' in inputs and hasattr(module, 'step'):
            module.step(inputs['info_bus'])
            return {}
        
        return {}
    
    def handle_module_failure(self, module_name: str, error: Exception):
        """Handle module execution failure"""
        error_msg = str(error)
        tb = traceback.format_exc()
        
        self.logger.error(
            format_operator_message(
                "💥", "MODULE FAILURE",
                instrument=module_name,
                details=error_msg,
                context="execution"
            )
        )
        
        # Record in InfoBus
        self.info_bus.record_module_failure(module_name, error_msg)
        
        # Store error details for debugging
        self.info_bus.set(
            f'error_{module_name}',
            {
                'error': error_msg,
                'traceback': tb,
                'timestamp': time.time()
            },
            module='Orchestrator',
            thesis=f"Module {module_name} failed during execution"
        )
    
    def _generate_execution_report(self, results: Dict[str, Any], 
                                 total_time: float) -> str:
        """Generate execution performance report"""
        lines = [
            "=" * 60,
            "ORCHESTRATOR EXECUTION REPORT",
            "=" * 60,
            f"Total execution time: {total_time:.1f}ms",
            f"Modules executed: {len(results)}",
            ""
        ]
        
        # Stage timings
        lines.append("Stage Timings:")
        for stage_idx, timings in enumerate(self.stage_timings.items()):
            stage_name, times = timings
            if times:
                avg_time = np.mean(times[-10:])  # Last 10 executions
                lines.append(f"  {stage_name}: {avg_time:.1f}ms avg")
        
        # Slow modules
        slow_modules = []
        for module_name in self.modules:
            timings = list(self.info_bus._latency_history.get(module_name, []))
            if timings and np.mean(timings[-10:]) > 100:
                slow_modules.append(f"{module_name} ({np.mean(timings[-10:]):.0f}ms)")
        
        if slow_modules:
            lines.extend([
                "",
                "⚠️ Slow Modules:",
                *[f"  - {m}" for m in slow_modules]
            ])
        
        # Failed modules
        if self.info_bus._module_disabled:
            lines.extend([
                "",
                "❌ Disabled Modules:",
                *[f"  - {m}" for m in self.info_bus._module_disabled]
            ])
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """Get module dependency graph for visualization"""
        graph = {}
        
        for module, deps in self.module_dependencies.items():
            graph[module] = list(deps)
        
        return graph
    
    def get_module_by_name(self, name: str) -> Optional[BaseModule]:
        """Get module instance by name"""
        return self.modules.get(name)
    
    def get_voting_modules(self) -> List[BaseModule]:
        """Get all voting member modules"""
        return [self.modules[name] for name in self.voting_members 
                if name in self.modules]
    
    def enable_module(self, module_name: str):
        """Enable a previously disabled module"""
        if module_name in self.modules:
            self.info_bus.reset_module_failures(module_name)
            self.logger.info(f"Module {module_name} enabled")
    
    def disable_module(self, module_name: str):
        """Manually disable a module"""
        if module_name in self.modules:
            self.info_bus._module_disabled.add(module_name)
            self.logger.warning(f"Module {module_name} manually disabled")
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary for monitoring"""
        return {
            'total_modules': len(self.modules),
            'active_modules': len([m for m in self.modules 
                                  if self.info_bus.is_module_enabled(m)]),
            'disabled_modules': list(self.info_bus._module_disabled),
            'voting_members': len(self.voting_members),
            'execution_stages': len(self.execution_stages),
            'average_stage_times': {
                stage: np.mean(times[-10:]) if times else 0
                for stage, times in self.stage_timings.items()
            }
        }