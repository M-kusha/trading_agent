#!/usr/bin/env python3
"""
Diagnostic script to check module registration status in the SmartInfoBus system.
This will help identify why modules aren't being properly registered with the InfoBus.
"""

import sys
import os
import importlib
import inspect
from typing import Dict, List, Set, Type, Any
from collections import defaultdict

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from modules.core.module_base import BaseModule, module
    from modules.core.module_system import ModuleOrchestrator
    from modules.utils.info_bus import InfoBusManager, SmartInfoBus
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)

def get_module_paths() -> List[str]:
    """Get all module paths from the configuration"""
    try:
        from modules.core.module_system import ModuleConfig
        config = ModuleConfig()
        return config.module_paths
    except Exception as e:
        print(f"Error getting module paths: {e}")
        return [
            'modules/auditing',
            'modules/external', 
            'modules/features',
            'modules/market',
            'modules/memory',
            'modules/meta',
            'modules/models',
            'modules/monitoring',
            'modules/position',
            'modules/reward',
            'modules/risk',
            'modules/simulation',
            'modules/strategy',
            'modules/trading_modes',
            'modules/utils',
            'modules/visualization',
            'modules/voting'
        ]

def discover_modules() -> Dict[str, Type[BaseModule]]:
    """Discover all modules in the system"""
    discovered: Dict[str, Type[BaseModule]] = {}
    module_paths = get_module_paths()
    
    for path_str in module_paths:
        path = os.path.join(os.path.dirname(__file__), path_str.replace('/', os.sep))
        if not os.path.exists(path):
            print(f"Module path does not exist: {path}")
            continue
            
        for py_file in os.listdir(path):
            if not py_file.endswith('.py') or py_file.startswith('_'):
                continue
                
            module_name = f"{path_str.replace('/', '.')}.{py_file[:-3]}"
            try:
                mod = importlib.import_module(module_name)
                for name, obj in inspect.getmembers(mod):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, BaseModule) and 
                        obj is not BaseModule and 
                        hasattr(obj, '__module_metadata__')):
                        discovered[name] = obj
                        print(f"Discovered module: {name}")
            except ImportError as e:
                print(f"Failed to import {module_name}: {e}")
            except Exception as e:
                print(f"Error discovering modules in {py_file}: {e}")
    
    return discovered

def check_info_bus_registration():
    """Check what's registered with InfoBusManager"""
    try:
        bus = InfoBusManager.get_instance()
        print("\n=== InfoBus Registration Status ===")
        
        # Check providers
        providers = getattr(bus, '_providers', {})
        print(f"Total providers: {len(providers)}")
        for key, provider_set in providers.items():
            print(f"  {key}: {list(provider_set)}")
            
        # Check consumers  
        consumers = getattr(bus, '_consumers', {})
        print(f"Total consumers: {len(consumers)}")
        for key, consumer_set in consumers.items():
            print(f"  {key}: {list(consumer_set)}")
            
    except Exception as e:
        print(f"Error checking InfoBus registration: {e}")

def check_module_orchestrator():
    """Check ModuleOrchestrator state"""
    try:
        orchestrator = ModuleOrchestrator.get_instance()
        print("\n=== ModuleOrchestrator Status ===")
        
        print(f"Total modules: {len(orchestrator.modules)}")
        print(f"Modules: {list(orchestrator.modules.keys())}")
        
        print(f"Total metadata: {len(orchestrator.metadata)}")
        print(f"Total circuit breakers: {len(orchestrator.circuit_breakers)}")
        
        # Check execution plan
        print(f"Execution stages: {len(orchestrator.execution_stages)} stages")
        for i, stage in enumerate(orchestrator.execution_stages):
            print(f"  Stage {i}: {len(stage)} modules - {stage}")
            
    except Exception as e:
        print(f"Error checking ModuleOrchestrator: {e}")

def check_individual_module_registration():
    """Check registration of specific problematic modules"""
    problematic_modules = [
        'PPOAgent', 'MarketAwarePPONetwork', 'EnhancedPPONetwork',
        'DependencyInspector', 'IntegrationValidator', 'HealthMonitor'
    ]
    
    print("\n=== Individual Module Registration Check ===")
    
    try:
        bus = InfoBusManager.get_instance()
        orchestrator = ModuleOrchestrator.get_instance()
        
        for module_name in problematic_modules:
            print(f"\nChecking {module_name}:")
            
            # Check if in orchestrator
            in_orchestrator = module_name in orchestrator.modules
            print(f"  In ModuleOrchestrator: {in_orchestrator}")
            
            if in_orchestrator:
                module_obj = orchestrator.modules[module_name]
                metadata = orchestrator.metadata[module_name]
                print(f"  Provides: {metadata.provides}")
                print(f"  Requires: {metadata.requires}")
                
                # Check InfoBus registration for each provided key
                for key in metadata.provides:
                    providers = bus.get_providers(key)
                    print(f"    Key '{key}' providers: {list(providers)}")
                    
    except Exception as e:
        print(f"Error checking individual modules: {e}")

def main():
    print("SmartInfoBus Module Registration Diagnostic")
    print("=" * 50)
    
    # Discover all modules
    discovered_modules = discover_modules()
    print(f"\nTotal discovered modules: {len(discovered_modules)}")
    
    # Check InfoBus registration
    check_info_bus_registration()
    
    # Check ModuleOrchestrator
    check_module_orchestrator()
    
    # Check specific modules
    check_individual_module_registration()
    
    print("\n=== Recommendations ===")
    print("1. Ensure all modules have proper @module decorators")
    print("2. Check that ModuleOrchestrator.initialize() is called")
    print("3. Verify module discovery paths are correct")
    print("4. Look for exceptions during module registration")
    print("5. Check circular dependencies in the dependency graph")

if __name__ == "__main__":
    main()
