#!/usr/bin/env python3
"""
Robust script to analyze circular dependencies in the SmartInfoBus system.
"""

from modules.utils.info_bus import InfoBusManager
import json
from pathlib import Path
import traceback

def main():
    print("Analyzing circular dependencies...")
    
    # Get the SmartInfoBus instance
    try:
        smart_bus = InfoBusManager.get_instance()
        print("SmartInfoBus instance obtained successfully.")
    except Exception as e:
        print(f"Failed to get SmartInfoBus instance: {e}")
        return
    
    try:
        # Check if the method exists
        if not hasattr(smart_bus, 'find_circular_dependencies'):
            print("SmartInfoBus does not have find_circular_dependencies method.")
            # Fallback: try to get dependency graph and analyze manually
            _analyze_dependency_graph(smart_bus)
            return
            
        # Find circular dependencies
        circular_deps = smart_bus.find_circular_dependencies()
        
        if circular_deps:
            print(f"Found {len(circular_deps)} circular dependencies:")
            for i, dep_cycle in enumerate(circular_deps, 1):
                print(f"Cycle {i}: {dep_cycle}")
        else:
            print("No circular dependencies found.")
            
        # Save detailed report
        output_path = "logs/circular_dependencies.json"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'circular_dependencies': circular_deps,
                'total_cycles': len(circular_deps)
            }, f, indent=2)
            
        print(f"Circular dependencies report saved to: {output_path}")
        
    except Exception as e:
        print(f"Error analyzing circular dependencies: {e}")
        print(traceback.format_exc())
        # Fallback: try to get dependency graph and analyze manually
        _analyze_dependency_graph(smart_bus)

def _analyze_dependency_graph(smart_bus):
    """Fallback method to analyze dependency graph manually."""
    try:
        print("Attempting to analyze dependency graph manually...")
        
        # Get providers and consumers maps
        providers_map = {}
        consumers_map = {}
        
        if hasattr(smart_bus, '_providers'):
            providers_map = {k: list(v) for k, v in smart_bus._providers.items()}
        if hasattr(smart_bus, '_consumers'):
            consumers_map = {k: list(v) for k, v in smart_bus._consumers.items()}
        
        # Save dependency graph
        graph_path = "logs/dependency_graph.json"
        Path(graph_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(graph_path, 'w', encoding='utf-8') as f:
            json.dump({
                'providers': providers_map,
                'consumers': consumers_map
            }, f, indent=2)
            
        print(f"Dependency graph saved to: {graph_path}")
        
        # Basic analysis
        print(f"Total provider keys: {len(providers_map)}")
        print(f"Total consumer keys: {len(consumers_map)}")
        
        # Check for potential circular dependencies
        circular_candidates = []
        for key, providers in providers_map.items():
            if key in consumers_map and any(prov in consumers_map.get(key, []) for prov in providers):
                circular_candidates.append(key)
        
        if circular_candidates:
            print(f"Potential circular dependency candidates: {circular_candidates}")
        else:
            print("No obvious circular dependency candidates found.")
            
    except Exception as e:
        print(f"Error in fallback analysis: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
