#!/usr/bin/env python3
"""
Script to analyze circular dependencies in the SmartInfoBus system.
"""

from modules.utils.info_bus import InfoBusManager
import json
from pathlib import Path

def main():
    print("Analyzing circular dependencies...")
    
    # Get the SmartInfoBus instance
    smart_bus = InfoBusManager.get_instance()
    
    try:
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
        # Fallback: try to get dependency graph and analyze manually
        try:
            providers_map = {k: list(v) for k, v in smart_bus._providers.items()}
            consumers_map = {k: list(v) for k, v in smart_bus._consumers.items()}
            
            graph_path = "logs/dependency_graph.json"
            with open(graph_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'providers': providers_map,
                    'consumers': consumers_map
                }, f, indent=2)
                
            print(f"Dependency graph saved to: {graph_path}")
            
        except Exception as graph_error:
            print(f"Could not save dependency graph: {graph_error}")

if __name__ == "__main__":
    main()
