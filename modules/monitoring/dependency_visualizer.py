# ─────────────────────────────────────────────────────────────
# File: modules/monitoring/dependency_visualizer.py
# [ROCKET] DEPENDENCY VISUALIZATION SYSTEM
# NASA/MILITARY GRADE - ZERO ERROR TOLERANCE
# ─────────────────────────────────────────────────────────────

import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import time

from modules.core.module_system import ModuleOrchestrator
from modules.utils.audit_utils import RotatingLogger, format_operator_message

class DependencyVisualizer:
    """
    Visualizes module dependencies and execution flow.
    Provides performance optimization recommendations.
    """
    
    def __init__(self, orchestrator: Optional[ModuleOrchestrator] = None):
        self.orchestrator = orchestrator or ModuleOrchestrator.get_instance()
        self.graph = nx.DiGraph()
        self.performance_data = {}
        
        # Setup logging
        self.logger = RotatingLogger(
            name="DependencyVisualizer",
            log_path="logs/visualization/dependencies.log",
            max_lines=2000,
            operator_mode=True
        )
        
        # Build initial graph
        self._build_dependency_graph()
    
    def _build_dependency_graph(self):
        """Build dependency graph from orchestrator"""
        try:
            # Clear existing graph
            self.graph.clear()
            
            # Add modules as nodes
            for module_name in self.orchestrator.modules:
                self.graph.add_node(module_name)
            
            # Add dependencies as edges
            for module_name, metadata in self.orchestrator.metadata.items():
                for required in metadata.requires:
                    providers = self.orchestrator.smart_bus.get_providers(required)
                    for provider in providers:
                        if provider in self.orchestrator.modules:
                            self.graph.add_edge(provider, module_name)
            
            self.logger.info(
                format_operator_message(
                    "[STATS]", "DEPENDENCY GRAPH BUILT",
                    details=f"{len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges",
                    context="visualization"
                )
            )
            
        except Exception as e:
            self.logger.error(f"Failed to build dependency graph: {e}")
    
    def update_performance_metrics(self, module_name: str, metrics: Dict[str, float]):
        """Update performance metrics for a module"""
        self.performance_data[module_name] = {
            'avg_latency_ms': metrics.get('avg_latency_ms', 0),
            'error_rate': metrics.get('error_rate', 0),
            'success_rate': metrics.get('success_rate', 1.0),
            'last_updated': time.time()
        }
    
    def get_graph_data(self) -> Dict[str, Any]:
        """Get graph data for visualization"""
        return {
            'nodes': list(self.graph.nodes()),
            'edges': list(self.graph.edges()),
            'performance': self.performance_data,
            'circular_dependencies': list(nx.simple_cycles(self.graph)),
            'isolated_modules': list(nx.isolates(self.graph)),
            'critical_path': self._find_critical_path()
        }
    
    def _find_critical_path(self) -> List[str]:
        """Find critical execution path"""
        try:
            # Use topological sort to find critical path
            sorted_nodes = list(nx.topological_sort(self.graph))
            critical_path = []
            
            for node in sorted_nodes:
                metadata = self.orchestrator.metadata.get(node)
                if metadata and hasattr(metadata, 'critical') and metadata.critical:
                    critical_path.append(node)
            
            return critical_path
        except nx.NetworkXError:
            # Graph has cycles
            return []
    
    def optimize_execution_stages(self) -> List[List[str]]:
        """Optimize execution stages for parallel processing"""
        try:
            # Find strongly connected components
            components = list(nx.strongly_connected_components(self.graph))
            
            # Create execution stages
            stages = []
            processed = set()
            
            for component in components:
                if len(component) == 1:
                    # Single module - can be parallel
                    module = list(component)[0]
                    if module not in processed:
                        stages.append([module])
                        processed.add(module)
                else:
                    # Multiple modules - need sequential execution
                    stage = list(component)
                    stages.append(stage)
                    processed.update(stage)
            
            self.logger.info(
                format_operator_message(
                    "[FAST]", "EXECUTION STAGES OPTIMIZED",
                    details=f"{len(stages)} stages created",
                    context="optimization"
                )
            )
            
            return stages
            
        except Exception as e:
            self.logger.error(f"Failed to optimize execution stages: {e}")
            return []
    
    def generate_visualization(self, output_path: str = "dependency_graph.png"):
        """Generate dependency graph visualization"""
        try:
            plt.figure(figsize=(12, 8))
            
            # Position nodes using spring layout
            pos = nx.spring_layout(self.graph, k=1, iterations=50)
            
            # Draw nodes
            nx.draw_networkx_nodes(self.graph, pos, 
                                 node_color='lightblue',
                                 node_size=1000)
            
            # Draw edges
            nx.draw_networkx_edges(self.graph, pos, 
                                 edge_color='gray',
                                 arrows=True,
                                 arrowsize=20)
            
            # Add labels
            nx.draw_networkx_labels(self.graph, pos)
            
            # Add title
            plt.title("Module Dependency Graph", fontsize=16)
            plt.axis('off')
            
            # Save
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(
                format_operator_message(
                    "[CHART]", "VISUALIZATION GENERATED",
                    details=output_path,
                    context="visualization"
                )
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate visualization: {e}")
    
    def analyze_bottlenecks(self) -> List[Dict[str, Any]]:
        """Analyze performance bottlenecks"""
        bottlenecks = []
        
        for module_name, metrics in self.performance_data.items():
            if metrics['avg_latency_ms'] > 200:  # 200ms threshold
                bottlenecks.append({
                    'module': module_name,
                    'issue': 'high_latency',
                    'value': metrics['avg_latency_ms'],
                    'threshold': 200,
                    'recommendation': 'Optimize processing logic or increase timeout'
                })
            
            if metrics['error_rate'] > 0.1:  # 10% error rate threshold
                bottlenecks.append({
                    'module': module_name,
                    'issue': 'high_error_rate',
                    'value': metrics['error_rate'],
                    'threshold': 0.1,
                    'recommendation': 'Investigate error causes and add error handling'
                })
        
        return bottlenecks
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations"""
        recommendations = []
        
        # Check for circular dependencies
        cycles = list(nx.simple_cycles(self.graph))
        if cycles:
            recommendations.append(f"Found {len(cycles)} circular dependencies - consider refactoring")
        
        # Check for isolated modules
        isolated = list(nx.isolates(self.graph))
        if isolated:
            recommendations.append(f"Found {len(isolated)} isolated modules - check if they're needed")
        
        # Check for bottlenecks
        bottlenecks = self.analyze_bottlenecks()
        for bottleneck in bottlenecks:
            recommendations.append(f"{bottleneck['module']}: {bottleneck['recommendation']}")
        
        return recommendations