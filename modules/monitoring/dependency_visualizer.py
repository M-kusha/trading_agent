# ─────────────────────────────────────────────────────────────
# File: modules/monitoring/dependency_visualizer.py
# 🚀 Dependency graph visualization for SmartInfoBus
# ─────────────────────────────────────────────────────────────

import json
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import graphviz

from modules.utils.info_bus import SmartInfoBus, InfoBusManager
from modules.core.module_orchestrator import ModuleOrchestrator


class DependencyVisualizer:
    """
    Visualizes module dependencies and data flow in SmartInfoBus.
    Helps identify circular dependencies and optimization opportunities.
    """
    
    def __init__(self, orchestrator: ModuleOrchestrator):
        self.orchestrator = orchestrator
        self.smart_bus = InfoBusManager.get_instance()
        
        # Graph representations
        self.module_graph = None
        self.data_flow_graph = None
        
        # Analysis results
        self.circular_dependencies = []
        self.isolated_modules = []
        self.critical_paths = []
        
        # Build graphs
        self._build_graphs()
    
    def _build_graphs(self):
        """Build dependency and data flow graphs"""
        # Module dependency graph
        self.module_graph = nx.DiGraph()
        
        # Add nodes
        for module_name, metadata in self.orchestrator.metadata.items():
            self.module_graph.add_node(
                module_name,
                category=metadata.category,
                is_voting=metadata.is_voting_member,
                priority=metadata.priority,
                provides=metadata.provides,
                requires=metadata.requires
            )
        
        # Add edges based on dependencies
        for module, deps in self.orchestrator.module_dependencies.items():
            for dep in deps:
                if dep in self.module_graph.nodes:
                    self.module_graph.add_edge(dep, module)
        
        # Data flow graph
        self.data_flow_graph = nx.DiGraph()
        
        # Add data nodes and connections
        for module_name, metadata in self.orchestrator.metadata.items():
            # Add module node
            self.data_flow_graph.add_node(
                f"module:{module_name}",
                type='module',
                category=metadata.category
            )
            
            # Add data nodes for provides
            for data_key in metadata.provides:
                self.data_flow_graph.add_node(
                    f"data:{data_key}",
                    type='data'
                )
                self.data_flow_graph.add_edge(
                    f"module:{module_name}",
                    f"data:{data_key}"
                )
            
            # Add edges for requires
            for data_key in metadata.requires:
                self.data_flow_graph.add_node(
                    f"data:{data_key}",
                    type='data'
                )
                self.data_flow_graph.add_edge(
                    f"data:{data_key}",
                    f"module:{module_name}"
                )
    
    def analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze dependency structure"""
        analysis = {
            'total_modules': self.module_graph.number_of_nodes(),
            'total_dependencies': self.module_graph.number_of_edges(),
            'circular_dependencies': self._find_circular_dependencies(),
            'isolated_modules': self._find_isolated_modules(),
            'critical_paths': self._find_critical_paths(),
            'dependency_depth': self._calculate_dependency_depth(),
            'module_importance': self._calculate_module_importance()
        }
        
        return analysis
    
    def _find_circular_dependencies(self) -> List[List[str]]:
        """Find circular dependencies in the graph"""
        cycles = list(nx.simple_cycles(self.module_graph))
        self.circular_dependencies = cycles
        return cycles
    
    def _find_isolated_modules(self) -> List[str]:
        """Find modules with no dependencies"""
        isolated = []
        for node in self.module_graph.nodes():
            if (self.module_graph.in_degree(node) == 0 and 
                self.module_graph.out_degree(node) == 0):
                isolated.append(node)
        
        self.isolated_modules = isolated
        return isolated
    
    def _find_critical_paths(self) -> List[List[str]]:
        """Find critical execution paths"""
        # Find longest paths in DAG (after removing cycles)
        dag = self.module_graph.copy()
        
        # Remove cycles
        for cycle in self.circular_dependencies:
            if len(cycle) > 1:
                dag.remove_edge(cycle[-1], cycle[0])
        
        # Find all paths from sources to sinks
        sources = [n for n in dag.nodes() if dag.in_degree(n) == 0]
        sinks = [n for n in dag.nodes() if dag.out_degree(n) == 0]
        
        critical_paths = []
        for source in sources:
            for sink in sinks:
                try:
                    paths = list(nx.all_simple_paths(dag, source, sink))
                    critical_paths.extend(paths)
                except:
                    pass
        
        # Sort by length (longest first)
        critical_paths.sort(key=len, reverse=True)
        self.critical_paths = critical_paths[:10]  # Top 10
        
        return self.critical_paths
    
    def _calculate_dependency_depth(self) -> Dict[str, int]:
        """Calculate dependency depth for each module"""
        depths = {}
        
        # Use topological sort to calculate depths
        try:
            topo_order = list(nx.topological_sort(self.module_graph))
            
            for node in topo_order:
                predecessors = list(self.module_graph.predecessors(node))
                if not predecessors:
                    depths[node] = 0
                else:
                    depths[node] = max(depths.get(p, 0) for p in predecessors) + 1
        except nx.NetworkXError:
            # Graph has cycles, use approximation
            for node in self.module_graph.nodes():
                depths[node] = len(nx.ancestors(self.module_graph, node))
        
        return depths
    
    def _calculate_module_importance(self) -> Dict[str, float]:
        """Calculate importance score for each module"""
        # Use PageRank algorithm
        try:
            importance = nx.pagerank(self.module_graph)
        except:
            # Fallback to degree centrality
            importance = nx.degree_centrality(self.module_graph)
        
        # Sort by importance
        sorted_importance = dict(sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        return sorted_importance
    
    def visualize_module_graph(self, output_path: str = "module_dependencies.png",
                             highlight_cycles: bool = True):
        """Create visual representation of module dependencies"""
        plt.figure(figsize=(20, 16))
        
        # Layout
        pos = nx.spring_layout(self.module_graph, k=3, iterations=50)
        
        # Color nodes by category
        categories = set(nx.get_node_attributes(self.module_graph, 'category').values())
        colors = plt.cm.tab20(range(len(categories)))
        category_colors = dict(zip(categories, colors))
        
        node_colors = []
        for node in self.module_graph.nodes():
            category = self.module_graph.nodes[node].get('category', 'unknown')
            node_colors.append(category_colors.get(category, 'gray'))
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.module_graph, pos,
            node_color=node_colors,
            node_size=1000,
            alpha=0.8
        )
        
# Draw edges
        edge_colors = []
        edge_widths = []
        
        for edge in self.module_graph.edges():
            # Highlight circular dependencies
            is_circular = False
            if highlight_cycles:
                for cycle in self.circular_dependencies:
                    if edge[0] in cycle and edge[1] in cycle:
                        idx0 = cycle.index(edge[0])
                        idx1 = cycle.index(edge[1])
                        if (idx1 - idx0) % len(cycle) == 1:
                            is_circular = True
                            break
            
            edge_colors.append('red' if is_circular else 'gray')
            edge_widths.append(3 if is_circular else 1)
        
        nx.draw_networkx_edges(
            self.module_graph, pos,
            edge_color=edge_colors,
            width=edge_widths,
            alpha=0.6,
            arrows=True,
            arrowsize=20
        )
        
        # Draw labels
        labels = {}
        for node in self.module_graph.nodes():
            # Shorten long names
            if len(node) > 15:
                labels[node] = node[:12] + "..."
            else:
                labels[node] = node
        
        nx.draw_networkx_labels(
            self.module_graph, pos,
            labels,
            font_size=8
        )
        
        # Add legend
        legend_elements = []
        for category, color in category_colors.items():
            legend_elements.append(plt.Line2D(
                [0], [0], marker='o', color='w',
                markerfacecolor=color, markersize=10,
                label=category
            ))
        
        plt.legend(
            handles=legend_elements,
            loc='upper left',
            bbox_to_anchor=(1, 1)
        )
        
        plt.title("Module Dependency Graph", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_data_flow(self, output_path: str = "data_flow.png"):
        """Visualize data flow between modules"""
        plt.figure(figsize=(24, 18))
        
        # Use hierarchical layout
        pos = nx.spring_layout(self.data_flow_graph, k=4, iterations=100)
        
        # Separate module and data nodes
        module_nodes = [n for n in self.data_flow_graph.nodes() 
                       if self.data_flow_graph.nodes[n].get('type') == 'module']
        data_nodes = [n for n in self.data_flow_graph.nodes() 
                     if self.data_flow_graph.nodes[n].get('type') == 'data']
        
        # Draw module nodes
        nx.draw_networkx_nodes(
            self.data_flow_graph, pos,
            nodelist=module_nodes,
            node_color='lightblue',
            node_shape='s',
            node_size=1500,
            alpha=0.8
        )
        
        # Draw data nodes
        nx.draw_networkx_nodes(
            self.data_flow_graph, pos,
            nodelist=data_nodes,
            node_color='lightgreen',
            node_shape='o',
            node_size=800,
            alpha=0.8
        )
        
        # Draw edges with different styles
        produces_edges = [(u, v) for u, v in self.data_flow_graph.edges() 
                         if u.startswith('module:')]
        consumes_edges = [(u, v) for u, v in self.data_flow_graph.edges() 
                         if u.startswith('data:')]
        
        nx.draw_networkx_edges(
            self.data_flow_graph, pos,
            edgelist=produces_edges,
            edge_color='blue',
            width=2,
            alpha=0.6,
            arrows=True,
            arrowsize=15
        )
        
        nx.draw_networkx_edges(
            self.data_flow_graph, pos,
            edgelist=consumes_edges,
            edge_color='green',
            width=2,
            alpha=0.6,
            arrows=True,
            arrowsize=15,
            style='dashed'
        )
        
        # Simplified labels
        labels = {}
        for node in self.data_flow_graph.nodes():
            if node.startswith('module:'):
                label = node[7:]  # Remove 'module:' prefix
            elif node.startswith('data:'):
                label = node[5:]  # Remove 'data:' prefix
            else:
                label = node
            
            # Truncate long labels
            if len(label) > 20:
                label = label[:17] + "..."
            
            labels[node] = label
        
        nx.draw_networkx_labels(
            self.data_flow_graph, pos,
            labels,
            font_size=7
        )
        
        plt.title("Data Flow Graph", fontsize=16)
        plt.axis('off')
        
        # Add legend
        plt.plot([], [], 's', color='lightblue', markersize=10, label='Modules')
        plt.plot([], [], 'o', color='lightgreen', markersize=10, label='Data')
        plt.plot([], [], '-', color='blue', linewidth=2, label='Produces')
        plt.plot([], [], '--', color='green', linewidth=2, label='Consumes')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_graphviz_dot(self, output_path: str = "dependencies.dot"):
        """Generate Graphviz DOT file for external visualization"""
        dot = graphviz.Digraph(comment='Module Dependencies')
        dot.attr(rankdir='LR')
        
        # Add nodes with attributes
        for node in self.module_graph.nodes():
            attrs = self.module_graph.nodes[node]
            
            # Style based on attributes
            if attrs.get('is_voting'):
                dot.node(node, shape='diamond', style='filled', fillcolor='lightcoral')
            elif attrs.get('category') == 'risk':
                dot.node(node, shape='box', style='filled', fillcolor='yellow')
            else:
                dot.node(node, shape='ellipse', style='filled', fillcolor='lightblue')
        
        # Add edges
        for edge in self.module_graph.edges():
            # Check if edge is part of a cycle
            is_circular = False
            for cycle in self.circular_dependencies:
                if edge[0] in cycle and edge[1] in cycle:
                    is_circular = True
                    break
            
            if is_circular:
                dot.edge(edge[0], edge[1], color='red', penwidth='2')
            else:
                dot.edge(edge[0], edge[1])
        
        # Save DOT file
        with open(output_path, 'w') as f:
            f.write(dot.source)
    
    def get_optimization_suggestions(self) -> List[str]:
        """Generate optimization suggestions based on dependency analysis"""
        suggestions = []
        
        # Check for circular dependencies
        if self.circular_dependencies:
            suggestions.append(
                f"⚠️ Found {len(self.circular_dependencies)} circular dependencies:"
            )
            for cycle in self.circular_dependencies[:3]:  # Show first 3
                suggestions.append(f"  • {' → '.join(cycle)} → {cycle[0]}")
            suggestions.append("  Consider refactoring to break these cycles")
        
        # Check for bottlenecks (high centrality)
        importance = self._calculate_module_importance()
        top_modules = list(importance.items())[:3]
        
        if top_modules:
            suggestions.append("\n📊 Critical modules (highest centrality):")
            for module, score in top_modules:
                in_degree = self.module_graph.in_degree(module)
                out_degree = self.module_graph.out_degree(module)
                suggestions.append(
                    f"  • {module}: {in_degree} dependencies, "
                    f"{out_degree} dependents"
                )
        
        # Check for isolated modules
        if self.isolated_modules:
            suggestions.append(f"\n🏝️ Found {len(self.isolated_modules)} isolated modules:")
            for module in self.isolated_modules[:5]:
                suggestions.append(f"  • {module}")
            suggestions.append("  Consider removing or integrating these modules")
        
        # Check for deep dependency chains
        depths = self._calculate_dependency_depth()
        max_depth = max(depths.values()) if depths else 0
        
        if max_depth > 5:
            deep_modules = [m for m, d in depths.items() if d == max_depth]
            suggestions.append(f"\n🔗 Deep dependency chain (depth={max_depth}):")
            suggestions.append(f"  • Deepest modules: {', '.join(deep_modules[:3])}")
            suggestions.append("  Consider flattening the dependency hierarchy")
        
        # Check for modules with too many dependencies
        for module in self.module_graph.nodes():
            in_degree = self.module_graph.in_degree(module)
            if in_degree > 5:
                suggestions.append(
                    f"\n⚡ Module '{module}' has {in_degree} dependencies - "
                    "consider reducing coupling"
                )
        
        return suggestions
    
    def export_analysis(self, output_dir: str = "dependency_analysis"):
        """Export complete dependency analysis"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate all visualizations
        self.visualize_module_graph(str(output_path / "module_dependencies.png"))
        self.visualize_data_flow(str(output_path / "data_flow.png"))
        self.generate_graphviz_dot(str(output_path / "dependencies.dot"))
        
        # Export analysis results
        analysis = self.analyze_dependencies()
        analysis['optimization_suggestions'] = self.get_optimization_suggestions()
        
        with open(output_path / "analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Generate text report
        report = self._generate_text_report(analysis)
        with open(output_path / "report.txt", 'w') as f:
            f.write(report)
    
    def _generate_text_report(self, analysis: Dict[str, Any]) -> str:
        """Generate human-readable text report"""
        lines = [
            "=" * 60,
            "SMARTINFOBUS DEPENDENCY ANALYSIS REPORT",
            "=" * 60,
            f"\nTotal Modules: {analysis['total_modules']}",
            f"Total Dependencies: {analysis['total_dependencies']}",
            f"\nCircular Dependencies: {len(analysis['circular_dependencies'])}",
            f"Isolated Modules: {len(analysis['isolated_modules'])}",
            f"Critical Paths: {len(analysis['critical_paths'])}",
            "\n" + "-" * 60,
            "\nOPTIMIZATION SUGGESTIONS:",
            "-" * 60
        ]
        
        lines.extend(analysis['optimization_suggestions'])
        
        lines.extend([
            "\n" + "-" * 60,
            "\nMODULE IMPORTANCE RANKING:",
            "-" * 60
        ])
        
        for i, (module, score) in enumerate(
            list(analysis['module_importance'].items())[:10], 1
        ):
            lines.append(f"{i:2d}. {module:<30} Score: {score:.4f}")
        
        lines.extend([
            "\n" + "-" * 60,
            "\nDEPENDENCY DEPTH:",
            "-" * 60
        ])
        
        sorted_depths = sorted(
            analysis['dependency_depth'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for module, depth in sorted_depths[:10]:
            lines.append(f"  {module:<30} Depth: {depth}")
        
        lines.append("\n" + "=" * 60)
        
        return "\n".join(lines)