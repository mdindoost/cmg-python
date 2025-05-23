#!/usr/bin/env python3
"""
Basic usage examples for CMG-Python library.

This script demonstrates the fundamental usage of the CMG Steiner Group algorithm
with simple examples and explanations.
"""

import numpy as np
import scipy.sparse as sp
from cmg import CMGSteinerSolver, create_laplacian_from_edges, create_test_graphs

# Optional visualization imports
try:
    from cmg.visualization.plotting import plot_graph_decomposition, plot_conductance_analysis
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
    print("Visualization not available. Install matplotlib and networkx for plots.")


def example_1_weak_connection():
    """Example 1: Simple weak connection graph."""
    
    print("=" * 60)
    print("Example 1: Weak Connection Graph")
    print("=" * 60)
    print("Graph structure: 1 -- 2 ~~~ 3 -- 4")
    print("Where ~~~ represents a weak connection")
    print()
    
    # Create graph with weak connection
    edges = [
        (0, 1, 1.0),    # Strong connection: node 1 -- node 2
        (1, 2, 0.01),   # Weak connection: node 2 ~~~ node 3
        (2, 3, 1.0)     # Strong connection: node 3 -- node 4
    ]
    n = 4
    
    # Create Laplacian matrix
    A = create_laplacian_from_edges(edges, n)
    
    print("Laplacian matrix:")
    print(A.toarray())
    print()
    
    # Create solver and run decomposition
    solver = CMGSteinerSolver(gamma=5.0, verbose=True)
    component_indices, num_components = solver.steiner_group(A)
    
    print(f"\nDecomposition result:")
    print(f"Number of components: {num_components}")
    print(f"Component assignment: {component_indices}")
    
    # Show detailed component information
    solver.visualize_components(component_indices)
    
    # Get and display statistics
    stats = solver.get_statistics()
    print(f"\nAlgorithm statistics:")
    print(f"  Average weighted degree: {stats['avg_weighted_degree']:.6f}")
    print(f"  High-degree nodes: {stats['high_degree_nodes']}")
    print(f"  Edges in final forest: {stats['forest_edges_final']}")
    print(f"  Computation time: {solver.last_decomposition_time:.4f} seconds")
    
    if HAS_VISUALIZATION:
        plot_graph_decomposition(A, component_indices, 
                               title="Example 1: Weak Connection Graph")
    
    return A, component_indices


def example_2_two_triangles():
    """Example 2: Two triangles connected by weak bridge."""
    
    print("\n" + "=" * 60)
    print("Example 2: Two Triangles with Weak Bridge")
    print("=" * 60)
    print("Graph structure: Triangle1 ~~~ Triangle2")
    print("Strong internal connections, weak bridge")
    print()
    
    # Create two triangles with weak bridge
    edges = [
        # Triangle 1: nodes 0, 1, 2
        (0, 1, 2.0), (1, 2, 2.0), (2, 0, 2.0),
        
        # Weak bridge
        (2, 3, 0.05),
        
        # Triangle 2: nodes 3, 4, 5  
        (3, 4, 2.0), (4, 5, 2.0), (5, 3, 2.0)
    ]
    n = 6
    
    A = create_laplacian_from_edges(edges, n)
    
    print("Graph has 6 nodes and 7 edges")
    print("Triangle 1: nodes 1-2-3 (strong connections)")
    print("Triangle 2: nodes 4-5-6 (strong connections)")
    print("Bridge: node 3 ~~~ node 4 (weak connection)")
    print()
    
    # Run decomposition
    solver = CMGSteinerSolver(gamma=5.0, verbose=True)
    component_indices, num_components = solver.steiner_group(A)
    
    # Display results
    solver.visualize_components(component_indices, 
                               node_names=[f"node{i+1}" for i in range(n)])
    
    if HAS_VISUALIZATION:
        plot_graph_decomposition(A, component_indices,
                               title="Example 2: Two Triangles")
        plot_conductance_analysis(A, component_indices,
                                title="Example 2: Conductance Analysis")
    
    return A, component_indices


def example_3_parameter_effects():
    """Example 3: Effect of gamma parameter."""
    
    print("\n" + "=" * 60)
    print("Example 3: Effect of Gamma Parameter")
    print("=" * 60)
    print("Testing different gamma values on the same graph")
    print()
    
    # Use the weak connection graph
    edges = [(0, 1, 1.0), (1, 2, 0.01), (2, 3, 1.0)]
    A = create_laplacian_from_edges(edges, 4)
    
    gamma_values = [4.1, 5.0, 7.0, 10.0]
    
    print(f"{'Gamma':<8} {'Components':<12} {'Avg Conductance':<15} {'Time (ms)':<10}")
    print("-" * 50)
    
    for gamma in gamma_values:
        solver = CMGSteinerSolver(gamma=gamma, verbose=False)
        component_indices, num_components = solver.steiner_group(A)
        
        stats = solver.get_statistics()
        avg_conductance = stats.get('avg_conductance', float('inf'))
        time_ms = solver.last_decomposition_time * 1000
        
        print(f"{gamma:<8.1f} {num_components:<12} {avg_conductance:<15.6f} {time_ms:<10.2f}")
    
    print("\nObservation: Gamma parameter affects high-degree node detection")
    print("Higher gamma → fewer nodes considered high-degree → less edge removal")


def example_4_using_test_graphs():
    """Example 4: Using built-in test graphs."""
    
    print("\n" + "=" * 60)
    print("Example 4: Built-in Test Graphs")
    print("=" * 60)
    
    # Get all test graphs
    test_graphs = create_test_graphs()
    
    solver = CMGSteinerSolver(gamma=5.0, verbose=False)
    
    print(f"{'Graph':<20} {'Nodes':<8} {'Edges':<8} {'Components':<12} {'Expected':<10}")
    print("-" * 70)
    
    for name, graph_data in test_graphs.items():
        A = create_laplacian_from_edges(graph_data['edges'], graph_data['n'])
        component_indices, num_components = solver.steiner_group(A)
        
        expected = graph_data.get('expected_components', 'N/A')
        
        print(f"{name:<20} {graph_data['n']:<8} {len(graph_data['edges']):<8} "
              f"{num_components:<12} {expected:<10}")
    
    print("\nThese test graphs demonstrate different scenarios:")
    print("- weak_connection: Simple path with weak link")
    print("- two_triangles: Strong clusters with weak bridge")  
    print("- star_periphery: Central star with weak periphery")
    print("- cluster_chain: Chain of clusters")
    print("- dense_connected: Dense graph (should stay connected)")


def example_5_custom_graph():
    """Example 5: Creating and analyzing a custom graph."""
    
    print("\n" + "=" * 60)
    print("Example 5: Custom Graph Analysis")
    print("=" * 60)
    
    # Create a more complex custom graph
    # Structure: two dense clusters connected by a single weak edge
    edges = [
        # Cluster 1: complete subgraph on nodes 0,1,2,3
        (0, 1, 1.5), (0, 2, 1.8), (0, 3, 1.2),
        (1, 2, 1.6), (1, 3, 1.7), (2, 3, 1.4),
        
        # Weak connection
        (3, 4, 0.02),
        
        # Cluster 2: complete subgraph on nodes 4,5,6,7
        (4, 5, 1.3), (4, 6, 1.9), (4, 7, 1.1),
        (5, 6, 1.4), (5, 7, 1.8), (6, 7, 1.6)
    ]
    n = 8
    
    A = create_laplacian_from_edges(edges, n)
    
    print(f"Created custom graph with {n} nodes and {len(edges)} edges")
    print("Two dense clusters (4 nodes each) connected by weak link")
    print()
    
    # Analyze the graph
    solver = CMGSteinerSolver(gamma=5.0, verbose=True)
    component_indices, num_components = solver.steiner_group(A)
    
    # Get detailed component information
    component_details = solver.get_component_details(component_indices, A)
    
    print("\nDetailed component analysis:")
    for comp_id, details in component_details.items():
        print(f"Component {comp_id}:")
        print(f"  Nodes: {[n+1 for n in details['nodes']]}")
        print(f"  Size: {details['size']}")
        print(f"  Conductance: {details['conductance']:.6f}")
        print(f"  Internal edges: {details['internal_edges']}")
        print(f"  Avg internal weight: {details['avg_internal_weight']:.3f}")
    
    if HAS_VISUALIZATION:
        plot_graph_decomposition(A, component_indices,
                               title="Example 5: Custom Dense Clusters")
    
    return A, component_indices


def main():
    """Run all basic examples."""
    
    print("CMG-Python Basic Usage Examples")
    print("These examples demonstrate core functionality of the CMG library.")
    print("Each example builds on previous concepts.")
    print()
    
    # Run examples
    try:
        example_1_weak_connection()
        example_2_two_triangles()
        example_3_parameter_effects()
        example_4_using_test_graphs()
        example_5_custom_graph()
        
        print("\n" + "=" * 60)
        print("All Examples Completed Successfully!")
        print("=" * 60)
        print("\nKey takeaways:")
        print("1. CMG detects and splits graphs at weak connections")
        print("2. The gamma parameter controls high-degree node detection")
        print("3. Built-in test graphs provide validation scenarios")
        print("4. Detailed statistics help understand algorithm behavior")
        print("5. Visualization helps interpret decomposition results")
        
        if not HAS_VISUALIZATION:
            print("\nInstall matplotlib and networkx for visualization:")
            print("  pip install matplotlib networkx")
    
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
