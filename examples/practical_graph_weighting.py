#!/usr/bin/env python3
"""
Practical Graph Weighting for Real-World CMG Usage
=================================================

This module provides simple, robust methods to add meaningful weights
to unweighted graphs for better CMG community detection performance.

Key principle: Keep it simple and reliable, not perfect.

Usage:
    from practical_graph_weighting import weight_unweighted_graph
    
    # Your unweighted graph
    G = nx.karate_club_graph()
    
    # Add meaningful weights
    weighted_edges = weight_unweighted_graph(G, method='auto')
    
    # Use with CMG
    A = create_laplacian_from_edges(weighted_edges, G.number_of_nodes())
    solver = CMGSteinerSolver(gamma=5.0)
    components, num_communities = solver.steiner_group(A)
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict
from collections import defaultdict


def weight_unweighted_graph(G: nx.Graph, method='auto', verbose=False) -> List[Tuple[int, int, float]]:
    """
    Convert unweighted graph to weighted graph for better CMG performance.
    
    Args:
        G: NetworkX graph (unweighted)
        method: 'auto', 'simple', 'jaccard', 'resource_allocation', 'common_neighbors'
        verbose: Print information about weighting process
        
    Returns:
        List of (u, v, weight) tuples
    """
    if verbose:
        print(f"🔸 Weighting graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Ensure consecutive node IDs
    G = _ensure_consecutive_nodes(G)
    
    # Choose method automatically if 'auto'
    if method == 'auto':
        method = _choose_best_method(G, verbose)
    
    # Apply chosen weighting method
    if method == 'simple':
        return _simple_weighting(G, verbose)
    elif method == 'jaccard':
        return _jaccard_weighting(G, verbose)
    elif method == 'resource_allocation':
        return _resource_allocation_weighting(G, verbose)
    elif method == 'common_neighbors':
        return _common_neighbors_weighting(G, verbose)
    else:
        raise ValueError(f"Unknown method: {method}")


def _ensure_consecutive_nodes(G: nx.Graph) -> nx.Graph:
    """Ensure graph has consecutive 0-based node IDs."""
    nodes = sorted(G.nodes())
    if nodes == list(range(len(nodes))):
        return G  # Already consecutive
    
    # Create mapping
    mapping = {old: new for new, old in enumerate(nodes)}
    return nx.relabel_nodes(G, mapping)


def _choose_best_method(G: nx.Graph, verbose=False) -> str:
    """
    Automatically choose the best weighting method based on graph properties.
    
    This is based on our test results and graph theory insights.
    """
    density = nx.density(G)
    avg_clustering = nx.average_clustering(G)
    num_nodes = G.number_of_nodes()
    
    if verbose:
        print(f"   Graph analysis: density={density:.3f}, clustering={avg_clustering:.3f}")
    
    # Decision tree based on empirical results
    if avg_clustering > 0.6 and density > 0.3:
        # Dense, highly clustered graph (like cliques)
        method = 'jaccard'
        reason = "dense clustered structure"
    elif density < 0.1:
        # Sparse graph
        method = 'common_neighbors'
        reason = "sparse structure"
    elif num_nodes > 100:
        # Large graph - use efficient method
        method = 'simple'
        reason = "large graph efficiency"
    else:
        # General case - most robust in our tests
        method = 'resource_allocation'
        reason = "general robustness"
    
    if verbose:
        print(f"   Selected method: {method} ({reason})")
    
    return method


def _simple_weighting(G: nx.Graph, verbose=False) -> List[Tuple[int, int, float]]:
    """
    Simple weighting: slightly boost edges based on local connectivity.
    Fast and reliable for large graphs.
    """
    if verbose:
        print("   Using simple weighting (degree-based)")
    
    edges = []
    for u, v in G.edges():
        # Weight based on geometric mean of degrees
        degree_u = G.degree(u)
        degree_v = G.degree(v)
        
        if degree_u > 0 and degree_v > 0:
            weight = 1.0 + 0.5 * np.sqrt(degree_u * degree_v) / max(degree_u, degree_v)
        else:
            weight = 1.0
        
        edges.append((u, v, weight))
    
    return edges


def _jaccard_weighting(G: nx.Graph, verbose=False) -> List[Tuple[int, int, float]]:
    """
    Jaccard similarity weighting: best for dense, clustered graphs.
    """
    if verbose:
        print("   Using Jaccard similarity weighting")
    
    edges = []
    for u, v in G.edges():
        # Calculate Jaccard similarity
        u_neighbors = set(G.neighbors(u))
        v_neighbors = set(G.neighbors(v))
        
        intersection = len(u_neighbors & v_neighbors)
        union = len(u_neighbors | v_neighbors)
        
        if union > 0:
            jaccard = intersection / union
            weight = 1.0 + 2.0 * jaccard  # Scale factor of 2
        else:
            weight = 1.0
        
        edges.append((u, v, weight))
    
    return edges


def _resource_allocation_weighting(G: nx.Graph, verbose=False) -> List[Tuple[int, int, float]]:
    """
    Resource allocation weighting: most robust across different graph types.
    """
    if verbose:
        print("   Using resource allocation weighting")
    
    edges = []
    for u, v in G.edges():
        # Resource allocation index
        ra_score = 0.0
        common_neighbors = list(nx.common_neighbors(G, u, v))
        
        for z in common_neighbors:
            degree_z = G.degree(z)
            if degree_z > 0:
                ra_score += 1.0 / degree_z
        
        weight = 1.0 + ra_score
        edges.append((u, v, weight))
    
    return edges


def _common_neighbors_weighting(G: nx.Graph, verbose=False) -> List[Tuple[int, int, float]]:
    """
    Common neighbors weighting: good for sparse graphs.
    """
    if verbose:
        print("   Using common neighbors weighting")
    
    edges = []
    for u, v in G.edges():
        # Count common neighbors
        common_neighbors = len(list(nx.common_neighbors(G, u, v)))
        
        # Weight based on common neighbors
        weight = 1.0 + 0.5 * common_neighbors
        
        edges.append((u, v, weight))
    
    return edges


def test_weighting_on_examples():
    """
    Test the weighting function on common example graphs.
    """
    print("🧪 Testing Practical Graph Weighting on Real Examples")
    print("=" * 60)
    
    # Import CMG for testing
    try:
        from cmg import CMGSteinerSolver, create_laplacian_from_edges
    except ImportError:
        print("❌ CMG not available for testing")
        return
    
    # Test graphs
    test_cases = [
        ("Karate Club", nx.karate_club_graph(), [
            list(range(12)),  # Simplified ground truth
            list(range(12, 34))
        ]),
        ("Les Miserables", _create_les_miserables_like(), None),
        ("Small World", nx.watts_strogatz_graph(20, 4, 0.3), None),
        ("Scale Free", nx.barabasi_albert_graph(25, 3), None),
    ]
    
    methods_to_test = ['auto', 'simple', 'jaccard', 'resource_allocation', 'common_neighbors']
    
    results = []
    
    for graph_name, G, true_communities in test_cases:
        print(f"\n📊 Testing on {graph_name}")
        print(f"   Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        print(f"   Density: {nx.density(G):.3f}")
        print("-" * 50)
        
        graph_results = {}
        
        for method in methods_to_test:
            try:
                # Apply weighting
                weighted_edges = weight_unweighted_graph(G, method=method, verbose=False)
                
                # Test with CMG
                A = create_laplacian_from_edges(weighted_edges, G.number_of_nodes())
                solver = CMGSteinerSolver(gamma=5.0, verbose=False)
                components, num_communities = solver.steiner_group(A)
                
                # Calculate weight statistics
                weights = [w for u, v, w in weighted_edges]
                weight_range = max(weights) / min(weights) if min(weights) > 0 else 1.0
                
                graph_results[method] = {
                    'num_communities': num_communities,
                    'weight_range': weight_range,
                    'avg_weight': np.mean(weights),
                    'success': True
                }
                
                print(f"   {method:20s}: {num_communities:2d} communities | "
                      f"weights: {np.mean(weights):.2f}±{np.std(weights):.2f} | "
                      f"range: {weight_range:.1f}x")
                
            except Exception as e:
                graph_results[method] = {'error': str(e), 'success': False}
                print(f"   {method:20s}: ❌ Error - {str(e)[:30]}...")
        
        results.append((graph_name, graph_results))
        
        # Identify best method for this graph
        successful_methods = [(m, r) for m, r in graph_results.items() if r.get('success', False)]
        if successful_methods:
            # Simple heuristic: prefer moderate number of communities
            best_method = min(successful_methods, 
                            key=lambda x: abs(x[1]['num_communities'] - np.sqrt(G.number_of_nodes())))
            print(f"   🎯 Best for this graph: {best_method[0]} ({best_method[1]['num_communities']} communities)")
    
    # Overall summary
    print(f"\n🏆 PRACTICAL WEIGHTING SUMMARY")
    print("=" * 40)
    
    # Count successes per method
    method_successes = defaultdict(int)
    method_total = defaultdict(int)
    
    for graph_name, graph_results in results:
        for method, result in graph_results.items():
            method_total[method] += 1
            if result.get('success', False):
                method_successes[method] += 1
    
    print("\n📈 Method Reliability:")
    for method in methods_to_test:
        success_rate = method_successes[method] / method_total[method] * 100 if method_total[method] > 0 else 0
        print(f"   {method:20s}: {success_rate:5.1f}% success ({method_successes[method]}/{method_total[method]})")
    
    # Recommendation
    best_overall = max(method_successes.items(), key=lambda x: x[1])
    print(f"\n💡 RECOMMENDATION:")
    print(f"   🥇 Most reliable: {best_overall[0]} ({best_overall[1]} successes)")
    print(f"   🤖 For automatic use: 'auto' method (adapts to graph properties)")
    
    return results


def _create_les_miserables_like():
    """Create a Les Miserables-like social network for testing."""
    G = nx.Graph()
    
    # Create character groups
    groups = [
        list(range(10)),      # Main characters
        list(range(10, 18)),  # Secondary characters  
        list(range(18, 25)),  # Minor characters
    ]
    
    # Dense connections within groups
    for group in groups:
        for i in group:
            for j in group:
                if i < j and np.random.random() > 0.3:
                    G.add_edge(i, j)
    
    # Sparse connections between groups
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            # Connect a few nodes between groups
            for _ in range(2):
                u = np.random.choice(groups[i])
                v = np.random.choice(groups[j])
                G.add_edge(u, v)
    
    return G


# Convenience function for direct use
def apply_cmg_with_weighting(G: nx.Graph, method='auto', gamma=5.0, verbose=False):
    """
    Complete workflow: unweighted graph → weighted → CMG communities.
    
    Args:
        G: Unweighted NetworkX graph
        method: Weighting method ('auto', 'simple', 'jaccard', etc.)
        gamma: CMG gamma parameter
        verbose: Print progress information
        
    Returns:
        (components, num_communities, weighted_edges)
    """
    try:
        from cmg import CMGSteinerSolver, create_laplacian_from_edges
    except ImportError:
        raise ImportError("CMG package not available")
    
    if verbose:
        print(f"🚀 Running CMG with automatic weighting")
        print(f"   Input: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Step 1: Add meaningful weights
    weighted_edges = weight_unweighted_graph(G, method=method, verbose=verbose)
    
    if verbose:
        weights = [w for u, v, w in weighted_edges]
        print(f"   Weights: {np.mean(weights):.2f} ± {np.std(weights):.2f}")
    
    # Step 2: Create Laplacian
    A = create_laplacian_from_edges(weighted_edges, G.number_of_nodes())
    
    # Step 3: Run CMG
    solver = CMGSteinerSolver(gamma=gamma, verbose=False)
    components, num_communities = solver.steiner_group(A)
    
    if verbose:
        print(f"   Result: {num_communities} communities")
    
    return components, num_communities, weighted_edges


def main():
    """Test the practical weighting approach."""
    test_weighting_on_examples()
    
    print(f"\n" + "="*60)
    print("🎯 PRACTICAL SOLUTION READY!")
    print("Use weight_unweighted_graph() or apply_cmg_with_weighting()")
    print("for automatic weighting of real-world unweighted graphs.")


if __name__ == "__main__":
    main()
