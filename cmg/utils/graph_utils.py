"""
Graph utilities for CMG algorithms.

This module provides helper functions for creating and manipulating graphs
used in CMG algorithms.
"""

import numpy as np
import scipy.sparse as sp
from typing import List, Tuple, Dict, Optional, Union


def create_laplacian_from_edges(edges: List[Tuple[int, int, float]], n: int) -> sp.csr_matrix:
    """
    Create a Laplacian matrix from a list of edges.
    
    The Laplacian matrix L is defined as:
    - L[i,j] = -w[i,j] for i ≠ j (negative edge weights)
    - L[i,i] = sum of weights incident to node i (degree)
    
    Args:
        edges: List of (u, v, weight) tuples
        n: Number of nodes
        
    Returns:
        scipy.sparse.csr_matrix: Laplacian matrix
        
    Example:
        >>> edges = [(0, 1, 1.0), (1, 2, 0.5)]
        >>> L = create_laplacian_from_edges(edges, 3)
        >>> print(L.toarray())
        [[ 1.  -1.   0. ]
         [-1.   1.5 -0.5]
         [ 0.  -0.5  0.5]]
    """
    rows, cols, data = [], [], []
    
    # Add off-diagonal entries (negative edge weights)
    for u, v, weight in edges:
        rows.extend([u, v])
        cols.extend([v, u])
        data.extend([-weight, -weight])
    
    # Calculate degrees and add diagonal entries
    degrees = np.zeros(n)
    for u, v, weight in edges:
        degrees[u] += weight
        degrees[v] += weight
    
    for i in range(n):
        rows.append(i)
        cols.append(i)
        data.append(degrees[i])
    
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def create_adjacency_from_edges(edges: List[Tuple[int, int, float]], n: int) -> sp.csr_matrix:
    """
    Create an adjacency matrix from a list of edges.
    
    Args:
        edges: List of (u, v, weight) tuples
        n: Number of nodes
        
    Returns:
        scipy.sparse.csr_matrix: Adjacency matrix
    """
    rows, cols, data = [], [], []
    
    for u, v, weight in edges:
        rows.extend([u, v])
        cols.extend([v, u])
        data.extend([weight, weight])
    
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def laplacian_to_adjacency(L: sp.spmatrix) -> sp.csr_matrix:
    """
    Convert a Laplacian matrix to an adjacency matrix.
    
    Args:
        L: Laplacian matrix
        
    Returns:
        scipy.sparse.csr_matrix: Adjacency matrix
    """
    A = -L.copy()
    A.setdiag(0)
    A.eliminate_zeros()
    A.data = np.abs(A.data)  # Ensure positive weights
    return A


def adjacency_to_laplacian(A: sp.spmatrix) -> sp.csr_matrix:
    """
    Convert an adjacency matrix to a Laplacian matrix.
    
    Args:
        A: Adjacency matrix
        
    Returns:
        scipy.sparse.csr_matrix: Laplacian matrix
    """
    if not sp.issparse(A):
        A = sp.csr_matrix(A)
    
    # Calculate degrees
    degrees = np.array(A.sum(axis=1)).flatten()
    
    # Create Laplacian: L = D - A
    L = sp.diags(degrees) - A
    
    return L.tocsr()


def create_test_graphs() -> Dict[str, Dict]:
    """
    Create a collection of test graphs for CMG algorithm testing.
    
    Returns:
        dict: Dictionary of test graphs with metadata
    """
    test_graphs = {}
    
    # 1. Simple weak connection (path graph)
    test_graphs['weak_connection'] = {
        'edges': [(0, 1, 1.0), (1, 2, 0.01), (2, 3, 1.0)],
        'n': 4,
        'description': 'Path with weak connection: 1--2~~~3--4',
        'expected_components': 2,
        'expected_split': [[0, 1], [2, 3]]
    }
    
    # 2. Two triangles connected by weak bridge
    test_graphs['two_triangles'] = {
        'edges': [
            # Triangle 1: strong internal connections
            (0, 1, 2.0), (1, 2, 2.0), (2, 0, 2.0),
            # Weak bridge
            (2, 3, 0.05),
            # Triangle 2: strong internal connections
            (3, 4, 2.0), (4, 5, 2.0), (5, 3, 2.0)
        ],
        'n': 6,
        'description': 'Two triangles connected by weak bridge',
        'expected_components': 2,
        'expected_split': [[0, 1, 2], [3, 4, 5]]
    }
    
    # 3. Star graph with weak periphery
    test_graphs['star_periphery'] = {
        'edges': [
            # Central star (strong connections)
            (0, 1, 3.0), (0, 2, 3.0), (0, 3, 3.0), (0, 4, 3.0),
            # Weak peripheral connections
            (1, 5, 0.1), (2, 6, 0.1), (3, 7, 0.1), (4, 8, 0.1)
        ],
        'n': 9,
        'description': 'Star with weak peripheral connections',
        'expected_components': 'variable',  # Depends on parameters
    }
    
    # 4. Chain of clusters
    test_graphs['cluster_chain'] = {
        'edges': [
            # Cluster 1
            (0, 1, 2.0), (1, 2, 2.0),
            # Weak connection
            (2, 3, 0.1),
            # Cluster 2  
            (3, 4, 2.0), (4, 5, 2.0),
            # Weak connection
            (5, 6, 0.1),
            # Cluster 3
            (6, 7, 2.0), (7, 8, 2.0)
        ],
        'n': 9,
        'description': 'Chain of three clusters connected by weak links',
        'expected_components': 3,
        'expected_split': [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    }
    
    # 5. Grid with weak connection
    test_graphs['weak_grid'] = {
        'edges': [
            # Left 2x2 grid (strong)
            (0, 1, 1.0), (0, 2, 1.0), (1, 3, 1.0), (2, 3, 1.0),
            # Weak connection
            (1, 4, 0.01),
            # Right 2x2 grid (strong)
            (4, 5, 1.0), (4, 6, 1.0), (5, 7, 1.0), (6, 7, 1.0)
        ],
        'n': 8,
        'description': 'Two 2x2 grids connected by weak link',
        'expected_components': 2,
        'expected_split': [[0, 1, 2, 3], [4, 5, 6, 7]]
    }
    
    # 6. Highly connected graph (should stay as one component)
    test_graphs['dense_connected'] = {
        'edges': [
            (0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0),
            (1, 2, 1.0), (1, 3, 1.0), (2, 3, 1.0)
        ],
        'n': 4,
        'description': 'Complete graph (should remain connected)',
        'expected_components': 1,
        'expected_split': [[0, 1, 2, 3]]
    }
    
    return test_graphs


def create_random_graph(n: int, p: float, weight_range: Tuple[float, float] = (0.1, 2.0), 
                       seed: Optional[int] = None) -> Tuple[List[Tuple[int, int, float]], int]:
    """
    Create a random graph with specified connection probability.
    
    Args:
        n: Number of nodes
        p: Edge probability (0 to 1)
        weight_range: Range of edge weights (min_weight, max_weight)
        seed: Random seed for reproducibility
        
    Returns:
        tuple: (edges, n) where edges is list of (u, v, weight) tuples
    """
    if seed is not None:
        np.random.seed(seed)
    
    edges = []
    min_weight, max_weight = weight_range
    
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.random() < p:
                weight = min_weight + np.random.random() * (max_weight - min_weight)
                edges.append((i, j, weight))
    
    return edges, n


def create_clustered_graph(cluster_sizes: List[int], 
                          intra_cluster_p: float = 0.8,
                          inter_cluster_p: float = 0.1,
                          intra_weight_range: Tuple[float, float] = (1.0, 2.0),
                          inter_weight_range: Tuple[float, float] = (0.01, 0.1),
                          seed: Optional[int] = None) -> Tuple[List[Tuple[int, int, float]], int]:
    """
    Create a graph with predefined cluster structure.
    
    Args:
        cluster_sizes: List of sizes for each cluster
        intra_cluster_p: Probability of edges within clusters
        inter_cluster_p: Probability of edges between clusters
        intra_weight_range: Weight range for intra-cluster edges
        inter_weight_range: Weight range for inter-cluster edges
        seed: Random seed
        
    Returns:
        tuple: (edges, total_nodes)
    """
    if seed is not None:
        np.random.seed(seed)
    
    edges = []
    total_nodes = sum(cluster_sizes)
    
    # Create cluster node mappings
    clusters = []
    node_offset = 0
    for size in cluster_sizes:
        cluster_nodes = list(range(node_offset, node_offset + size))
        clusters.append(cluster_nodes)
        node_offset += size
    
    # Add intra-cluster edges (strong connections)
    for cluster in clusters:
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                if np.random.random() < intra_cluster_p:
                    weight = (intra_weight_range[0] + 
                             np.random.random() * (intra_weight_range[1] - intra_weight_range[0]))
                    edges.append((cluster[i], cluster[j], weight))
    
    # Add inter-cluster edges (weak connections)
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            cluster_i, cluster_j = clusters[i], clusters[j]
            
            for node_i in cluster_i:
                for node_j in cluster_j:
                    if np.random.random() < inter_cluster_p:
                        weight = (inter_weight_range[0] + 
                                 np.random.random() * (inter_weight_range[1] - inter_weight_range[0]))
                        edges.append((node_i, node_j, weight))
    
    return edges, total_nodes


def validate_graph(edges: List[Tuple[int, int, float]], n: int) -> Dict[str, any]:
    """
    Validate a graph and return diagnostic information.
    
    Args:
        edges: List of edges
        n: Number of nodes
        
    Returns:
        dict: Validation results and graph statistics
    """
    validation = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {}
    }
    
    # Check for valid node indices
    all_nodes = set()
    for u, v, weight in edges:
        all_nodes.update([u, v])
        
        if not (0 <= u < n):
            validation['errors'].append(f"Node {u} out of range [0, {n-1}]")
            validation['is_valid'] = False
            
        if not (0 <= v < n):
            validation['errors'].append(f"Node {v} out of range [0, {n-1}]")
            validation['is_valid'] = False
            
        if weight <= 0:
            validation['warnings'].append(f"Non-positive weight {weight} on edge ({u}, {v})")
    
    # Check for self-loops
    self_loops = [(u, v) for u, v, _ in edges if u == v]
    if self_loops:
        validation['warnings'].append(f"Found {len(self_loops)} self-loops")
    
    # Calculate statistics
    weights = [w for _, _, w in edges]
    degrees = np.zeros(n)
    
    for u, v, weight in edges:
        degrees[u] += weight
        degrees[v] += weight
    
    validation['statistics'] = {
        'num_nodes': n,
        'num_edges': len(edges),
        'num_isolated_nodes': np.sum(degrees == 0),
        'min_weight': min(weights) if weights else 0,
        'max_weight': max(weights) if weights else 0,
        'avg_weight': np.mean(weights) if weights else 0,
        'min_degree': np.min(degrees),
        'max_degree': np.max(degrees),
        'avg_degree': np.mean(degrees),
        'density': len(edges) / (n * (n - 1) / 2) if n > 1 else 0
    }
    
    return validation


def benchmark_graph_sizes() -> Dict[str, Dict]:
    """
    Create a set of graphs of increasing sizes for benchmarking.
    
    Returns:
        dict: Dictionary of benchmark graphs
    """
    benchmark_graphs = {}
    
    sizes = [10, 50, 100, 500, 1000]
    
    for size in sizes:
        # Create a graph with natural cluster structure
        if size <= 100:
            cluster_sizes = [size // 4] * 4
        else:
            cluster_sizes = [size // 10] * 10
            
        edges, n = create_clustered_graph(
            cluster_sizes=cluster_sizes,
            intra_cluster_p=0.7,
            inter_cluster_p=0.05,
            seed=42
        )
        
        benchmark_graphs[f'clustered_{size}'] = {
            'edges': edges,
            'n': n,
            'description': f'Clustered graph with {n} nodes',
            'type': 'clustered'
        }
        
        # Also create a random graph of the same size
        edges_random, n_random = create_random_graph(size, p=0.1, seed=42)
        
        benchmark_graphs[f'random_{size}'] = {
            'edges': edges_random,
            'n': n_random,
            'description': f'Random graph with {n_random} nodes',
            'type': 'random'
        }
    
    return benchmark_graphs
