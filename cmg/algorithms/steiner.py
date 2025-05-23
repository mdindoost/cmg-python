"""
Core CMG Steiner Group algorithm implementation.

This module contains the main CMGSteinerSolver class that implements the 
Decompose-Graph algorithm from the CMG paper.
"""

import numpy as np
import scipy.sparse as sp
from collections import defaultdict, deque
import time
import warnings
from typing import Tuple, List, Optional, Union


class CMGSteinerSolver:
    """
    CMG Steiner Group solver implementing the Decompose-Graph algorithm.
    
    This class implements the core algorithm from:
    "Combinatorial preconditioners and multilevel solvers for problems 
    in computer vision and image processing" by Koutis, Miller, and Tolliver.
    
    The algorithm creates graph decompositions with bounded conductance,
    suitable for multigrid preconditioning and hierarchical solving.
    """
    
    def __init__(self, gamma: float = 5.0, verbose: bool = True):
        """
        Initialize CMG Steiner solver.
        
        Args:
            gamma: Parameter for high-degree node detection (must be > 4)
            verbose: Whether to print progress information
            
        Raises:
            ValueError: If gamma <= 4
        """
        if gamma <= 4.0:
            raise ValueError("Parameter gamma must be > 4 (paper requirement)")
            
        self.gamma = gamma
        self.verbose = verbose
        self.last_decomposition_time = None
        self.last_statistics = {}
    
    def weighted_degree(self, A: sp.spmatrix, node: int) -> float:
        """
        Calculate weighted degree of a node.
        
        The weighted degree is defined as:
        wd(v) = vol(v) / max_{u∈N(v)} w(u,v)
        
        where vol(v) is the total weight incident to node v.
        
        Args:
            A: Sparse adjacency matrix
            node: Node index
            
        Returns:
            float: Weighted degree of the node
        """
        if not sp.issparse(A):
            A = sp.csr_matrix(A)
        
        neighbors = A[node].nonzero()[1]
        if len(neighbors) == 0:
            return 0.0
        
        weights = np.abs(A[node, neighbors].toarray().flatten())
        vol_v = np.sum(weights)
        max_weight = np.max(weights)
        
        return vol_v / max_weight if max_weight > 0 else 0.0
    
    def average_weighted_degree(self, A: sp.spmatrix) -> float:
        """
        Calculate average weighted degree of the graph.
        
        awd(G) = (1/n) * Σ_{v∈V} wd(v)
        
        Args:
            A: Sparse adjacency matrix
            
        Returns:
            float: Average weighted degree
        """
        n = A.shape[0]
        total_wd = sum(self.weighted_degree(A, i) for i in range(n))
        return total_wd / n if n > 0 else 0.0
    
    def volume(self, A: sp.spmatrix, node: int) -> float:
        """
        Calculate volume (total incident weight) of a node.
        
        Args:
            A: Sparse matrix
            node: Node index
            
        Returns:
            float: Total incident weight
        """
        incident_weights = A[node].toarray().flatten()
        return np.sum(np.abs(incident_weights))
    
    def conductance(self, A: sp.spmatrix, cluster: List[int]) -> float:
        """
        Calculate conductance of a cluster.
        
        φ(S) = w(S, V\S) / min(w(S), w(V\S))
        
        where w(S, V\S) is the weight of edges crossing the cut,
        and w(S) is the total weight within cluster S.
        
        Args:
            A: Sparse matrix (should be Laplacian)
            cluster: List of node indices in the cluster
            
        Returns:
            float: Conductance of the cluster
        """
        if len(cluster) == 0 or len(cluster) == A.shape[0]:
            return float('inf')
        
        cluster_set = set(cluster)
        n = A.shape[0]
        complement = [i for i in range(n) if i not in cluster_set]
        
        w_cluster = 0.0
        w_complement = 0.0
        w_cut = 0.0
        
        # Calculate cluster volume and cut weight
        for node in cluster:
            neighbors = A[node].nonzero()[1]
            weights = np.abs(A[node, neighbors].toarray().flatten())
            
            for neighbor, weight in zip(neighbors, weights):
                w_cluster += weight
                if neighbor not in cluster_set:
                    w_cut += weight
        
        # Calculate complement volume
        for node in complement:
            neighbors = A[node].nonzero()[1]
            weights = np.abs(A[node, neighbors].toarray().flatten())
            w_complement += sum(weights)
        
        # Avoid double counting (each edge counted twice)
        w_cluster /= 2.0
        w_complement /= 2.0
        w_cut /= 2.0
        
        min_volume = min(w_cluster, w_complement)
        return w_cut / min_volume if min_volume > 0 else float('inf')
    
    def build_heaviest_edge_forest(self, A: sp.spmatrix) -> List[Tuple[int, int]]:
        """
        Build forest by keeping the heaviest incident edge for each vertex.
        
        This implements Step 2 of the Decompose-Graph algorithm.
        
        Args:
            A: Sparse adjacency matrix
            
        Returns:
            List of edges (u, v) in the forest
        """
        n = A.shape[0]
        forest_edges = []
        
        for node in range(n):
            neighbors = A[node].nonzero()[1]
            if len(neighbors) == 0:
                continue
            
            weights = np.abs(A[node, neighbors].toarray().flatten())
            max_weight_idx = np.argmax(weights)
            heaviest_neighbor = neighbors[max_weight_idx]
            
            # Add edge if not already present (avoid duplicates)
            edge = tuple(sorted([node, heaviest_neighbor]))
            if edge not in forest_edges:
                forest_edges.append(edge)
        
        return forest_edges
    
    def connected_components_from_edges(self, edges: List[Tuple[int, int]], n: int) -> List[List[int]]:
        """
        Find connected components given a list of edges.
        
        Args:
            edges: List of edges (u, v)
            n: Number of nodes
            
        Returns:
            List of components, where each component is a list of node indices
        """
        # Build adjacency list
        adj_list = defaultdict(list)
        for u, v in edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
        
        visited = np.zeros(n, dtype=bool)
        components = []
        
        for start in range(n):
            if not visited[start]:
                # BFS to find connected component
                component = []
                queue = deque([start])
                visited[start] = True
                
                while queue:
                    node = queue.popleft()
                    component.append(node)
                    
                    for neighbor in adj_list[node]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor)
                
                components.append(component)
        
        return components
    
    def steiner_group(self, A: sp.spmatrix) -> Tuple[np.ndarray, int]:
        """
        Main Steiner group decomposition algorithm.
        
        Implements the Decompose-Graph algorithm from Section 3.3 of the CMG paper.
        
        Args:
            A: Input matrix (Laplacian or adjacency matrix)
            
        Returns:
            tuple: (component_indices, num_components)
                component_indices: Array mapping each node to its cluster (0-based)
                num_components: Number of clusters found
                
        Raises:
            ValueError: If matrix is not square or is empty
        """
        start_time = time.time()
        
        # Input validation
        if not sp.issparse(A):
            A = sp.csr_matrix(A)
        
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square")
        
        if A.shape[0] == 0:
            raise ValueError("Matrix cannot be empty")
        
        n = A.shape[0]
        
        if self.verbose:
            print(f"CMG Steiner Group Decomposition:")
            print(f"  Graph: {n} nodes, {A.nnz} edges")
        
        # Convert Laplacian to adjacency if needed
        if np.all(A.diagonal() >= 0):
            # This looks like a Laplacian matrix
            A_adj = -A.copy()
            A_adj.setdiag(0)
            A_adj.eliminate_zeros()
        else:
            A_adj = A.copy()
        
        # Ensure positive weights
        A_adj.data = np.abs(A_adj.data)
        
        # Step 1: Calculate weighted degrees and identify high-degree nodes
        awd = self.average_weighted_degree(A_adj)
        
        high_degree_nodes = []
        weighted_degrees = {}
        
        for v in range(n):
            wd_v = self.weighted_degree(A_adj, v)
            weighted_degrees[v] = wd_v
            if wd_v > self.gamma * awd:
                high_degree_nodes.append(v)
        
        if self.verbose:
            print(f"  Average weighted degree: {awd:.6f}")
            print(f"  High-degree nodes (wd > {self.gamma} * awd): {len(high_degree_nodes)}")
        
        # Step 2: Build forest of heaviest edges
        forest_edges = self.build_heaviest_edge_forest(A_adj)
        
        if self.verbose:
            print(f"  Initial forest: {len(forest_edges)} edges")
        
        # Step 3: Remove problematic edges from high-degree nodes
        # Check condition: vol_T(w) < vol_G(w)/awd(A)
        edges_to_remove = []
        
        for edge in forest_edges:
            u, v = edge
            for w in [u, v]:
                if w in high_degree_nodes:
                    # Calculate volumes
                    edge_weight = abs(A_adj[u, v])
                    vol_T_w = edge_weight  # Volume in tree (simplified)
                    vol_G_w = self.volume(A_adj, w)  # Volume in original graph
                    
                    if vol_T_w < vol_G_w / awd:
                        edges_to_remove.append(edge)
                        break
        
        final_forest_edges = [e for e in forest_edges if e not in edges_to_remove]
        
        if self.verbose:
            print(f"  Removed {len(edges_to_remove)} problematic edges")
            print(f"  Final forest: {len(final_forest_edges)} edges")
        
        # Step 4: Find connected components in the final forest
        components = self.connected_components_from_edges(final_forest_edges, n)
        
        # Create component indices array
        component_indices = np.zeros(n, dtype=np.int32)
        for comp_id, component in enumerate(components):
            for node in component:
                component_indices[node] = comp_id
        
        num_components = len(components)
        
        # Calculate detailed statistics
        self.last_decomposition_time = time.time() - start_time
        
        component_sizes = [len(comp) for comp in components]
        conductances = []
        
        for component in components:
            if len(component) > 1:
                cond = self.conductance(A, component)  # Use original matrix for conductance
                conductances.append(cond)
        
        self.last_statistics = {
            'num_components': num_components,
            'component_sizes': component_sizes,
            'avg_component_size': np.mean(component_sizes),
            'conductances': conductances,
            'avg_conductance': np.mean(conductances) if conductances else float('inf'),
            'high_degree_nodes': len(high_degree_nodes),
            'avg_weighted_degree': awd,
            'forest_edges_initial': len(forest_edges),
            'forest_edges_final': len(final_forest_edges),
            'edges_removed': len(edges_to_remove),
            'weighted_degrees': weighted_degrees,
        }
        
        if self.verbose:
            print(f"  Result: {num_components} components")
            print(f"  Component sizes: {component_sizes}")
            if conductances:
                print(f"  Average conductance: {np.mean(conductances):.6f}")
            print(f"  Computation time: {self.last_decomposition_time:.4f} seconds")
        
        return component_indices, num_components
    
    def get_statistics(self) -> dict:
        """
        Get detailed statistics from the last decomposition.
        
        Returns:
            dict: Dictionary containing algorithm statistics
        """
        return self.last_statistics.copy()
    
    def visualize_components(self, component_indices: np.ndarray, 
                           node_names: Optional[List[str]] = None) -> None:
        """
        Print a visualization of the component assignment.
        
        Args:
            component_indices: Array of component assignments
            node_names: Optional list of custom node names
        """
        unique_components = np.unique(component_indices)
        
        print("\nComponent Assignment:")
        print("=" * 50)
        
        for comp_id in unique_components:
            nodes = np.where(component_indices == comp_id)[0]
            
            if node_names:
                display_nodes = [node_names[i] for i in nodes]
            else:
                display_nodes = [f"node{i+1}" for i in nodes]
            
            print(f"Component {comp_id}: {display_nodes} (size: {len(nodes)})")
            
            # Show conductance if available
            if hasattr(self, 'last_statistics') and 'conductances' in self.last_statistics:
                conductances = self.last_statistics['conductances']
                if comp_id < len(conductances):
                    print(f"  Conductance: {conductances[comp_id]:.6f}")
    
    def get_component_details(self, component_indices: np.ndarray, A: sp.spmatrix) -> dict:
        """
        Get detailed information about each component.
        
        Args:
            component_indices: Array of component assignments
            A: Original matrix
            
        Returns:
            dict: Detailed component information
        """
        unique_components = np.unique(component_indices)
        details = {}
        
        for comp_id in unique_components:
            nodes = np.where(component_indices == comp_id)[0].tolist()
            
            # Calculate component-specific metrics
            if len(nodes) > 1:
                conductance = self.conductance(A, nodes)
                
                # Calculate internal edges
                internal_edges = 0
                total_weight = 0.0
                
                for node in nodes:
                    neighbors = A[node].nonzero()[1]
                    weights = A[node, neighbors].toarray().flatten()
                    
                    for neighbor, weight in zip(neighbors, weights):
                        if neighbor in nodes and node < neighbor:  # Avoid double counting
                            internal_edges += 1
                            total_weight += abs(weight)
                
                details[comp_id] = {
                    'nodes': nodes,
                    'size': len(nodes),
                    'conductance': conductance,
                    'internal_edges': internal_edges,
                    'total_internal_weight': total_weight,
                    'avg_internal_weight': total_weight / internal_edges if internal_edges > 0 else 0.0
                }
            else:
                # Singleton component
                details[comp_id] = {
                    'nodes': nodes,
                    'size': 1,
                    'conductance': float('inf'),
                    'internal_edges': 0,
                    'total_internal_weight': 0.0,
                    'avg_internal_weight': 0.0
                }
        
        return details
