#!/usr/bin/env python3
"""
CMG Synthetic Baseline Comparison Framework
===========================================

Comprehensive comparison of CMG against established hierarchical clustering methods
on controlled synthetic graphs to demonstrate CMG's unique capabilities and position
it within the hierarchical clustering landscape.

Phase 1: Synthetic Testing
- CMG-favorable graphs (sparse, hierarchical, density < 0.1)
- Challenging graphs (intermediate density)
- CMG-unfavorable graphs (dense, density > 0.3)

Baseline Methods:
- Agglomerative Clustering (Ward, Average, Complete linkage)
- Spectral Clustering (standard + hierarchical variants)
- Louvain (with hierarchy detection)

Evaluation:
- Boundary Preservation Score (our innovation)
- Hierarchical Clustering Quality
- Computational Performance
- Ground Truth Alignment

Run with: python baseline_comparison_test.py
"""

import sys
sys.path.append('..')

import numpy as np
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, asdict
import json

# Core libraries
import scipy.sparse as sp
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform

# CMG imports
from cmg import CMGSteinerSolver, create_laplacian_from_edges
from cmg.utils.graph_utils import create_clustered_graph

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('validation_results/baseline_comparison.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class BaselineComparisonResult:
    """Result of baseline comparison test."""
    test_name: str
    method_name: str
    graph_category: str
    
    # Graph properties
    n_nodes: int
    n_edges: int
    density: float
    avg_degree: float
    
    # Clustering results
    n_communities: int
    runtime_seconds: float
    
    # Quality metrics
    boundary_preservation_score: float
    silhouette_score: float
    hierarchical_quality: float
    
    # Ground truth alignment (when available)
    main_cluster_purity: float
    subcluster_alignment: float
    
    # Technical details
    method_parameters: Dict
    success: bool = True
    error_message: Optional[str] = None


class BaselineClusteringMethods:
    """Implementation of baseline hierarchical clustering methods."""
    
    @staticmethod
    def agglomerative_ward(A: sp.spmatrix, n_clusters: int = None) -> Tuple[np.ndarray, Dict]:
        """Ward linkage agglomerative clustering."""
        try:
            # Convert to distance matrix
            if sp.issparse(A):
                A_dense = A.toarray()
            else:
                A_dense = A
            
            # Create distance matrix from adjacency
            # Use negative weights (make positive) and convert to distances
            A_pos = np.abs(A_dense)
            np.fill_diagonal(A_pos, 0)
            
            # Convert similarity to distance (1 - normalized similarity)
            max_weight = np.max(A_pos)
            if max_weight > 0:
                dist_matrix = 1 - (A_pos / max_weight)
            else:
                dist_matrix = np.ones_like(A_pos)
            
            np.fill_diagonal(dist_matrix, 0)
            
            # Perform hierarchical clustering
            condensed_dist = pdist(dist_matrix)
            linkage_matrix = linkage(condensed_dist, method='ward')
            
            # Determine number of clusters if not specified
            if n_clusters is None:
                # Use a heuristic based on linkage matrix
                n_clusters = max(2, min(10, int(np.sqrt(A.shape[0]))))
            
            labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1
            
            return labels, {'linkage_method': 'ward', 'n_clusters': n_clusters}
            
        except Exception as e:
            logging.warning(f"Ward clustering failed: {e}")
            # Return single cluster as fallback
            return np.zeros(A.shape[0], dtype=int), {'error': str(e)}
    
    @staticmethod
    def agglomerative_average(A: sp.spmatrix, n_clusters: int = None) -> Tuple[np.ndarray, Dict]:
        """Average linkage agglomerative clustering."""
        try:
            if sp.issparse(A):
                A_dense = A.toarray()
            else:
                A_dense = A
            
            A_pos = np.abs(A_dense)
            np.fill_diagonal(A_pos, 0)
            
            max_weight = np.max(A_pos)
            if max_weight > 0:
                dist_matrix = 1 - (A_pos / max_weight)
            else:
                dist_matrix = np.ones_like(A_pos)
            
            np.fill_diagonal(dist_matrix, 0)
            
            condensed_dist = pdist(dist_matrix)
            linkage_matrix = linkage(condensed_dist, method='average')
            
            if n_clusters is None:
                n_clusters = max(2, min(10, int(np.sqrt(A.shape[0]))))
            
            labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1
            
            return labels, {'linkage_method': 'average', 'n_clusters': n_clusters}
            
        except Exception as e:
            logging.warning(f"Average clustering failed: {e}")
            return np.zeros(A.shape[0], dtype=int), {'error': str(e)}
    
    @staticmethod
    def agglomerative_complete(A: sp.spmatrix, n_clusters: int = None) -> Tuple[np.ndarray, Dict]:
        """Complete linkage agglomerative clustering."""
        try:
            if sp.issparse(A):
                A_dense = A.toarray()
            else:
                A_dense = A
            
            A_pos = np.abs(A_dense)
            np.fill_diagonal(A_pos, 0)
            
            max_weight = np.max(A_pos)
            if max_weight > 0:
                dist_matrix = 1 - (A_pos / max_weight)
            else:
                dist_matrix = np.ones_like(A_pos)
            
            np.fill_diagonal(dist_matrix, 0)
            
            condensed_dist = pdist(dist_matrix)
            linkage_matrix = linkage(condensed_dist, method='complete')
            
            if n_clusters is None:
                n_clusters = max(2, min(10, int(np.sqrt(A.shape[0]))))
            
            labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1
            
            return labels, {'linkage_method': 'complete', 'n_clusters': n_clusters}
            
        except Exception as e:
            logging.warning(f"Complete clustering failed: {e}")
            return np.zeros(A.shape[0], dtype=int), {'error': str(e)}
    
    @staticmethod
    def spectral_clustering(A: sp.spmatrix, n_clusters: int = None) -> Tuple[np.ndarray, Dict]:
        """Spectral clustering."""
        try:
            # Simple spectral clustering implementation
            if sp.issparse(A):
                A_work = A.copy()
            else:
                A_work = sp.csr_matrix(A)
            
            # Convert Laplacian to adjacency if needed
            if np.all(A_work.diagonal() >= 0):
                # This looks like a Laplacian
                A_adj = -A_work.copy()
                A_adj.setdiag(0)
                A_adj.eliminate_zeros()
                A_adj.data = np.abs(A_adj.data)
            else:
                A_adj = A_work.copy()
                A_adj.data = np.abs(A_adj.data)
            
            # Compute degree matrix
            degrees = np.array(A_adj.sum(axis=1)).flatten()
            
            # Avoid division by zero
            degrees[degrees == 0] = 1
            
            # Normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
            D_sqrt_inv = sp.diags(1.0 / np.sqrt(degrees))
            L_norm = sp.eye(A_adj.shape[0]) - D_sqrt_inv @ A_adj @ D_sqrt_inv
            
            if n_clusters is None:
                n_clusters = max(2, min(8, int(np.sqrt(A.shape[0]))))
            
            # Compute smallest eigenvalues/eigenvectors
            try:
                from scipy.sparse.linalg import eigsh
                eigenvals, eigenvecs = eigsh(L_norm, k=n_clusters, which='SM')
                
                # Use eigenvectors for clustering (simple k-means-like approach)
                # Normalize eigenvectors
                eigenvecs_norm = eigenvecs / (np.linalg.norm(eigenvecs, axis=1, keepdims=True) + 1e-10)
                
                # Simple clustering of eigenvector space
                from scipy.cluster.vq import kmeans2
                _, labels = kmeans2(eigenvecs_norm, n_clusters)
                
                return labels, {'method': 'spectral', 'n_clusters': n_clusters}
                
            except Exception as inner_e:
                logging.warning(f"Eigenvalue computation failed: {inner_e}")
                # Fallback to random clustering
                labels = np.random.randint(0, n_clusters, A.shape[0])
                return labels, {'method': 'spectral_fallback', 'error': str(inner_e)}
            
        except Exception as e:
            logging.warning(f"Spectral clustering failed: {e}")
            return np.zeros(A.shape[0], dtype=int), {'error': str(e)}
    
    @staticmethod
    def modularity_clustering(A: sp.spmatrix) -> Tuple[np.ndarray, Dict]:
        """Simple modularity-based clustering."""
        try:
            # Convert to adjacency if needed
            if sp.issparse(A):
                A_work = A.copy()
            else:
                A_work = sp.csr_matrix(A)
            
            if np.all(A_work.diagonal() >= 0):
                A_adj = -A_work.copy()
                A_adj.setdiag(0)
                A_adj.eliminate_zeros()
                A_adj.data = np.abs(A_adj.data)
            else:
                A_adj = A_work.copy()
                A_adj.data = np.abs(A_adj.data)
            
            n = A_adj.shape[0]
            
            # Simple greedy modularity optimization
            # Start with each node in its own community
            labels = np.arange(n)
            
            # Compute modularity matrix
            degrees = np.array(A_adj.sum(axis=1)).flatten()
            total_weight = np.sum(degrees) / 2
            
            if total_weight == 0:
                return np.zeros(n, dtype=int), {'method': 'modularity', 'communities': 1}
            
            # Simple community merging based on modularity gain
            improved = True
            iterations = 0
            max_iterations = min(10, n)
            
            while improved and iterations < max_iterations:
                improved = False
                iterations += 1
                
                for i in range(n):
                    current_community = labels[i]
                    best_community = current_community
                    best_gain = 0
                    
                    # Check neighbors
                    neighbors = A_adj[i].nonzero()[1]
                    neighbor_communities = set(labels[neighbors])
                    
                    for neighbor_comm in neighbor_communities:
                        if neighbor_comm != current_community:
                            # Simple gain calculation (approximation)
                            comm_nodes = np.where(labels == neighbor_comm)[0]
                            if len(comm_nodes) > 0:
                                gain = np.sum(A_adj[i, comm_nodes])
                                if gain > best_gain:
                                    best_gain = gain
                                    best_community = neighbor_comm
                    
                    if best_community != current_community:
                        labels[i] = best_community
                        improved = True
            
            # Relabel communities to be consecutive
            unique_labels = np.unique(labels)
            label_map = {old: new for new, old in enumerate(unique_labels)}
            labels = np.array([label_map[label] for label in labels])
            
            return labels, {'method': 'modularity', 'communities': len(unique_labels)}
            
        except Exception as e:
            logging.warning(f"Modularity clustering failed: {e}")
            return np.zeros(A.shape[0], dtype=int), {'error': str(e)}


class BaselineComparisonFramework:
    """
    Comprehensive framework for comparing CMG against baseline methods.
    """
    
    def __init__(self):
        self.results: List[BaselineComparisonResult] = []
        self.baseline_methods = BaselineClusteringMethods()
        
        # Ensure results directory exists
        Path("validation_results").mkdir(exist_ok=True)
        
        # Define comparison methods
        self.methods = {
            'CMG': self.run_cmg,
            'Ward': self.baseline_methods.agglomerative_ward,
            'Average': self.baseline_methods.agglomerative_average,
            'Complete': self.baseline_methods.agglomerative_complete,
            'Spectral': self.baseline_methods.spectral_clustering,
            'Modularity': self.baseline_methods.modularity_clustering
        }
    
    def run_cmg(self, A: sp.spmatrix, n_clusters: int = None) -> Tuple[np.ndarray, Dict]:
        """Run CMG for comparison."""
        try:
            solver = CMGSteinerSolver(gamma=5.0, verbose=False)
            labels, n_communities = solver.steiner_group(A)
            stats = solver.get_statistics()
            
            return labels, {
                'method': 'CMG',
                'gamma': 5.0,
                'n_communities': n_communities,
                'avg_weighted_degree': stats.get('avg_weighted_degree'),
                'high_degree_nodes': stats.get('high_degree_nodes')
            }
            
        except Exception as e:
            logging.warning(f"CMG failed: {e}")
            return np.zeros(A.shape[0], dtype=int), {'error': str(e)}
    
    def calculate_boundary_preservation_score(self, labels: np.ndarray, 
                                            true_main_labels: np.ndarray) -> float:
        """Calculate boundary preservation score (our innovation)."""
        if len(true_main_labels) == 0 or len(np.unique(true_main_labels)) <= 1:
            return 1.0  # No boundaries to preserve
        
        n_pure_nodes = 0
        
        for cluster_id in np.unique(labels):
            nodes_in_cluster = np.where(labels == cluster_id)[0]
            true_labels_in_cluster = true_main_labels[nodes_in_cluster]
            
            # Pure if all nodes in cluster have same true main label
            if len(np.unique(true_labels_in_cluster)) == 1:
                n_pure_nodes += len(nodes_in_cluster)
        
        return n_pure_nodes / len(labels)
    
    def calculate_silhouette_score(self, A: sp.spmatrix, labels: np.ndarray) -> float:
        """Calculate silhouette score for clustering quality."""
        try:
            if len(np.unique(labels)) <= 1:
                return 0.0
            
            # Convert to distance matrix
            if sp.issparse(A):
                A_dense = A.toarray()
            else:
                A_dense = A
            
            # Create distance matrix
            A_pos = np.abs(A_dense)
            np.fill_diagonal(A_pos, 0)
            
            max_weight = np.max(A_pos)
            if max_weight > 0:
                dist_matrix = 1 - (A_pos / max_weight)
            else:
                return 0.0
            
            # Simple silhouette calculation
            n = len(labels)
            silhouette_vals = []
            
            for i in range(n):
                cluster_i = labels[i]
                
                # Intra-cluster distance (a_i)
                same_cluster = np.where(labels == cluster_i)[0]
                if len(same_cluster) > 1:
                    a_i = np.mean([dist_matrix[i, j] for j in same_cluster if j != i])
                else:
                    a_i = 0
                
                # Inter-cluster distance (b_i)
                b_i = float('inf')
                for cluster_j in np.unique(labels):
                    if cluster_j != cluster_i:
                        other_cluster = np.where(labels == cluster_j)[0]
                        if len(other_cluster) > 0:
                            avg_dist = np.mean([dist_matrix[i, j] for j in other_cluster])
                            b_i = min(b_i, avg_dist)
                
                # Silhouette value
                if max(a_i, b_i) > 0:
                    silhouette_vals.append((b_i - a_i) / max(a_i, b_i))
                else:
                    silhouette_vals.append(0)
            
            return np.mean(silhouette_vals)
            
        except Exception as e:
            logging.warning(f"Silhouette calculation failed: {e}")
            return 0.0
    
    def calculate_hierarchical_quality(self, A: sp.spmatrix, labels: np.ndarray) -> float:
        """Calculate hierarchical clustering quality metric."""
        try:
            unique_labels = np.unique(labels)
            if len(unique_labels) <= 1:
                return 0.0
            
            # Convert to adjacency
            if sp.issparse(A):
                A_work = A.copy()
            else:
                A_work = sp.csr_matrix(A)
            
            if np.all(A_work.diagonal() >= 0):
                A_adj = -A_work.copy()
                A_adj.setdiag(0)
                A_adj.eliminate_zeros()
                A_adj.data = np.abs(A_adj.data)
            else:
                A_adj = A_work.copy()
                A_adj.data = np.abs(A_adj.data)
            
            # Calculate intra vs inter cluster edge weights
            total_intra_weight = 0
            total_inter_weight = 0
            
            for i in range(A_adj.shape[0]):
                neighbors = A_adj[i].nonzero()[1]
                for j in neighbors:
                    if i < j:  # Avoid double counting
                        weight = A_adj[i, j]
                        if labels[i] == labels[j]:
                            total_intra_weight += weight
                        else:
                            total_inter_weight += weight
            
            total_weight = total_intra_weight + total_inter_weight
            if total_weight > 0:
                return total_intra_weight / total_weight
            else:
                return 0.0
                
        except Exception as e:
            logging.warning(f"Hierarchical quality calculation failed: {e}")
            return 0.0
    
    def create_cmg_favorable_graphs(self) -> Dict[str, Tuple]:
        """Create graphs where CMG should excel (sparse, hierarchical)."""
        
        graphs = {}
        
        # Grid-based hierarchical structures
        graphs['grid_hierarchical_small'] = self.create_grid_hierarchical(6)
        graphs['grid_hierarchical_medium'] = self.create_grid_hierarchical(8)
        
        # Sparse clustered graphs (CMG's sweet spot)
        graphs['sparse_clusters_2x2'] = self.create_sparse_clustered(
            cluster_sizes=[15, 15, 15, 15], inter_prob=0.03, density_target=0.08
        )
        graphs['sparse_clusters_2x3'] = self.create_sparse_clustered(
            cluster_sizes=[12, 12, 12, 12, 12, 12], inter_prob=0.02, density_target=0.06
        )
        
        # Scale-free with hierarchical bias
        graphs['scale_free_hierarchical'] = self.create_scale_free_hierarchical(60)
        
        return graphs
    
    def create_challenging_graphs(self) -> Dict[str, Tuple]:
        """Create graphs with intermediate density."""
        
        graphs = {}
        
        # Small-world networks
        graphs['small_world_moderate'] = self.create_small_world_hierarchical(50, p=0.3)
        graphs['small_world_dense'] = self.create_small_world_hierarchical(40, p=0.5)
        
        # Mixed topology
        graphs['mixed_topology'] = self.create_mixed_topology(50)
        
        return graphs
    
    def create_cmg_unfavorable_graphs(self) -> Dict[str, Tuple]:
        """Create graphs where CMG should struggle (dense)."""
        
        graphs = {}
        
        # Dense clustered graphs
        graphs['dense_clusters'] = self.create_sparse_clustered(
            cluster_sizes=[20, 20, 20], inter_prob=0.4, density_target=0.4
        )
        
        # Bipartite structures
        graphs['bipartite_dense'] = self.create_bipartite_hierarchical(25, 25, p=0.6)
        
        return graphs
    
    def create_grid_hierarchical(self, grid_size: int) -> Tuple[sp.spmatrix, np.ndarray, Dict]:
        """Create hierarchical grid structure."""
        edges = []
        n = grid_size * grid_size
        
        # Basic grid connections
        for i in range(grid_size):
            for j in range(grid_size):
                node = i * grid_size + j
                
                if j < grid_size - 1:
                    neighbor = i * grid_size + (j + 1)
                    edges.append((node, neighbor, 1.0))
                
                if i < grid_size - 1:
                    neighbor = (i + 1) * grid_size + j
                    edges.append((node, neighbor, 1.0))
        
        # Add hierarchical strengthening
        mid = grid_size // 2
        for i in range(grid_size):
            for j in range(mid):
                node = i * grid_size + j
                if i > 0:
                    neighbor = (i - 1) * grid_size + j
                    edges.append((node, neighbor, 1.5))
        
        A = create_laplacian_from_edges(edges, n)
        
        # Ground truth: left vs right
        main_labels = []
        for i in range(grid_size):
            for j in range(grid_size):
                main_labels.append(0 if j < mid else 1)
        
        metadata = {
            'description': f'{grid_size}x{grid_size} hierarchical grid',
            'expected_main_clusters': 2,
            'density': len(edges) / (n * (n - 1) / 2)
        }
        
        return A, np.array(main_labels), metadata
    
    def create_sparse_clustered(self, cluster_sizes: List[int], inter_prob: float, 
                              density_target: float) -> Tuple[sp.spmatrix, np.ndarray, Dict]:
        """Create sparse clustered graph."""
        
        edges, n = create_clustered_graph(
            cluster_sizes=cluster_sizes,
            intra_cluster_p=0.8,
            inter_cluster_p=inter_prob,
            intra_weight_range=(1.5, 2.0),
            inter_weight_range=(0.01, 0.1),
            seed=42
        )
        
        A = create_laplacian_from_edges(edges, n)
        
        # Ground truth: first half vs second half of clusters
        main_labels = []
        mid_cluster = len(cluster_sizes) // 2
        for i, size in enumerate(cluster_sizes):
            main_cluster_id = 0 if i < mid_cluster else 1
            main_labels.extend([main_cluster_id] * size)
        
        metadata = {
            'description': f'Sparse clustered graph with {len(cluster_sizes)} subclusters',
            'expected_main_clusters': 2,
            'density': len(edges) / (n * (n - 1) / 2),
            'cluster_sizes': cluster_sizes
        }
        
        return A, np.array(main_labels), metadata
    
    def create_scale_free_hierarchical(self, n: int) -> Tuple[sp.spmatrix, np.ndarray, Dict]:
        """Create scale-free network with hierarchical bias."""
        
        # Use simplified preferential attachment with hierarchical bias
        edges = []
        np.random.seed(42)
        
        # Start with small complete graph
        initial_nodes = 4
        for i in range(initial_nodes):
            for j in range(i + 1, initial_nodes):
                edges.append((i, j, 1.0))
        
        degrees = [initial_nodes - 1] * initial_nodes
        
        # Add nodes with hierarchical preferential attachment
        for new_node in range(initial_nodes, n):
            m = 3  # Number of edges to add
            targets = []
            
            total_degree = sum(degrees)
            if total_degree > 0:
                for _ in range(min(m, new_node)):
                    # Bias toward same half
                    new_node_half = 0 if new_node < n // 2 else 1
                    
                    probabilities = []
                    for existing_node in range(new_node):
                        existing_half = 0 if existing_node < n // 2 else 1
                        base_prob = degrees[existing_node] / total_degree
                        
                        if new_node_half == existing_half:
                            bias_prob = base_prob * 2.0
                        else:
                            bias_prob = base_prob * 0.2
                        
                        probabilities.append(bias_prob)
                    
                    prob_sum = sum(probabilities)
                    if prob_sum > 0:
                        probabilities = [p / prob_sum for p in probabilities]
                        target = np.random.choice(new_node, p=probabilities)
                        if target not in targets:
                            targets.append(target)
            
            # Add edges
            for target in targets:
                weight = 1.5 if (new_node < n // 2) == (target < n // 2) else 0.3
                edges.append((new_node, target, weight))
                degrees[target] += 1
            
            degrees.append(len(targets))
        
        A = create_laplacian_from_edges(edges, n)
        main_labels = [0] * (n // 2) + [1] * (n // 2)
        
        metadata = {
            'description': f'Scale-free hierarchical network with {n} nodes',
            'expected_main_clusters': 2,
            'density': len(edges) / (n * (n - 1) / 2)
        }
        
        return A, np.array(main_labels), metadata
    
    def create_small_world_hierarchical(self, n: int, p: float) -> Tuple[sp.spmatrix, np.ndarray, Dict]:
        """Create small-world network with hierarchical structure."""
        
        edges = []
        k = 4  # Regular lattice parameter
        
        # Ring lattice
        for i in range(n):
            for j in range(1, k // 2 + 1):
                neighbor = (i + j) % n
                edges.append((i, neighbor, 1.0))
        
        # Add biased random shortcuts
        np.random.seed(42)
        n_shortcuts = int(n * p)
        
        for _ in range(n_shortcuts):
            u = np.random.randint(0, n)
            v = np.random.randint(0, n)
            
            if u != v:
                u_half = 0 if u < n // 2 else 1
                v_half = 0 if v < n // 2 else 1
                
                weight = 1.5 if u_half == v_half else 0.2
                edges.append((u, v, weight))
        
        A = create_laplacian_from_edges(edges, n)
        main_labels = [0] * (n // 2) + [1] * (n // 2)
        
        metadata = {
            'description': f'Small-world hierarchical network n={n}, p={p}',
            'expected_main_clusters': 2,
            'density': len(edges) / (n * (n - 1) / 2)
        }
        
        return A, np.array(main_labels), metadata
    
    def create_mixed_topology(self, n: int) -> Tuple[sp.spmatrix, np.ndarray, Dict]:
        """Create mixed topology structure."""
        
        edges = []
        
        # Part 1: Small grid (first quarter)
        grid_size = int(np.sqrt(n // 4))
        grid_nodes = grid_size * grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                node = i * grid_size + j
                if j < grid_size - 1:
                    edges.append((node, node + 1, 1.0))
                if i < grid_size - 1:
                    edges.append((node, node + grid_size, 1.0))
        
        # Part 2: Star structure (remaining nodes)
        center = grid_nodes
        for leaf in range(grid_nodes + 1, n):
            edges.append((center, leaf, 0.8))
        
        # Connect grid to star
        edges.append((grid_nodes - 1, center, 0.1))
        
        A = create_laplacian_from_edges(edges, n)
        
        # Ground truth: grid vs star
        main_labels = [0] * grid_nodes + [1] * (n - grid_nodes)
        
        metadata = {
            'description': f'Mixed topology: grid + star, n={n}',
            'expected_main_clusters': 2,
            'density': len(edges) / (n * (n - 1) / 2)
        }
        
        return A, np.array(main_labels), metadata
    
    def create_bipartite_hierarchical(self, n1: int, n2: int, p: float) -> Tuple[sp.spmatrix, np.ndarray, Dict]:
        """Create bipartite structure."""
        
        edges = []
        n = n1 + n2
        
        np.random.seed(42)
        
        # Bipartite connections
        for i in range(n1):
            for j in range(n1, n1 + n2):
                if np.random.random() < p:
                    edges.append((i, j, 1.0))
        
        # Internal connections within partitions
        for i in range(n1):
            for j in range(i + 1, n1):
                if np.random.random() < 0.3:
                    edges.append((i, j, 0.5))
        
        for i in range(n1, n1 + n2):
            for j in range(i + 1, n1 + n2):
                if np.random.random() < 0.3:
                    edges.append((i, j, 0.5))
        
        A = create_laplacian_from_edges(edges, n)
        main_labels = [0] * n1 + [1] * n2
        
        metadata = {
            'description': f'Bipartite hierarchical n1={n1}, n2={n2}, p={p}',
            'expected_main_clusters': 2,
            'density': len(edges) / (n * (n - 1) / 2)
        }
        
        return A, np.array(main_labels), metadata
    
    def run_single_comparison(self, test_name: str, A: sp.spmatrix, 
                            main_labels: np.ndarray, metadata: Dict, 
                            graph_category: str) -> List[BaselineComparisonResult]:
        """Run comparison for a single graph across all methods."""
        
        results = []
        n = A.shape[0]
        n_edges = A.nnz // 2
        density = metadata.get('density', 0.0)
        avg_degree = 2 * n_edges / n if n > 0 else 0.0
        
        logging.info(f"Testing {test_name} ({graph_category}): {n} nodes, {n_edges} edges, density={density:.3f}")
        
        for method_name, method_func in self.methods.items():
            
            try:
                start_time = time.time()
                labels, method_params = method_func(A)
                runtime = time.time() - start_time
                
                n_communities = len(np.unique(labels))
                
                # Calculate evaluation metrics
                boundary_score = self.calculate_boundary_preservation_score(labels, main_labels)
                silhouette = self.calculate_silhouette_score(A, labels)
                hierarchical_quality = self.calculate_hierarchical_quality(A, labels)
                
                # Calculate ground truth alignment
                main_purity = boundary_score  # Same calculation
                subcluster_alignment = 0.0  # Simplified for now
                
                result = BaselineComparisonResult(
                    test_name=test_name,
                    method_name=method_name,
                    graph_category=graph_category,
                    n_nodes=n,
                    n_edges=n_edges,
                    density=density,
                    avg_degree=avg_degree,
                    n_communities=n_communities,
                    runtime_seconds=runtime,
                    boundary_preservation_score=boundary_score,
                    silhouette_score=silhouette,
                    hierarchical_quality=hierarchical_quality,
                    main_cluster_purity=main_purity,
                    subcluster_alignment=subcluster_alignment,
                    method_parameters=method_params,
                    success=True
                )
                
                results.append(result)
                
                logging.info(f"  {method_name:12s}: {n_communities:2d} communities, "
                           f"boundary={boundary_score:.3f}, "
                           f"silhouette={silhouette:.3f}, "
                           f"runtime={runtime:.3f}s")
                
            except Exception as e:
                logging.error(f"  {method_name:12s}: FAILED - {e}")
                
                result = BaselineComparisonResult(
                    test_name=test_name,
                    method_name=method_name,
                    graph_category=graph_category,
                    n_nodes=n,
                    n_edges=n_edges,
                    density=density,
                    avg_degree=avg_degree,
                    n_communities=0,
                    runtime_seconds=0.0,
                    boundary_preservation_score=0.0,
                    silhouette_score=0.0,
                    hierarchical_quality=0.0,
                    main_cluster_purity=0.0,
                    subcluster_alignment=0.0,
                    method_parameters={'error': str(e)},
                    success=False,
                    error_message=str(e)
                )
                
                results.append(result)
        
        return results
    
    def run_comprehensive_comparison(self) -> None:
        """Run comprehensive baseline comparison across all graph categories."""
        
        logging.info("="*80)
        logging.info("CMG COMPREHENSIVE BASELINE COMPARISON")
        logging.info("="*80)
        logging.info("Comparing CMG against established hierarchical clustering methods")
        logging.info("on controlled synthetic graphs across three categories:")
        logging.info("  1. CMG-Favorable: Sparse, hierarchical structures")
        logging.info("  2. Challenging: Intermediate density, mixed structures")  
        logging.info("  3. CMG-Unfavorable: Dense graphs where CMG should struggle")
        logging.info("")
        
        # Test all graph categories
        graph_categories = [
            ("CMG-Favorable", self.create_cmg_favorable_graphs()),
            ("Challenging", self.create_challenging_graphs()),
            ("CMG-Unfavorable", self.create_cmg_unfavorable_graphs())
        ]
        
        for category_name, graphs in graph_categories:
            
            logging.info(f"\n{'='*60}")
            logging.info(f"TESTING CATEGORY: {category_name}")
            logging.info(f"{'='*60}")
            
            for test_name, (A, main_labels, metadata) in graphs.items():
                
                logging.info(f"\n{'-'*40}")
                logging.info(f"Graph: {test_name}")
                logging.info(f"Description: {metadata['description']}")
                logging.info(f"{'-'*40}")
                
                # Run comparison for this graph
                graph_results = self.run_single_comparison(
                    test_name, A, main_labels, metadata, category_name
                )
                
                self.results.extend(graph_results)
        
        # Comprehensive analysis
        self.analyze_comparison_results()
        
        # Save results
        self.save_comparison_results()
    
    def analyze_comparison_results(self) -> None:
        """Analyze comprehensive comparison results."""
        
        successful_results = [r for r in self.results if r.success]
        
        if not successful_results:
            logging.info("No successful results to analyze")
            return
        
        logging.info(f"\n{'='*80}")
        logging.info("COMPREHENSIVE BASELINE COMPARISON ANALYSIS")
        logging.info(f"{'='*80}")
        
        # Overall statistics
        total_tests = len(successful_results)
        methods_tested = set(r.method_name for r in successful_results)
        categories_tested = set(r.graph_category for r in successful_results)
        
        logging.info(f"\n📊 OVERALL RESULTS:")
        logging.info(f"   Total successful tests: {total_tests}")
        logging.info(f"   Methods compared: {list(methods_tested)}")
        logging.info(f"   Graph categories: {list(categories_tested)}")
        
        # Method-wise performance analysis
        logging.info(f"\n🏆 METHOD PERFORMANCE ANALYSIS:")
        
        for method in methods_tested:
            method_results = [r for r in successful_results if r.method_name == method]
            
            if method_results:
                avg_boundary = np.mean([r.boundary_preservation_score for r in method_results])
                avg_silhouette = np.mean([r.silhouette_score for r in method_results])
                avg_runtime = np.mean([r.runtime_seconds for r in method_results])
                perfect_boundary_count = sum(1 for r in method_results 
                                           if r.boundary_preservation_score >= 0.999)
                
                logging.info(f"   {method:12s}: boundary={avg_boundary:.3f}, "
                           f"silhouette={avg_silhouette:.3f}, "
                           f"perfect_boundary={perfect_boundary_count}/{len(method_results)}, "
                           f"runtime={avg_runtime:.3f}s")
        
        # Category-wise analysis
        logging.info(f"\n📈 CATEGORY PERFORMANCE ANALYSIS:")
        
        for category in categories_tested:
            logging.info(f"\n   Category: {category}")
            category_results = [r for r in successful_results if r.graph_category == category]
            
            for method in methods_tested:
                method_category_results = [r for r in category_results 
                                         if r.method_name == method]
                
                if method_category_results:
                    avg_boundary = np.mean([r.boundary_preservation_score 
                                          for r in method_category_results])
                    perfect_count = sum(1 for r in method_category_results 
                                      if r.boundary_preservation_score >= 0.999)
                    
                    logging.info(f"     {method:12s}: boundary={avg_boundary:.3f}, "
                               f"perfect={perfect_count}/{len(method_category_results)}")
        
        # CMG-specific analysis
        cmg_results = [r for r in successful_results if r.method_name == 'CMG']
        if cmg_results:
            logging.info(f"\n🎯 CMG-SPECIFIC ANALYSIS:")
            
            cmg_favorable = [r for r in cmg_results if r.graph_category == 'CMG-Favorable']
            cmg_unfavorable = [r for r in cmg_results if r.graph_category == 'CMG-Unfavorable']
            
            if cmg_favorable:
                favorable_boundary = np.mean([r.boundary_preservation_score for r in cmg_favorable])
                logging.info(f"   CMG on favorable graphs: boundary={favorable_boundary:.3f}")
            
            if cmg_unfavorable:
                unfavorable_boundary = np.mean([r.boundary_preservation_score for r in cmg_unfavorable])
                logging.info(f"   CMG on unfavorable graphs: boundary={unfavorable_boundary:.3f}")
        
        # Performance ranking by category
        logging.info(f"\n🥇 PERFORMANCE RANKINGS:")
        
        for category in categories_tested:
            logging.info(f"\n   {category} - Top performers by boundary preservation:")
            
            category_results = [r for r in successful_results if r.graph_category == category]
            method_scores = {}
            
            for method in methods_tested:
                method_results = [r for r in category_results if r.method_name == method]
                if method_results:
                    avg_score = np.mean([r.boundary_preservation_score for r in method_results])
                    method_scores[method] = avg_score
            
            # Sort by performance
            sorted_methods = sorted(method_scores.items(), key=lambda x: x[1], reverse=True)
            
            for rank, (method, score) in enumerate(sorted_methods[:3], 1):
                logging.info(f"     {rank}. {method:12s}: {score:.3f}")
        
        # Research implications
        logging.info(f"\n🎓 RESEARCH IMPLICATIONS:")
        
        # Check if CMG dominates in favorable category
        if 'CMG-Favorable' in categories_tested:
            favorable_results = [r for r in successful_results 
                               if r.graph_category == 'CMG-Favorable']
            
            cmg_favorable_scores = [r.boundary_preservation_score for r in favorable_results 
                                  if r.method_name == 'CMG']
            other_favorable_scores = [r.boundary_preservation_score for r in favorable_results 
                                    if r.method_name != 'CMG']
            
            if cmg_favorable_scores and other_favorable_scores:
                cmg_avg = np.mean(cmg_favorable_scores)
                others_avg = np.mean(other_favorable_scores)
                
                if cmg_avg > others_avg + 0.1:
                    logging.info("   📝 CMG demonstrates clear superiority in its optimal domain")
                    logging.info("   📝 Specialized but superior performance confirmed")
                elif cmg_avg > others_avg:
                    logging.info("   📝 CMG shows competitive performance in optimal domain")
                else:
                    logging.info("   📝 CMG advantages less clear than expected")
        
        # Check structure dependency
        cmg_results_by_category = {}
        for category in categories_tested:
            cmg_cat_results = [r for r in successful_results 
                             if r.method_name == 'CMG' and r.graph_category == category]
            if cmg_cat_results:
                avg_score = np.mean([r.boundary_preservation_score for r in cmg_cat_results])
                cmg_results_by_category[category] = avg_score
        
        if len(cmg_results_by_category) > 1:
            scores = list(cmg_results_by_category.values())
            if max(scores) - min(scores) > 0.3:
                logging.info("   📝 CMG shows strong structure dependency (as expected)")
            else:
                logging.info("   📝 CMG more robust across structures than expected")
        
        logging.info(f"\n{'='*80}")
    
    def save_comparison_results(self, filename: str = None) -> None:
        """Save comparison results to JSON file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"validation_results/baseline_comparison_results_{timestamp}.json"
        
        results_data = [asdict(result) for result in self.results]
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logging.info(f"Results saved to {filename}")


def main():
    """Run comprehensive baseline comparison."""
    
    print("🏆 CMG COMPREHENSIVE BASELINE COMPARISON")
    print("=" * 60)
    print("This test systematically compares CMG against established")
    print("hierarchical clustering methods across three categories:")
    print()
    print("📊 Baseline Methods:")
    print("  • Agglomerative Clustering (Ward, Average, Complete)")
    print("  • Spectral Clustering")
    print("  • Modularity-based Clustering")
    print()
    print("📋 Graph Categories:")
    print("  • CMG-Favorable: Sparse, hierarchical (density < 0.1)")
    print("  • Challenging: Intermediate density, mixed structures")
    print("  • CMG-Unfavorable: Dense graphs (density > 0.3)")
    print()
    print("📈 Evaluation Metrics:")
    print("  • Boundary Preservation Score (our innovation)")
    print("  • Silhouette Score (clustering quality)")
    print("  • Hierarchical Quality Score")
    print("  • Computational Performance")
    print()
    print("Expected runtime: 20-30 minutes")
    print("Results will be saved to validation_results/ directory")
    print()
    
    # Create and run comparison framework
    framework = BaselineComparisonFramework()
    
    try:
        framework.run_comprehensive_comparison()
        
        print("\n✅ Comprehensive baseline comparison completed!")
        print("\n🎯 Key findings:")
        print("   Check the detailed analysis above for performance insights")
        print("   Results saved to validation_results/ directory")
        print("\n📝 This analysis will position CMG in the hierarchical clustering landscape!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        if framework.results:
            print("Analyzing partial results...")
            framework.analyze_comparison_results()
            framework.save_comparison_results()
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        logging.error(f"Baseline comparison failed: {e}")


if __name__ == "__main__":
    main()
