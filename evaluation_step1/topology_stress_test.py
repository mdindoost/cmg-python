#!/usr/bin/env python3
"""
CMG Topology Stress Test
=========================

Systematically tests CMG's hierarchical clustering behavior across diverse graph topologies
to understand how algorithm performance varies with structural properties.

This test explores:
1. Classical graph topologies (grid, ring, tree, star, etc.)
2. Real-world network structures (small-world, scale-free, etc.) 
3. Hierarchical arrangements of different base topologies
4. Mixed and hybrid topological structures
5. Sparse vs dense connectivity patterns

Goal: Understand CMG's behavior across the full spectrum of graph structures.

Run with: python topology_stress_test.py
"""

import sys
sys.path.append('..')

import numpy as np
import time
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass, asdict
import json

# CMG imports
from cmg import CMGSteinerSolver, create_laplacian_from_edges
from cmg.utils.graph_utils import create_clustered_graph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('validation_results/topology_stress_test.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class TopologyTestResult:
    """Result of topology stress test."""
    topology_name: str
    topology_type: str
    n_nodes: int
    n_edges: int
    n_communities: int
    main_cluster_purity: float
    boundary_preserved: bool
    runtime_seconds: float
    
    # Topological properties
    avg_degree: float
    density: float
    structure_description: str
    expected_behavior: str
    
    # CMG-specific metrics
    gamma: float
    avg_weighted_degree: Optional[float] = None
    high_degree_nodes: Optional[int] = None
    forest_edges_initial: Optional[int] = None
    forest_edges_final: Optional[int] = None
    
    # Ground truth comparison (when available)
    expected_main_clusters: Optional[int] = None
    expected_subclusters: Optional[int] = None
    subcluster_alignment: Optional[float] = None
    
    success: bool = True
    error_message: Optional[str] = None


class TopologyStressTester:
    """
    Comprehensive topology testing for CMG across diverse graph structures.
    """
    
    def __init__(self):
        self.results: List[TopologyTestResult] = []
        self.gamma = 5.0  # Fixed gamma since insensitivity confirmed
        
        # Ensure results directory exists
        Path("validation_results").mkdir(exist_ok=True)
        
    def calculate_main_cluster_purity(self, cmg_labels: np.ndarray, 
                                    true_main_labels: np.ndarray) -> float:
        """Calculate main cluster purity."""
        if len(true_main_labels) == 0:
            return 1.0  # No ground truth available
            
        n_pure_nodes = 0
        
        for cmg_cluster_id in np.unique(cmg_labels):
            nodes_in_cmg = np.where(cmg_labels == cmg_cluster_id)[0]
            true_labels_in_cmg = true_main_labels[nodes_in_cmg]
            
            if len(np.unique(true_labels_in_cmg)) == 1:
                n_pure_nodes += len(nodes_in_cmg)
        
        return n_pure_nodes / len(cmg_labels)
    
    def create_grid_topology(self, grid_size: int = 8) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Create hierarchical grid topology."""
        
        edges = []
        n = grid_size * grid_size
        
        # Create grid connections
        for i in range(grid_size):
            for j in range(grid_size):
                node = i * grid_size + j
                
                # Right neighbor
                if j < grid_size - 1:
                    neighbor = i * grid_size + (j + 1)
                    edges.append((node, neighbor, 1.0))
                
                # Bottom neighbor  
                if i < grid_size - 1:
                    neighbor = (i + 1) * grid_size + j
                    edges.append((node, neighbor, 1.0))
        
        # Add weak connections across middle to create hierarchy
        mid = grid_size // 2
        
        # Strengthen left half
        for i in range(grid_size):
            for j in range(mid):
                node = i * grid_size + j
                # Add extra internal connections
                if i > 0:
                    neighbor = (i - 1) * grid_size + j
                    edges.append((node, neighbor, 2.0))
                if j > 0:
                    neighbor = i * grid_size + (j - 1) 
                    edges.append((node, neighbor, 2.0))
        
        # Strengthen right half
        for i in range(grid_size):
            for j in range(mid, grid_size):
                node = i * grid_size + j
                # Add extra internal connections
                if i > 0:
                    neighbor = (i - 1) * grid_size + j
                    edges.append((node, neighbor, 2.0))
                if j < grid_size - 1:
                    neighbor = i * grid_size + (j + 1)
                    edges.append((node, neighbor, 2.0))
        
        A = create_laplacian_from_edges(edges, n)
        
        # Ground truth: left half vs right half
        main_labels = []
        for i in range(grid_size):
            for j in range(grid_size):
                main_labels.append(0 if j < mid else 1)
        
        metadata = {
            'structure_description': f'{grid_size}x{grid_size} grid with hierarchical strengthening',
            'expected_behavior': 'Split into left/right halves',
            'expected_main_clusters': 2,
            'grid_size': grid_size
        }
        
        return A, np.array(main_labels), metadata
    
    def create_ring_topology(self, ring_size: int = 40) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Create hierarchical ring topology."""
        
        edges = []
        
        # Basic ring
        for i in range(ring_size):
            next_node = (i + 1) % ring_size
            edges.append((i, next_node, 1.0))
        
        # Add shortcuts to create hierarchy
        # Strengthen first half of ring
        for i in range(ring_size // 2):
            next_node = (i + 2) % (ring_size // 2)  # Skip connections within first half
            edges.append((i, next_node, 2.0))
        
        # Strengthen second half of ring
        offset = ring_size // 2
        for i in range(ring_size // 2):
            node = offset + i
            next_node = offset + ((i + 2) % (ring_size // 2))
            edges.append((node, next_node, 2.0))
        
        A = create_laplacian_from_edges(edges, ring_size)
        
        # Ground truth: first half vs second half
        main_labels = [0] * (ring_size // 2) + [1] * (ring_size // 2)
        
        metadata = {
            'structure_description': f'Ring of {ring_size} nodes with hierarchical shortcuts',
            'expected_behavior': 'Split ring into two halves',
            'expected_main_clusters': 2,
            'ring_size': ring_size
        }
        
        return A, np.array(main_labels), metadata
    
    def create_tree_topology(self, depth: int = 4, branching: int = 3) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Create hierarchical tree topology."""
        
        edges = []
        n = sum(branching ** i for i in range(depth + 1))  # Total nodes in tree
        
        # Build tree level by level
        node_id = 0
        level_starts = [0]
        
        for level in range(depth):
            level_start = level_starts[level]
            level_size = branching ** level
            next_level_start = node_id + level_size
            level_starts.append(next_level_start)
            
            # Connect each node in current level to its children
            for i in range(level_size):
                parent = level_start + i
                for child_idx in range(branching):
                    if next_level_start + i * branching + child_idx < n:
                        child = next_level_start + i * branching + child_idx
                        edges.append((parent, child, 1.0))
        
        # Add hierarchy by strengthening left vs right subtrees
        if depth >= 2:
            # Strengthen connections in left subtree of root
            left_subtree_start = 1  # Root's left child
            left_subtree_size = (n - 1) // 2
            
            for i in range(left_subtree_start, min(left_subtree_start + left_subtree_size, n)):
                for j in range(i + 1, min(left_subtree_start + left_subtree_size, n)):
                    if abs(i - j) <= branching:  # Connect nearby nodes
                        edges.append((i, j, 2.0))
        
        A = create_laplacian_from_edges(edges, n)
        
        # Ground truth: left subtree vs right subtree (when depth >= 2)
        if depth >= 2:
            main_labels = []
            for i in range(n):
                if i == 0:  # Root
                    main_labels.append(0)
                elif i <= n // 2:
                    main_labels.append(0)  # Left subtree
                else:
                    main_labels.append(1)  # Right subtree
        else:
            main_labels = [0] * n  # All same cluster for small trees
        
        metadata = {
            'structure_description': f'Tree depth={depth}, branching={branching}',
            'expected_behavior': 'Split into left/right subtrees' if depth >= 2 else 'Single cluster',
            'expected_main_clusters': 2 if depth >= 2 else 1,
            'tree_depth': depth,
            'branching_factor': branching
        }
        
        return A, np.array(main_labels), metadata
    
    def create_star_topology(self, n_leaves: int = 30) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Create star topology with hierarchical structure."""
        
        edges = []
        n = n_leaves + 1  # leaves + center
        
        # Basic star: center connected to all leaves
        center = 0
        for leaf in range(1, n):
            edges.append((center, leaf, 1.0))
        
        # Add hierarchy by creating two groups of leaves
        mid = n // 2
        
        # Group 1: leaves 1 to mid
        for i in range(1, mid):
            for j in range(i + 1, mid):
                edges.append((i, j, 0.5))  # Weak connections within group 1
        
        # Group 2: leaves mid to n
        for i in range(mid, n):
            for j in range(i + 1, n):
                edges.append((i, j, 0.5))  # Weak connections within group 2
        
        A = create_laplacian_from_edges(edges, n)
        
        # Ground truth: center + group 1 vs group 2
        main_labels = [0] + [0] * (mid - 1) + [1] * (n - mid)
        
        metadata = {
            'structure_description': f'Star with {n_leaves} leaves, hierarchically grouped',
            'expected_behavior': 'Split star into two leaf groups',
            'expected_main_clusters': 2,
            'n_leaves': n_leaves
        }
        
        return A, np.array(main_labels), metadata
    
    def create_small_world_topology(self, n: int = 50, k: int = 4, p: float = 0.3) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Create small-world topology with hierarchical structure."""
        
        edges = []
        
        # Start with ring lattice
        for i in range(n):
            for j in range(1, k // 2 + 1):
                neighbor = (i + j) % n
                edges.append((i, neighbor, 1.0))
        
        # Rewire with probability p, but create hierarchy
        np.random.seed(42)  # Reproducible
        
        # Add random shortcuts, but bias toward creating two main clusters
        n_shortcuts = int(n * p)
        
        for _ in range(n_shortcuts):
            u = np.random.randint(0, n)
            v = np.random.randint(0, n)
            
            if u != v:
                # Bias shortcuts to be within the same half
                u_half = 0 if u < n // 2 else 1
                v_half = 0 if v < n // 2 else 1
                
                if u_half == v_half:
                    weight = 2.0  # Strengthen intra-half connections
                else:
                    weight = 0.1  # Weaken inter-half connections
                
                edges.append((u, v, weight))
        
        A = create_laplacian_from_edges(edges, n)
        
        # Ground truth: first half vs second half
        main_labels = [0] * (n // 2) + [1] * (n // 2)
        
        metadata = {
            'structure_description': f'Small-world n={n}, k={k}, p={p} with hierarchical bias',
            'expected_behavior': 'Split into two halves despite small-world properties',
            'expected_main_clusters': 2,
            'small_world_params': {'n': n, 'k': k, 'p': p}
        }
        
        return A, np.array(main_labels), metadata
    
    def create_scale_free_topology(self, n: int = 50, m: int = 3) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Create scale-free topology with hierarchical structure."""
        
        edges = []
        
        # Simplified preferential attachment with hierarchical bias
        np.random.seed(42)  # Reproducible
        
        # Start with a small complete graph
        initial_nodes = min(m + 1, n)
        for i in range(initial_nodes):
            for j in range(i + 1, initial_nodes):
                edges.append((i, j, 1.0))
        
        # Track node degrees for preferential attachment
        degrees = [initial_nodes - 1] * initial_nodes
        
        # Add remaining nodes with preferential attachment
        for new_node in range(initial_nodes, n):
            # Choose m existing nodes to connect to based on degree
            total_degree = sum(degrees)
            
            if total_degree == 0:
                # Connect to random nodes if no degrees yet
                targets = list(range(min(m, new_node)))
            else:
                # Preferential attachment with hierarchical bias
                targets = []
                for _ in range(min(m, new_node)):
                    # Bias toward same half of the network
                    new_node_half = 0 if new_node < n // 2 else 1
                    
                    probabilities = []
                    for existing_node in range(new_node):
                        existing_half = 0 if existing_node < n // 2 else 1
                        base_prob = degrees[existing_node] / total_degree
                        
                        if new_node_half == existing_half:
                            bias_prob = base_prob * 3.0  # Prefer same half
                        else:
                            bias_prob = base_prob * 0.3  # Discourage different half
                        
                        probabilities.append(bias_prob)
                    
                    # Normalize probabilities
                    prob_sum = sum(probabilities)
                    if prob_sum > 0:
                        probabilities = [p / prob_sum for p in probabilities]
                        
                        # Sample target
                        target = np.random.choice(new_node, p=probabilities)
                        if target not in targets:
                            targets.append(target)
            
            # Add edges to selected targets
            for target in targets:
                weight = 2.0 if (new_node < n // 2) == (target < n // 2) else 0.5
                edges.append((new_node, target, weight))
                degrees[target] += 1
            
            degrees.append(len(targets))
        
        A = create_laplacian_from_edges(edges, n)
        
        # Ground truth: first half vs second half
        main_labels = [0] * (n // 2) + [1] * (n // 2)
        
        metadata = {
            'structure_description': f'Scale-free n={n}, m={m} with hierarchical bias',
            'expected_behavior': 'Split despite scale-free structure',
            'expected_main_clusters': 2,
            'scale_free_params': {'n': n, 'm': m}
        }
        
        return A, np.array(main_labels), metadata
    
    def create_bipartite_topology(self, n1: int = 20, n2: int = 30, p: float = 0.3) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Create bipartite topology."""
        
        edges = []
        n = n1 + n2
        
        # Create bipartite connections
        np.random.seed(42)
        
        for i in range(n1):
            for j in range(n1, n1 + n2):
                if np.random.random() < p:
                    edges.append((i, j, 1.0))
        
        # Add some internal structure within each partition
        # Partition 1 internal connections
        for i in range(n1):
            for j in range(i + 1, n1):
                if np.random.random() < 0.2:  # Sparse internal
                    edges.append((i, j, 0.5))
        
        # Partition 2 internal connections
        for i in range(n1, n1 + n2):
            for j in range(i + 1, n1 + n2):
                if np.random.random() < 0.2:  # Sparse internal
                    edges.append((i, j, 0.5))
        
        A = create_laplacian_from_edges(edges, n)
        
        # Ground truth: partition 1 vs partition 2
        main_labels = [0] * n1 + [1] * n2
        
        metadata = {
            'structure_description': f'Bipartite n1={n1}, n2={n2}, p={p}',
            'expected_behavior': 'Split into two partitions',
            'expected_main_clusters': 2,
            'bipartite_params': {'n1': n1, 'n2': n2, 'p': p}
        }
        
        return A, np.array(main_labels), metadata
    
    def test_single_topology(self, topology_name: str, create_func, 
                           expected_main_clusters: int = None) -> TopologyTestResult:
        """Test CMG on a single topology."""
        
        try:
            # Create topology
            A, main_labels, metadata = create_func()
            
            n = A.shape[0]
            n_edges = A.nnz // 2  # Undirected graph
            
            # Calculate basic graph properties
            degrees = np.array(A.sum(axis=1)).flatten()
            avg_degree = np.mean(degrees)
            density = n_edges / (n * (n - 1) / 2) if n > 1 else 0.0
            
            # Test CMG
            solver = CMGSteinerSolver(gamma=self.gamma, verbose=False)
            start_time = time.time()
            cmg_labels, n_communities = solver.steiner_group(A)
            runtime = time.time() - start_time
            
            # Get CMG statistics
            stats = solver.get_statistics()
            
            # Calculate hierarchical metrics
            if len(main_labels) > 0 and len(np.unique(main_labels)) > 1:
                main_purity = self.calculate_main_cluster_purity(cmg_labels, main_labels)
                boundary_preserved = main_purity >= 0.999
            else:
                main_purity = 1.0  # No hierarchy to preserve
                boundary_preserved = True
            
            # Create result
            result = TopologyTestResult(
                topology_name=topology_name,
                topology_type=metadata.get('structure_description', 'Unknown'),
                n_nodes=n,
                n_edges=n_edges,
                n_communities=n_communities,
                main_cluster_purity=main_purity,
                boundary_preserved=boundary_preserved,
                runtime_seconds=runtime,
                
                # Topological properties
                avg_degree=avg_degree,
                density=density,
                structure_description=metadata.get('structure_description', ''),
                expected_behavior=metadata.get('expected_behavior', ''),
                
                # CMG-specific metrics
                gamma=self.gamma,
                avg_weighted_degree=stats.get('avg_weighted_degree'),
                high_degree_nodes=stats.get('high_degree_nodes'),
                forest_edges_initial=stats.get('forest_edges_initial'),
                forest_edges_final=stats.get('forest_edges_final'),
                
                # Ground truth comparison
                expected_main_clusters=metadata.get('expected_main_clusters'),
                success=True
            )
            
            return result
            
        except Exception as e:
            logging.error(f"Failed to test topology {topology_name}: {e}")
            
            return TopologyTestResult(
                topology_name=topology_name,
                topology_type="Failed",
                n_nodes=0,
                n_edges=0,
                n_communities=0,
                main_cluster_purity=0.0,
                boundary_preserved=False,
                runtime_seconds=0.0,
                avg_degree=0.0,
                density=0.0,
                structure_description="Test failed",
                expected_behavior="Unknown",
                gamma=self.gamma,
                success=False,
                error_message=str(e)
            )
    
    def run_topology_stress_test(self) -> None:
        """Run comprehensive topology stress test."""
        
        logging.info("="*80)
        logging.info("CMG TOPOLOGY STRESS TEST")
        logging.info("="*80)
        logging.info("Testing CMG across diverse graph topologies and structures...")
        logging.info("")
        
        # Define topology test suite
        topology_tests = [
            ("Grid 6x6", lambda: self.create_grid_topology(6)),
            ("Grid 8x8", lambda: self.create_grid_topology(8)),
            ("Ring 40", lambda: self.create_ring_topology(40)),
            ("Ring 60", lambda: self.create_ring_topology(60)),
            ("Binary Tree d=3", lambda: self.create_tree_topology(3, 2)),
            ("Ternary Tree d=3", lambda: self.create_tree_topology(3, 3)),
            ("Tree d=4 b=2", lambda: self.create_tree_topology(4, 2)),
            ("Star 30 leaves", lambda: self.create_star_topology(30)),
            ("Star 50 leaves", lambda: self.create_star_topology(50)),
            ("Small World", lambda: self.create_small_world_topology(50, 4, 0.3)),
            ("Small World Dense", lambda: self.create_small_world_topology(40, 6, 0.5)),
            ("Scale Free", lambda: self.create_scale_free_topology(50, 3)),
            ("Scale Free Dense", lambda: self.create_scale_free_topology(40, 5)),
            ("Bipartite", lambda: self.create_bipartite_topology(20, 30, 0.3)),
            ("Bipartite Dense", lambda: self.create_bipartite_topology(25, 25, 0.6))
        ]
        
        # Run all topology tests
        for topology_name, create_func in topology_tests:
            
            logging.info(f"\n{'='*60}")
            logging.info(f"TESTING: {topology_name}")
            logging.info(f"{'='*60}")
            
            result = self.test_single_topology(topology_name, create_func)
            self.results.append(result)
            
            if result.success:
                logging.info(f"✅ {topology_name}:")
                logging.info(f"   Nodes: {result.n_nodes}, Edges: {result.n_edges}")
                logging.info(f"   Communities: {result.n_communities}")
                logging.info(f"   Purity: {result.main_cluster_purity:.6f}")
                logging.info(f"   Boundary preserved: {result.boundary_preserved}")
                logging.info(f"   Avg degree: {result.avg_degree:.2f}, Density: {result.density:.4f}")
                logging.info(f"   Runtime: {result.runtime_seconds:.3f}s")
                logging.info(f"   Expected: {result.expected_behavior}")
            else:
                logging.error(f"❌ {topology_name}: {result.error_message}")
        
        # Comprehensive analysis
        self.analyze_topology_results()
        
        # Save results
        self.save_topology_results()
    
    def analyze_topology_results(self) -> None:
        """Analyze topology test results comprehensively."""
        
        successful_results = [r for r in self.results if r.success]
        
        if not successful_results:
            logging.info("No successful topology tests to analyze")
            return
        
        logging.info(f"\n{'='*80}")
        logging.info("TOPOLOGY STRESS TEST ANALYSIS")
        logging.info(f"{'='*80}")
        
        # Overall statistics
        total_tests = len(successful_results)
        perfect_boundary_count = sum(1 for r in successful_results if r.boundary_preserved)
        
        logging.info(f"\n📊 OVERALL RESULTS:")
        logging.info(f"   Total successful tests: {total_tests}")
        logging.info(f"   Perfect boundary preservation: {perfect_boundary_count}/{total_tests} "
                    f"({perfect_boundary_count/total_tests:.1%})")
        
        # Purity analysis
        purities = [r.main_cluster_purity for r in successful_results]
        min_purity = min(purities)
        max_purity = max(purities)
        avg_purity = np.mean(purities)
        
        logging.info(f"\n📈 BOUNDARY PRESERVATION ANALYSIS:")
        logging.info(f"   Minimum purity: {min_purity:.6f}")
        logging.info(f"   Maximum purity: {max_purity:.6f}")
        logging.info(f"   Average purity: {avg_purity:.6f}")
        
        # Topology-specific analysis
        logging.info(f"\n🏗️  TOPOLOGY-SPECIFIC PERFORMANCE:")
        
        topology_groups = {}
        for result in successful_results:
            topology_type = result.topology_name.split()[0]  # Grid, Ring, Tree, etc.
            if topology_type not in topology_groups:
                topology_groups[topology_type] = []
            topology_groups[topology_type].append(result)
        
        for topology_type, group_results in topology_groups.items():
            group_purities = [r.main_cluster_purity for r in group_results]
            group_avg_purity = np.mean(group_purities)
            group_boundary_count = sum(1 for r in group_results if r.boundary_preserved)
            
            logging.info(f"   {topology_type}: {group_boundary_count}/{len(group_results)} perfect, "
                        f"avg_purity={group_avg_purity:.3f}")
        
        # Structure vs performance analysis
        logging.info(f"\n📐 STRUCTURE vs PERFORMANCE:")
        
        # Density analysis
        low_density = [r for r in successful_results if r.density < 0.1]
        high_density = [r for r in successful_results if r.density > 0.3]
        
        if low_density:
            low_density_purity = np.mean([r.main_cluster_purity for r in low_density])
            logging.info(f"   Low density graphs (<0.1): avg_purity={low_density_purity:.3f}")
        
        if high_density:
            high_density_purity = np.mean([r.main_cluster_purity for r in high_density])
            logging.info(f"   High density graphs (>0.3): avg_purity={high_density_purity:.3f}")
        
        # Degree analysis
        low_degree = [r for r in successful_results if r.avg_degree < 5]
        high_degree = [r for r in successful_results if r.avg_degree > 10]
        
        if low_degree:
            low_degree_purity = np.mean([r.main_cluster_purity for r in low_degree])
            logging.info(f"   Low avg degree (<5): avg_purity={low_degree_purity:.3f}")
        
        if high_degree:
            high_degree_purity = np.mean([r.main_cluster_purity for r in high_degree])
            logging.info(f"   High avg degree (>10): avg_purity={high_degree_purity:.3f}")
        
        # Runtime analysis
        runtimes = [r.runtime_seconds for r in successful_results]
        avg_runtime = np.mean(runtimes)
        max_runtime = max(runtimes)
        
        logging.info(f"\n⏱️  PERFORMANCE ANALYSIS:")
        logging.info(f"   Average runtime: {avg_runtime:.3f}s")
        logging.info(f"   Maximum runtime: {max_runtime:.3f}s")
        
        # Over-segmentation analysis
        over_segmentation_ratios = []
        for result in successful_results:
            if result.expected_main_clusters and result.expected_main_clusters > 0:
                ratio = result.n_communities / result.expected_main_clusters
                over_segmentation_ratios.append(ratio)
        
        if over_segmentation_ratios:
            avg_over_seg = np.mean(over_segmentation_ratios)
            logging.info(f"   Average over-segmentation: {avg_over_seg:.2f}x")
        
        # Research implications
        logging.info(f"\n🎓 RESEARCH IMPLICATIONS:")
        
        if perfect_boundary_count / total_tests > 0.9:
            logging.info("   📝 CMG demonstrates consistent hierarchical behavior across topologies")
            logging.info("   📝 Algorithm is topology-agnostic for boundary preservation")
        elif perfect_boundary_count / total_tests > 0.7:
            logging.info("   📝 CMG shows good hierarchical behavior across most topologies")
            logging.info("   📝 Some topological structures more challenging than others")
        else:
            logging.info("   📝 CMG's hierarchical behavior is topology-dependent")
            logging.info("   📝 Need to identify which structures work best")
        
        if avg_purity > 0.95:
            logging.info("   📝 Strong evidence for robust hierarchical clustering across structures")
        elif avg_purity > 0.8:
            logging.info("   📝 Generally good hierarchical clustering with some variation")
        else:
            logging.info("   📝 Hierarchical clustering quality varies significantly with topology")
        
        logging.info(f"\n{'='*80}")
    
    def save_topology_results(self, filename: str = None) -> None:
        """Save results to JSON file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"validation_results/topology_stress_results_{timestamp}.json"
        
        results_data = [asdict(result) for result in self.results]
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logging.info(f"Results saved to {filename}")


def main():
    """Run comprehensive topology stress test."""
    
    print("🏗️  CMG TOPOLOGY STRESS TEST")
    print("=" * 60)
    print("This test evaluates CMG's hierarchical clustering behavior")
    print("across diverse graph topologies and structural patterns.")
    print()
    print("Testing topologies:")
    print("  • Grid structures (2D lattices)")  
    print("  • Ring structures (circular)")
    print("  • Tree structures (hierarchical)")
    print("  • Star structures (centralized)")
    print("  • Small-world networks")
    print("  • Scale-free networks")
    print("  • Bipartite structures")
    print()
    print("Expected runtime: 15-20 minutes")
    print()
    
    # Create and run tester
    tester = TopologyStressTester()
    
    try:
        tester.run_topology_stress_test()
        
        print("\n✅ Topology stress test completed!")
        print("\n🎯 Key findings:")
        print("   Check the analysis above for topology-specific insights")
        print("   Results saved to validation_results/ directory")
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        if tester.results:
            print("Analyzing partial results...")
            tester.analyze_topology_results()
            tester.save_topology_results()
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        logging.error(f"Topology stress test failed: {e}")


if __name__ == "__main__":
    main()
