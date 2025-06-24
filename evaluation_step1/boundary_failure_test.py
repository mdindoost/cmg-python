#!/usr/bin/env python3
"""
CMG Boundary Preservation Failure Detection Test
=================================================

This test systematically tries to BREAK CMG's perfect boundary preservation
to understand the limits and conditions under which it might fail.

We test increasingly challenging conditions:
1. Very strong inter-cluster connections
2. Identical cluster properties 
3. Degenerate graph structures
4. Extreme weight ratios
5. Pathological topologies

Goal: Either confirm robustness OR find failure conditions to understand limits.

Run with: python boundary_failure_test.py
"""

import sys
sys.path.append('..')

import numpy as np
import time
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
        logging.FileHandler('boundary_failure_test.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class BoundaryTestResult:
    """Result of boundary preservation test."""
    test_name: str
    test_type: str
    n_nodes: int
    n_communities: int
    main_cluster_purity: float
    boundary_preserved: bool
    runtime_seconds: float
    challenge_description: str
    challenge_parameters: Dict
    success: bool
    error_message: Optional[str] = None


class BoundaryFailureTester:
    """
    Systematically test conditions that might break CMG's boundary preservation.
    """
    
    def __init__(self):
        self.results: List[BoundaryTestResult] = []
        self.gamma = 5.0  # Use single gamma > 4 (since insensitivity confirmed)
        
    def calculate_main_cluster_purity(self, cmg_labels: np.ndarray, 
                                    true_main_labels: np.ndarray) -> float:
        """Calculate main cluster purity."""
        n_pure_nodes = 0
        
        for cmg_cluster_id in np.unique(cmg_labels):
            nodes_in_cmg = np.where(cmg_labels == cmg_cluster_id)[0]
            true_labels_in_cmg = true_main_labels[nodes_in_cmg]
            
            if len(np.unique(true_labels_in_cmg)) == 1:
                n_pure_nodes += len(nodes_in_cmg)
        
        return n_pure_nodes / len(cmg_labels)
    
    def test_strong_inter_cluster_connections(self) -> List[BoundaryTestResult]:
        """Test with increasingly strong connections between main clusters."""
        
        logging.info("🔥 Testing Strong Inter-cluster Connections")
        logging.info("Goal: Find if strong bridges break boundary preservation")
        
        results = []
        
        # Progressive increase in inter-cluster connection strength
        inter_probs = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]  # Up to 90% connection probability
        
        for inter_p in inter_probs:
            
            try:
                # Create graph with strong inter-cluster connections
                edges, n = create_clustered_graph(
                    cluster_sizes=[25, 25, 25, 25],  # 4 subclusters
                    intra_cluster_p=0.8,           # Keep intra-cluster strong
                    inter_cluster_p=inter_p,       # Variable inter-cluster strength
                    intra_weight_range=(2.0, 3.0), # Strong intra weights
                    inter_weight_range=(1.5, 2.5), # Strong inter weights (not weak!)
                    seed=42
                )
                
                if not edges:
                    continue
                
                A = create_laplacian_from_edges(edges, n)
                
                # Ground truth: 2 main clusters
                main_labels = np.array([0]*50 + [1]*50)
                
                # Test CMG
                solver = CMGSteinerSolver(gamma=self.gamma, verbose=False)
                start_time = time.time()
                cmg_labels, n_communities = solver.steiner_group(A)
                runtime = time.time() - start_time
                
                # Calculate purity
                purity = self.calculate_main_cluster_purity(cmg_labels, main_labels)
                boundary_preserved = purity >= 0.999
                
                result = BoundaryTestResult(
                    test_name=f"strong_inter_p{inter_p}",
                    test_type="strong_connections",
                    n_nodes=n,
                    n_communities=n_communities,
                    main_cluster_purity=purity,
                    boundary_preserved=boundary_preserved,
                    runtime_seconds=runtime,
                    challenge_description=f"Inter-cluster probability {inter_p:.1%}",
                    challenge_parameters={"inter_prob": inter_p, "inter_weight_range": (1.5, 2.5)},
                    success=True
                )
                
                results.append(result)
                
                logging.info(f"  Inter-p {inter_p:.1%}: purity={purity:.6f}, "
                           f"preserved={boundary_preserved}, communities={n_communities}")
                
                # Early exit if boundary breaks
                if not boundary_preserved:
                    logging.warning(f"🚨 BOUNDARY PRESERVATION BROKEN at inter_p={inter_p:.1%}")
                    break
                    
            except Exception as e:
                logging.error(f"Failed inter_p={inter_p}: {e}")
                
                result = BoundaryTestResult(
                    test_name=f"strong_inter_p{inter_p}",
                    test_type="strong_connections",
                    n_nodes=0,
                    n_communities=0,
                    main_cluster_purity=0.0,
                    boundary_preserved=False,
                    runtime_seconds=0.0,
                    challenge_description=f"Inter-cluster probability {inter_p:.1%}",
                    challenge_parameters={"inter_prob": inter_p},
                    success=False,
                    error_message=str(e)
                )
                results.append(result)
        
        return results
    
    def test_identical_cluster_properties(self) -> List[BoundaryTestResult]:
        """Test with identical cluster sizes and connection patterns."""
        
        logging.info("🔥 Testing Identical Cluster Properties")
        logging.info("Goal: Check if identical structure confuses CMG")
        
        results = []
        
        # Different levels of similarity
        similarity_tests = [
            {
                "name": "identical_sizes",
                "cluster_sizes": [25, 25, 25, 25],  # Perfectly identical
                "intra_p": 0.8,
                "inter_p": 0.05,
                "description": "Identical cluster sizes"
            },
            {
                "name": "identical_structure",
                "cluster_sizes": [20, 20, 20, 20],  # Identical + symmetric weights
                "intra_p": 0.8,
                "inter_p": 0.05,
                "description": "Identical sizes and structure"
            },
            {
                "name": "highly_similar",
                "cluster_sizes": [24, 25, 25, 26],  # Nearly identical
                "intra_p": 0.8,
                "inter_p": 0.05,
                "description": "Highly similar cluster sizes"
            }
        ]
        
        for test_config in similarity_tests:
            
            try:
                edges, n = create_clustered_graph(
                    cluster_sizes=test_config["cluster_sizes"],
                    intra_cluster_p=test_config["intra_p"],
                    inter_cluster_p=test_config["inter_p"],
                    seed=42
                )
                
                if not edges:
                    continue
                
                A = create_laplacian_from_edges(edges, n)
                
                # Ground truth: 2 main clusters
                cluster_sizes = test_config["cluster_sizes"]
                main_labels = []
                for i, size in enumerate(cluster_sizes):
                    main_cluster_id = 0 if i < 2 else 1  # First 2 subclusters = main cluster 0
                    main_labels.extend([main_cluster_id] * size)
                main_labels = np.array(main_labels)
                
                # Test CMG
                solver = CMGSteinerSolver(gamma=self.gamma, verbose=False)
                start_time = time.time()
                cmg_labels, n_communities = solver.steiner_group(A)
                runtime = time.time() - start_time
                
                # Calculate purity
                purity = self.calculate_main_cluster_purity(cmg_labels, main_labels)
                boundary_preserved = purity >= 0.999
                
                result = BoundaryTestResult(
                    test_name=test_config["name"],
                    test_type="identical_properties",
                    n_nodes=n,
                    n_communities=n_communities,
                    main_cluster_purity=purity,
                    boundary_preserved=boundary_preserved,
                    runtime_seconds=runtime,
                    challenge_description=test_config["description"],
                    challenge_parameters=test_config,
                    success=True
                )
                
                results.append(result)
                
                logging.info(f"  {test_config['name']}: purity={purity:.6f}, "
                           f"preserved={boundary_preserved}")
                
            except Exception as e:
                logging.error(f"Failed {test_config['name']}: {e}")
                
                result = BoundaryTestResult(
                    test_name=test_config["name"],
                    test_type="identical_properties",
                    n_nodes=0,
                    n_communities=0,
                    main_cluster_purity=0.0,
                    boundary_preserved=False,
                    runtime_seconds=0.0,
                    challenge_description=test_config["description"],
                    challenge_parameters=test_config,
                    success=False,
                    error_message=str(e)
                )
                results.append(result)
        
        return results
    
    def test_extreme_weight_ratios(self) -> List[BoundaryTestResult]:
        """Test with extreme weight ratios between intra and inter cluster connections."""
        
        logging.info("🔥 Testing Extreme Weight Ratios")
        logging.info("Goal: Test if extreme weight differences break algorithm")
        
        results = []
        
        # Test different weight ratio scenarios
        weight_tests = [
            {
                "name": "huge_intra_tiny_inter",
                "intra_range": (100.0, 200.0),
                "inter_range": (0.001, 0.01),
                "description": "Huge intra weights, tiny inter weights"
            },
            {
                "name": "tiny_intra_huge_inter", 
                "intra_range": (0.001, 0.01),
                "inter_range": (100.0, 200.0),
                "description": "Tiny intra weights, huge inter weights"
            },
            {
                "name": "extreme_ratio_1M",
                "intra_range": (1000.0, 1000.0),
                "inter_range": (0.001, 0.001),
                "description": "1 million to 1 weight ratio"
            },
            {
                "name": "all_equal_weights",
                "intra_range": (1.0, 1.0),
                "inter_range": (1.0, 1.0),
                "description": "All edges have identical weights"
            }
        ]
        
        for test_config in weight_tests:
            
            try:
                edges, n = create_clustered_graph(
                    cluster_sizes=[25, 25, 25, 25],
                    intra_cluster_p=0.8,
                    inter_cluster_p=0.05,
                    intra_weight_range=test_config["intra_range"],
                    inter_weight_range=test_config["inter_range"],
                    seed=42
                )
                
                if not edges:
                    continue
                
                A = create_laplacian_from_edges(edges, n)
                
                # Ground truth
                main_labels = np.array([0]*50 + [1]*50)
                
                # Test CMG
                solver = CMGSteinerSolver(gamma=self.gamma, verbose=False)
                start_time = time.time()
                cmg_labels, n_communities = solver.steiner_group(A)
                runtime = time.time() - start_time
                
                # Calculate purity
                purity = self.calculate_main_cluster_purity(cmg_labels, main_labels)
                boundary_preserved = purity >= 0.999
                
                result = BoundaryTestResult(
                    test_name=test_config["name"],
                    test_type="extreme_weights",
                    n_nodes=n,
                    n_communities=n_communities,
                    main_cluster_purity=purity,
                    boundary_preserved=boundary_preserved,
                    runtime_seconds=runtime,
                    challenge_description=test_config["description"],
                    challenge_parameters=test_config,
                    success=True
                )
                
                results.append(result)
                
                logging.info(f"  {test_config['name']}: purity={purity:.6f}, "
                           f"preserved={boundary_preserved}")
                
            except Exception as e:
                logging.error(f"Failed {test_config['name']}: {e}")
                
                result = BoundaryTestResult(
                    test_name=test_config["name"],
                    test_type="extreme_weights",
                    n_nodes=0,
                    n_communities=0,
                    main_cluster_purity=0.0,
                    boundary_preserved=False,
                    runtime_seconds=0.0,
                    challenge_description=test_config["description"],
                    challenge_parameters=test_config,
                    success=False,
                    error_message=str(e)
                )
                results.append(result)
        
        return results
    
    def test_degenerate_structures(self) -> List[BoundaryTestResult]:
        """Test degenerate graph structures that might confuse CMG."""
        
        logging.info("🔥 Testing Degenerate Graph Structures")
        logging.info("Goal: Test pathological cases")
        
        results = []
        
        # Create degenerate test cases manually
        degenerate_tests = [
            {
                "name": "single_bridge",
                "description": "Two clusters connected by single edge",
                "create_func": self._create_single_bridge_graph
            },
            {
                "name": "star_clusters", 
                "description": "Star-shaped clusters",
                "create_func": self._create_star_clusters_graph
            },
            {
                "name": "chain_clusters",
                "description": "Chain of small clusters",
                "create_func": self._create_chain_clusters_graph
            }
        ]
        
        for test_config in degenerate_tests:
            
            try:
                A, main_labels = test_config["create_func"]()
                
                if A is None:
                    continue
                
                n = A.shape[0]
                
                # Test CMG
                solver = CMGSteinerSolver(gamma=self.gamma, verbose=False)
                start_time = time.time()
                cmg_labels, n_communities = solver.steiner_group(A)
                runtime = time.time() - start_time
                
                # Calculate purity
                purity = self.calculate_main_cluster_purity(cmg_labels, main_labels)
                boundary_preserved = purity >= 0.999
                
                result = BoundaryTestResult(
                    test_name=test_config["name"],
                    test_type="degenerate_structure",
                    n_nodes=n,
                    n_communities=n_communities,
                    main_cluster_purity=purity,
                    boundary_preserved=boundary_preserved,
                    runtime_seconds=runtime,
                    challenge_description=test_config["description"],
                    challenge_parameters={},
                    success=True
                )
                
                results.append(result)
                
                logging.info(f"  {test_config['name']}: purity={purity:.6f}, "
                           f"preserved={boundary_preserved}")
                
            except Exception as e:
                logging.error(f"Failed {test_config['name']}: {e}")
                
                result = BoundaryTestResult(
                    test_name=test_config["name"],
                    test_type="degenerate_structure", 
                    n_nodes=0,
                    n_communities=0,
                    main_cluster_purity=0.0,
                    boundary_preserved=False,
                    runtime_seconds=0.0,
                    challenge_description=test_config["description"],
                    challenge_parameters={},
                    success=False,
                    error_message=str(e)
                )
                results.append(result)
        
        return results
    
    def _create_single_bridge_graph(self) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Create two dense clusters connected by single edge."""
        
        # Two complete subgraphs connected by one edge
        edges = []
        
        # Cluster 1: complete graph on nodes 0-19
        for i in range(20):
            for j in range(i+1, 20):
                edges.append((i, j, 2.0))
        
        # Cluster 2: complete graph on nodes 20-39
        for i in range(20, 40):
            for j in range(i+1, 40):
                edges.append((i, j, 2.0))
        
        # Single bridge
        edges.append((19, 20, 0.1))
        
        A = create_laplacian_from_edges(edges, 40)
        main_labels = np.array([0]*20 + [1]*20)
        
        return A, main_labels
    
    def _create_star_clusters_graph(self) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Create star-shaped clusters."""
        
        edges = []
        
        # Star 1: center at node 0, leaves 1-15
        for i in range(1, 16):
            edges.append((0, i, 2.0))
        
        # Star 2: center at node 16, leaves 17-31
        for i in range(17, 32):
            edges.append((16, i, 2.0))
        
        # Bridge between star centers
        edges.append((0, 16, 0.1))
        
        A = create_laplacian_from_edges(edges, 32)
        main_labels = np.array([0]*16 + [1]*16)
        
        return A, main_labels
    
    def _create_chain_clusters_graph(self) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Create chain of small dense clusters."""
        
        edges = []
        
        # 6 clusters of 5 nodes each
        cluster_size = 5
        n_clusters = 6
        
        for cluster_id in range(n_clusters):
            start_node = cluster_id * cluster_size
            
            # Make each cluster complete
            for i in range(start_node, start_node + cluster_size):
                for j in range(i+1, start_node + cluster_size):
                    edges.append((i, j, 2.0))
            
            # Connect to next cluster
            if cluster_id < n_clusters - 1:
                bridge_node1 = start_node + cluster_size - 1
                bridge_node2 = (cluster_id + 1) * cluster_size
                edges.append((bridge_node1, bridge_node2, 0.1))
        
        A = create_laplacian_from_edges(edges, n_clusters * cluster_size)
        
        # Main clusters: first 3 clusters vs last 3 clusters
        main_labels = [0] * (3 * cluster_size) + [1] * (3 * cluster_size)
        main_labels = np.array(main_labels)
        
        return A, main_labels
    
    def run_all_boundary_tests(self) -> None:
        """Run comprehensive boundary preservation failure detection."""
        
        logging.info("="*80)
        logging.info("CMG BOUNDARY PRESERVATION FAILURE DETECTION")
        logging.info("="*80)
        logging.info("Testing increasingly challenging conditions to find limits...")
        logging.info("")
        
        # Run all test categories
        test_categories = [
            ("Strong Inter-cluster Connections", self.test_strong_inter_cluster_connections),
            ("Identical Cluster Properties", self.test_identical_cluster_properties),
            ("Extreme Weight Ratios", self.test_extreme_weight_ratios),
            ("Degenerate Structures", self.test_degenerate_structures)
        ]
        
        all_results = []
        boundary_failures = []
        
        for category_name, test_func in test_categories:
            logging.info(f"\n{'='*60}")
            logging.info(f"TESTING: {category_name}")
            logging.info(f"{'='*60}")
            
            category_results = test_func()
            all_results.extend(category_results)
            
            # Check for boundary failures
            failures = [r for r in category_results if r.success and not r.boundary_preserved]
            if failures:
                boundary_failures.extend(failures)
                logging.warning(f"🚨 BOUNDARY FAILURES FOUND in {category_name}")
            else:
                logging.info(f"✅ No boundary failures in {category_name}")
        
        self.results = all_results
        
        # Comprehensive analysis
        self.analyze_boundary_test_results(boundary_failures)
        
        # Save results
        self.save_boundary_results()
    
    def analyze_boundary_test_results(self, boundary_failures: List[BoundaryTestResult]) -> None:
        """Analyze boundary test results comprehensively."""
        
        successful_results = [r for r in self.results if r.success]
        
        logging.info(f"\n{'='*80}")
        logging.info("BOUNDARY PRESERVATION ANALYSIS")
        logging.info(f"{'='*80}")
        
        # Overall statistics
        total_tests = len(successful_results)
        perfect_boundary_count = sum(1 for r in successful_results if r.boundary_preserved)
        
        logging.info(f"\n📊 OVERALL RESULTS:")
        logging.info(f"   Total successful tests: {total_tests}")
        logging.info(f"   Perfect boundary preservation: {perfect_boundary_count}/{total_tests} "
                    f"({perfect_boundary_count/total_tests:.1%})")
        
        if boundary_failures:
            logging.info(f"   Boundary failures: {len(boundary_failures)}")
            logging.info(f"\n🚨 FAILURE ANALYSIS:")
            
            for failure in boundary_failures:
                logging.info(f"     {failure.test_name}: purity={failure.main_cluster_purity:.6f}")
                logging.info(f"       Challenge: {failure.challenge_description}")
                
        else:
            logging.info(f"   ✅ NO BOUNDARY FAILURES DETECTED")
        
        # Purity statistics
        purities = [r.main_cluster_purity for r in successful_results]
        min_purity = min(purities) if purities else 0.0
        max_purity = max(purities) if purities else 0.0
        avg_purity = np.mean(purities) if purities else 0.0
        
        logging.info(f"\n📈 PURITY ANALYSIS:")
        logging.info(f"   Minimum purity: {min_purity:.6f}")
        logging.info(f"   Maximum purity: {max_purity:.6f}")
        logging.info(f"   Average purity: {avg_purity:.6f}")
        
        # Research implications
        logging.info(f"\n🎓 RESEARCH IMPLICATIONS:")
        
        if len(boundary_failures) == 0:
            logging.info("   📝 CMG demonstrates exceptional robustness")
            logging.info("   📝 Perfect boundary preservation across all tested challenges")
            logging.info("   📝 Strong evidence for algorithmic reliability")
        else:
            logging.info("   📝 CMG has discoverable failure conditions")
            logging.info("   📝 Boundary preservation has limits under extreme conditions")
            logging.info("   📝 Important to document these limitations")
        
        if min_purity >= 0.999:
            logging.info("   📝 Even 'worst case' maintains near-perfect purity")
        elif min_purity >= 0.95:
            logging.info("   📝 Strong boundary preservation even under stress")
        else:
            logging.info("   📝 Some conditions significantly degrade performance")
        
        logging.info(f"\n{'='*80}")
    
    def save_boundary_results(self, filename: str = None) -> None:
        """Save results to JSON file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"boundary_failure_results_{timestamp}.json"
        
        results_data = [asdict(result) for result in self.results]
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logging.info(f"Results saved to {filename}")


def main():
    """Run boundary preservation failure detection tests."""
    
    print("🔥 CMG BOUNDARY PRESERVATION FAILURE DETECTION")
    print("=" * 60)
    print("This test systematically tries to BREAK CMG's perfect boundary")
    print("preservation to understand its limits and robustness.")
    print()
    print("Testing increasingly challenging conditions:")
    print("  • Very strong inter-cluster connections")
    print("  • Identical cluster properties")
    print("  • Extreme weight ratios")
    print("  • Degenerate graph structures")
    print()
    print("Expected runtime: 10-15 minutes")
    print()
    
    # Create and run tester
    tester = BoundaryFailureTester()
    
    try:
        tester.run_all_boundary_tests()
        
        print("\n✅ Boundary failure detection completed!")
        print("\n🎯 Key findings:")
        print("   Check the analysis above for robustness assessment")
        print("   Results saved to JSON file for further analysis")
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        if tester.results:
            print("Analyzing partial results...")
            tester.analyze_boundary_test_results([])
            tester.save_boundary_results()
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        logging.error(f"Boundary failure test failed: {e}")


if __name__ == "__main__":
    main()
