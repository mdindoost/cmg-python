#!/usr/bin/env python3
"""
CMG Scale Stress Testing
========================

Systematically tests CMG's hierarchical clustering behavior across increasing graph sizes
to validate boundary preservation claims and discover performance limits.

This test pushes beyond the current <100 node testing to validate:
1. Perfect boundary preservation at scale
2. Gamma parameter insensitivity persistence  
3. Runtime and memory scaling behavior
4. Algorithmic breakdown points

Run with: python scale_stress_test.py
"""

import sys
sys.path.append('..')

import numpy as np
import time
import psutil
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import traceback
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
        logging.FileHandler('scale_stress_test.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class ScaleTestResult:
    """Data structure for storing scale test results."""
    graph_name: str
    n_nodes: int
    n_edges: int
    gamma: float
    n_communities: int
    main_cluster_purity: float
    boundary_preservation: bool
    runtime_seconds: float
    memory_mb: float
    peak_memory_mb: float
    success: bool
    error_message: Optional[str] = None
    
    # Detailed statistics
    avg_weighted_degree: Optional[float] = None
    high_degree_nodes: Optional[int] = None
    forest_edges_initial: Optional[int] = None
    forest_edges_final: Optional[int] = None
    edges_removed: Optional[int] = None
    
    # Ground truth comparison
    expected_main_clusters: Optional[int] = None
    expected_subclusters: Optional[int] = None
    subcluster_alignment: Optional[float] = None


class ScaleStressTester:
    """
    Comprehensive scale testing for CMG hierarchical clustering.
    """
    
    def __init__(self):
        self.results: List[ScaleTestResult] = []
        self.gamma_values = [4.1, 5.0, 10.0, 20.0]  # Test gamma insensitivity
        
        # Scale progression: start conservative, then push limits
        self.scale_progression = [
            # Validation range (confirm current findings)
            50, 75, 100,
            # Extension range (modest scaling)
            150, 200, 300, 500,
            # Stress range (find limits) 
            750, 1000, 1500, 2000,
            # Extreme range (if computational resources allow)
            3000, 5000, 7500, 10000
        ]
        
        # Memory tracking
        self.process = psutil.Process(os.getpid())
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def create_hierarchical_test_graph(self, n_nodes: int, seed: int = 42) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Create hierarchical test graph with specified number of nodes.
        
        Strategy: Create 2 main clusters, each containing multiple subclusters
        Scale by increasing subcluster count while keeping individual subclusters manageable.
        """
        try:
            # Design hierarchical structure based on node count
            if n_nodes < 100:
                # Small graphs: 2 main clusters, 2 subclusters each
                subclusters_per_main = 2
                subcluster_size = n_nodes // 4
            elif n_nodes < 500:
                # Medium graphs: 2 main clusters, 3-4 subclusters each  
                subclusters_per_main = 3
                subcluster_size = n_nodes // 6
            else:
                # Large graphs: 2 main clusters, adaptive subclusters
                subclusters_per_main = max(3, n_nodes // 200)  # Scale subclusters with size
                subcluster_size = n_nodes // (2 * subclusters_per_main)
            
            # Ensure valid cluster sizes
            if subcluster_size < 5:
                subcluster_size = 5
                subclusters_per_main = n_nodes // 10
            
            total_subclusters = 2 * subclusters_per_main
            cluster_sizes = [subcluster_size] * total_subclusters
            
            # Adjust for exact node count
            actual_nodes = sum(cluster_sizes)
            if actual_nodes != n_nodes:
                # Distribute extra nodes to first few clusters
                extra_nodes = n_nodes - actual_nodes
                for i in range(min(len(cluster_sizes), abs(extra_nodes))):
                    if extra_nodes > 0:
                        cluster_sizes[i] += 1
                        extra_nodes -= 1
                    elif extra_nodes < 0:
                        cluster_sizes[i] = max(1, cluster_sizes[i] - 1)
                        extra_nodes += 1
            
            # Generate graph
            edges, actual_n = create_clustered_graph(
                cluster_sizes=cluster_sizes,
                intra_cluster_p=0.8,    # Strong within subclusters
                inter_cluster_p=0.05,   # Weak between subclusters (but some connection)
                intra_weight_range=(1.5, 2.5),  # Strong intra weights
                inter_weight_range=(0.01, 0.1), # Weak inter weights
                seed=seed
            )
            
            if not edges:
                return None, {}
            
            # Create Laplacian
            A = create_laplacian_from_edges(edges, actual_n)
            
            # Generate ground truth labels
            main_cluster_labels = []
            subcluster_labels = []
            node_idx = 0
            
            for main_cluster_id in range(2):  # 2 main clusters
                for sub_id in range(subclusters_per_main):
                    global_subcluster_id = main_cluster_id * subclusters_per_main + sub_id
                    subcluster_size_actual = cluster_sizes[global_subcluster_id]
                    
                    for _ in range(subcluster_size_actual):
                        main_cluster_labels.append(main_cluster_id)
                        subcluster_labels.append(global_subcluster_id)
                        node_idx += 1
            
            metadata = {
                'n_nodes': actual_n,
                'n_edges': len(edges),
                'n_main_clusters': 2,
                'n_subclusters': total_subclusters,
                'subclusters_per_main': subclusters_per_main,
                'cluster_sizes': cluster_sizes,
                'main_cluster_labels': np.array(main_cluster_labels),
                'subcluster_labels': np.array(subcluster_labels),
                'structure_type': 'hierarchical_2_main'
            }
            
            return A, metadata
            
        except Exception as e:
            logging.error(f"Failed to create graph with {n_nodes} nodes: {e}")
            return None, {}
    
    def calculate_main_cluster_purity(self, cmg_labels: np.ndarray, 
                                    true_main_labels: np.ndarray) -> float:
        """Calculate main cluster purity (perfect = 1.0)."""
        n_pure_nodes = 0
        
        for cmg_cluster_id in np.unique(cmg_labels):
            nodes_in_cmg = np.where(cmg_labels == cmg_cluster_id)[0]
            true_labels_in_cmg = true_main_labels[nodes_in_cmg]
            
            # Pure if all nodes in CMG cluster have same true main label
            if len(np.unique(true_labels_in_cmg)) == 1:
                n_pure_nodes += len(nodes_in_cmg)
        
        return n_pure_nodes / len(cmg_labels)
    
    def calculate_subcluster_alignment(self, cmg_labels: np.ndarray,
                                     true_subcluster_labels: np.ndarray) -> float:
        """Calculate subcluster alignment score."""
        total_correctly_assigned = 0
        
        for true_sub_id in np.unique(true_subcluster_labels):
            nodes_in_true_sub = np.where(true_subcluster_labels == true_sub_id)[0]
            cmg_labels_for_sub = cmg_labels[nodes_in_true_sub]
            
            # Find most common CMG label for this true subcluster
            unique_cmg, counts = np.unique(cmg_labels_for_sub, return_counts=True)
            max_count = np.max(counts)
            total_correctly_assigned += max_count
        
        return total_correctly_assigned / len(cmg_labels)
    
    def test_single_scale(self, n_nodes: int) -> List[ScaleTestResult]:
        """Test CMG on a single scale with multiple gamma values."""
        
        logging.info(f"Testing scale: {n_nodes} nodes")
        scale_results = []
        
        # Create test graph
        A, metadata = self.create_hierarchical_test_graph(n_nodes)
        
        if A is None:
            logging.error(f"Failed to create graph for {n_nodes} nodes")
            return []
        
        actual_nodes = metadata['n_nodes']
        actual_edges = metadata['n_edges']
        
        logging.info(f"Created graph: {actual_nodes} nodes, {actual_edges} edges")
        
        # Test each gamma value
        for gamma in self.gamma_values:
            
            # Memory tracking
            memory_before = self.get_memory_usage()
            peak_memory = memory_before
            
            try:
                # Run CMG
                solver = CMGSteinerSolver(gamma=gamma, verbose=False)
                
                start_time = time.time()
                cmg_labels, n_communities = solver.steiner_group(A)
                runtime = time.time() - start_time
                
                # Track peak memory during execution
                memory_after = self.get_memory_usage()
                peak_memory = max(peak_memory, memory_after)
                
                # Get CMG statistics
                stats = solver.get_statistics()
                
                # Calculate hierarchical metrics
                main_purity = self.calculate_main_cluster_purity(
                    cmg_labels, metadata['main_cluster_labels']
                )
                
                subcluster_alignment = self.calculate_subcluster_alignment(
                    cmg_labels, metadata['subcluster_labels']
                )
                
                boundary_preservation = main_purity >= 0.999  # Nearly perfect
                
                # Create result record
                result = ScaleTestResult(
                    graph_name=f"hierarchical_{actual_nodes}nodes",
                    n_nodes=actual_nodes,
                    n_edges=actual_edges,
                    gamma=gamma,
                    n_communities=n_communities,
                    main_cluster_purity=main_purity,
                    boundary_preservation=boundary_preservation,
                    runtime_seconds=runtime,
                    memory_mb=memory_after - memory_before,
                    peak_memory_mb=peak_memory,
                    success=True,
                    
                    # Detailed statistics
                    avg_weighted_degree=stats.get('avg_weighted_degree'),
                    high_degree_nodes=stats.get('high_degree_nodes'),
                    forest_edges_initial=stats.get('forest_edges_initial'),
                    forest_edges_final=stats.get('forest_edges_final'),
                    edges_removed=stats.get('edges_removed'),
                    
                    # Ground truth comparison
                    expected_main_clusters=metadata['n_main_clusters'],
                    expected_subclusters=metadata['n_subclusters'],
                    subcluster_alignment=subcluster_alignment
                )
                
                scale_results.append(result)
                
                logging.info(f"  γ={gamma:5.1f}: {n_communities:3d} communities, "
                           f"purity={main_purity:.3f}, runtime={runtime:.3f}s, "
                           f"memory={memory_after-memory_before:.1f}MB")
                
                # Check for concerning memory usage
                if peak_memory > 4000:  # > 4GB
                    logging.warning(f"High memory usage: {peak_memory:.1f}MB")
                
                # Check for very long runtime
                if runtime > 300:  # > 5 minutes
                    logging.warning(f"Long runtime: {runtime:.1f}s")
                
            except Exception as e:
                logging.error(f"CMG failed for {actual_nodes} nodes, γ={gamma}: {e}")
                
                # Record failure
                result = ScaleTestResult(
                    graph_name=f"hierarchical_{actual_nodes}nodes",
                    n_nodes=actual_nodes,
                    n_edges=actual_edges,
                    gamma=gamma,
                    n_communities=0,
                    main_cluster_purity=0.0,
                    boundary_preservation=False,
                    runtime_seconds=0.0,
                    memory_mb=0.0,
                    peak_memory_mb=peak_memory,
                    success=False,
                    error_message=str(e),
                    expected_main_clusters=metadata['n_main_clusters'],
                    expected_subclusters=metadata['n_subclusters']
                )
                
                scale_results.append(result)
        
        return scale_results
    
    def run_scale_stress_test(self, max_memory_mb: float = 8000, 
                            max_runtime_minutes: float = 10) -> None:
        """
        Run comprehensive scale stress test.
        
        Args:
            max_memory_mb: Stop if memory usage exceeds this limit
            max_runtime_minutes: Stop if single test exceeds this time
        """
        
        logging.info("="*80)
        logging.info("CMG SCALE STRESS TEST STARTING")
        logging.info("="*80)
        logging.info(f"Testing scales: {self.scale_progression}")
        logging.info(f"Gamma values: {self.gamma_values}")
        logging.info(f"Memory limit: {max_memory_mb}MB")
        logging.info(f"Runtime limit: {max_runtime_minutes} minutes")
        logging.info("")
        
        total_tests = 0
        successful_tests = 0
        failed_tests = 0
        
        for n_nodes in self.scale_progression:
            
            logging.info(f"\n{'='*60}")
            logging.info(f"TESTING SCALE: {n_nodes} NODES")
            logging.info(f"{'='*60}")
            
            # Check if we should continue based on previous results
            if self.results:
                recent_failures = [r for r in self.results[-4:] if not r.success]
                if len(recent_failures) >= 3:
                    logging.warning("Multiple recent failures - stopping scale test")
                    break
                
                recent_memory = [r.peak_memory_mb for r in self.results[-2:] if r.success]
                if recent_memory and max(recent_memory) > max_memory_mb:
                    logging.warning(f"Memory limit exceeded - stopping at {n_nodes} nodes")
                    break
            
            # Run tests for this scale
            scale_results = self.test_single_scale(n_nodes)
            
            if not scale_results:
                logging.error(f"Failed to test scale {n_nodes}")
                failed_tests += len(self.gamma_values)
                continue
            
            # Update counters
            total_tests += len(scale_results)
            successful_tests += sum(1 for r in scale_results if r.success)
            failed_tests += sum(1 for r in scale_results if not r.success)
            
            # Store results
            self.results.extend(scale_results)
            
            # Check gamma insensitivity for this scale
            successful_scale_results = [r for r in scale_results if r.success]
            if len(successful_scale_results) > 1:
                communities = [r.n_communities for r in successful_scale_results]
                purities = [r.main_cluster_purity for r in successful_scale_results]
                
                gamma_insensitive = len(set(communities)) == 1
                perfect_purity = all(p >= 0.999 for p in purities)
                
                logging.info(f"\n  Scale {n_nodes} Summary:")
                logging.info(f"    Gamma insensitive: {gamma_insensitive}")
                logging.info(f"    Perfect boundary preservation: {perfect_purity}")
                logging.info(f"    Average runtime: {np.mean([r.runtime_seconds for r in successful_scale_results]):.3f}s")
                logging.info(f"    Peak memory: {max([r.peak_memory_mb for r in successful_scale_results]):.1f}MB")
                
                # Early stopping conditions
                max_runtime = max([r.runtime_seconds for r in successful_scale_results])
                if max_runtime > max_runtime_minutes * 60:
                    logging.warning(f"Runtime limit exceeded - stopping at {n_nodes} nodes")
                    break
        
        logging.info(f"\n{'='*80}")
        logging.info("SCALE STRESS TEST COMPLETED")
        logging.info(f"{'='*80}")
        logging.info(f"Total tests: {total_tests}")
        logging.info(f"Successful: {successful_tests}")
        logging.info(f"Failed: {failed_tests}")
        logging.info(f"Success rate: {successful_tests/total_tests:.1%}")
        
        # Save results
        self.save_results()
        self.analyze_results()
    
    def save_results(self, filename: str = None) -> None:
        """Save results to JSON file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"scale_stress_results_{timestamp}.json"
        
        # Convert results to serializable format
        results_data = [asdict(result) for result in self.results]
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logging.info(f"Results saved to {filename}")
    
    def analyze_results(self) -> None:
        """Analyze and report comprehensive results."""
        
        if not self.results:
            logging.info("No results to analyze")
            return
        
        successful_results = [r for r in self.results if r.success]
        
        if not successful_results:
            logging.info("No successful results to analyze")
            return
        
        logging.info(f"\n{'='*80}")
        logging.info("COMPREHENSIVE SCALE ANALYSIS")
        logging.info(f"{'='*80}")
        
        # 1. Boundary Preservation Analysis
        perfect_boundary_count = sum(1 for r in successful_results if r.boundary_preservation)
        logging.info(f"\n🎯 BOUNDARY PRESERVATION:")
        logging.info(f"   Perfect preservation: {perfect_boundary_count}/{len(successful_results)} "
                    f"({perfect_boundary_count/len(successful_results):.1%})")
        
        min_purity = min(r.main_cluster_purity for r in successful_results)
        max_purity = max(r.main_cluster_purity for r in successful_results)
        avg_purity = np.mean([r.main_cluster_purity for r in successful_results])
        
        logging.info(f"   Purity range: {min_purity:.6f} - {max_purity:.6f}")
        logging.info(f"   Average purity: {avg_purity:.6f}")
        
        if min_purity >= 0.999:
            logging.info("   ✅ PERFECT BOUNDARY PRESERVATION CONFIRMED AT ALL SCALES")
        elif avg_purity >= 0.95:
            logging.info("   ✅ EXCELLENT BOUNDARY PRESERVATION")
        else:
            logging.info("   ⚠️  BOUNDARY PRESERVATION DEGRADES AT SCALE")
        
        # 2. Gamma Insensitivity Analysis
        logging.info(f"\n🔧 GAMMA PARAMETER ANALYSIS:")
        
        # Group by node count
        by_scale = {}
        for result in successful_results:
            scale = result.n_nodes
            if scale not in by_scale:
                by_scale[scale] = []
            by_scale[scale].append(result)
        
        gamma_insensitive_scales = 0
        for scale, scale_results in by_scale.items():
            if len(scale_results) > 1:
                communities = [r.n_communities for r in scale_results]
                if len(set(communities)) == 1:
                    gamma_insensitive_scales += 1
        
        logging.info(f"   Gamma-insensitive scales: {gamma_insensitive_scales}/{len(by_scale)}")
        
        if gamma_insensitive_scales == len(by_scale):
            logging.info("   ⚠️  GAMMA PARAMETER COMPLETELY NON-FUNCTIONAL")
        elif gamma_insensitive_scales > len(by_scale) * 0.8:
            logging.info("   ⚠️  GAMMA PARAMETER MOSTLY NON-FUNCTIONAL")
        else:
            logging.info("   ✅ GAMMA PARAMETER SHOWS SOME SENSITIVITY")
        
        # 3. Scalability Analysis
        logging.info(f"\n📈 SCALABILITY ANALYSIS:")
        
        max_nodes = max(r.n_nodes for r in successful_results)
        max_runtime = max(r.runtime_seconds for r in successful_results)
        max_memory = max(r.peak_memory_mb for r in successful_results)
        
        logging.info(f"   Largest successful test: {max_nodes} nodes")
        logging.info(f"   Maximum runtime: {max_runtime:.2f} seconds")
        logging.info(f"   Peak memory usage: {max_memory:.1f} MB")
        
        # Runtime scaling analysis
        runtimes_by_scale = [(r.n_nodes, r.runtime_seconds) for r in successful_results]
        runtimes_by_scale.sort()
        
        if len(runtimes_by_scale) > 3:
            small_runtime = np.mean([rt for nodes, rt in runtimes_by_scale[:3]])
            large_runtime = np.mean([rt for nodes, rt in runtimes_by_scale[-3:]])
            
            if large_runtime > small_runtime * 100:
                logging.info("   ⚠️  RUNTIME SCALING CONCERN: >100x increase")
            elif large_runtime > small_runtime * 10:
                logging.info("   ⚠️  RUNTIME SCALING: ~10x increase")
            else:
                logging.info("   ✅ REASONABLE RUNTIME SCALING")
        
        # 4. Algorithm Behavior Analysis
        logging.info(f"\n🔍 ALGORITHM BEHAVIOR:")
        
        avg_communities = np.mean([r.n_communities for r in successful_results])
        communities_range = (min(r.n_communities for r in successful_results),
                           max(r.n_communities for r in successful_results))
        
        logging.info(f"   Average communities: {avg_communities:.1f}")
        logging.info(f"   Communities range: {communities_range[0]} - {communities_range[1]}")
        
        # Over-segmentation analysis
        over_segmentation_ratios = []
        for result in successful_results:
            if result.expected_subclusters:
                ratio = result.n_communities / result.expected_subclusters
                over_segmentation_ratios.append(ratio)
        
        if over_segmentation_ratios:
            avg_over_seg = np.mean(over_segmentation_ratios)
            logging.info(f"   Average over-segmentation ratio: {avg_over_seg:.2f}x")
            
            if avg_over_seg > 3:
                logging.info("   ⚠️  SEVERE OVER-SEGMENTATION")
            elif avg_over_seg > 1.5:
                logging.info("   ⚠️  MODERATE OVER-SEGMENTATION")
            else:
                logging.info("   ✅ REASONABLE SEGMENTATION")
        
        # 5. Research Implications
        logging.info(f"\n🎓 RESEARCH IMPLICATIONS:")
        
        if perfect_boundary_count / len(successful_results) > 0.95:
            logging.info("   📝 CMG demonstrates exceptional hierarchical clustering at scale")
            logging.info("   📝 Perfect boundary preservation is a unique algorithmic property")
            logging.info("   📝 Strong foundation for hierarchical clustering publication")
        
        if gamma_insensitive_scales == len(by_scale):
            logging.info("   📝 Gamma parameter investigation is critical research priority")
            logging.info("   📝 Algorithm analysis needed to understand parameter role")
        
        if max_nodes >= 1000:
            logging.info("   📝 Algorithm scales to practical problem sizes")
        else:
            logging.info("   📝 Scalability limitations need investigation")
        
        logging.info(f"\n{'='*80}")


def main():
    """Run comprehensive scale stress test."""
    
    print("🚀 CMG SCALE STRESS TEST")
    print("=" * 50)
    print("This test will systematically validate CMG's hierarchical clustering")
    print("behavior across increasing graph sizes to:")
    print("  • Confirm perfect boundary preservation at scale")
    print("  • Test gamma parameter insensitivity persistence")
    print("  • Identify performance and algorithmic limits")
    print("  • Validate research claims comprehensively")
    print()
    
    # Configuration
    max_memory_gb = 8  # Adjust based on your system
    max_runtime_minutes = 10  # Per individual test
    
    print(f"Configuration:")
    print(f"  Memory limit: {max_memory_gb}GB")
    print(f"  Runtime limit: {max_runtime_minutes} minutes per test")
    print(f"  Expected total time: 30-60 minutes")
    print()
    
    # Create and run tester
    tester = ScaleStressTester()
    
    try:
        tester.run_scale_stress_test(
            max_memory_mb=max_memory_gb * 1000,
            max_runtime_minutes=max_runtime_minutes
        )
        
        print("\n✅ Scale stress test completed successfully!")
        print("\n🎯 Key findings should be in the analysis above.")
        print("📊 Detailed results saved to JSON file.")
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        if tester.results:
            print("Analyzing partial results...")
            tester.analyze_results()
            tester.save_results()
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        logging.error(f"Scale stress test failed: {e}")
        logging.error(traceback.format_exc())


if __name__ == "__main__":
    main()
