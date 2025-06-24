#!/usr/bin/env python3
"""
Gamma Parameter Weight Sensitivity Investigation
==============================================

Tests the hypothesis that gamma insensitivity is caused by unweighted graphs.
Systematically compares CMG behavior across:
1. Unweighted graphs (uniform weights = 1.0)
2. Moderately weighted graphs (weights vary 10x)
3. Heavily weighted graphs (weights vary 100x+)
4. Extreme weight variations (orders of magnitude)

Run with: python gamma_weight_investigation.py
"""

import sys
sys.path.append('..')

import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import json

# CMG imports
from cmg import CMGSteinerSolver, create_laplacian_from_edges
from cmg.utils.graph_utils import create_clustered_graph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gamma_weight_investigation.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class WeightTestResult:
    """Results for weight sensitivity testing."""
    test_name: str
    weight_scenario: str
    gamma: float
    n_communities: int
    main_cluster_purity: float
    
    # Weighted degree statistics
    avg_weighted_degree: float
    min_weighted_degree: float
    max_weighted_degree: float
    high_degree_nodes: int
    gamma_threshold: float
    
    # Edge weight statistics
    min_edge_weight: float
    max_edge_weight: float
    weight_range_ratio: float
    
    # Algorithm behavior
    forest_edges_initial: int
    forest_edges_final: int
    edges_removed: int
    runtime: float


class GammaWeightInvestigator:
    """
    Investigates gamma parameter sensitivity across different edge weight scenarios.
    """
    
    def __init__(self):
        self.results: List[WeightTestResult] = []
        
        # Test comprehensive gamma range
        self.gamma_values = [4.1, 5.0, 7.0, 10.0, 15.0, 20.0, 50.0, 100.0]
        
        # Weight scenarios to test
        self.weight_scenarios = {
            'unweighted': 'All edge weights = 1.0 (current testing)',
            'light_variation': 'Weights vary 2-5x (realistic networks)',
            'moderate_variation': 'Weights vary 10-20x (some heterogeneity)',
            'heavy_variation': 'Weights vary 100x (strong heterogeneity)',
            'extreme_variation': 'Weights vary 1000x+ (pathological case)',
            'bimodal_weights': 'Two weight classes: very light + very heavy'
        }
    
    def create_test_graph_with_weights(self, weight_scenario: str, seed: int = 42) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Create hierarchical test graph with specified weight distribution.
        """
        
        # Base graph structure (2 main clusters, 4 subclusters)
        cluster_sizes = [15, 15, 15, 15]  # 60 nodes total
        
        # Create base topology
        edges, n = create_clustered_graph(
            cluster_sizes=cluster_sizes,
            intra_cluster_p=0.8,
            inter_cluster_p=0.05,
            seed=seed
        )
        
        if not edges:
            return None, {}
        
        # Modify weights based on scenario
        modified_edges = []
        np.random.seed(seed)
        
        for u, v, original_weight in edges:
            
            if weight_scenario == 'unweighted':
                # All weights = 1.0
                new_weight = 1.0
                
            elif weight_scenario == 'light_variation':
                # Weights in range [0.5, 2.5] (5x variation)
                new_weight = 0.5 + np.random.random() * 2.0
                
            elif weight_scenario == 'moderate_variation':
                # Weights in range [0.1, 2.0] (20x variation)
                new_weight = 0.1 + np.random.random() * 1.9
                
            elif weight_scenario == 'heavy_variation':
                # Weights in range [0.01, 1.0] (100x variation)
                new_weight = 0.01 + np.random.random() * 0.99
                
            elif weight_scenario == 'extreme_variation':
                # Log-uniform distribution: [0.001, 10.0] (10,000x variation)
                log_weight = np.random.uniform(-3, 1)  # log10 scale
                new_weight = 10 ** log_weight
                
            elif weight_scenario == 'bimodal_weights':
                # Two classes: very light (0.01) or heavy (10.0)
                if np.random.random() < 0.7:
                    new_weight = 0.01  # Light edges (70%)
                else:
                    new_weight = 10.0  # Heavy edges (30%)
            
            else:
                new_weight = original_weight
            
            modified_edges.append((u, v, new_weight))
        
        # Create Laplacian with modified weights
        A = create_laplacian_from_edges(modified_edges, n)
        
        # Ground truth (2 main clusters: [0,1] vs [2,3])
        main_labels = [0]*30 + [1]*30  # First 2 subclusters vs last 2
        subcluster_labels = [0]*15 + [1]*15 + [2]*15 + [3]*15
        
        # Calculate weight statistics
        weights = [w for _, _, w in modified_edges]
        min_weight = min(weights)
        max_weight = max(weights)
        weight_range_ratio = max_weight / min_weight if min_weight > 0 else float('inf')
        
        metadata = {
            'n_nodes': n,
            'n_edges': len(modified_edges),
            'weight_scenario': weight_scenario,
            'main_cluster_labels': np.array(main_labels),
            'subcluster_labels': np.array(subcluster_labels),
            'edge_weights': weights,
            'min_edge_weight': min_weight,
            'max_edge_weight': max_weight,
            'weight_range_ratio': weight_range_ratio
        }
        
        return A, metadata
    
    def calculate_weighted_degree_statistics(self, A: np.ndarray, solver: CMGSteinerSolver) -> Dict:
        """Calculate detailed weighted degree statistics."""
        
        # Convert to adjacency matrix for weighted degree calculation
        A_adj = -A.copy()
        A_adj.setdiag(0)
        A_adj.eliminate_zeros()
        A_adj.data = np.abs(A_adj.data)
        
        n = A.shape[0]
        weighted_degrees = []
        
        for node in range(n):
            wd = solver.weighted_degree(A_adj, node)
            weighted_degrees.append(wd)
        
        avg_wd = np.mean(weighted_degrees)
        
        return {
            'weighted_degrees': weighted_degrees,
            'avg_weighted_degree': avg_wd,
            'min_weighted_degree': min(weighted_degrees),
            'max_weighted_degree': max(weighted_degrees),
            'std_weighted_degree': np.std(weighted_degrees)
        }
    
    def test_weight_scenario(self, weight_scenario: str) -> List[WeightTestResult]:
        """Test CMG across gamma values for a specific weight scenario."""
        
        logging.info(f"\n{'='*60}")
        logging.info(f"TESTING WEIGHT SCENARIO: {weight_scenario.upper()}")
        logging.info(f"{'='*60}")
        logging.info(f"Description: {self.weight_scenarios[weight_scenario]}")
        
        # Create test graph
        A, metadata = self.create_test_graph_with_weights(weight_scenario)
        
        if A is None:
            logging.error(f"Failed to create graph for scenario: {weight_scenario}")
            return []
        
        logging.info(f"Created graph: {metadata['n_nodes']} nodes, {metadata['n_edges']} edges")
        logging.info(f"Edge weights: {metadata['min_edge_weight']:.6f} - {metadata['max_edge_weight']:.6f}")
        logging.info(f"Weight range ratio: {metadata['weight_range_ratio']:.1f}x")
        
        scenario_results = []
        
        # Test each gamma value
        for gamma in self.gamma_values:
            
            try:
                # Create solver and run CMG
                solver = CMGSteinerSolver(gamma=gamma, verbose=False)
                
                start_time = time.time()
                cmg_labels, n_communities = solver.steiner_group(A)
                runtime = time.time() - start_time
                
                # Get algorithm statistics
                stats = solver.get_statistics()
                
                # Calculate weighted degree statistics
                wd_stats = self.calculate_weighted_degree_statistics(A, solver)
                
                # Calculate main cluster purity
                main_labels = metadata['main_cluster_labels']
                purity = self._calculate_purity(cmg_labels, main_labels)
                
                # Calculate gamma threshold
                gamma_threshold = gamma * wd_stats['avg_weighted_degree']
                
                # Create result record
                result = WeightTestResult(
                    test_name=f"{weight_scenario}_gamma{gamma}",
                    weight_scenario=weight_scenario,
                    gamma=gamma,
                    n_communities=n_communities,
                    main_cluster_purity=purity,
                    
                    # Weighted degree statistics
                    avg_weighted_degree=wd_stats['avg_weighted_degree'],
                    min_weighted_degree=wd_stats['min_weighted_degree'],
                    max_weighted_degree=wd_stats['max_weighted_degree'],
                    high_degree_nodes=stats.get('high_degree_nodes', 0),
                    gamma_threshold=gamma_threshold,
                    
                    # Edge weight statistics
                    min_edge_weight=metadata['min_edge_weight'],
                    max_edge_weight=metadata['max_edge_weight'],
                    weight_range_ratio=metadata['weight_range_ratio'],
                    
                    # Algorithm behavior
                    forest_edges_initial=stats.get('forest_edges_initial', 0),
                    forest_edges_final=stats.get('forest_edges_final', 0),
                    edges_removed=stats.get('edges_removed', 0),
                    runtime=runtime
                )
                
                scenario_results.append(result)
                
                logging.info(f"  γ={gamma:6.1f}: {n_communities:2d} communities, "
                           f"purity={purity:.3f}, high_deg_nodes={stats.get('high_degree_nodes', 0):2d}, "
                           f"avg_wd={wd_stats['avg_weighted_degree']:.3f}")
                
            except Exception as e:
                logging.error(f"  γ={gamma:6.1f}: FAILED - {e}")
        
        # Analyze gamma sensitivity for this scenario
        if len(scenario_results) > 1:
            communities = [r.n_communities for r in scenario_results]
            purities = [r.main_cluster_purity for r in scenario_results]
            high_degree_counts = [r.high_degree_nodes for r in scenario_results]
            
            gamma_sensitive = len(set(communities)) > 1
            purity_variation = max(purities) - min(purities)
            
            logging.info(f"\n  Scenario Analysis:")
            logging.info(f"    Gamma sensitive: {gamma_sensitive}")
            logging.info(f"    Communities range: {min(communities)} - {max(communities)}")
            logging.info(f"    Purity variation: {purity_variation:.6f}")
            logging.info(f"    High-degree nodes range: {min(high_degree_counts)} - {max(high_degree_counts)}")
            
            if gamma_sensitive:
                logging.info(f"    ✅ GAMMA PARAMETER IS FUNCTIONAL in {weight_scenario}")
            else:
                logging.info(f"    ⚠️  GAMMA PARAMETER STILL INSENSITIVE in {weight_scenario}")
        
        return scenario_results
    
    def _calculate_purity(self, cmg_labels: np.ndarray, true_labels: np.ndarray) -> float:
        """Calculate main cluster purity."""
        n_pure_nodes = 0
        
        for cmg_cluster_id in np.unique(cmg_labels):
            nodes_in_cmg = np.where(cmg_labels == cmg_cluster_id)[0]
            true_labels_in_cmg = true_labels[nodes_in_cmg]
            
            if len(np.unique(true_labels_in_cmg)) == 1:
                n_pure_nodes += len(nodes_in_cmg)
        
        return n_pure_nodes / len(cmg_labels)
    
    def run_comprehensive_weight_investigation(self) -> None:
        """Run comprehensive investigation across all weight scenarios."""
        
        logging.info("="*80)
        logging.info("CMG GAMMA PARAMETER WEIGHT SENSITIVITY INVESTIGATION")
        logging.info("="*80)
        logging.info("Testing hypothesis: Gamma insensitivity is caused by unweighted graphs")
        logging.info(f"Gamma values tested: {self.gamma_values}")
        logging.info(f"Weight scenarios: {list(self.weight_scenarios.keys())}")
        logging.info("")
        
        # Test each weight scenario
        for scenario in self.weight_scenarios.keys():
            scenario_results = self.test_weight_scenario(scenario)
            self.results.extend(scenario_results)
        
        # Comprehensive analysis
        self.analyze_weight_sensitivity()
        
        # Save results
        self.save_results()
    
    def analyze_weight_sensitivity(self) -> None:
        """Analyze gamma sensitivity across all weight scenarios."""
        
        logging.info(f"\n{'='*80}")
        logging.info("COMPREHENSIVE WEIGHT SENSITIVITY ANALYSIS")
        logging.info(f"{'='*80}")
        
        # Group results by scenario
        by_scenario = {}
        for result in self.results:
            scenario = result.weight_scenario
            if scenario not in by_scenario:
                by_scenario[scenario] = []
            by_scenario[scenario].append(result)
        
        logging.info(f"\n📊 SCENARIO-BY-SCENARIO ANALYSIS:")
        
        gamma_sensitive_scenarios = []
        
        for scenario, results in by_scenario.items():
            if len(results) < 2:
                continue
            
            communities = [r.n_communities for r in results]
            purities = [r.main_cluster_purity for r in results]
            high_degree_counts = [r.high_degree_nodes for r in results]
            weight_ratio = results[0].weight_range_ratio
            
            gamma_sensitive = len(set(communities)) > 1
            purity_stable = (max(purities) - min(purities)) < 0.01
            
            logging.info(f"\n  {scenario.upper()}:")
            logging.info(f"    Weight range ratio: {weight_ratio:.1f}x")
            logging.info(f"    Communities: {min(communities)}-{max(communities)} "
                        f"(unique: {len(set(communities))})")
            logging.info(f"    Purities: {min(purities):.3f}-{max(purities):.3f}")
            logging.info(f"    High-degree nodes: {min(high_degree_counts)}-{max(high_degree_counts)}")
            logging.info(f"    Gamma sensitive: {gamma_sensitive}")
            
            if gamma_sensitive:
                gamma_sensitive_scenarios.append(scenario)
                logging.info(f"    ✅ GAMMA WORKS in {scenario}")
            else:
                logging.info(f"    ❌ GAMMA INSENSITIVE in {scenario}")
        
        # Overall conclusions
        logging.info(f"\n🎯 OVERALL CONCLUSIONS:")
        
        total_scenarios = len(by_scenario)
        sensitive_scenarios = len(gamma_sensitive_scenarios)
        
        logging.info(f"   Gamma-sensitive scenarios: {sensitive_scenarios}/{total_scenarios}")
        
        if sensitive_scenarios == 0:
            logging.info("   ❌ GAMMA PARAMETER IS FUNDAMENTALLY BROKEN")
            logging.info("   📝 Implementation bug or theoretical misunderstanding")
            
        elif sensitive_scenarios == total_scenarios:
            logging.info("   ✅ GAMMA PARAMETER WORKS WITH PROPER EDGE WEIGHTS")
            logging.info("   📝 Unweighted graphs cause gamma insensitivity")
            logging.info("   📝 CMG requires weight variation to function properly")
            
        elif 'unweighted' not in gamma_sensitive_scenarios:
            logging.info("   ✅ HYPOTHESIS CONFIRMED: Unweighted graphs cause insensitivity")
            logging.info("   📝 CMG needs weight variation for parameter control")
            
        else:
            logging.info("   ⚠️  PARTIAL GAMMA SENSITIVITY")
            logging.info("   📝 Weight variation helps but doesn't fully solve the issue")
        
        # Technical insights
        logging.info(f"\n🔬 TECHNICAL INSIGHTS:")
        
        if 'unweighted' in by_scenario and 'extreme_variation' in by_scenario:
            unweighted_results = by_scenario['unweighted']
            extreme_results = by_scenario['extreme_variation']
            
            unweighted_communities = set(r.n_communities for r in unweighted_results)
            extreme_communities = set(r.n_communities for r in extreme_results)
            
            if len(unweighted_communities) == 1 and len(extreme_communities) > 1:
                logging.info("   📝 Edge weight variation enables gamma parameter control")
                logging.info("   📝 Weighted degree calculation requires weight heterogeneity")
                
        # Research implications
        logging.info(f"\n🎓 RESEARCH IMPLICATIONS:")
        
        if sensitive_scenarios > 0:
            logging.info("   📝 CMG gamma parameter CAN work with proper graphs")
            logging.info("   📝 Previous testing used inappropriate (unweighted) graphs")
            logging.info("   📝 Real-world networks typically have weight variation")
            logging.info("   📝 Algorithm behavior depends critically on edge weight distribution")
        
        if sensitive_scenarios == 0:
            logging.info("   📝 CMG gamma parameter has fundamental implementation issues")
            logging.info("   📝 Algorithm investigation and potential fixes needed")
            logging.info("   📝 Current algorithm may differ from paper specification")
        
        logging.info(f"\n{'='*80}")
    
    def save_results(self, filename: str = None) -> None:
        """Save results to JSON file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"gamma_weight_investigation_{timestamp}.json"
        
        # Convert results to serializable format
        results_data = []
        for result in self.results:
            result_dict = {
                'test_name': result.test_name,
                'weight_scenario': result.weight_scenario,
                'gamma': result.gamma,
                'n_communities': result.n_communities,
                'main_cluster_purity': result.main_cluster_purity,
                'avg_weighted_degree': result.avg_weighted_degree,
                'min_weighted_degree': result.min_weighted_degree,
                'max_weighted_degree': result.max_weighted_degree,
                'high_degree_nodes': result.high_degree_nodes,
                'gamma_threshold': result.gamma_threshold,
                'min_edge_weight': result.min_edge_weight,
                'max_edge_weight': result.max_edge_weight,
                'weight_range_ratio': result.weight_range_ratio,
                'forest_edges_initial': result.forest_edges_initial,
                'forest_edges_final': result.forest_edges_final,
                'edges_removed': result.edges_removed,
                'runtime': result.runtime
            }
            results_data.append(result_dict)
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logging.info(f"Results saved to {filename}")


def main():
    """Run gamma parameter weight sensitivity investigation."""
    
    print("🔬 CMG GAMMA PARAMETER WEIGHT SENSITIVITY INVESTIGATION")
    print("=" * 65)
    print("Testing hypothesis: Gamma insensitivity is caused by unweighted graphs")
    print()
    print("This investigation will test CMG across multiple edge weight scenarios:")
    print("  • Unweighted graphs (current testing)")
    print("  • Light weight variation (2-5x)")
    print("  • Moderate weight variation (10-20x)")
    print("  • Heavy weight variation (100x)")
    print("  • Extreme weight variation (1000x+)")
    print("  • Bimodal weight distributions")
    print()
    print("Expected runtime: 10-15 minutes")
    print("=" * 65)
    print()
    
    investigator = GammaWeightInvestigator()
    
    try:
        investigator.run_comprehensive_weight_investigation()
        
        print("\n✅ Weight sensitivity investigation completed!")
        print("\n🎯 If your hypothesis is correct, you should see:")
        print("   • Gamma insensitivity in unweighted scenarios")
        print("   • Gamma sensitivity in heavily weighted scenarios")
        print("   • Transition point where parameter becomes functional")
        
    except KeyboardInterrupt:
        print("\n⚠️  Investigation interrupted by user")
        if investigator.results:
            print("Analyzing partial results...")
            investigator.analyze_weight_sensitivity()
            investigator.save_results()
    
    except Exception as e:
        print(f"\n❌ Investigation failed: {e}")
        logging.error(f"Weight investigation failed: {e}")


if __name__ == "__main__":
    main()
