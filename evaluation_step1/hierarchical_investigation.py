#!/usr/bin/env python3
"""
CMG Hierarchical Clustering Investigation
=========================================

This script systematically tests the hypothesis that CMG is actually
a hierarchical clustering algorithm that finds fine-grained substructure
within coarse clusters.
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# CMG imports
from cmg import CMGSteinerSolver, create_laplacian_from_edges
from cmg.utils.graph_utils import create_clustered_graph

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class HierarchicalClusteringInvestigator:
    """
    Investigates CMG's hierarchical clustering behavior.
    """
    
    def __init__(self):
        self.results = {
            'hierarchical_tests': [],
            'substructure_analysis': [],
            'comparison_tests': []
        }
        
        # Test different gamma values to confirm insensitivity
        self.gamma_values = [4.1, 5.0, 10.0, 20.0, 50.0, 100.0]
        
    def create_hierarchical_test_graphs(self) -> Dict:
        """
        Create graphs with known hierarchical structure to test CMG.
        
        Returns:
            Dictionary of test graphs with known hierarchical ground truth
        """
        logging.info("Creating hierarchical test graphs...")
        
        test_graphs = {}
        
        # 1. Simple 2-level hierarchy: 2 main clusters, each with 2 subclusters
        hierarchical_configs = [
            {
                'name': 'simple_2x2',
                'main_clusters': 2,
                'subclusters_per_main': 2,
                'subcluster_size': 10,
                'description': '2 main clusters, each with 2 subclusters of 10 nodes'
            },
            {
                'name': 'complex_2x3', 
                'main_clusters': 2,
                'subclusters_per_main': 3,
                'subcluster_size': 8,
                'description': '2 main clusters, each with 3 subclusters of 8 nodes'
            },
            {
                'name': 'unbalanced_hier',
                'main_clusters': 2,
                'subclusters_per_main': [2, 4],  # Unbalanced
                'subcluster_sizes': [12, 8, 6, 6, 6, 6],
                'description': 'Unbalanced: cluster 1 has 2 subclusters, cluster 2 has 4'
            }
        ]
        
        for config in hierarchical_configs[:2]:  # Start with first two
            test_graphs.update(self._create_balanced_hierarchical_graph(config))
        
        # 3. Create unbalanced hierarchical graph
        test_graphs.update(self._create_unbalanced_hierarchical_graph())
        
        # 4. Create graphs with different inter-cluster strengths
        test_graphs.update(self._create_varied_strength_graphs())
        
        logging.info(f"Created {len(test_graphs)} hierarchical test graphs")
        return test_graphs
    
    def _create_balanced_hierarchical_graph(self, config: Dict) -> Dict:
        """Create balanced hierarchical graphs."""
        
        graphs = {}
        
        for seed in [42, 123]:
            # Create subclusters
            subcluster_size = config['subcluster_size']
            n_subclusters = config['main_clusters'] * config['subclusters_per_main']
            
            # All subclusters have same size for balanced case
            cluster_sizes = [subcluster_size] * n_subclusters
            
            edges, n = create_clustered_graph(
                cluster_sizes=cluster_sizes,
                intra_cluster_p=0.8,    # High within subcluster
                inter_cluster_p=0.1,    # Medium between subclusters 
                seed=seed
            )
            
            if not edges:
                continue
            
            # Create hierarchical ground truth
            # Level 1: Main clusters (every subclusters_per_main subclusters belong to same main cluster)
            main_cluster_labels = []
            subcluster_labels = []
            
            for main_cluster_id in range(config['main_clusters']):
                for sub_id in range(config['subclusters_per_main']):
                    subcluster_id = main_cluster_id * config['subclusters_per_main'] + sub_id
                    for _ in range(subcluster_size):
                        main_cluster_labels.append(main_cluster_id)
                        subcluster_labels.append(subcluster_id)
            
            A = create_laplacian_from_edges(edges, n)
            
            metadata = {
                'type': 'hierarchical',
                'n_nodes': n,
                'n_main_clusters': config['main_clusters'],
                'n_subclusters': n_subclusters,
                'subcluster_size': subcluster_size,
                'description': config['description'],
                'seed': seed,
                'main_cluster_labels': np.array(main_cluster_labels),
                'subcluster_labels': np.array(subcluster_labels),
                'config': config
            }
            
            graph_name = f"hierarchical_{config['name']}_seed{seed}"
            graphs[graph_name] = (A, metadata)
        
        return graphs
    
    def _create_unbalanced_hierarchical_graph(self) -> Dict:
        """Create unbalanced hierarchical graph."""
        
        graphs = {}
        
        # Cluster 1: 2 subclusters of [12, 8]
        # Cluster 2: 4 subclusters of [6, 6, 6, 6]
        cluster_sizes = [12, 8, 6, 6, 6, 6]
        
        edges, n = create_clustered_graph(
            cluster_sizes=cluster_sizes,
            intra_cluster_p=0.8,
            inter_cluster_p=0.05,
            seed=42
        )
        
        if edges:
            # Ground truth: main clusters [0,0,...,1,1,1,1]
            main_labels = [0]*20 + [1]*24  # 20 nodes in cluster 1, 24 in cluster 2
            subcluster_labels = ([0]*12 + [1]*8 +     # Subclusters 0,1 in main cluster 0  
                               [2]*6 + [3]*6 + [4]*6 + [5]*6)  # Subclusters 2,3,4,5 in main cluster 1
            
            A = create_laplacian_from_edges(edges, n)
            
            metadata = {
                'type': 'hierarchical_unbalanced',
                'n_nodes': n,
                'n_main_clusters': 2,
                'n_subclusters': 6,
                'description': 'Unbalanced: 2 vs 4 subclusters',
                'main_cluster_labels': np.array(main_labels),
                'subcluster_labels': np.array(subcluster_labels),
                'cluster_sizes': cluster_sizes
            }
            
            graphs['hierarchical_unbalanced'] = (A, metadata)
        
        return graphs
    
    def _create_varied_strength_graphs(self) -> Dict:
        """Create graphs with different inter-cluster connection strengths."""
        
        graphs = {}
        
        # Test different inter-cluster probabilities
        inter_probs = [0.01, 0.05, 0.1, 0.2]  # Very weak to moderate
        
        for inter_p in inter_probs:
            edges, n = create_clustered_graph(
                cluster_sizes=[15, 15, 15, 15],  # 4 subclusters
                intra_cluster_p=0.8,
                inter_cluster_p=inter_p,
                seed=42
            )
            
            if edges:
                # Ground truth: 2 main clusters, each with 2 subclusters
                main_labels = [0]*30 + [1]*30
                subcluster_labels = [0]*15 + [1]*15 + [2]*15 + [3]*15
                
                A = create_laplacian_from_edges(edges, n)
                
                metadata = {
                    'type': 'varied_strength',
                    'n_nodes': n,
                    'inter_prob': inter_p,
                    'description': f'4 subclusters with inter_p={inter_p}',
                    'main_cluster_labels': np.array(main_labels),
                    'subcluster_labels': np.array(subcluster_labels)
                }
                
                graphs[f'varied_strength_p{inter_p}'] = (A, metadata)
        
        return graphs
    
    def test_cmg_hierarchical_behavior(self, test_graphs: Dict) -> None:
        """
        Test CMG on hierarchical graphs and analyze its behavior.
        """
        logging.info("Testing CMG hierarchical behavior...")
        
        for graph_name, (A, metadata) in test_graphs.items():
            logging.info(f"Testing {graph_name}")
            
            # Test multiple gamma values
            for gamma in self.gamma_values:
                try:
                    solver = CMGSteinerSolver(gamma=gamma, verbose=False)
                    start_time = time.time()
                    components, num_communities = solver.steiner_group(A)
                    runtime = time.time() - start_time
                    
                    # Analyze hierarchical structure
                    analysis = self._analyze_hierarchical_structure(
                        components, metadata, graph_name, gamma
                    )
                    
                    result = {
                        'graph_name': graph_name,
                        'gamma': gamma,
                        'num_communities': num_communities,
                        'runtime': runtime,
                        'hierarchical_analysis': analysis,
                        **metadata
                    }
                    
                    self.results['hierarchical_tests'].append(result)
                    
                except Exception as e:
                    logging.error(f"CMG failed on {graph_name} with gamma={gamma}: {e}")
        
        self._print_hierarchical_analysis()
    
    def _analyze_hierarchical_structure(self, cmg_labels: np.ndarray, 
                                      metadata: Dict, graph_name: str, gamma: float) -> Dict:
        """
        Analyze how well CMG's clustering aligns with hierarchical structure.
        """
        main_labels = metadata['main_cluster_labels']
        subcluster_labels = metadata['subcluster_labels']
        
        # Check if CMG respects main cluster boundaries
        main_cluster_purity = self._calculate_cluster_purity(cmg_labels, main_labels)
        
        # Check if CMG finds the subclusters
        subcluster_alignment = self._calculate_subcluster_alignment(cmg_labels, subcluster_labels)
        
        # Analyze CMG cluster composition
        cmg_composition = self._analyze_cmg_composition(cmg_labels, main_labels, subcluster_labels)
        
        return {
            'main_cluster_purity': main_cluster_purity,
            'subcluster_alignment': subcluster_alignment, 
            'cmg_composition': cmg_composition,
            'respects_hierarchy': main_cluster_purity > 0.95  # Almost perfect separation
        }
    
    def _calculate_cluster_purity(self, cmg_labels: np.ndarray, true_labels: np.ndarray) -> float:
        """
        Calculate how well CMG clusters respect true cluster boundaries.
        Returns 1.0 if no CMG cluster mixes nodes from different true clusters.
        """
        n_pure_nodes = 0
        
        for cmg_cluster_id in np.unique(cmg_labels):
            nodes_in_cmg_cluster = np.where(cmg_labels == cmg_cluster_id)[0]
            true_labels_in_cluster = true_labels[nodes_in_cmg_cluster]
            
            # If all nodes in this CMG cluster have the same true label, it's pure
            if len(np.unique(true_labels_in_cluster)) == 1:
                n_pure_nodes += len(nodes_in_cmg_cluster)
        
        return n_pure_nodes / len(cmg_labels)
    
    def _calculate_subcluster_alignment(self, cmg_labels: np.ndarray, 
                                      subcluster_labels: np.ndarray) -> float:
        """
        Calculate how well CMG clusters align with known subclusters.
        """
        # For each true subcluster, find the CMG cluster that contains most of its nodes
        total_correctly_assigned = 0
        
        for true_subcluster_id in np.unique(subcluster_labels):
            nodes_in_true_subcluster = np.where(subcluster_labels == true_subcluster_id)[0]
            cmg_labels_for_subcluster = cmg_labels[nodes_in_true_subcluster]
            
            # Find the most common CMG label for this true subcluster
            unique_cmg_labels, counts = np.unique(cmg_labels_for_subcluster, return_counts=True)
            most_common_cmg_label = unique_cmg_labels[np.argmax(counts)]
            max_count = np.max(counts)
            
            total_correctly_assigned += max_count
        
        return total_correctly_assigned / len(cmg_labels)
    
    def _analyze_cmg_composition(self, cmg_labels: np.ndarray, 
                               main_labels: np.ndarray, subcluster_labels: np.ndarray) -> Dict:
        """
        Analyze the composition of each CMG cluster.
        """
        composition = {}
        
        for cmg_cluster_id in np.unique(cmg_labels):
            nodes_in_cmg = np.where(cmg_labels == cmg_cluster_id)[0]
            
            # Count main clusters represented
            main_clusters_in_cmg = main_labels[nodes_in_cmg]
            main_cluster_counts = {int(k): int(v) for k, v in 
                                 zip(*np.unique(main_clusters_in_cmg, return_counts=True))}
            
            # Count subclusters represented  
            subclusters_in_cmg = subcluster_labels[nodes_in_cmg]
            subcluster_counts = {int(k): int(v) for k, v in 
                               zip(*np.unique(subclusters_in_cmg, return_counts=True))}
            
            composition[int(cmg_cluster_id)] = {
                'size': len(nodes_in_cmg),
                'main_clusters': main_cluster_counts,
                'subclusters': subcluster_counts,
                'is_pure': len(main_cluster_counts) == 1  # Only from one main cluster
            }
        
        return composition
    
    def _print_hierarchical_analysis(self):
        """Print comprehensive analysis of hierarchical clustering behavior."""
        
        if not self.results['hierarchical_tests']:
            print("No hierarchical test results to analyze")
            return
        
        print("\n" + "="*80)
        print("CMG HIERARCHICAL CLUSTERING ANALYSIS")
        print("="*80)
        
        # Group results by graph
        graphs = {}
        for result in self.results['hierarchical_tests']:
            graph_name = result['graph_name']
            if graph_name not in graphs:
                graphs[graph_name] = []
            graphs[graph_name].append(result)
        
        # Analyze each graph
        for graph_name, graph_results in graphs.items():
            print(f"\n📊 GRAPH: {graph_name}")
            print("-" * 50)
            
            # Print graph metadata
            first_result = graph_results[0]
            print(f"   Description: {first_result['description']}")
            print(f"   Nodes: {first_result['n_nodes']}")
            print(f"   Main clusters: {first_result['n_main_clusters']}")
            print(f"   Subclusters: {first_result['n_subclusters']}")
            
            # Analyze gamma sensitivity
            gamma_results = {}
            for result in graph_results:
                gamma = result['gamma']
                gamma_results[gamma] = result
            
            print(f"\n   Gamma Sensitivity Analysis:")
            all_same = True
            first_num_communities = list(gamma_results.values())[0]['num_communities']
            
            for gamma in sorted(gamma_results.keys()):
                result = gamma_results[gamma]
                num_comm = result['num_communities']
                purity = result['hierarchical_analysis']['main_cluster_purity']
                respects_hier = result['hierarchical_analysis']['respects_hierarchy']
                
                print(f"     γ={gamma:5.1f}: {num_comm:2d} communities, "
                      f"purity={purity:.3f}, respects_hierarchy={respects_hier}")
                
                if num_comm != first_num_communities:
                    all_same = False
            
            if all_same:
                print(f"     ⚠️  GAMMA INSENSITIVE: All gamma values produce identical results")
            else:
                print(f"     ✅ GAMMA SENSITIVE: Different gamma values produce different results")
            
            # Detailed analysis for gamma=5.0
            if 5.0 in gamma_results:
                result = gamma_results[5.0]
                analysis = result['hierarchical_analysis']
                
                print(f"\n   Detailed Analysis (γ=5.0):")
                print(f"     Main cluster purity: {analysis['main_cluster_purity']:.3f}")
                print(f"     Subcluster alignment: {analysis['subcluster_alignment']:.3f}")
                print(f"     Respects hierarchy: {analysis['respects_hierarchy']}")
                
                # Show CMG cluster composition
                composition = analysis['cmg_composition']
                pure_clusters = sum(1 for info in composition.values() if info['is_pure'])
                total_clusters = len(composition)
                
                print(f"     Pure clusters: {pure_clusters}/{total_clusters}")
                
                # Show a few example clusters
                print(f"     Example CMG clusters:")
                for cmg_id, info in list(composition.items())[:3]:
                    main_repr = list(info['main_clusters'].keys())
                    sub_repr = list(info['subclusters'].keys())
                    pure_str = "PURE" if info['is_pure'] else "MIXED"
                    print(f"       Cluster {cmg_id}: size={info['size']}, "
                          f"from_main={main_repr}, from_sub={sub_repr} ({pure_str})")
        
        # Overall conclusions
        print(f"\n🎯 OVERALL CONCLUSIONS:")
        
        # Check if CMG consistently respects hierarchy
        respects_hierarchy_count = sum(1 for result in self.results['hierarchical_tests'] 
                                     if result['hierarchical_analysis']['respects_hierarchy'])
        total_tests = len(self.results['hierarchical_tests'])
        
        print(f"   Hierarchy respect rate: {respects_hierarchy_count}/{total_tests} "
              f"({respects_hierarchy_count/total_tests:.1%})")
        
        # Check gamma insensitivity across all graphs
        gamma_insensitive_graphs = 0
        for graph_name, graph_results in graphs.items():
            communities = [r['num_communities'] for r in graph_results]
            if len(set(communities)) == 1:
                gamma_insensitive_graphs += 1
        
        print(f"   Gamma-insensitive graphs: {gamma_insensitive_graphs}/{len(graphs)}")
        
        # Average purity
        avg_purity = np.mean([r['hierarchical_analysis']['main_cluster_purity'] 
                             for r in self.results['hierarchical_tests']])
        print(f"   Average main cluster purity: {avg_purity:.3f}")
        
        if avg_purity > 0.95:
            print("   ✅ CMG EXCELLENT at respecting main cluster boundaries!")
        elif avg_purity > 0.8:
            print("   ✅ CMG GOOD at respecting main cluster boundaries")
        else:
            print("   ❌ CMG struggles with hierarchical structure")
        
        if gamma_insensitive_graphs == len(graphs):
            print("   ⚠️  CRITICAL: Gamma parameter appears non-functional")
        
        print("\n💡 RESEARCH IMPLICATIONS:")
        if avg_purity > 0.9:
            print("   📝 CMG could be positioned as hierarchical clustering algorithm")
            print("   📝 Focus on substructure detection capabilities")
            print("   📝 Compare against hierarchical clustering baselines")
        else:
            print("   📝 CMG has limitations for hierarchical clustering")
            print("   📝 Focus on understanding algorithm behavior")
        
        print("=" * 80)

def main():
    """Main hierarchical clustering investigation."""
    
    print("🔬 CMG Hierarchical Clustering Investigation")
    print("=" * 60)
    print("Testing hypothesis: CMG is a hierarchical clustering algorithm")
    print("that finds fine-grained substructure within coarse clusters.")
    print()
    
    # Create investigator
    investigator = HierarchicalClusteringInvestigator()
    
    # Create test graphs with known hierarchical structure
    test_graphs = investigator.create_hierarchical_test_graphs()
    
    print(f"Created {len(test_graphs)} hierarchical test graphs")
    print("Expected runtime: 5-10 minutes")
    print()
    
    # Test CMG on hierarchical graphs
    investigator.test_cmg_hierarchical_behavior(test_graphs)
    
    print("\n✅ Hierarchical clustering investigation completed!")
    print("\n🎯 This analysis will reveal if CMG is actually a hierarchical clustering algorithm!")

if __name__ == "__main__":
    main()
