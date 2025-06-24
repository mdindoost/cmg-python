#!/usr/bin/env python3
"""
CMG Systematic Evaluation - Step 1
==================================

Comprehensive evaluation to understand when and why CMG works well.

This script conducts systematic evaluation across different graph types
to characterize CMG's strengths, weaknesses, and optimal use cases.
"""

import sys
import os
sys.path.append('..')  # Add parent directory to path for CMG imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path
import pickle
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# CMG imports
from cmg import CMGSteinerSolver, create_laplacian_from_edges
from cmg.utils.graph_utils import create_test_graphs, create_random_graph, create_clustered_graph

# Additional imports for baselines
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("⚠️  NetworkX not available. Install with: pip install networkx")

try:
    from sklearn.cluster import SpectralClustering
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️  scikit-learn not available. Install with: pip install scikit-learn")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)

class CMGSystematicEvaluator:
    """
    Systematic evaluator for CMG algorithm across different graph types.
    
    This class implements comprehensive evaluation methodology to understand:
    1. When CMG works well vs poorly
    2. What graph properties predict CMG success  
    3. How CMG compares to baseline methods
    4. Parameter sensitivity analysis
    """
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize CMG solver with default parameters
        self.cmg_solver = CMGSteinerSolver(gamma=5.0, verbose=False)
        
        # Results storage
        self.results = {
            'cmg_results': [],
            'baseline_results': [],
            'graph_properties': [],
            'timing_results': []
        }
        
        logging.info("CMG Systematic Evaluator initialized")
    
    def evaluate_graph(self, graph_name: str, A, graph_properties: Dict) -> Dict:
        """
        Evaluate CMG on a single graph and collect comprehensive metrics.
        
        Args:
            graph_name: Identifier for the graph
            A: Graph Laplacian matrix
            graph_properties: Pre-computed graph properties
            
        Returns:
            Dictionary with evaluation results
        """
        logging.info(f"Evaluating graph: {graph_name}")
        
        n = A.shape[0]
        results = {
            'graph_name': graph_name,
            'n_nodes': n,
            'n_edges': A.nnz // 2,  # Laplacian has double entries
            **graph_properties
        }
        
        # Test different gamma values
        gamma_values = [2.0, 3.0, 4.1, 5.0, 7.0, 10.0, 15.0, 20.0]
        
        for gamma in gamma_values:
            try:
                # CMG evaluation
                solver = CMGSteinerSolver(gamma=gamma, verbose=False)
                
                start_time = time.time()
                components, num_communities = solver.steiner_group(A)
                cmg_time = time.time() - start_time
                
                # Get detailed statistics
                stats = solver.get_statistics()
                
                # Calculate additional metrics
                cmg_results = {
                    'gamma': gamma,
                    'num_communities': num_communities,
                    'cmg_time': cmg_time,
                    'avg_conductance': stats.get('avg_conductance', float('inf')),
                    'avg_component_size': stats.get('avg_component_size', 0),
                    'high_degree_nodes': stats.get('high_degree_nodes', 0),
                    'edges_removed': stats.get('edges_removed', 0)
                }
                
                # Store results
                self.results['cmg_results'].append({
                    **results,
                    **cmg_results
                })
                
            except Exception as e:
                logging.error(f"CMG failed on {graph_name} with gamma={gamma}: {e}")
                continue
        
        # Baseline comparisons (if available)
        if HAS_SKLEARN and n < 5000:  # Spectral clustering can be slow
            try:
                self._evaluate_spectral_clustering(graph_name, A, results)
            except Exception as e:
                logging.warning(f"Spectral clustering failed on {graph_name}: {e}")
        
        return results
    
    def _evaluate_spectral_clustering(self, graph_name: str, A, base_results: Dict):
        """Evaluate spectral clustering baseline."""
        
        # Convert Laplacian to adjacency matrix
        A_adj = -A.copy()
        A_adj.setdiag(0)
        A_adj.eliminate_zeros()
        A_adj.data = np.abs(A_adj.data)
        
        # Test different numbers of clusters
        for n_clusters in [2, 3, 4, 5, 8, 10]:
            if n_clusters >= A.shape[0]:
                continue
                
            try:
                start_time = time.time()
                spectral = SpectralClustering(
                    n_clusters=n_clusters, 
                    affinity='precomputed',
                    random_state=42
                )
                labels = spectral.fit_predict(A_adj.toarray())
                spectral_time = time.time() - start_time
                
                self.results['baseline_results'].append({
                    **base_results,
                    'method': 'spectral_clustering',
                    'n_clusters': n_clusters,
                    'time': spectral_time,
                    'labels': labels.tolist()
                })
                
            except Exception as e:
                logging.warning(f"Spectral clustering failed with {n_clusters} clusters: {e}")
                continue
    
    def calculate_graph_properties(self, A, graph_type: str = "unknown") -> Dict:
        """
        Calculate comprehensive graph properties for analysis.
        
        Args:
            A: Graph Laplacian matrix
            graph_type: Type of graph for context
            
        Returns:
            Dictionary of graph properties
        """
        n = A.shape[0]
        
        # Convert to adjacency matrix for analysis
        A_adj = -A.copy()
        A_adj.setdiag(0)
        A_adj.eliminate_zeros()
        A_adj.data = np.abs(A_adj.data)
        
        # Basic properties
        n_edges = A_adj.nnz // 2
        density = n_edges / (n * (n - 1) / 2) if n > 1 else 0
        
        # Degree statistics
        degrees = np.array(A_adj.sum(axis=1)).flatten()
        
        properties = {
            'graph_type': graph_type,
            'density': density,
            'avg_degree': np.mean(degrees),
            'max_degree': np.max(degrees),
            'min_degree': np.min(degrees),
            'degree_std': np.std(degrees),
            'degree_skewness': self._calculate_skewness(degrees)
        }
        
        # Spectral properties (for smaller graphs)
        if n < 1000:
            try:
                # Calculate Laplacian eigenvalues
                eigenvals = np.linalg.eigvals(A.toarray())
                eigenvals = np.sort(np.real(eigenvals[eigenvals > 1e-10]))
                
                if len(eigenvals) > 1:
                    properties.update({
                        'spectral_gap': eigenvals[1] - eigenvals[0] if len(eigenvals) > 1 else 0,
                        'algebraic_connectivity': eigenvals[1] if len(eigenvals) > 1 else 0
                    })
            except Exception as e:
                logging.warning(f"Failed to compute spectral properties: {e}")
        
        # NetworkX properties (if available and graph not too large)
        if HAS_NETWORKX and n < 2000:
            try:
                G = nx.from_scipy_sparse_array(A_adj)
                
                if nx.is_connected(G):
                    properties.update({
                        'avg_clustering': nx.average_clustering(G),
                        'diameter': nx.diameter(G) if n < 500 else -1  # Expensive for large graphs
                    })
                else:
                    properties.update({
                        'avg_clustering': nx.average_clustering(G),
                        'diameter': -1,
                        'num_components': nx.number_connected_components(G)
                    })
                    
            except Exception as e:
                logging.warning(f"Failed to compute NetworkX properties: {e}")
        
        return properties
    
    def _calculate_skewness(self, data):
        """Calculate skewness of data."""
        if len(data) < 3:
            return 0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def generate_test_graphs(self) -> Dict:
        """
        Generate comprehensive set of test graphs for evaluation.
        
        Returns:
            Dictionary mapping graph names to (matrix, properties) tuples
        """
        logging.info("Generating test graphs...")
        
        graphs = {}
        
        # 1. Built-in test graphs
        test_graphs = create_test_graphs()
        for name, graph_data in test_graphs.items():
            A = create_laplacian_from_edges(graph_data['edges'], graph_data['n'])
            properties = self.calculate_graph_properties(A, 'builtin_test')
            graphs[f"builtin_{name}"] = (A, properties)
        
        # 2. Grid graphs (structured)
        for size in [10, 20, 30]:
            edges = []
            n = size * size
            for i in range(size):
                for j in range(size):
                    node = i * size + j
                    # Right neighbor
                    if j < size - 1:
                        edges.append((node, node + 1, 1.0))
                    # Bottom neighbor  
                    if i < size - 1:
                        edges.append((node, node + size, 1.0))
            
            A = create_laplacian_from_edges(edges, n)
            properties = self.calculate_graph_properties(A, 'grid')
            graphs[f"grid_{size}x{size}"] = (A, properties)
        
        # 3. Random graphs
        for n in [50, 100, 200]:
            for p in [0.1, 0.2, 0.3]:
                edges, _ = create_random_graph(n, p, seed=42)
                if edges:  # Only if graph is not empty
                    A = create_laplacian_from_edges(edges, n)
                    properties = self.calculate_graph_properties(A, 'random')
                    graphs[f"random_n{n}_p{p}"] = (A, properties)
        
        # 4. Clustered graphs (CMG should work well here)
        cluster_configs = [
            ([10, 10, 10], "small_clusters"),
            ([20, 20, 20], "medium_clusters"), 
            ([15, 15, 15, 15], "four_clusters")
        ]
        
        for cluster_sizes, desc in cluster_configs:
            edges, n = create_clustered_graph(
                cluster_sizes=cluster_sizes,
                intra_cluster_p=0.8,
                inter_cluster_p=0.05,
                seed=42
            )
            A = create_laplacian_from_edges(edges, n)
            properties = self.calculate_graph_properties(A, 'clustered')
            graphs[f"clustered_{desc}"] = (A, properties)
        
        logging.info(f"Generated {len(graphs)} test graphs")
        return graphs
    
    def run_systematic_evaluation(self):
        """
        Run the complete systematic evaluation.
        """
        logging.info("Starting systematic CMG evaluation...")
        
        # Generate test graphs
        graphs = self.generate_test_graphs()
        
        # Evaluate each graph
        for graph_name, (A, properties) in graphs.items():
            try:
                self.evaluate_graph(graph_name, A, properties)
            except Exception as e:
                logging.error(f"Failed to evaluate {graph_name}: {e}")
                continue
        
        # Save results
        self.save_results()
        
        # Generate initial analysis
        self.generate_summary_report()
        
        logging.info("Systematic evaluation completed!")
    
    def save_results(self):
        """Save all results to files."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save as pickle for complete data
        with open(self.results_dir / f"cmg_evaluation_results_{timestamp}.pkl", 'wb') as f:
            pickle.dump(self.results, f)
        
        # Save as CSV for analysis
        if self.results['cmg_results']:
            df_cmg = pd.DataFrame(self.results['cmg_results'])
            df_cmg.to_csv(self.results_dir / f"cmg_results_{timestamp}.csv", index=False)
        
        if self.results['baseline_results']:
            df_baseline = pd.DataFrame(self.results['baseline_results'])
            df_baseline.to_csv(self.results_dir / f"baseline_results_{timestamp}.csv", index=False)
        
        logging.info(f"Results saved with timestamp: {timestamp}")
    
    def generate_summary_report(self):
        """Generate summary report of findings."""
        if not self.results['cmg_results']:
            logging.warning("No CMG results to analyze")
            return
        
        df = pd.DataFrame(self.results['cmg_results'])
        
        print("\n" + "="*60)
        print("CMG SYSTEMATIC EVALUATION - SUMMARY REPORT")
        print("="*60)
        
        print(f"\n📊 EVALUATION OVERVIEW:")
        print(f"   Total graphs evaluated: {df['graph_name'].nunique()}")
        print(f"   Total CMG runs: {len(df)}")
        print(f"   Graph types: {', '.join(df['graph_type'].unique())}")
        
        print(f"\n🎯 CMG PERFORMANCE BY GRAPH TYPE:")
        for graph_type in df['graph_type'].unique():
            subset = df[df['graph_type'] == graph_type]
            avg_communities = subset['num_communities'].mean()
            avg_time = subset['cmg_time'].mean()
            avg_conductance = subset[subset['avg_conductance'] != float('inf')]['avg_conductance'].mean()
            
            print(f"   {graph_type}:")
            print(f"     Avg communities: {avg_communities:.1f}")
            print(f"     Avg time: {avg_time:.4f}s") 
            print(f"     Avg conductance: {avg_conductance:.6f}" if not np.isnan(avg_conductance) else "     Avg conductance: N/A")
        
        print(f"\n⚙️ PARAMETER SENSITIVITY:")
        gamma_analysis = df.groupby('gamma').agg({
            'num_communities': 'mean',
            'avg_conductance': lambda x: x[x != float('inf')].mean(),
            'cmg_time': 'mean'
        }).round(4)
        print(gamma_analysis)
        
        print(f"\n🏆 BEST PERFORMING CONFIGURATIONS:")
        # Find configurations with good conductance and reasonable community count
        good_results = df[
            (df['avg_conductance'] != float('inf')) & 
            (df['avg_conductance'] < 1.0) &
            (df['num_communities'] > 1) &
            (df['num_communities'] < df['n_nodes'] / 2)
        ]
        
        if len(good_results) > 0:
            best = good_results.nsmallest(5, 'avg_conductance')[
                ['graph_name', 'gamma', 'num_communities', 'avg_conductance', 'cmg_time']
            ]
            print(best.to_string(index=False))
        else:
            print("   No configurations met quality criteria")
        
        print("\n" + "="*60)


def main():
    """Main evaluation script."""
    print("🔍 CMG Systematic Evaluation - Step 1")
    print("=====================================")
    
    # Create evaluator
    evaluator = CMGSystematicEvaluator()
    
    # Run systematic evaluation
    evaluator.run_systematic_evaluation()
    
    print("\n✅ Evaluation completed!")
    print("\n💡 Next steps:")
    print("   1. Review the summary report above")
    print("   2. Check results/ directory for detailed CSV files")
    print("   3. Run analysis notebooks for deeper insights")
    print("   4. Use findings to guide Step 2 (theoretical analysis)")


if __name__ == "__main__":
    main()
