#!/bin/bash
# CMG Systematic Evaluation Setup
# Creates folder structure and initial files for Step 1 analysis

echo "🔍 Setting up CMG Systematic Evaluation (Step 1)"
echo "================================================"

# Navigate to your cmg-python directory
cd ~/cmg-python

# Create main evaluation directory
mkdir -p evaluation_step1
cd evaluation_step1

# Create subdirectories for organization
mkdir -p {datasets,benchmarks,results,analysis,notebooks,scripts}

# Create subdirectories for different graph types
mkdir -p datasets/{grid_graphs,social_networks,road_networks,mesh_graphs,image_graphs,random_graphs,synthetic}

# Create results subdirectories
mkdir -p results/{cmg_results,baseline_results,comparisons,plots,reports}

# Create analysis subdirectories  
mkdir -p analysis/{statistical_tests,graph_properties,performance_analysis,parameter_sensitivity}

echo "📁 Created directory structure:"
tree -L 3 .

echo ""
echo "📋 Creating initial evaluation files..."

# Create main evaluation script
cat > evaluate_cmg_systematic.py << 'EOF'
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
EOF

# Create Jupyter notebook for analysis
cat > notebooks/step1_analysis.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# CMG Systematic Evaluation - Analysis Notebook\n",
    "\n",
    "This notebook analyzes the results from Step 1 systematic evaluation to understand:\n",
    "1. When CMG works well vs poorly\n",
    "2. What graph properties predict CMG success\n",
    "3. Parameter sensitivity analysis\n",
    "4. Comparison with baseline methods"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from pathlib import Path\n",
    "import pickle\n",
    "\n",
    "# Set up plotting\n",
    "plt.style.use('seaborn-v0_8')\n",
    "sns.set_palette('husl')\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load results\n",
    "results_dir = Path('../results')\n",
    "\n",
    "# Find most recent results file\n",
    "csv_files = list(results_dir.glob('cmg_results_*.csv'))\n",
    "if csv_files:\n",
    "    latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)\n",
    "    df = pd.read_csv(latest_file)\n",
    "    print(f\"Loaded {len(df)} results from {latest_file.name}\")\n",
    "else:\n",
    "    print(\"No results files found. Run evaluate_cmg_systematic.py first.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Overall Performance Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Performance by graph type\n",
    "fig, axes = plt.subplots(2, 2, figsize=(15, 10))\n",
    "\n",
    "# Number of communities by graph type\n",
    "sns.boxplot(data=df, x='graph_type', y='num_communities', ax=axes[0,0])\n",
    "axes[0,0].set_title('Number of Communities by Graph Type')\n",
    "axes[0,0].tick_params(axis='x', rotation=45)\n",
    "\n",
    "# Runtime by graph type\n",
    "sns.boxplot(data=df, x='graph_type', y='cmg_time', ax=axes[0,1])\n",
    "axes[0,1].set_title('Runtime by Graph Type')\n",
    "axes[0,1].tick_params(axis='x', rotation=45)\n",
    "\n",
    "# Conductance by graph type (excluding infinite values)\n",
    "df_finite_cond = df[df['avg_conductance'] != float('inf')]\n",
    "if len(df_finite_cond) > 0:\n",
    "    sns.boxplot(data=df_finite_cond, x='graph_type', y='avg_conductance', ax=axes[1,0])\n",
    "    axes[1,0].set_title('Conductance by Graph Type')\n",
    "    axes[1,0].tick_params(axis='x', rotation=45)\n",
    "\n",
    "# Parameter sensitivity\n",
    "sns.boxplot(data=df, x='gamma', y='num_communities', ax=axes[1,1])\n",
    "axes[1,1].set_title('Communities vs Gamma Parameter')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Graph Properties vs Performance Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Correlation analysis\n",
    "numeric_cols = df.select_dtypes(include=[np.number]).columns\n",
    "correlation_matrix = df[numeric_cols].corr()\n",
    "\n",
    "plt.figure(figsize=(12, 10))\n",
    "sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')\n",
    "plt.title('Correlation Matrix: Graph Properties vs CMG Performance')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Success Criteria Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define success criteria\n",
    "df['is_successful'] = (\n",
    "    (df['avg_conductance'] != float('inf')) & \n",
    "    (df['avg_conductance'] < 1.0) &\n",
    "    (df['num_communities'] > 1) &\n",
    "    (df['num_communities'] < df['n_nodes'] / 2)\n",
    ")\n",
    "\n",
    "print(\"Success Rate Analysis:\")\n",
    "print(f\"Overall success rate: {df['is_successful'].mean():.2%}\")\n",
    "print(\"\\nSuccess rate by graph type:\")\n",
    "print(df.groupby('graph_type')['is_successful'].mean().sort_values(ascending=False))\n",
    "\n",
    "print(\"\\nSuccess rate by gamma:\")\n",
    "print(df.groupby('gamma')['is_successful'].mean().sort_values(ascending=False))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Insights and Recommendations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Find best performing configurations\n",
    "successful_runs = df[df['is_successful']]\n",
    "\n",
    "if len(successful_runs) > 0:\n",
    "    print(\"🏆 TOP PERFORMING CONFIGURATIONS:\")\n",
    "    top_configs = successful_runs.nsmallest(10, 'avg_conductance')[[\n",
    "        'graph_name', 'graph_type', 'gamma', 'num_communities', \n",
    "        'avg_conductance', 'cmg_time', 'density'\n",
    "    ]]\n",
    "    print(top_configs.to_string(index=False))\n",
    "    \n",
    "    print(\"\\n🔍 KEY INSIGHTS:\")\n",
    "    best_graph_types = successful_runs['graph_type'].value_counts()\n",
    "    print(f\"Best graph types: {best_graph_types.head(3).to_dict()}\")\n",
    "    \n",
    "    best_gamma = successful_runs.groupby('gamma')['avg_conductance'].mean().idxmin()\n",
    "    print(f\"Best gamma value: {best_gamma}\")\n",
    "    \n",
    "    avg_density = successful_runs['density'].mean()\n",
    "    print(f\"Avg density of successful graphs: {avg_density:.4f}\")\n",
    "else:\n",
    "    print(\"❌ No configurations met success criteria.\")\n",
    "    print(\"Consider relaxing criteria or investigating why CMG struggles.\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

# Create requirements file for evaluation
cat > requirements_evaluation.txt << 'EOF'
# Additional requirements for systematic evaluation
pandas>=1.3.0
matplotlib>=3.3.0
seaborn>=0.11.0
jupyter>=1.0.0
scikit-learn>=0.24.0
networkx>=2.5
# For statistical analysis
scipy>=1.7.0
statsmodels>=0.12.0
EOF

# Create README for evaluation
cat > README.md << 'EOF'
# CMG Systematic Evaluation - Step 1

This directory contains a comprehensive evaluation framework to understand when and why CMG works well.

## 📁 Directory Structure

```
evaluation_step1/
├── datasets/           # Test graph datasets organized by type
├── benchmarks/         # Benchmark comparison scripts  
├── results/           # Evaluation results (CSV, pickle files)
├── analysis/          # Statistical analysis scripts
├── notebooks/         # Jupyter notebooks for interactive analysis
├── scripts/           # Utility scripts
├── evaluate_cmg_systematic.py  # Main evaluation script
└── README.md          # This file
```

## 🚀 Quick Start

1. **Install additional requirements:**
   ```bash
   pip install -r requirements_evaluation.txt
   ```

2. **Run systematic evaluation:**
   ```bash
   python evaluate_cmg_systematic.py
   ```

3. **Analyze results:**
   ```bash
   jupyter notebook notebooks/step1_analysis.ipynb
   ```

## 📊 What Gets Evaluated

### Graph Types Tested:
- **Grid graphs**: Regular lattice structures
- **Random graphs**: Erdős–Rényi graphs with different densities  
- **Clustered graphs**: Graphs with known community structure
- **Built-in test graphs**: From your existing test suite

### Metrics Collected:
- **Performance**: Runtime, scalability
- **Quality**: Conductance, community structure
- **Graph properties**: Density, degree distribution, spectral properties
- **Parameter sensitivity**: Different γ values

### Comparisons:
- **Spectral clustering**: Standard baseline
- **Parameter variations**: Different CMG settings
- **Graph property analysis**: What makes CMG work well

## 📈 Expected Outcomes

After running this evaluation, you'll have:

1. **Clear characterization** of when CMG works well vs poorly
2. **Data-driven insights** about optimal parameters
3. **Graph property predictors** for CMG success
4. **Baseline comparisons** showing CMG's strengths/weaknesses
5. **Foundation for Step 2** theoretical analysis

## 🔧 Customization

### Adding New Graph Types:
Edit `generate_test_graphs()` in `evaluate_cmg_systematic.py`

### Adding New Baselines:
Implement comparison methods in `CMGSystematicEvaluator`

### Modifying Metrics:
Update `evaluate_graph()` method to collect additional metrics

## 📋 Next Steps

1. Run the evaluation and review results
2. Use Jupyter notebook for deeper analysis
3. Identify CMG's "sweet spot" from the data
4. Proceed to Step 2 with data-driven insights

## 📝 Notes

- First run may take 10-30 minutes depending on your system
- Results are automatically saved with timestamps
- All graphs are generated deterministically (seed=42) for reproducibility
- Large graphs (>5000 nodes) skip expensive baseline comparisons
EOF

# Create additional analysis scripts
mkdir -p scripts

cat > scripts/parameter_sensitivity_analysis.py << 'EOF'
#!/usr/bin/env python3
"""
Parameter Sensitivity Analysis for CMG
=====================================

This script performs detailed analysis of how CMG parameters affect performance
across different graph types.
"""

import sys
sys.path.append('../..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_parameter_sensitivity(results_file):
    """
    Analyze how gamma parameter affects CMG performance across graph types.
    """
    df = pd.read_csv(results_file)
    
    print("🔧 Parameter Sensitivity Analysis")
    print("=" * 50)
    
    # Analysis by graph type and gamma
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Number of communities vs gamma
    for graph_type in df['graph_type'].unique():
        subset = df[df['graph_type'] == graph_type]
        gamma_means = subset.groupby('gamma')['num_communities'].mean()
        axes[0,0].plot(gamma_means.index, gamma_means.values, 
                       marker='o', label=graph_type)
    
    axes[0,0].set_xlabel('Gamma Parameter')
    axes[0,0].set_ylabel('Average Number of Communities')
    axes[0,0].set_title('Communities vs Gamma by Graph Type')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Runtime vs gamma
    for graph_type in df['graph_type'].unique():
        subset = df[df['graph_type'] == graph_type]
        gamma_means = subset.groupby('gamma')['cmg_time'].mean()
        axes[0,1].plot(gamma_means.index, gamma_means.values, 
                       marker='s', label=graph_type)
    
    axes[0,1].set_xlabel('Gamma Parameter')
    axes[0,1].set_ylabel('Average Runtime (seconds)')
    axes[0,1].set_title('Runtime vs Gamma by Graph Type')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Conductance vs gamma (finite values only)
    df_finite = df[df['avg_conductance'] != float('inf')]
    if len(df_finite) > 0:
        for graph_type in df_finite['graph_type'].unique():
            subset = df_finite[df_finite['graph_type'] == graph_type]
            if len(subset) > 0:
                gamma_means = subset.groupby('gamma')['avg_conductance'].mean()
                axes[1,0].plot(gamma_means.index, gamma_means.values, 
                               marker='^', label=graph_type)
        
        axes[1,0].set_xlabel('Gamma Parameter')
        axes[1,0].set_ylabel('Average Conductance')
        axes[1,0].set_title('Conductance vs Gamma by Graph Type')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
    
    # 4. Success rate vs gamma
    df['is_successful'] = (
        (df['avg_conductance'] != float('inf')) & 
        (df['avg_conductance'] < 1.0) &
        (df['num_communities'] > 1) &
        (df['num_communities'] < df['n_nodes'] / 2)
    )
    
    success_by_gamma = df.groupby(['graph_type', 'gamma'])['is_successful'].mean().unstack()
    success_by_gamma.plot(kind='bar', ax=axes[1,1])
    axes[1,1].set_xlabel('Graph Type')
    axes[1,1].set_ylabel('Success Rate')
    axes[1,1].set_title('Success Rate by Graph Type and Gamma')
    axes[1,1].legend(title='Gamma', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Print recommendations
    print("\n🎯 PARAMETER RECOMMENDATIONS:")
    
    # Best gamma overall
    best_gamma_overall = df.groupby('gamma')['is_successful'].mean().idxmax()
    print(f"Best gamma overall: {best_gamma_overall}")
    
    # Best gamma by graph type
    print("\nBest gamma by graph type:")
    for graph_type in df['graph_type'].unique():
        subset = df[df['graph_type'] == graph_type]
        if len(subset) > 0:
            best_gamma = subset.groupby('gamma')['is_successful'].mean().idxmax()
            success_rate = subset.groupby('gamma')['is_successful'].mean().max()
            print(f"  {graph_type}: γ={best_gamma} (success rate: {success_rate:.2%})")

if __name__ == "__main__":
    # Find most recent results
    results_dir = Path('../results')
    csv_files = list(results_dir.glob('cmg_results_*.csv'))
    
    if csv_files:
        latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
        analyze_parameter_sensitivity(latest_file)
    else:
        print("No results files found. Run evaluate_cmg_systematic.py first.")
EOF

cat > scripts/graph_property_analysis.py << 'EOF'
#!/usr/bin/env python3
"""
Graph Property Analysis for CMG
==============================

This script analyzes which graph properties are most predictive of CMG success.
"""

import sys
sys.path.append('../..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def analyze_predictive_properties(results_file):
    """
    Analyze which graph properties best predict CMG success.
    """
    df = pd.read_csv(results_file)
    
    print("🔍 Graph Property Predictive Analysis")
    print("=" * 50)
    
    # Define success criteria
    df['is_successful'] = (
        (df['avg_conductance'] != float('inf')) & 
        (df['avg_conductance'] < 1.0) &
        (df['num_communities'] > 1) &
        (df['num_communities'] < df['n_nodes'] / 2)
    )
    
    # Prepare features for machine learning
    feature_cols = [
        'n_nodes', 'n_edges', 'density', 'avg_degree', 'max_degree', 
        'min_degree', 'degree_std', 'degree_skewness', 'gamma'
    ]
    
    # Only use rows with all features available
    df_ml = df.dropna(subset=feature_cols + ['is_successful'])
    
    if len(df_ml) < 20:
        print("Insufficient data for machine learning analysis")
        return
    
    X = df_ml[feature_cols]
    y = df_ml['is_successful']
    
    # Train random forest to identify important features
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🎯 FEATURE IMPORTANCE (Random Forest):")
    print(feature_importance.to_string(index=False))
    
    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Feature Importance for Predicting CMG Success')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.show()
    
    # Performance on test set
    y_pred = rf.predict(X_test)
    print(f"\n📊 MODEL PERFORMANCE:")
    print(f"Accuracy: {rf.score(X_test, y_test):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Analyze successful vs unsuccessful cases
    print(f"\n📈 SUCCESSFUL vs UNSUCCESSFUL ANALYSIS:")
    
    successful = df_ml[df_ml['is_successful']]
    unsuccessful = df_ml[~df_ml['is_successful']]
    
    print(f"Successful cases: {len(successful)}")
    print(f"Unsuccessful cases: {len(unsuccessful)}")
    
    if len(successful) > 0 and len(unsuccessful) > 0:
        print(f"\nProperty differences (successful vs unsuccessful):")
        for col in ['density', 'avg_degree', 'degree_std', 'n_nodes']:
            if col in df_ml.columns:
                succ_mean = successful[col].mean()
                unsucc_mean = unsuccessful[col].mean()
                print(f"  {col}: {succ_mean:.4f} vs {unsucc_mean:.4f}")
    
    # Correlation analysis
    plt.figure(figsize=(12, 8))
    
    # Select numeric columns for correlation
    numeric_cols = df_ml.select_dtypes(include=[np.number]).columns
    corr_with_success = df_ml[numeric_cols].corrwith(df_ml['is_successful'].astype(int))
    
    # Plot correlation with success
    corr_with_success = corr_with_success.drop('is_successful').sort_values(key=abs, ascending=False)
    
    plt.subplot(1, 2, 1)
    corr_with_success.plot(kind='barh')
    plt.title('Correlation with CMG Success')
    plt.xlabel('Correlation Coefficient')
    
    # Heatmap of all correlations
    plt.subplot(1, 2, 2)
    correlation_matrix = df_ml[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Property Correlation Matrix')
    
    plt.tight_layout()
    plt.show()
    
    return feature_importance, rf

if __name__ == "__main__":
    # Find most recent results
    results_dir = Path('../results')
    csv_files = list(results_dir.glob('cmg_results_*.csv'))
    
    if csv_files:
        latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
        analyze_predictive_properties(latest_file)
    else:
        print("No results files found. Run evaluate_cmg_systematic.py first.")
EOF

# Create script to run all evaluations
cat > run_step1_evaluation.sh << 'EOF'
#!/bin/bash
# Complete Step 1 Evaluation Runner
# =================================

echo "🚀 Starting CMG Step 1 Systematic Evaluation"
echo "============================================="

# Check if we're in the right directory
if [ ! -f "evaluate_cmg_systematic.py" ]; then
    echo "❌ Error: evaluate_cmg_systematic.py not found!"
    echo "Please run this script from the evaluation_step1 directory"
    exit 1
fi

# Install requirements if needed
echo "📦 Checking requirements..."
pip install -r requirements_evaluation.txt

echo ""
echo "1️⃣ Running systematic evaluation..."
python evaluate_cmg_systematic.py

echo ""
echo "2️⃣ Running parameter sensitivity analysis..."
python scripts/parameter_sensitivity_analysis.py

echo ""
echo "3️⃣ Running graph property analysis..."
python scripts/graph_property_analysis.py

echo ""
echo "✅ Step 1 evaluation completed!"
echo ""
echo "📋 Next steps:"
echo "   1. Review the analysis outputs above"
echo "   2. Open Jupyter notebook for interactive analysis:"
echo "      jupyter notebook notebooks/step1_analysis.ipynb"
echo "   3. Check results/ directory for detailed data files"
echo "   4. Use insights to plan Step 2 (theoretical analysis)"
echo ""
echo "🎯 Key questions to answer from this data:"
echo "   - On which graph types does CMG work best?"
echo "   - What gamma values are optimal for different graphs?"
echo "   - Which graph properties predict CMG success?"
echo "   - Where does CMG fail and why?"
EOF

chmod +x run_step1_evaluation.sh

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "📋 What was created:"
echo "   📁 evaluation_step1/ - Main evaluation directory"
echo "   🐍 evaluate_cmg_systematic.py - Comprehensive evaluation script"
echo "   📓 notebooks/step1_analysis.ipynb - Interactive analysis notebook"
echo "   🔧 scripts/ - Additional analysis tools"
echo "   🚀 run_step1_evaluation.sh - One-click evaluation runner"
echo ""
echo "🎯 To start the evaluation:"
echo "   cd evaluation_step1"
echo "   ./run_step1_evaluation.sh"
echo ""
echo "⏱️  Expected runtime: 10-30 minutes (depending on your system)"
echo ""
echo "💡 This will give you data-driven insights about:"
echo "   ✓ When CMG works well vs poorly"
echo "   ✓ Optimal parameters for different graph types"
echo "   ✓ Graph properties that predict success"
echo "   ✓ Comparison with baseline methods"
