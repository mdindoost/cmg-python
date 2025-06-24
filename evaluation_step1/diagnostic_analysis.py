#!/usr/bin/env python3
"""
Diagnostic Analysis of CMG Validation Results
=============================================

This script analyzes the validation results to understand why all methods
are getting ARI = 0.0 and what's actually happening with the clustering.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_validation_results():
    """Analyze the validation results in detail."""
    
    # Load the results
    results_dir = Path('validation_results')
    csv_files = list(results_dir.glob('focused_validation_*.csv'))
    
    if not csv_files:
        print("❌ No validation results found!")
        return
    
    latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
    df = pd.read_csv(latest_file)
    
    print("🔍 DIAGNOSTIC ANALYSIS OF VALIDATION RESULTS")
    print("=" * 60)
    print(f"Data source: {latest_file.name}")
    print(f"Total rows: {len(df)}")
    print()
    
    # 1. Basic data overview
    print("📊 DATA OVERVIEW:")
    print(f"   Unique graphs: {df['graph_name'].nunique()}")
    print(f"   Methods tested: {df['method'].nunique()}")
    print(f"   Method types: {df['method_type'].unique()}")
    print(f"   Graph types: {df['type'].unique()}")
    print()
    
    # 2. Examine the number of communities found
    print("🔢 COMMUNITIES DETECTED:")
    print("By method:")
    method_communities = df.groupby('method')['n_communities'].agg(['mean', 'min', 'max', 'std']).round(2)
    print(method_communities)
    print()
    
    print("By graph:")
    graph_communities = df.groupby('graph_name')['n_communities'].agg(['mean', 'min', 'max']).round(2)
    print(graph_communities)
    print()
    
    # 3. Look at ARI scores in detail
    print("📈 ARI SCORES ANALYSIS:")
    print("ARI distribution by method:")
    ari_stats = df.groupby('method')['ari'].agg(['count', 'mean', 'min', 'max', 'std']).round(4)
    print(ari_stats)
    print()
    
    # Check if ALL ARI scores are actually 0
    non_zero_ari = df[df['ari'] != 0.0]
    print(f"Non-zero ARI scores: {len(non_zero_ari)}/{len(df)}")
    if len(non_zero_ari) > 0:
        print("Non-zero ARI cases:")
        print(non_zero_ari[['graph_name', 'method', 'ari', 'n_communities']].to_string())
    print()
    
    # 4. Examine specific graph cases
    print("🔍 DETAILED GRAPH ANALYSIS:")
    for graph_name in df['graph_name'].unique()[:3]:  # Look at first 3 graphs
        print(f"\nGraph: {graph_name}")
        graph_data = df[df['graph_name'] == graph_name]
        
        # Get graph metadata
        metadata = graph_data.iloc[0]
        print(f"   Nodes: {metadata['n_nodes']}")
        print(f"   True clusters: {metadata['n_clusters']}")
        print(f"   Cluster sizes: {metadata['cluster_sizes']}")
        print(f"   Difficulty: {metadata['difficulty']}")
        
        # Show results for each method
        print("   Results by method:")
        for _, row in graph_data.iterrows():
            print(f"     {row['method']:15s}: {row['n_communities']:2d} communities, ARI={row['ari']:.4f}")
    
    # 5. Runtime analysis
    print("\n⏱️ RUNTIME ANALYSIS:")
    runtime_stats = df.groupby('method')['runtime'].agg(['mean', 'std']).round(4)
    print(runtime_stats)
    print()
    
    # 6. Check for patterns in failures
    print("🚩 FAILURE PATTERN ANALYSIS:")
    
    # Look at relationship between n_communities and ground truth
    print("Expected vs Detected Communities:")
    community_comparison = df.groupby(['n_clusters', 'method']).agg({
        'n_communities': 'mean',
        'ari': 'mean'
    }).round(3)
    print(community_comparison)
    print()
    
    # 7. Graph difficulty analysis
    print("📊 DIFFICULTY LEVEL ANALYSIS:")
    if 'difficulty' in df.columns:
        difficulty_analysis = df.groupby(['difficulty', 'method_type']).agg({
            'ari': 'mean',
            'n_communities': 'mean'
        }).round(4)
        print(difficulty_analysis)
    print()
    
    # 8. Create visualizations
    create_diagnostic_plots(df)
    
    # 9. Recommendations
    print("💡 DIAGNOSTIC RECOMMENDATIONS:")
    
    # Check if CMG is finding any communities at all
    cmg_data = df[df['method_type'] == 'CMG']
    avg_cmg_communities = cmg_data['n_communities'].mean()
    
    if avg_cmg_communities < 1.5:
        print("   ❌ CMG is mostly finding 1 community (no splitting)")
        print("   🔧 Try lower gamma values (4.1, 4.5)")
        print("   🔧 Check if graphs actually have the expected structure")
    elif avg_cmg_communities > 10:
        print("   ❌ CMG is over-segmenting (too many communities)")
        print("   🔧 Try higher gamma values (15, 20)")
        print("   🔧 Check for noise in graph generation")
    else:
        print("   ⚠️  CMG is finding reasonable number of communities")
        print("   🔧 Problem might be in community assignment quality")
        print("   🔧 Check if clusters align with ground truth")
    
    # Check baseline performance
    baseline_data = df[df['method_type'] == 'baseline']
    if len(baseline_data) > 0:
        avg_baseline_ari = baseline_data['ari'].mean()
        if avg_baseline_ari == 0.0:
            print("   ❌ All baselines also failing (ARI = 0)")
            print("   🔧 Graph generation or evaluation metrics might be broken")
        else:
            print(f"   ℹ️  Baselines achieving ARI = {avg_baseline_ari:.4f}")
    
    print("\n🎯 NEXT STEPS:")
    print("   1. Run debug_single_graph.py to examine one graph in detail")
    print("   2. Visualize the actual graph structure")
    print("   3. Check if ground truth labels are correct")
    print("   4. Test with different CMG parameters")

def create_diagnostic_plots(df):
    """Create diagnostic visualizations."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Number of communities by method
    sns.boxplot(data=df, x='method', y='n_communities', ax=axes[0,0])
    axes[0,0].set_title('Number of Communities by Method')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Plot 2: ARI scores by method
    sns.boxplot(data=df, x='method', y='ari', ax=axes[0,1])
    axes[0,1].set_title('ARI Scores by Method')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Runtime by method
    sns.boxplot(data=df, x='method', y='runtime', ax=axes[1,0])
    axes[1,0].set_title('Runtime by Method')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Plot 4: Communities vs Expected
    df_plot = df.copy()
    df_plot['expected_communities'] = df_plot['n_clusters']
    sns.scatterplot(data=df_plot, x='expected_communities', y='n_communities', 
                   hue='method_type', ax=axes[1,1])
    axes[1,1].set_title('Detected vs Expected Communities')
    axes[1,1].plot([1, 5], [1, 5], 'k--', alpha=0.5, label='Perfect match')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('validation_results/diagnostic_plots.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("📊 Diagnostic plots saved as 'diagnostic_plots.png'")

def debug_single_graph():
    """Debug a single graph to understand what's happening."""
    
    print("\n🔬 DEBUGGING SINGLE GRAPH:")
    print("=" * 40)
    
    # Import necessary modules
    import sys
    sys.path.append('..')
    from cmg import CMGSteinerSolver
    from cmg.utils.graph_utils import create_clustered_graph, create_laplacian_from_edges
    from sklearn.metrics import adjusted_rand_score
    
    # Create the same graph as in validation
    print("Creating test graph...")
    edges, n = create_clustered_graph(
        cluster_sizes=[25, 25],
        intra_cluster_p=0.8,
        inter_cluster_p=0.05,
        seed=42
    )
    
    A = create_laplacian_from_edges(edges, n)
    true_labels = np.array([0]*25 + [1]*25)
    
    print(f"Graph created: {n} nodes, {len(edges)} edges")
    print(f"Expected: 2 clusters of size [25, 25]")
    print(f"True labels: {true_labels[:10]}...{true_labels[-10:]}")
    print()
    
    # Test CMG with different gamma values
    gamma_values = [4.1, 5.0, 7.0, 10.0, 15.0]
    
    print("Testing CMG with different gamma values:")
    for gamma in gamma_values:
        try:
            solver = CMGSteinerSolver(gamma=gamma, verbose=False)
            components, num_communities = solver.steiner_group(A)
            
            ari = adjusted_rand_score(true_labels, components)
            
            print(f"   γ={gamma:4.1f}: {num_communities:2d} communities, ARI={ari:.4f}")
            
            if gamma == 5.0:  # Show details for gamma=5.0
                print(f"   Details for γ=5.0:")
                print(f"     CMG labels: {components[:10]}...{components[-10:]}")
                
                # Show cluster sizes
                unique_labels, counts = np.unique(components, return_counts=True)
                cluster_info = [(label, count) for label, count in zip(unique_labels, counts)]
                print(f"     Cluster sizes: {cluster_info}")
            
        except Exception as e:
            print(f"   γ={gamma:4.1f}: Failed - {e}")
    
    print()
    
    # Test baselines for comparison
    print("Testing baseline methods:")
    
    # Test random clustering
    np.random.seed(42)
    random_labels = np.random.randint(0, 2, n)
    random_ari = adjusted_rand_score(true_labels, random_labels)
    print(f"   Random (k=2): ARI={random_ari:.4f}")
    
    # Test if graph actually has the expected structure
    print("\nGraph structure analysis:")
    
    # Check connectivity within and between clusters
    cluster1_nodes = list(range(25))
    cluster2_nodes = list(range(25, 50))
    
    intra_edges_1 = sum(1 for u, v, w in edges if u in cluster1_nodes and v in cluster1_nodes)
    intra_edges_2 = sum(1 for u, v, w in edges if u in cluster2_nodes and v in cluster2_nodes)
    inter_edges = sum(1 for u, v, w in edges if 
                     (u in cluster1_nodes and v in cluster2_nodes) or
                     (u in cluster2_nodes and v in cluster1_nodes))
    
    print(f"   Intra-cluster edges (cluster 1): {intra_edges_1}")
    print(f"   Intra-cluster edges (cluster 2): {intra_edges_2}")
    print(f"   Inter-cluster edges: {inter_edges}")
    print(f"   Ratio (intra/inter): {(intra_edges_1 + intra_edges_2) / max(inter_edges, 1):.2f}")
    
    if inter_edges == 0:
        print("   ⚠️  No inter-cluster edges - graph is disconnected!")
    elif inter_edges > (intra_edges_1 + intra_edges_2) * 0.5:
        print("   ⚠️  Too many inter-cluster edges - structure not clear")
    else:
        print("   ✅ Graph structure looks reasonable for clustering")

if __name__ == "__main__":
    # Run diagnostic analysis
    analyze_validation_results()
    
    # Run single graph debug
    debug_single_graph()
