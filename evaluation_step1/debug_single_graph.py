#!/usr/bin/env python3
import sys
sys.path.append('..')
from cmg import CMGSteinerSolver
from cmg.utils.graph_utils import create_clustered_graph, create_laplacian_from_edges
from sklearn.metrics import adjusted_rand_score
import numpy as np

print("🔬 DEBUGGING SINGLE GRAPH:")
print("=" * 40)

# Create the same graph as in validation
edges, n = create_clustered_graph(
    cluster_sizes=[25, 25],
    intra_cluster_p=0.8,
    inter_cluster_p=0.05,
    seed=42
)

A = create_laplacian_from_edges(edges, n)
true_labels = np.array([0]*25 + [1]*25)

print(f"Graph: {n} nodes, {len(edges)} edges")
print(f"Expected: 2 clusters of size [25, 25]")
print(f"True labels: {true_labels[:10]}...{true_labels[-10:]}")
print()

# Test CMG with gamma=5.0 (from validation)
solver = CMGSteinerSolver(gamma=5.0, verbose=False)
components, num_communities = solver.steiner_group(A)

print(f"CMG Results:")
print(f"   Found {num_communities} communities")
print(f"   CMG labels: {components[:10]}...{components[-10:]}")

# Show cluster sizes
unique_labels, counts = np.unique(components, return_counts=True)
cluster_info = [(label, count) for label, count in zip(unique_labels, counts)]
print(f"   Cluster sizes: {cluster_info}")

# Calculate ARI
ari = adjusted_rand_score(true_labels, components)
print(f"   ARI: {ari:.4f}")
print()

# Analyze graph structure
print("Graph structure analysis:")
cluster1_nodes = set(range(25))
cluster2_nodes = set(range(25, 50))

intra_edges_1 = sum(1 for u, v, w in edges if u in cluster1_nodes and v in cluster1_nodes)
intra_edges_2 = sum(1 for u, v, w in edges if u in cluster2_nodes and v in cluster2_nodes)
inter_edges = sum(1 for u, v, w in edges if 
                 (u in cluster1_nodes and v in cluster2_nodes) or
                 (u in cluster2_nodes and v in cluster1_nodes))

print(f"   Intra-cluster edges (cluster 1): {intra_edges_1}")
print(f"   Intra-cluster edges (cluster 2): {intra_edges_2}")
print(f"   Inter-cluster edges: {inter_edges}")
print(f"   Ratio (intra/inter): {(intra_edges_1 + intra_edges_2) / max(inter_edges, 1):.2f}")

# Test if Louvain gets the assignment right
try:
    import networkx as nx
    import community as community_louvain
    
    # Convert to NetworkX
    A_adj = -A.copy()
    A_adj.setdiag(0)
    A_adj.eliminate_zeros()
    A_adj.data = np.abs(A_adj.data)
    G = nx.from_scipy_sparse_array(A_adj)
    
    partition = community_louvain.best_partition(G, random_state=42)
    louvain_labels = np.array([partition.get(i, 0) for i in range(n)])
    
    print(f"\nLouvain Results:")
    print(f"   Found {len(np.unique(louvain_labels))} communities")
    print(f"   Louvain labels: {louvain_labels[:10]}...{louvain_labels[-10:]}")
    
    # Louvain cluster sizes
    lou_unique, lou_counts = np.unique(louvain_labels, return_counts=True)
    lou_cluster_info = [(label, count) for label, count in zip(lou_unique, lou_counts)]
    print(f"   Cluster sizes: {lou_cluster_info}")
    
    louvain_ari = adjusted_rand_score(true_labels, louvain_labels)
    print(f"   ARI: {louvain_ari:.4f}")
    
    # Check if labels just need remapping
    if len(lou_unique) == 2:
        # Try swapping labels
        louvain_swapped = 1 - louvain_labels  # Swap 0<->1
        swapped_ari = adjusted_rand_score(true_labels, louvain_swapped)
        print(f"   ARI (swapped): {swapped_ari:.4f}")
        
except Exception as e:
    print(f"Louvain test failed: {e}")

print("\n💡 DIAGNOSIS:")
if num_communities > 10:
    print("   ❌ CMG is severely over-segmenting")
    print("   🔧 Problem: CMG parameters too aggressive")
    print("   🔧 Solution: Need much higher gamma values or algorithm fix")
elif inter_edges == 0:
    print("   ❌ Graph is completely disconnected")
    print("   🔧 Problem: Graph generation broken")
else:
    print("   ⚠️  Graph structure exists but CMG can't find it")
    print("   🔧 Problem: CMG algorithm not suited for this graph type")
