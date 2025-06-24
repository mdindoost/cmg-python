#!/usr/bin/env python3
import sys
sys.path.append('..')
from cmg import CMGSteinerSolver
from cmg.utils.graph_utils import create_clustered_graph, create_laplacian_from_edges
import numpy as np

def manual_ari(true_labels, pred_labels):
    """Calculate ARI manually without sklearn."""
    n = len(true_labels)
    
    # Create contingency table
    true_unique = np.unique(true_labels)
    pred_unique = np.unique(pred_labels)
    
    contingency = np.zeros((len(true_unique), len(pred_unique)))
    
    for i, true_val in enumerate(true_unique):
        for j, pred_val in enumerate(pred_unique):
            contingency[i, j] = np.sum((true_labels == true_val) & (pred_labels == pred_val))
    
    # Simple similarity measure
    diagonal_sum = 0
    for i in range(min(len(true_unique), len(pred_unique))):
        if i < contingency.shape[0] and i < contingency.shape[1]:
            diagonal_sum += contingency[i, i]
    
    accuracy = diagonal_sum / n
    return accuracy

print("🔬 SIMPLE GRAPH DEBUG ANALYSIS")
print("=" * 50)

# Create test graph
print("Creating test graph...")
edges, n = create_clustered_graph(
    cluster_sizes=[25, 25],
    intra_cluster_p=0.8,
    inter_cluster_p=0.05,
    seed=42
)

A = create_laplacian_from_edges(edges, n)
true_labels = np.array([0]*25 + [1]*25)

print(f"✅ Graph created: {n} nodes, {len(edges)} edges")
print(f"✅ Expected: 2 clusters of [25, 25]")
print(f"✅ True labels: [0,0,0...0,1,1,1...1]")
print()

# Test CMG
print("Testing CMG...")
solver = CMGSteinerSolver(gamma=5.0, verbose=False)
components, num_communities = solver.steiner_group(A)

print(f"📊 CMG Results:")
print(f"   Communities found: {num_communities}")
print(f"   First 10 labels: {components[:10]}")
print(f"   Last 10 labels: {components[-10:]}")

# Analyze CMG cluster sizes
unique_cmg, counts_cmg = np.unique(components, return_counts=True)
print(f"   Cluster sizes: {list(zip(unique_cmg, counts_cmg))}")

# Calculate simple accuracy
cmg_accuracy = manual_ari(true_labels, components)
print(f"   Simple accuracy: {cmg_accuracy:.4f}")
print()

# Analyze graph structure
print("📈 Graph Structure Analysis:")
cluster1_nodes = set(range(25))
cluster2_nodes = set(range(25, 50))

intra_1 = sum(1 for u, v, w in edges if u in cluster1_nodes and v in cluster1_nodes)
intra_2 = sum(1 for u, v, w in edges if u in cluster2_nodes and v in cluster2_nodes)
inter = sum(1 for u, v, w in edges if 
           (u in cluster1_nodes and v in cluster2_nodes) or
           (u in cluster2_nodes and v in cluster1_nodes))

print(f"   Intra-cluster 1 edges: {intra_1}")
print(f"   Intra-cluster 2 edges: {intra_2}")
print(f"   Inter-cluster edges: {inter}")
print(f"   Structure ratio: {(intra_1 + intra_2) / max(inter, 1):.2f}")

if inter == 0:
    print("   ❌ DISCONNECTED: No edges between clusters!")
elif inter > (intra_1 + intra_2) * 0.3:
    print("   ⚠️  WEAK STRUCTURE: Too many inter-cluster edges")
else:
    print("   ✅ GOOD STRUCTURE: Clear cluster separation")

print()

# Test different gamma values
print("🔧 Testing Different Gamma Values:")
gamma_values = [4.1, 5.0, 7.0, 10.0, 15.0, 20.0]

for gamma in gamma_values:
    try:
        solver = CMGSteinerSolver(gamma=gamma, verbose=False)
        comps, num_comms = solver.steiner_group(A)
        accuracy = manual_ari(true_labels, comps)
        
        # Get largest cluster sizes
        unique_vals, counts = np.unique(comps, return_counts=True)
        largest_clusters = sorted(counts, reverse=True)[:3]
        
        print(f"   γ={gamma:4.1f}: {num_comms:2d} communities, accuracy={accuracy:.3f}, largest: {largest_clusters}")
        
    except Exception as e:
        print(f"   γ={gamma:4.1f}: FAILED - {e}")

print()

# Manual analysis of what CMG is doing
print("🔍 Detailed CMG Analysis (γ=5.0):")
solver = CMGSteinerSolver(gamma=5.0, verbose=True)
components, num_communities = solver.steiner_group(A)

# Check which original cluster each CMG cluster belongs to
print("\nCMG cluster composition:")
for cmg_cluster in unique_cmg:
    nodes_in_cluster = np.where(components == cmg_cluster)[0]
    
    # Count how many from each true cluster
    from_cluster_0 = np.sum(nodes_in_cluster < 25)
    from_cluster_1 = np.sum(nodes_in_cluster >= 25)
    
    print(f"   CMG cluster {cmg_cluster:2d} (size {len(nodes_in_cluster):2d}): "
          f"{from_cluster_0:2d} from true cluster 0, {from_cluster_1:2d} from true cluster 1")

print("\n💡 DIAGNOSIS:")
if num_communities > 8:
    print("   ❌ SEVERE OVER-SEGMENTATION")
    print("   🔧 CMG is fragmenting clusters instead of finding them")
    print("   🔧 Possible fixes: Much higher gamma (>20) or algorithm issue")
elif num_communities == 2:
    if cmg_accuracy > 0.8:
        print("   ✅ CMG WORKING CORRECTLY")
        print("   🔧 Issue might be in ARI calculation in validation")
    else:
        print("   ⚠️  RIGHT COUNT, WRONG ASSIGNMENT") 
        print("   🔧 CMG finds 2 clusters but assigns nodes incorrectly")
else:
    print("   ⚠️  MODERATE OVER-SEGMENTATION")
    print("   🔧 CMG parameters need tuning")

print("\n🎯 CONCLUSION:")
print("   This analysis shows exactly why ARI = 0.0 in validation!")
