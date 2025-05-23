#!/usr/bin/env python3
"""
CMG-Python Quick Test Script
===========================
Edit this file to test your own graphs!
"""

from cmg import CMGSteinerSolver, create_laplacian_from_edges

def my_graph_analysis():
    """🔧 EDIT THIS FUNCTION TO TEST YOUR GRAPH."""
    
    # ✏️ Step 1: Define your graph edges
    # Format: (from_node, to_node, connection_weight)
    my_edges = [
        (0, 1, 2.0),    # Strong connection
        (1, 2, 0.1),    # Weak connection (potential cut point)
        (2, 3, 2.0),    # Strong connection
        (3, 4, 1.0),    # Medium connection
    ]
    
    # ✏️ Step 2: Set number of nodes
    num_nodes = 5
    
    # ✏️ Step 3: Choose sensitivity (4.1 = sensitive, 15.0 = less sensitive)
    gamma = 5.0
    
    print(f"🔍 Analyzing your graph:")
    print(f"   Nodes: {num_nodes}")
    print(f"   Edges: {len(my_edges)}")
    print(f"   Sensitivity (gamma): {gamma}")
    print()
    
    # Convert to matrix and analyze
    A = create_laplacian_from_edges(my_edges, num_nodes)
    solver = CMGSteinerSolver(gamma=gamma, verbose=True)
    components, num_communities = solver.steiner_group(A)
    
    # Show results
    print(f"\n📊 Results:")
    print(f"   Communities found: {num_communities}")
    print(f"   Node assignments: {components}")
    
    # Group nodes by community
    communities = {}
    for node, community_id in enumerate(components):
        if community_id not in communities:
            communities[community_id] = []
        communities[community_id].append(node)
    
    print(f"\n🏘️  Community Structure:")
    for community_id, nodes in communities.items():
        print(f"   Community {community_id}: nodes {nodes}")
    
    return components, num_communities

def quick_test():
    """Quick validation that CMG is working."""
    edges = [(0, 1, 1.0), (1, 2, 0.01), (2, 3, 1.0)]  # Weak bridge
    A = create_laplacian_from_edges(edges, 4)
    solver = CMGSteinerSolver(verbose=False)
    components, num_comp = solver.steiner_group(A)
    
    print("✅ CMG is working correctly!")
    print(f"   Test case: {num_comp} communities found")
    return True

if __name__ == "__main__":
    print("🚀 CMG-Python Graph Analysis")
    print("=" * 40)
    
    # Quick validation
    quick_test()
    print()
    
    # Analyze your graph
    my_graph_analysis()
    
    print("\n💡 To customize:")
    print("   Edit the my_graph_analysis() function above")
    print("   Change my_edges, num_nodes, and gamma values")
