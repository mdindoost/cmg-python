#!/usr/bin/env python3
"""
Simple Two Cliques Test - Minimal Version
=========================================

Basic test of two cliques with weak bridge, no dependencies required.
"""

from cmg import CMGSteinerSolver, create_laplacian_from_edges

def create_two_cliques_simple():
    """Create two triangular cliques connected by weak bridge."""
    
    edges = []
    
    # Clique 1: Triangle (nodes 0, 1, 2) - strong connections
    edges.extend([
        (0, 1, 2.0),  # Strong
        (1, 2, 2.0),  # Strong  
        (2, 0, 2.0),  # Strong
    ])
    
    # Clique 2: Triangle (nodes 3, 4, 5) - strong connections
    edges.extend([
        (3, 4, 2.0),  # Strong
        (4, 5, 2.0),  # Strong
        (5, 3, 2.0),  # Strong
    ])
    
    # Weak bridge connecting the cliques
    edges.append((2, 3, 0.01))  # Very weak bridge
    
    return edges, 6  # 6 total nodes

def test_cmg_detection():
    """Test CMG's ability to detect the two communities."""
    
    print("🔍 Testing Two Cliques with Weak Bridge")
    print("=" * 40)
    
    # Create graph
    edges, num_nodes = create_two_cliques_simple()
    
    print(f"Graph structure:")
    print(f"  Clique 1: nodes 0-1-2 (triangle, weight=2.0)")
    print(f"  Clique 2: nodes 3-4-5 (triangle, weight=2.0)")  
    print(f"  Bridge:   2 ↔ 3 (weight=0.01)")
    print(f"  Total: {num_nodes} nodes, {len(edges)} edges")
    
    # Test different gamma values
    gamma_values = [4.1, 5.0, 7.0, 10.0]
    
    print(f"\nCMG Analysis:")
    print("-" * 25)
    
    for gamma in gamma_values:
        # Convert to Laplacian and run CMG
        A = create_laplacian_from_edges(edges, num_nodes)
        solver = CMGSteinerSolver(gamma=gamma, verbose=False)
        components, num_communities = solver.steiner_group(A)
        
        # Analyze results
        communities = {}
        for node, comp_id in enumerate(components):
            if comp_id not in communities:
                communities[comp_id] = []
            communities[comp_id].append(node)
        
        print(f"γ={gamma:4.1f}: {num_communities} communities")
        for comp_id, nodes in communities.items():
            print(f"         Community {comp_id}: {nodes}")
        
        # Check if detection is correct
        if num_communities == 2:
            clique1_nodes = set([0, 1, 2])
            clique2_nodes = set([3, 4, 5])
            
            comm_sets = [set(nodes) for nodes in communities.values()]
            
            if clique1_nodes in comm_sets and clique2_nodes in comm_sets:
                print(f"         ✅ Perfect detection!")
            else:
                print(f"         ⚠️  Suboptimal grouping")
        elif num_communities == 1:
            print(f"         ❌ Bridge too weak, merged communities")
        else:
            print(f"         ❌ Over-segmentation")
        
        print()
    
    return communities

def ascii_visualization():
    """Simple ASCII art visualization."""
    
    print("📊 Graph Visualization:")
    print("=" * 25)
    print()
    print("    Clique 1          Clique 2")
    print("       0                 3    ")
    print("      /|\\               /|\\   ")
    print("   2.0| |2.0         2.0| |2.0")
    print("     /2.0\\             /2.0\\  ")
    print("    2-----1           5-----4  ")
    print("           \\         /        ")
    print("            \\0.01   /         ")
    print("             \\     /          ")
    print("              \\   /           ")
    print("               \\ /            ")
    print("                X             ")
    print("             Bridge           ")
    print()
    print("Legend:")
    print("  Heavy lines (2.0): Strong intra-clique connections")
    print("  Dotted line (0.01): Weak inter-clique bridge")
    print()

def main():
    """Run the complete simple test."""
    
    print("🚀 Simple Two Cliques Test")
    print("=" * 30)
    print()
    
    # Show visualization
    ascii_visualization()
    
    # Run CMG test
    test_cmg_detection()
    
    print("💡 Key Takeaways:")
    print("  • CMG detects weak bridges between dense communities")
    print("  • Lower gamma = more sensitive to weak connections")  
    print("  • Strong intra-clique connections vs weak bridge = clear cut")
    print("  • Perfect for social networks, protein complexes, etc.")

if __name__ == "__main__":
    main()
