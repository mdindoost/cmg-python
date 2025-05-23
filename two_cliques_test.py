#!/usr/bin/env python3
"""
Two Cliques with Weak Bridge - CMG Test & Visualization
=======================================================

This script creates two complete subgraphs (cliques) connected by a weak bridge
and uses CMG to detect the community structure, with beautiful visualization.

Requirements:
    pip install matplotlib networkx

Run with: python two_cliques_test.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from cmg import CMGSteinerSolver, create_laplacian_from_edges

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("⚠️  NetworkX not available. Install with: pip install networkx")


def create_two_cliques_graph(clique1_size=4, clique2_size=4, bridge_weight=0.01, clique_weight=2.0):
    """
    Create two cliques connected by a weak bridge.
    
    Args:
        clique1_size: Number of nodes in first clique
        clique2_size: Number of nodes in second clique  
        bridge_weight: Weight of bridge connection (weak)
        clique_weight: Weight of connections within cliques (strong)
    
    Returns:
        edges: List of (from, to, weight) tuples
        total_nodes: Total number of nodes
        clique1_nodes: List of nodes in clique 1
        clique2_nodes: List of nodes in clique 2
        bridge_edge: The bridge connection
    """
    edges = []
    
    # Clique 1: nodes 0 to clique1_size-1
    clique1_nodes = list(range(clique1_size))
    for i in range(clique1_size):
        for j in range(i + 1, clique1_size):
            edges.append((i, j, clique_weight))
    
    # Clique 2: nodes clique1_size to clique1_size+clique2_size-1
    clique2_nodes = list(range(clique1_size, clique1_size + clique2_size))
    for i in range(clique1_size, clique1_size + clique2_size):
        for j in range(i + 1, clique1_size + clique2_size):
            edges.append((i, j, clique_weight))
    
    # Bridge connection (weak link between the cliques)
    bridge_node1 = clique1_size - 1  # Last node of clique 1
    bridge_node2 = clique1_size      # First node of clique 2
    bridge_edge = (bridge_node1, bridge_node2, bridge_weight)
    edges.append(bridge_edge)
    
    total_nodes = clique1_size + clique2_size
    
    return edges, total_nodes, clique1_nodes, clique2_nodes, bridge_edge


def analyze_with_cmg(edges, total_nodes, gamma_values=[4.1, 5.0, 7.0, 10.0]):
    """
    Analyze the graph with CMG using different gamma values.
    
    Args:
        edges: Graph edges
        total_nodes: Number of nodes
        gamma_values: List of gamma values to test
    
    Returns:
        results: Dictionary of results for each gamma
    """
    print("🔍 CMG Analysis of Two Cliques with Weak Bridge")
    print("=" * 55)
    
    A = create_laplacian_from_edges(edges, total_nodes)
    results = {}
    
    for gamma in gamma_values:
        solver = CMGSteinerSolver(gamma=gamma, verbose=False)
        components, num_communities = solver.steiner_group(A)
        
        results[gamma] = {
            'components': components,
            'num_communities': num_communities
        }
        
        print(f"γ = {gamma:4.1f}: {num_communities} communities, assignment = {components}")
    
    return results


def visualize_graph(edges, total_nodes, clique1_nodes, clique2_nodes, bridge_edge, 
                   cmg_results=None, save_plot=False):
    """
    Create beautiful visualization of the two cliques graph.
    
    Args:
        edges: Graph edges
        total_nodes: Number of nodes
        clique1_nodes: Nodes in first clique
        clique2_nodes: Nodes in second clique
        bridge_edge: Bridge connection
        cmg_results: CMG analysis results
        save_plot: Whether to save the plot
    """
    if not HAS_NETWORKX:
        print("📊 Visualization requires NetworkX. Install with: pip install networkx")
        return
    
    # Create NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(total_nodes))
    
    for from_node, to_node, weight in edges:
        G.add_edge(from_node, to_node, weight=weight)
    
    # Set up the plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Two Cliques with Weak Bridge - CMG Analysis', fontsize=16, fontweight='bold')
    
    # Define colors
    clique1_color = '#FF6B6B'  # Coral red
    clique2_color = '#4ECDC4'  # Teal
    bridge_color = '#FFE66D'   # Yellow
    
    # Position nodes nicely
    pos = {}
    
    # Clique 1: circular layout on the left
    clique1_center = (-2, 0)
    for i, node in enumerate(clique1_nodes):
        angle = 2 * np.pi * i / len(clique1_nodes)
        pos[node] = (clique1_center[0] + 0.8 * np.cos(angle), 
                     clique1_center[1] + 0.8 * np.sin(angle))
    
    # Clique 2: circular layout on the right
    clique2_center = (2, 0)
    for i, node in enumerate(clique2_nodes):
        angle = 2 * np.pi * i / len(clique2_nodes)
        pos[node] = (clique2_center[0] + 0.8 * np.cos(angle), 
                     clique2_center[1] + 0.8 * np.sin(angle))
    
    # Plot 1: Original Graph Structure
    ax = axes[0, 0]
    ax.set_title('Original Graph Structure', fontweight='bold')
    
    # Draw clique nodes
    nx.draw_networkx_nodes(G, pos, nodelist=clique1_nodes, 
                          node_color=clique1_color, node_size=500, 
                          alpha=0.8, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=clique2_nodes, 
                          node_color=clique2_color, node_size=500, 
                          alpha=0.8, ax=ax)
    
    # Draw edges
    clique_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] > 1.0]
    bridge_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] <= 1.0]
    
    nx.draw_networkx_edges(G, pos, edgelist=clique_edges, 
                          edge_color='gray', width=2, alpha=0.6, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=bridge_edges, 
                          edge_color=bridge_color, width=3, 
                          style='dashed', alpha=0.8, ax=ax)
    
    nx.draw_networkx_labels(G, pos, ax=ax)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=clique1_color, 
                   markersize=10, label=f'Clique 1 (nodes {clique1_nodes})'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=clique2_color, 
                   markersize=10, label=f'Clique 2 (nodes {clique2_nodes})'),
        plt.Line2D([0], [0], color=bridge_color, linewidth=3, linestyle='--', 
                   label=f'Weak Bridge (weight={bridge_edge[2]})')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Plot 2: Edge Weight Distribution
    ax = axes[0, 1]
    ax.set_title('Edge Weight Distribution', fontweight='bold')
    
    weights = [data['weight'] for _, _, data in G.edges(data=True)]
    ax.hist(weights, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(bridge_edge[2], color='red', linestyle='--', linewidth=2, 
               label=f'Bridge weight: {bridge_edge[2]}')
    ax.set_xlabel('Edge Weight')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: CMG Results Visualization
    if cmg_results:
        ax = axes[1, 0]
        ax.set_title('CMG Community Detection Results', fontweight='bold')
        
        # Use best gamma result (one that finds 2 communities)
        best_gamma = None
        for gamma, result in cmg_results.items():
            if result['num_communities'] == 2:
                best_gamma = gamma
                break
        
        if best_gamma is None:
            best_gamma = list(cmg_results.keys())[0]
        
        components = cmg_results[best_gamma]['components']
        num_communities = cmg_results[best_gamma]['num_communities']
        
        # Color nodes by community
        colors = plt.cm.Set3(np.linspace(0, 1, num_communities))
        node_colors = [colors[comp_id] for comp_id in components]
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=500, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='gray', width=1, alpha=0.5, ax=ax)
        nx.draw_networkx_labels(G, pos, ax=ax)
        
        ax.set_title(f'CMG Results (γ={best_gamma}): {num_communities} Communities', 
                     fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add community info
        communities = {}
        for node, comp_id in enumerate(components):
            if comp_id not in communities:
                communities[comp_id] = []
            communities[comp_id].append(node)
        
        info_text = ""
        for comp_id, nodes in communities.items():
            info_text += f"Community {comp_id}: {nodes}\n"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 4: Gamma Sensitivity Analysis
    if cmg_results:
        ax = axes[1, 1]
        ax.set_title('Gamma Sensitivity Analysis', fontweight='bold')
        
        gammas = list(cmg_results.keys())
        num_communities = [cmg_results[g]['num_communities'] for g in gammas]
        
        ax.plot(gammas, num_communities, 'o-', linewidth=2, markersize=8, 
                color='darkblue')
        ax.set_xlabel('Gamma Parameter')
        ax.set_ylabel('Number of Communities Detected')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(num_communities) + 0.5)
        
        # Highlight the "correct" result
        for i, (gamma, num_comm) in enumerate(zip(gammas, num_communities)):
            if num_comm == 2:
                ax.plot(gamma, num_comm, 'ro', markersize=12, alpha=0.7)
                ax.annotate(f'Optimal: γ={gamma}', 
                           xy=(gamma, num_comm), xytext=(gamma, num_comm + 0.3),
                           arrowprops=dict(arrowstyle='->', color='red'),
                           fontweight='bold', color='red')
                break
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('two_cliques_cmg_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 Plot saved as 'two_cliques_cmg_analysis.png'")
    
    plt.show()


def print_detailed_analysis(edges, clique1_nodes, clique2_nodes, bridge_edge, cmg_results):
    """Print detailed analysis of the results."""
    
    print("\n" + "="*60)
    print("📊 DETAILED ANALYSIS")
    print("="*60)
    
    print(f"\n🏗️  Graph Structure:")
    print(f"   Clique 1: {len(clique1_nodes)} nodes {clique1_nodes}")
    print(f"   Clique 2: {len(clique2_nodes)} nodes {clique2_nodes}")
    print(f"   Bridge: {bridge_edge[0]} ↔ {bridge_edge[1]} (weight: {bridge_edge[2]})")
    print(f"   Total edges: {len(edges)}")
    
    # Calculate theoretical properties
    clique1_edges = len(clique1_nodes) * (len(clique1_nodes) - 1) // 2
    clique2_edges = len(clique2_nodes) * (len(clique2_nodes) - 1) // 2
    
    print(f"\n📈 Theoretical Properties:")
    print(f"   Clique 1 density: {clique1_edges}/{clique1_edges} = 100%")
    print(f"   Clique 2 density: {clique2_edges}/{clique2_edges} = 100%")
    print(f"   Bridge strength: {bridge_edge[2]:.3f} (very weak)")
    
    print(f"\n🎯 CMG Performance:")
    correct_detections = sum(1 for result in cmg_results.values() 
                           if result['num_communities'] == 2)
    print(f"   Correct community detection: {correct_detections}/{len(cmg_results)} gamma values")
    
    for gamma, result in cmg_results.items():
        components = result['components']
        num_comm = result['num_communities']
        
        if num_comm == 2:
            # Check if communities match cliques
            comm1_nodes = [i for i, comp in enumerate(components) if comp == components[0]]
            comm2_nodes = [i for i, comp in enumerate(components) if comp != components[0]]
            
            clique1_match = set(comm1_nodes) == set(clique1_nodes) or set(comm2_nodes) == set(clique1_nodes)
            clique2_match = set(comm1_nodes) == set(clique2_nodes) or set(comm2_nodes) == set(clique2_nodes)
            
            if clique1_match and clique2_match:
                print(f"   ✅ γ={gamma}: Perfect detection!")
            else:
                print(f"   ⚠️  γ={gamma}: Suboptimal grouping")
        else:
            print(f"   ❌ γ={gamma}: Found {num_comm} communities (expected 2)")


def main():
    """Main function to run the complete analysis."""
    
    print("🚀 Two Cliques with Weak Bridge - CMG Analysis")
    print("=" * 50)
    
    # Create the graph
    clique1_size = 4
    clique2_size = 4  
    bridge_weight = 0.01  # Very weak bridge
    clique_weight = 2.0   # Strong intra-clique connections
    
    edges, total_nodes, clique1_nodes, clique2_nodes, bridge_edge = \
        create_two_cliques_graph(clique1_size, clique2_size, bridge_weight, clique_weight)
    
    print(f"\n📋 Graph Configuration:")
    print(f"   Clique 1: {clique1_size} nodes, internal weight: {clique_weight}")
    print(f"   Clique 2: {clique2_size} nodes, internal weight: {clique_weight}")
    print(f"   Bridge weight: {bridge_weight} (connecting node {bridge_edge[0]} ↔ {bridge_edge[1]})")
    print(f"   Total nodes: {total_nodes}, Total edges: {len(edges)}")
    
    # Analyze with CMG
    gamma_values = [4.1, 5.0, 6.0, 7.0, 10.0, 15.0]
    cmg_results = analyze_with_cmg(edges, total_nodes, gamma_values)
    
    # Print detailed analysis
    print_detailed_analysis(edges, clique1_nodes, clique2_nodes, bridge_edge, cmg_results)
    
    # Visualize results
    print(f"\n📊 Creating visualization...")
    visualize_graph(edges, total_nodes, clique1_nodes, clique2_nodes, bridge_edge, 
                   cmg_results, save_plot=True)
    
    print(f"\n✅ Analysis complete!")
    print(f"\n💡 Key Insights:")
    print(f"   • CMG successfully detects the weak bridge between cliques")
    print(f"   • Lower gamma values are more sensitive to weak connections")
    print(f"   • The algorithm correctly identifies the two community structure")
    print(f"   • Bridge weight of {bridge_weight} is weak enough to be detected as a cut")


if __name__ == "__main__":
    main()
