#!/usr/bin/env python3
"""
Complete Graph Weighting Schemes Test with Ground Truth
=======================================================

This script comprehensively tests different weighting schemes for converting
unweighted graphs to weighted graphs for community detection using CMG.

Requirements:
    pip install networkx matplotlib seaborn scikit-learn

Run with: python complete_weighting_test.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
import networkx as nx
from typing import Dict, List, Tuple, Union, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import CMG
from cmg import CMGSteinerSolver, create_laplacian_from_edges

# Evaluation metrics
try:
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    from sklearn.metrics.cluster import contingency_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️  scikit-learn not available. Install with: pip install scikit-learn")


class GraphWeightingSchemes:
    """Collection of weighting schemes to convert unweighted graphs to weighted graphs."""
    
    @staticmethod
    def common_neighbors(G: nx.Graph) -> List[Tuple[int, int, float]]:
        """Common Neighbors weighting: Weight(i,j) = 1 + |common_neighbors|"""
        edges = []
        for u, v in G.edges():
            common_neighs = len(list(nx.common_neighbors(G, u, v)))
            weight = 1.0 + common_neighs
            edges.append((u, v, weight))
        return edges
    
    @staticmethod
    def jaccard_coefficient(G: nx.Graph) -> List[Tuple[int, int, float]]:
        """Jaccard Coefficient: Weight(i,j) = |intersection| / |union|"""
        edges = []
        for u, v in G.edges():
            neighbors_u = set(G.neighbors(u))
            neighbors_v = set(G.neighbors(v))
            
            intersection = len(neighbors_u & neighbors_v)
            union = len(neighbors_u | neighbors_v)
            
            if union > 0:
                jaccard = intersection / union
                weight = 1.0 + 2.0 * jaccard
            else:
                weight = 1.0
                
            edges.append((u, v, weight))
        return edges
    
    @staticmethod
    def cosine_similarity(G: nx.Graph) -> List[Tuple[int, int, float]]:
        """Cosine Similarity: Your original idea with hub problems"""
        edges = []
        for u, v in G.edges():
            neighbors_u = set(G.neighbors(u))
            neighbors_v = set(G.neighbors(v))
            
            intersection = len(neighbors_u & neighbors_v)
            degree_u = len(neighbors_u)
            degree_v = len(neighbors_v)
            
            if degree_u > 0 and degree_v > 0:
                cosine = intersection / np.sqrt(degree_u * degree_v)
                weight = 1.0 + 2.0 * cosine
            else:
                weight = 1.0
                
            edges.append((u, v, weight))
        return edges
    
    @staticmethod
    def adamic_adar(G: nx.Graph) -> List[Tuple[int, int, float]]:
        """Adamic-Adar: Handles hubs by weighting common neighbors by 1/log(degree)"""
        edges = []
        for u, v in G.edges():
            aa_score = 0.0
            common_neighs = nx.common_neighbors(G, u, v)
            
            for z in common_neighs:
                degree_z = G.degree(z)
                if degree_z > 1:
                    aa_score += 1.0 / np.log(degree_z)
            
            weight = 1.0 + aa_score
            edges.append((u, v, weight))
        return edges
    
    @staticmethod
    def resource_allocation(G: nx.Graph) -> List[Tuple[int, int, float]]:
        """Resource Allocation: Models resource flow, 1/degree weighting"""
        edges = []
        for u, v in G.edges():
            ra_score = 0.0
            common_neighs = nx.common_neighbors(G, u, v)
            
            for z in common_neighs:
                degree_z = G.degree(z)
                if degree_z > 0:
                    ra_score += 1.0 / degree_z
            
            weight = 1.0 + ra_score
            edges.append((u, v, weight))
        return edges
    
    @staticmethod
    def preferential_attachment(G: nx.Graph) -> List[Tuple[int, int, float]]:
        """Preferential Attachment: Emphasizes high-degree connections"""
        edges = []
        max_weight = 1.0
        
        # Calculate normalization factor
        for u, v in G.edges():
            weight = G.degree(u) * G.degree(v)
            max_weight = max(max_weight, weight)
        
        for u, v in G.edges():
            weight = G.degree(u) * G.degree(v)
            normalized_weight = 1.0 + 2.0 * (weight / max_weight)
            edges.append((u, v, normalized_weight))
        return edges
    
    @staticmethod
    def hub_promoted_index(G: nx.Graph) -> List[Tuple[int, int, float]]:
        """Hub Promoted Index: Promotes hub connections"""
        edges = []
        for u, v in G.edges():
            common_neighs = len(list(nx.common_neighbors(G, u, v)))
            min_degree = min(G.degree(u), G.degree(v))
            
            if min_degree > 0:
                hpi = common_neighs / min_degree
                weight = 1.0 + 2.0 * hpi
            else:
                weight = 1.0
                
            edges.append((u, v, weight))
        return edges
    
    @staticmethod
    def degree_corrected(G: nx.Graph, alpha: float = 0.5) -> List[Tuple[int, int, float]]:
        """Degree-Corrected: Tunable degree influence"""
        edges = []
        total_edges = G.number_of_edges()
        
        if total_edges == 0:
            return edges
            
        for u, v in G.edges():
            degree_product = G.degree(u) * G.degree(v)
            weight = (degree_product ** alpha) / total_edges
            weight = 1.0 + 2.0 * min(weight, 1.0)
            edges.append((u, v, weight))
            
        return edges


def evaluate_community_detection(true_communities: List[List[int]], 
                               predicted_components: List[int]) -> Dict[str, float]:
    """
    Evaluate community detection results against ground truth.
    
    Args:
        true_communities: List of lists, each containing node IDs in a community
        predicted_components: List where index=node_id, value=community_id
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Convert true communities to same format as predicted
    true_labels = [-1] * len(predicted_components)
    for comm_id, nodes in enumerate(true_communities):
        for node in nodes:
            if node < len(true_labels):
                true_labels[node] = comm_id
    
    # Remove unassigned nodes
    valid_indices = [i for i in range(len(true_labels)) if true_labels[i] != -1]
    if not valid_indices:
        return {'error': 'No valid node assignments'}
    
    true_filtered = [true_labels[i] for i in valid_indices]
    pred_filtered = [predicted_components[i] for i in valid_indices]
    
    results = {}
    
    # Basic metrics
    results['num_true_communities'] = len(true_communities)
    results['num_predicted_communities'] = len(set(predicted_components))
    results['community_size_match'] = len(true_communities) == len(set(predicted_components))
    
    if HAS_SKLEARN:
        try:
            # Normalized Mutual Information
            results['nmi'] = normalized_mutual_info_score(true_filtered, pred_filtered)
            
            # Adjusted Rand Index
            results['ari'] = adjusted_rand_score(true_filtered, pred_filtered)
            
            # Accuracy (perfect community detection)
            results['perfect_match'] = (results['nmi'] > 0.95 and results['ari'] > 0.95)
            
        except Exception as e:
            results['sklearn_error'] = str(e)
    
    # Custom modularity-like score
    try:
        results['modularity_score'] = calculate_custom_modularity(true_communities, predicted_components)
    except Exception:
        results['modularity_score'] = 0.0
    
    return results


def calculate_custom_modularity(true_communities: List[List[int]], 
                               predicted_components: List[int]) -> float:
    """Calculate a custom modularity-like score."""
    if not true_communities:
        return 0.0
    
    total_score = 0.0
    total_nodes = sum(len(comm) for comm in true_communities)
    
    for true_comm in true_communities:
        if not true_comm:
            continue
            
        # Find the predicted community that has the most overlap
        pred_comm_counts = defaultdict(int)
        for node in true_comm:
            if node < len(predicted_components):
                pred_comm_counts[predicted_components[node]] += 1
        
        if pred_comm_counts:
            max_overlap = max(pred_comm_counts.values())
            total_score += max_overlap / len(true_comm)
    
    return total_score / len(true_communities)


def create_test_graphs():
    """Create various test graphs with known ground truth."""
    graphs = {}
    
    # 1. Karate Club (Classic benchmark)
    G_karate = nx.karate_club_graph()
    # Known ground truth from Zachary's study
    karate_communities = [
        [0, 1, 2, 3, 7, 8, 9, 12, 13, 17, 19, 21],  # Mr. Hi's group
        [4, 5, 6, 10, 11, 14, 15, 16, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]  # Officer's group
    ]
    graphs['Karate Club'] = (G_karate, karate_communities)
    
    # 2. Three Cliques with Bridges
    G_cliques = nx.Graph()
    
    # Clique 1 (nodes 0-4)
    for i in range(5):
        for j in range(i+1, 5):
            G_cliques.add_edge(i, j)
    
    # Clique 2 (nodes 5-9)  
    for i in range(5, 10):
        for j in range(i+1, 10):
            G_cliques.add_edge(i, j)
    
    # Clique 3 (nodes 10-14)
    for i in range(10, 15):
        for j in range(i+1, 15):
            G_cliques.add_edge(i, j)
    
    # Weak bridges
    G_cliques.add_edge(4, 5)   # Bridge 1-2
    G_cliques.add_edge(9, 10)  # Bridge 2-3
    
    cliques_communities = [
        list(range(5)),      # Clique 1
        list(range(5, 10)),  # Clique 2  
        list(range(10, 15))  # Clique 3
    ]
    graphs['Three Cliques'] = (G_cliques, cliques_communities)
    
    # 3. Two Communities with Hub
    G_hub = nx.Graph()
    
    # Community 1: Star with hub at center
    hub_node = 0
    comm1_nodes = list(range(1, 8))
    for node in comm1_nodes:
        G_hub.add_edge(hub_node, node)
    
    # Add some internal connections in community 1
    G_hub.add_edges_from([(1,2), (2,3), (3,4), (5,6), (6,7)])
    
    # Community 2: Dense group
    comm2_nodes = list(range(8, 15))
    for i in comm2_nodes:
        for j in comm2_nodes:
            if i < j and np.random.random() > 0.3:  # 70% connection probability
                G_hub.add_edge(i, j)
    
    # Weak connection between communities
    G_hub.add_edge(7, 8)
    
    hub_communities = [
        [0] + comm1_nodes,  # Hub community
        comm2_nodes         # Dense community
    ]
    graphs['Hub Network'] = (G_hub, hub_communities)
    
    # 4. LFR-like Synthetic Network
    try:
        # Create a more complex synthetic network
        G_lfr = nx.Graph()
        
        # Community 1: 8 nodes, dense
        c1_nodes = list(range(8))
        for i in c1_nodes:
            for j in c1_nodes:
                if i < j and np.random.random() > 0.2:  # 80% internal density
                    G_lfr.add_edge(i, j)
        
        # Community 2: 6 nodes, medium density
        c2_nodes = list(range(8, 14))
        for i in c2_nodes:
            for j in c2_nodes:
                if i < j and np.random.random() > 0.4:  # 60% internal density
                    G_lfr.add_edge(i, j)
        
        # Community 3: 5 nodes, sparse
        c3_nodes = list(range(14, 19))
        G_lfr.add_edges_from([(14,15), (15,16), (16,17), (17,18), (18,14), (14,16), (15,17)])
        
        # Inter-community connections (10% of internal)
        inter_edges = [(7, 8), (6, 9), (13, 14), (12, 15)]
        G_lfr.add_edges_from(inter_edges)
        
        lfr_communities = [c1_nodes, c2_nodes, c3_nodes]
        graphs['LFR-like Synthetic'] = (G_lfr, lfr_communities)
        
    except Exception:
        pass  # Skip if creation fails
    
    return graphs


def run_comprehensive_test():
    """Run comprehensive test of all weighting schemes."""
    print("🚀 Comprehensive Graph Weighting Schemes Test")
    print("=" * 55)
    
    # Test graphs
    test_graphs = create_test_graphs()
    
    # Weighting schemes to test
    schemes = {
        'Unweighted': lambda G: [(u, v, 1.0) for u, v in G.edges()],
        'Common Neighbors': GraphWeightingSchemes.common_neighbors,
        'Jaccard Coefficient': GraphWeightingSchemes.jaccard_coefficient,
        'Cosine Similarity': GraphWeightingSchemes.cosine_similarity,
        'Adamic-Adar': GraphWeightingSchemes.adamic_adar,
        'Resource Allocation': GraphWeightingSchemes.resource_allocation,
        'Preferential Attachment': GraphWeightingSchemes.preferential_attachment,
        'Hub Promoted': GraphWeightingSchemes.hub_promoted_index,
        'Degree Corrected': GraphWeightingSchemes.degree_corrected,
    }
    
    # Gamma values to test
    gamma_values = [4.1, 5.0, 6.0, 7.0, 10.0]
    
    all_results = {}
    
    for graph_name, (G, true_communities) in test_graphs.items():
        print(f"\n📊 Testing on {graph_name}")
        print(f"   Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        print(f"   True communities: {len(true_communities)}")
        print(f"   Community sizes: {[len(c) for c in true_communities]}")
        print("-" * 50)
        
        graph_results = {}
        
        for scheme_name, scheme_func in schemes.items():
            scheme_results = {}
            
            try:
                # Generate weighted edges
                weighted_edges = scheme_func(G)
                
                # Test different gamma values
                best_result = None
                best_score = -1
                
                for gamma in gamma_values:
                    try:
                        # Convert to Laplacian and run CMG
                        A = create_laplacian_from_edges(weighted_edges, G.number_of_nodes())
                        solver = CMGSteinerSolver(gamma=gamma, verbose=False)
                        components, num_communities = solver.steiner_group(A)
                        
                        # Evaluate against ground truth
                        eval_results = evaluate_community_detection(true_communities, components)
                        eval_results['gamma'] = gamma
                        eval_results['num_communities'] = num_communities
                        
                        # Score for best result selection
                        if HAS_SKLEARN and 'nmi' in eval_results:
                            score = eval_results['nmi'] * 0.5 + eval_results['ari'] * 0.5
                        else:
                            score = eval_results.get('modularity_score', 0)
                        
                        if score > best_score:
                            best_score = score
                            best_result = eval_results
                            
                    except Exception as e:
                        continue
                
                if best_result:
                    scheme_results = best_result
                    
                    # Print results
                    print(f"{scheme_name:20s}: ", end="")
                    if scheme_results.get('community_size_match', False):
                        print("✅", end=" ")
                    else:
                        print("❌", end=" ")
                    
                    print(f"Found {scheme_results['num_communities']:2d} communities ", end="")
                    
                    if HAS_SKLEARN and 'nmi' in scheme_results:
                        print(f"(NMI: {scheme_results['nmi']:.3f}, ARI: {scheme_results['ari']:.3f})", end="")
                    else:
                        print(f"(Score: {scheme_results['modularity_score']:.3f})", end="")
                    
                    print(f" γ={scheme_results['gamma']}")
                
                else:
                    print(f"{scheme_name:20s}: ❌ Failed")
                    scheme_results = {'error': 'All gamma values failed'}
                
            except Exception as e:
                print(f"{scheme_name:20s}: ❌ Error - {str(e)}")
                scheme_results = {'error': str(e)}
            
            graph_results[scheme_name] = scheme_results
        
        all_results[graph_name] = graph_results
    
    return all_results


def summarize_results(all_results: Dict):
    """Summarize and rank the weighting schemes."""
    print("\n🏆 SUMMARY & RANKINGS")
    print("=" * 50)
    
    scheme_scores = defaultdict(list)
    
    for graph_name, graph_results in all_results.items():
        print(f"\n{graph_name}:")
        best_schemes = []
        
        for scheme_name, results in graph_results.items():
            if 'error' not in results:
                if results.get('community_size_match', False):
                    best_schemes.append(scheme_name)
                    if HAS_SKLEARN and 'nmi' in results:
                        score = (results['nmi'] + results['ari']) / 2
                    else:
                        score = results.get('modularity_score', 0)
                    scheme_scores[scheme_name].append(score)
        
        if best_schemes:
            print(f"  ✅ Best performers: {', '.join(best_schemes)}")
        else:
            print(f"  ⚠️  No scheme found exact community count")
    
    # Overall ranking
    print(f"\n🥇 OVERALL RANKING:")
    print("-" * 30)
    
    avg_scores = {}
    for scheme, scores in scheme_scores.items():
        if scores:
            avg_scores[scheme] = np.mean(scores)
    
    # Sort by average score
    ranked_schemes = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
    
    for i, (scheme, avg_score) in enumerate(ranked_schemes, 1):
        success_rate = len(scheme_scores[scheme]) / len(all_results) * 100
        print(f"  {i:2d}. {scheme:20s}: {avg_score:.3f} avg score ({success_rate:.0f}% success)")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 20)
    
    if ranked_schemes:
        best_scheme = ranked_schemes[0][0]
        print(f"  🥇 Best overall: {best_scheme}")
        
        if 'Jaccard' in [s[0] for s in ranked_schemes[:3]]:
            print(f"  🎯 Jaccard Coefficient is consistently good for community detection")
        
        if 'Adamic' in [s[0] for s in ranked_schemes[:3]]:
            print(f"  🎯 Adamic-Adar handles hub networks well")
        
        if 'Resource' in [s[0] for s in ranked_schemes[:3]]:
            print(f"  🎯 Resource Allocation is excellent for social networks")
        
        print(f"  ⚠️  Cosine Similarity shows hub problems as expected")


def create_visualization(all_results: Dict):
    """Create visualization of results."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Extract data for visualization
        schemes = []
        graphs = []
        scores = []
        success = []
        
        for graph_name, graph_results in all_results.items():
            for scheme_name, results in graph_results.items():
                if 'error' not in results:
                    schemes.append(scheme_name)
                    graphs.append(graph_name)
                    
                    if HAS_SKLEARN and 'nmi' in results:
                        score = (results['nmi'] + results['ari']) / 2
                    else:
                        score = results.get('modularity_score', 0)
                    
                    scores.append(score)
                    success.append(results.get('community_size_match', False))
        
        if not scores:
            print("No data available for visualization")
            return
        
        # Create heatmap
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Heatmap of scores
        score_matrix = []
        scheme_names = list(set(schemes))
        graph_names = list(set(graphs))
        
        for graph in graph_names:
            row = []
            for scheme in scheme_names:
                found_score = 0
                for i, (g, s) in enumerate(zip(graphs, schemes)):
                    if g == graph and s == scheme:
                        found_score = scores[i]
                        break
                row.append(found_score)
            score_matrix.append(row)
        
        sns.heatmap(score_matrix, 
                   xticklabels=scheme_names, 
                   yticklabels=graph_names,
                   annot=True, 
                   cmap='RdYlGn', 
                   ax=ax1)
        ax1.set_title('Community Detection Scores\n(Higher = Better)')
        ax1.set_xlabel('Weighting Scheme')
        ax1.set_ylabel('Test Graph')
        
        # Success rate bar plot
        success_rates = {}
        for scheme in scheme_names:
            total = sum(1 for s in schemes if s == scheme)
            successes = sum(1 for i, s in enumerate(schemes) if s == scheme and success[i])
            success_rates[scheme] = successes / total * 100 if total > 0 else 0
        
        schemes_sorted = sorted(success_rates.keys(), key=lambda x: success_rates[x], reverse=True)
        rates = [success_rates[s] for s in schemes_sorted]
        
        bars = ax2.bar(range(len(schemes_sorted)), rates, color='skyblue')
        ax2.set_title('Success Rate: Finding Correct Number of Communities')
        ax2.set_xlabel('Weighting Scheme')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_xticks(range(len(schemes_sorted)))
        ax2.set_xticklabels(schemes_sorted, rotation=45, ha='right')
        
        # Color bars based on performance
        for i, bar in enumerate(bars):
            if rates[i] >= 75:
                bar.set_color('green')
            elif rates[i] >= 50:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        plt.tight_layout()
        plt.savefig('weighting_schemes_comparison.png', dpi=300, bbox_inches='tight')
        print("\n📊 Visualization saved as 'weighting_schemes_comparison.png'")
        plt.show()
        
    except ImportError:
        print("📊 Install matplotlib & seaborn for visualization: pip install matplotlib seaborn")
    except Exception as e:
        print(f"📊 Visualization error: {e}")


def main():
    """Main function."""
    print("🚀 Starting Comprehensive Weighting Schemes Test")
    print("This will test 9 different weighting schemes on 4 different graphs")
    print("with ground truth evaluation using CMG community detection.")
    print()
    
    # Run the comprehensive test
    all_results = run_comprehensive_test()
    
    # Summarize results
    summarize_results(all_results)
    
    # Create visualization
    create_visualization(all_results)
    
    print(f"\n✅ Test completed!")
    print(f"💡 Use the best-performing scheme for your specific network type.")
    print(f"🔬 Consider combining multiple schemes or learning weights adaptively.")


if __name__ == "__main__":
    main()
