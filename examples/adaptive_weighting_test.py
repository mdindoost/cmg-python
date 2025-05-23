#!/usr/bin/env python3
"""
Complete Adaptive Graph Weighting Test Suite (Single File)
==========================================================

This file contains both the adaptive weighting implementation and comprehensive tests.
No external dependencies on other files - everything is self-contained.

Usage: python complete_weighting_test.py
"""

import numpy as np
import networkx as nx
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Set
import time
import warnings
warnings.filterwarnings('ignore')

# Import CMG
from cmg import CMGSteinerSolver, create_laplacian_from_edges

# Import evaluation metrics
try:
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    from sklearn.metrics.cluster import contingency_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️  Install scikit-learn for full evaluation: pip install scikit-learn")


class AdaptiveGraphWeighting:
    """
    Adaptive weighting schemes that preserve natural graph structure
    for improved community detection.
    """
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        
    def analyze_graph_structure(self, G: nx.Graph) -> Dict:
        """Analyze the global structure of the graph to guide weighting."""
        stats = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'avg_clustering': nx.average_clustering(G),
            'connected_components': nx.number_connected_components(G),
        }
        
        # Calculate degree distribution
        degrees = [G.degree(n) for n in G.nodes()]
        stats['avg_degree'] = np.mean(degrees) if degrees else 0
        stats['degree_std'] = np.std(degrees) if degrees else 0
        stats['max_degree'] = max(degrees) if degrees else 0
        
        # Analyze triangles (sign of community structure)
        triangles = sum(nx.triangles(G).values()) // 3
        stats['triangle_density'] = triangles / max(G.number_of_edges(), 1)
        
        # Check for hub-like structure
        stats['has_hubs'] = stats['max_degree'] > 3 * stats['avg_degree']
        
        # Check for clique-like structure  
        stats['is_clique_like'] = stats['avg_clustering'] > 0.7
        
        # Check for sparse structure
        stats['is_sparse'] = stats['density'] < 0.1
        
        if self.verbose:
            print(f"📊 Graph Analysis:")
            print(f"   Nodes: {stats['num_nodes']}, Edges: {stats['num_edges']}")
            print(f"   Density: {stats['density']:.3f}, Clustering: {stats['avg_clustering']:.3f}")
            print(f"   Avg Degree: {stats['avg_degree']:.1f}, Has Hubs: {stats['has_hubs']}")
            print(f"   Triangle Density: {stats['triangle_density']:.3f}")
        
        return stats
    
    def structure_preserving_weights(self, G: nx.Graph) -> List[Tuple[int, int, float]]:
        """
        Main method: Adaptive weighting that preserves natural structure.
        
        Strategy:
        1. Analyze global graph properties
        2. For each edge, analyze local structure  
        3. Apply appropriate weighting based on context
        4. Normalize to prevent bias
        """
        if G.number_of_edges() == 0:
            return []
            
        graph_stats = self.analyze_graph_structure(G)
        
        # Choose primary strategy based on graph characteristics
        if graph_stats['is_clique_like']:
            return self._clique_preserving_weights(G, graph_stats)
        elif graph_stats['has_hubs']:
            return self._hub_aware_weights(G, graph_stats)
        elif graph_stats['is_sparse']:
            return self._sparse_structure_weights(G, graph_stats)
        else:
            return self._hybrid_weights(G, graph_stats)
    
    def _clique_preserving_weights(self, G: nx.Graph, stats: Dict) -> List[Tuple[int, int, float]]:
        """Optimal for graphs with dense clique-like communities."""
        edges = []
        
        if self.verbose:
            print("🔸 Using Clique-Preserving Weights")
        
        for u, v in G.edges():
            # Common neighbors indicate intra-clique edges
            common_neighbors = len(list(nx.common_neighbors(G, u, v)))
            
            # Jaccard coefficient for normalized similarity
            u_neighbors = set(G.neighbors(u))
            v_neighbors = set(G.neighbors(v))
            union_size = len(u_neighbors | v_neighbors)
            
            if union_size > 0:
                jaccard = len(u_neighbors & v_neighbors) / union_size
                # Exponential boost for high similarity (intra-clique)
                weight = 1.0 + 5.0 * (jaccard ** 2)
            else:
                weight = 1.0
            
            # Additional boost for triangle closure
            if common_neighbors > 0:
                triangle_boost = min(2.0, 1.0 + 0.5 * common_neighbors)
                weight *= triangle_boost
            
            edges.append((u, v, weight))
        
        return edges
    
    def _hub_aware_weights(self, G: nx.Graph, stats: Dict) -> List[Tuple[int, int, float]]:
        """Optimal for graphs with hub nodes connecting communities."""
        edges = []
        
        if self.verbose:
            print("🔸 Using Hub-Aware Weights")
        
        # Identify potential hub nodes
        degrees = {n: G.degree(n) for n in G.nodes()}
        avg_degree = stats['avg_degree']
        hub_threshold = max(3, avg_degree * 2)
        hubs = {n for n, d in degrees.items() if d >= hub_threshold}
        
        for u, v in G.edges():
            # Base weight using Resource Allocation (good for hubs)
            common_neighbors = list(nx.common_neighbors(G, u, v))
            ra_score = sum(1.0 / G.degree(z) for z in common_neighbors if G.degree(z) > 0)
            
            weight = 1.0 + ra_score
            
            # Penalty for hub-to-hub connections (likely inter-community)
            if u in hubs and v in hubs:
                weight *= 0.5  # Reduce weight of hub-hub edges
            
            # Boost for hub-to-non-hub in same neighborhood
            elif (u in hubs) != (v in hubs):  # One is hub, one isn't
                # Check if they share many neighbors (same community)
                shared_ratio = len(common_neighbors) / min(G.degree(u), G.degree(v)) if min(G.degree(u), G.degree(v)) > 0 else 0
                if shared_ratio > 0.3:
                    weight *= 1.5  # Boost intra-community hub edges
            
            edges.append((u, v, weight))
        
        return edges
    
    def _sparse_structure_weights(self, G: nx.Graph, stats: Dict) -> List[Tuple[int, int, float]]:
        """Optimal for sparse graphs where every edge matters."""
        edges = []
        
        if self.verbose:
            print("🔸 Using Sparse-Structure Weights")
        
        # In sparse graphs, focus on path-based similarity
        for u, v in G.edges():
            # Common neighbors are rare and important
            common_neighbors = list(nx.common_neighbors(G, u, v))
            
            # Base weight - higher baseline for sparse graphs
            weight = 2.0
            
            # Strong boost for any common neighbors
            if common_neighbors:
                cn_boost = len(common_neighbors) * 1.0
                weight += cn_boost
            
            # Path-based similarity (2-hop neighbors)
            u_2hop = set()
            for neighbor in G.neighbors(u):
                u_2hop.update(G.neighbors(neighbor))
            u_2hop.discard(u)
            u_2hop.discard(v)
            
            v_2hop = set()  
            for neighbor in G.neighbors(v):
                v_2hop.update(G.neighbors(neighbor))
            v_2hop.discard(v)
            v_2hop.discard(u)
            
            # 2-hop similarity
            if u_2hop and v_2hop:
                path_similarity = len(u_2hop & v_2hop) / len(u_2hop | v_2hop)
                weight += path_similarity
            
            edges.append((u, v, weight))
        
        return edges
    
    def _hybrid_weights(self, G: nx.Graph, stats: Dict) -> List[Tuple[int, int, float]]:
        """
        Hybrid approach combining multiple structural signals.
        Best for mixed or unknown graph types.
        """
        edges = []
        
        if self.verbose:
            print("🔸 Using Hybrid Multi-Signal Weights")
        
        for u, v in G.edges():
            # Signal 1: Common Neighbors (local density)
            common_neighbors = list(nx.common_neighbors(G, u, v))
            cn_score = len(common_neighbors)
            
            # Signal 2: Jaccard Similarity (normalized overlap)
            u_neighbors = set(G.neighbors(u))
            v_neighbors = set(G.neighbors(v))
            union_size = len(u_neighbors | v_neighbors)
            jaccard_score = len(u_neighbors & v_neighbors) / union_size if union_size > 0 else 0
            
            # Signal 3: Adamic-Adar (hub-corrected common neighbors)
            aa_score = sum(1.0 / np.log(G.degree(z)) for z in common_neighbors if G.degree(z) > 1)
            
            # Signal 4: Clustering contribution
            # How much does this edge contribute to local clustering?
            clustering_contrib = 0
            for z in common_neighbors:
                degree_z = G.degree(z)
                if degree_z > 1:
                    max_triangles = degree_z * (degree_z - 1) / 2
                    if max_triangles > 0:
                        clustering_contrib += 1.0 / max_triangles
            
            # Combine signals with learned weights
            weight = (1.0 +                          # Base weight
                     0.5 * cn_score +               # Common neighbors
                     2.0 * jaccard_score +          # Structural similarity  
                     0.3 * aa_score +               # Hub-corrected
                     1.0 * clustering_contrib)      # Clustering contribution
            
            edges.append((u, v, weight))
        
        return edges
    
    def topology_aware_weights(self, G: nx.Graph) -> List[Tuple[int, int, float]]:
        """
        Advanced method: Analyze local topology around each edge
        and apply context-specific weighting.
        """
        edges = []
        
        if self.verbose:
            print("🔸 Using Topology-Aware Weights")
        
        # Pre-compute local topology features for all nodes
        node_features = self._compute_node_features(G)
        
        for u, v in G.edges():
            # Analyze local topology around this edge
            edge_context = self._analyze_edge_context(G, u, v, node_features)
            
            # Apply context-specific weighting
            if edge_context['type'] == 'intra_clique':
                weight = 1.0 + 3.0 * edge_context['strength']
            elif edge_context['type'] == 'hub_spoke':
                weight = 1.0 + 1.5 * edge_context['strength']
            elif edge_context['type'] == 'bridge':
                weight = 1.0 + 0.5 * edge_context['strength']  # Lower weight for bridges
            elif edge_context['type'] == 'chain':
                weight = 1.0 + 1.0 * edge_context['strength']
            else:  # 'mixed' or unknown
                weight = 1.0 + 1.0 * edge_context['strength']
            
            edges.append((u, v, weight))
        
        return edges
    
    def _compute_node_features(self, G: nx.Graph) -> Dict:
        """Compute structural features for each node."""
        features = {}
        avg_degree = np.mean([G.degree(n) for n in G.nodes()]) if G.nodes() else 0
        
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            degree = len(neighbors)
            
            # Local clustering coefficient
            if degree >= 2:
                possible_edges = degree * (degree - 1) // 2
                actual_edges = sum(1 for i in range(len(neighbors)) 
                                 for j in range(i+1, len(neighbors))
                                 if G.has_edge(neighbors[i], neighbors[j]))
                clustering = actual_edges / possible_edges
            else:
                clustering = 0.0
            
            # Effective eccentricity (local centrality measure)
            if neighbors:
                neighbor_degrees = [G.degree(n) for n in neighbors]
                avg_neighbor_degree = np.mean(neighbor_degrees)
                degree_variance = np.var(neighbor_degrees)
            else:
                avg_neighbor_degree = 0
                degree_variance = 0
            
            features[node] = {
                'degree': degree,
                'clustering': clustering,
                'avg_neighbor_degree': avg_neighbor_degree,
                'degree_variance': degree_variance,
                'is_hub': degree > 3 * avg_degree
            }
        
        return features
    
    def _analyze_edge_context(self, G: nx.Graph, u: int, v: int, node_features: Dict) -> Dict:
        """Analyze the local context around an edge to determine its role."""
        common_neighbors = list(nx.common_neighbors(G, u, v))
        
        u_features = node_features[u]
        v_features = node_features[v]
        
        # Determine edge type based on local topology
        edge_type = 'mixed'
        strength = 0.5
        
        # Intra-clique: Both nodes have high clustering, many common neighbors
        if (u_features['clustering'] > 0.5 and v_features['clustering'] > 0.5 and 
            len(common_neighbors) >= 2):
            edge_type = 'intra_clique'
            strength = min(1.0, len(common_neighbors) / 5.0)
        
        # Hub-spoke: One node is hub, other is not, but they share neighbors
        elif (u_features['is_hub'] != v_features['is_hub'] and 
              len(common_neighbors) > 0):
            edge_type = 'hub_spoke'
            strength = min(1.0, len(common_neighbors) / 3.0)
        
        # Bridge: Few/no common neighbors, but both nodes have decent degree
        elif (len(common_neighbors) == 0 and 
              u_features['degree'] >= 2 and v_features['degree'] >= 2):
            edge_type = 'bridge'
            strength = 0.3  # Lower strength for bridges
        
        # Chain: Part of a path-like structure
        elif (u_features['degree'] <= 2 or v_features['degree'] <= 2):
            edge_type = 'chain'
            strength = 0.7
        
        return {
            'type': edge_type,
            'strength': strength,
            'common_neighbors': len(common_neighbors)
        }
    
    def ensemble_weights(self, G: nx.Graph) -> List[Tuple[int, int, float]]:
        """
        Ensemble method: Combine multiple weighting approaches
        for maximum robustness across different graph types.
        """
        if G.number_of_edges() == 0:
            return []
        
        if self.verbose:
            print("🔸 Using Ensemble Weights (Multi-Method Combination)")
        
        # Get weights from different methods
        structure_weights = {(u,v): w for u,v,w in self.structure_preserving_weights(G)}
        topology_weights = {(u,v): w for u,v,w in self.topology_aware_weights(G)}
        
        # Analyze graph to determine combination strategy
        stats = self.analyze_graph_structure(G)
        
        edges = []
        for u, v in G.edges():
            edge_key = (u, v) if (u, v) in structure_weights else (v, u)
            
            struct_w = structure_weights.get(edge_key, 1.0)
            topo_w = topology_weights.get(edge_key, 1.0)
            
            # Adaptive combination based on graph properties
            if stats['is_clique_like']:
                # For clique-like graphs, trust structure-preserving more
                weight = 0.7 * struct_w + 0.3 * topo_w
            elif stats['has_hubs']:
                # For hub networks, balance both approaches
                weight = 0.5 * struct_w + 0.5 * topo_w
            else:
                # For mixed graphs, slightly favor topology-aware
                weight = 0.4 * struct_w + 0.6 * topo_w
            
            edges.append((u, v, weight))
        
        return edges
    
    def natural_flow_weights(self, G: nx.Graph) -> List[Tuple[int, int, float]]:
        """
        Novel approach: Model information/random walk flow to identify
        natural community boundaries.
        """
        edges = []
        
        if self.verbose:
            print("🔸 Using Natural Flow Weights")
        
        # Compute personalized PageRank from each node
        flow_similarities = self._compute_flow_similarities(G)
        
        # Get graph statistics to calibrate flow weights
        avg_clustering = nx.average_clustering(G)
        density = nx.density(G)
        
        # Adjust flow sensitivity based on graph structure
        if avg_clustering > 0.7:  # Clique-like structure
            flow_multiplier = 1.0  # Less aggressive
            base_weight = 2.0      # Higher baseline
        else:  # Mixed or sparse structure
            flow_multiplier = 0.5  # More conservative
            base_weight = 1.5      # Moderate baseline
        
        for u, v in G.edges():
            # Adaptive base weight
            weight = base_weight
            
            # Flow similarity: nodes in same community have similar flow patterns
            if (u, v) in flow_similarities:
                flow_sim = flow_similarities[(u, v)]
                weight += flow_multiplier * flow_sim
            
            # Resistance distance approximation
            # Lower resistance = stronger connection within community
            resistance = self._approximate_resistance_distance(G, u, v)
            resistance_boost = 0.5 / (1.0 + resistance)  # Reduced impact
            weight += resistance_boost
            
            edges.append((u, v, weight))
        
        return edges
    
    def _compute_flow_similarities(self, G: nx.Graph) -> Dict:
        """Compute flow-based similarities between adjacent nodes."""
        similarities = {}
        
        # For efficiency, use a simplified flow model
        for u, v in G.edges():
            # Compute overlap in 2-hop neighborhoods (proxy for flow similarity)
            u_2hop = set()
            for n1 in G.neighbors(u):
                u_2hop.update(G.neighbors(n1))
            
            v_2hop = set()
            for n1 in G.neighbors(v):
                v_2hop.update(G.neighbors(n1))
            
            # Remove the nodes themselves
            u_2hop.discard(u)
            u_2hop.discard(v)
            v_2hop.discard(u)
            v_2hop.discard(v)
            
            if u_2hop or v_2hop:
                jaccard = len(u_2hop & v_2hop) / len(u_2hop | v_2hop)
                similarities[(u, v)] = jaccard
            else:
                similarities[(u, v)] = 0.0
        
        return similarities
    
    def _approximate_resistance_distance(self, G: nx.Graph, u: int, v: int) -> float:
        """Approximate resistance distance between two nodes."""
        # Simple approximation: inverse of number of disjoint paths
        try:
            # Use NetworkX's approximation
            paths = list(nx.node_disjoint_paths(G, u, v))
            if paths:
                return 1.0 / len(paths)
            else:
                return float('inf')
        except:
            # Fallback: inverse degree product
            deg_u, deg_v = G.degree(u), G.degree(v)
            if deg_u > 0 and deg_v > 0:
                return 1.0 / (deg_u * deg_v)
            else:
                return 1.0


class ComprehensiveWeightingTester:
    """Comprehensive testing framework for adaptive weighting schemes."""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.weighter = AdaptiveGraphWeighting(verbose=False)  # Quiet for batch testing
        self.results = {}
        
    def log(self, message, end='\n'):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message, end=end)
    
    def create_test_graphs(self):
        """Create a comprehensive set of test graphs with known ground truth."""
        graphs = {}
        
        # 1. Perfect Cliques (Easy Case)
        self.log("🔸 Creating Perfect Cliques Test Graph...")
        G_cliques = nx.Graph()
        
        # Three perfect cliques
        clique_size = 6
        for clique_id in range(3):
            start_node = clique_id * clique_size
            for i in range(start_node, start_node + clique_size):
                for j in range(i + 1, start_node + clique_size):
                    G_cliques.add_edge(i, j)
        
        # Single bridges between cliques
        G_cliques.add_edge(5, 6)   # Clique 1 -> Clique 2
        G_cliques.add_edge(11, 12) # Clique 2 -> Clique 3
        
        cliques_communities = [
            list(range(6)),        # Clique 1: 0-5
            list(range(6, 12)),    # Clique 2: 6-11  
            list(range(12, 18))    # Clique 3: 12-17
        ]
        graphs['Perfect Cliques'] = (G_cliques, cliques_communities)
        
        # 2. Hub-Spoke Network (Medium Case)
        self.log("🔸 Creating Hub-Spoke Network...")
        G_hub = nx.Graph()
        
        # Central hub with spokes
        hub = 0
        spokes = list(range(1, 7))
        for spoke in spokes:
            G_hub.add_edge(hub, spoke)
        
        # Add some internal connections in hub community
        G_hub.add_edges_from([(1, 2), (2, 3), (4, 5), (5, 6)])
        
        # Separate dense community
        dense_community = list(range(7, 13))
        for i in dense_community:
            for j in dense_community:
                if i < j and np.random.random() > 0.3:  # 70% connection probability
                    G_hub.add_edge(i, j)
        
        # Bridge between communities
        G_hub.add_edge(6, 7)
        
        hub_communities = [
            [hub] + spokes,     # Hub community: 0-6
            dense_community     # Dense community: 7-12
        ]
        graphs['Hub-Spoke Network'] = (G_hub, hub_communities)
        
        # 3. Chain of Communities (Hard Case)
        self.log("🔸 Creating Chain of Communities...")
        G_chain = nx.Graph()
        
        # Five small communities in a chain
        community_size = 4
        chain_communities = []
        
        for comm_id in range(5):
            start_node = comm_id * community_size
            community = list(range(start_node, start_node + community_size))
            chain_communities.append(community)
            
            # Make each community a cycle
            for i in range(community_size):
                next_i = (i + 1) % community_size
                G_chain.add_edge(community[i], community[next_i])
            
            # Add one internal edge for density
            if community_size >= 4:
                G_chain.add_edge(community[0], community[2])
        
        # Chain connections (bridges)
        for i in range(4):  # Connect communities 0-1, 1-2, 2-3, 3-4
            bridge_u = (i + 1) * community_size - 1  # Last node of community i
            bridge_v = (i + 1) * community_size      # First node of community i+1
            G_chain.add_edge(bridge_u, bridge_v)
        
        graphs['Chain of Communities'] = (G_chain, chain_communities)
        
        # 4. Mixed Density Network (Very Hard Case)
        self.log("🔸 Creating Mixed Density Network...")
        G_mixed = nx.Graph()
        
        # Dense core (clique-like)
        dense_core = list(range(8))
        for i in dense_core:
            for j in dense_core:
                if i < j and np.random.random() > 0.2:  # 80% density
                    G_mixed.add_edge(i, j)
        
        # Sparse periphery (tree-like)
        sparse_nodes = list(range(8, 16))
        # Create a tree structure
        for i in range(len(sparse_nodes) - 1):
            G_mixed.add_edge(sparse_nodes[i], sparse_nodes[i + 1])
        
        # Add a few random connections in sparse region
        G_mixed.add_edges_from([(8, 12), (10, 14), (9, 15)])
        
        # Medium density community
        medium_nodes = list(range(16, 22))
        for i in medium_nodes:
            for j in medium_nodes:
                if i < j and np.random.random() > 0.5:  # 50% density
                    G_mixed.add_edge(i, j)
        
        # Bridges between regions
        G_mixed.add_edge(7, 8)   # Dense -> Sparse
        G_mixed.add_edge(15, 16) # Sparse -> Medium
        
        mixed_communities = [dense_core, sparse_nodes, medium_nodes]
        graphs['Mixed Density Network'] = (G_mixed, mixed_communities)
        
        # 5. Star Forest (Edge Case)
        self.log("🔸 Creating Star Forest...")
        G_stars = nx.Graph()
        
        star_communities = []
        node_counter = 0
        
        # Create 4 star subgraphs
        for star_id in range(4):
            center = node_counter
            leaves = list(range(node_counter + 1, node_counter + 5))
            node_counter += 5
            
            # Connect center to all leaves
            for leaf in leaves:
                G_stars.add_edge(center, leaf)
            
            star_communities.append([center] + leaves)
        
        # Connect stars with single edges
        G_stars.add_edge(4, 5)   # Star 1 -> Star 2
        G_stars.add_edge(9, 10)  # Star 2 -> Star 3  
        G_stars.add_edge(14, 15) # Star 3 -> Star 4
        
        graphs['Star Forest'] = (G_stars, star_communities)
        
        # 6. Social Network Simulation
        self.log("🔸 Creating Social Network Simulation...")
        G_social = nx.Graph()
        
        # Family 1 (dense clique)
        family1 = list(range(6))
        for i in family1:
            for j in family1:
                if i < j:
                    G_social.add_edge(i, j)
        
        # Work colleagues (medium density)
        colleagues = list(range(6, 12))
        for i in colleagues:
            for j in colleagues:
                if i < j and np.random.random() > 0.4:  # 60% connections
                    G_social.add_edge(i, j)
        
        # Hobby group (sparse but connected)
        hobby = list(range(12, 18))
        # Ring structure with some shortcuts
        for i in range(len(hobby)):
            next_i = (i + 1) % len(hobby)
            G_social.add_edge(hobby[i], hobby[next_i])
        G_social.add_edges_from([(12, 15), (13, 16), (14, 17)])
        
        # Cross-group friendships (weak ties)
        G_social.add_edge(5, 6)   # Family -> Work
        G_social.add_edge(11, 12) # Work -> Hobby
        G_social.add_edge(2, 14)  # Family -> Hobby (weak)
        
        social_communities = [family1, colleagues, hobby]
        graphs['Social Network'] = (G_social, social_communities)
        
        return graphs
    
    def evaluate_method(self, G, true_communities, method_name, weighting_func):
        """Evaluate a single weighting method on a graph."""
        try:
            start_time = time.time()
            
            # Generate weights
            weighted_edges = weighting_func(G)
            weighting_time = time.time() - start_time
            
            # Create Laplacian and run CMG
            start_time = time.time()
            A = create_laplacian_from_edges(weighted_edges, G.number_of_nodes())
            solver = CMGSteinerSolver(gamma=5.0, verbose=False)
            components, num_communities = solver.steiner_group(A)
            cmg_time = time.time() - start_time
            
            # Evaluate results
            evaluation = self.evaluate_communities(true_communities, components)
            evaluation['weighting_time'] = weighting_time
            evaluation['cmg_time'] = cmg_time
            evaluation['total_time'] = weighting_time + cmg_time
            evaluation['num_edges'] = len(weighted_edges)
            
            # Weight statistics
            if weighted_edges:
                weights = [w for u, v, w in weighted_edges]
                evaluation['weight_stats'] = {
                    'min': min(weights),
                    'max': max(weights),
                    'mean': np.mean(weights),
                    'std': np.std(weights)
                }
            
            return evaluation
            
        except Exception as e:
            return {
                'error': str(e),
                'num_communities': 0,
                'success': False
            }
    
    def evaluate_communities(self, true_communities, predicted_components):
        """Evaluate predicted communities against ground truth."""
        evaluation = {
            'num_true_communities': len(true_communities),
            'num_predicted_communities': len(set(predicted_components)),
            'success': False
        }
        
        # Check if we have the right number of communities
        evaluation['correct_count'] = len(true_communities) == len(set(predicted_components))
        
        if not true_communities:
            return evaluation
        
        # Convert true communities to label format
        true_labels = [-1] * len(predicted_components)
        for comm_id, nodes in enumerate(true_communities):
            for node in nodes:
                if node < len(true_labels):
                    true_labels[node] = comm_id
        
        # Filter out unassigned nodes
        valid_indices = [i for i in range(len(true_labels)) if true_labels[i] != -1]
        if not valid_indices:
            return evaluation
        
        true_filtered = [true_labels[i] for i in valid_indices]
        pred_filtered = [predicted_components[i] for i in valid_indices]
        
        # Calculate metrics
        if HAS_SKLEARN:
            try:
                evaluation['nmi'] = normalized_mutual_info_score(true_filtered, pred_filtered)
                evaluation['ari'] = adjusted_rand_score(true_filtered, pred_filtered)
                evaluation['success'] = evaluation['nmi'] > 0.8 and evaluation['ari'] > 0.8
            except:
                evaluation['nmi'] = 0.0
                evaluation['ari'] = 0.0
        
        # Custom accuracy metric
        evaluation['custom_accuracy'] = self.calculate_custom_accuracy(true_communities, predicted_components)
        
        if not evaluation.get('success', False):
            evaluation['success'] = evaluation['custom_accuracy'] > 0.8 and evaluation['correct_count']
        
        return evaluation
    
    def calculate_custom_accuracy(self, true_communities, predicted_components):
        """Calculate custom accuracy based on community overlap."""
        if not true_communities:
            return 0.0
        
        total_score = 0.0
        for true_comm in true_communities:
            if not true_comm:
                continue
            
            # Find predicted community with maximum overlap
            pred_comm_counts = defaultdict(int)
            for node in true_comm:
                if node < len(predicted_components):
                    pred_comm_counts[predicted_components[node]] += 1
            
            if pred_comm_counts:
                max_overlap = max(pred_comm_counts.values())
                total_score += max_overlap / len(true_comm)
        
        return total_score / len(true_communities)
    
    def run_comprehensive_test(self):
        """Run the complete test suite."""
        self.log("🚀 Starting Comprehensive Adaptive Weighting Test Suite")
        self.log("=" * 65)
        
        # Create test graphs
        test_graphs = self.create_test_graphs()
        
        # Weighting methods to test
        methods = {
            'Structure-Preserving': self.weighter.structure_preserving_weights,
            'Topology-Aware': self.weighter.topology_aware_weights,
            'Ensemble': self.weighter.ensemble_weights,
            'Natural Flow': self.weighter.natural_flow_weights,
            'Baseline (Unit)': lambda G: [(u, v, 1.0) for u, v in G.edges()],
        }
        
        # Track overall performance
        method_scores = defaultdict(list)
        method_times = defaultdict(list)
        method_successes = defaultdict(int)
        
        # Test each graph
        for graph_name, (G, true_communities) in test_graphs.items():
            self.log(f"\n📊 Testing on {graph_name}")
            self.log(f"   Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
            self.log(f"   True communities: {len(true_communities)}")
            self.log(f"   Community sizes: {[len(c) for c in true_communities]}")
            self.log("-" * 60)
            
            graph_results = {}
            
            for method_name, method_func in methods.items():
                evaluation = self.evaluate_method(G, true_communities, method_name, method_func)
                graph_results[method_name] = evaluation
                
                # Print results
                status = "✅" if evaluation.get('success', False) else "❌"
                self.log(f"{method_name:20s}: {status} ", end="")
                
                if 'error' in evaluation:
                    self.log(f"Error - {evaluation['error']}")
                else:
                    self.log(f"Communities: {evaluation['num_predicted_communities']}", end="")
                    
                    if HAS_SKLEARN and 'nmi' in evaluation:
                        self.log(f" (NMI: {evaluation['nmi']:.3f}, ARI: {evaluation['ari']:.3f})", end="")
                    else:
                        self.log(f" (Accuracy: {evaluation['custom_accuracy']:.3f})", end="")
                    
                    self.log(f" Time: {evaluation['total_time']:.3f}s")
                    
                    # Track scores
                    if HAS_SKLEARN and 'nmi' in evaluation:
                        score = (evaluation['nmi'] + evaluation['ari']) / 2
                    else:
                        score = evaluation['custom_accuracy']
                    
                    method_scores[method_name].append(score)
                    method_times[method_name].append(evaluation['total_time'])
                    
                    if evaluation.get('success', False):
                        method_successes[method_name] += 1
            
            self.results[graph_name] = graph_results
        
        # Overall summary
        self.print_summary(method_scores, method_times, method_successes, len(test_graphs))
        
        return self.results
    
    def print_summary(self, method_scores, method_times, method_successes, total_graphs):
        """Print comprehensive summary of results."""
        self.log("\n🏆 COMPREHENSIVE SUMMARY")
        self.log("=" * 50)
        
        # Success rates
        self.log("\n📈 Success Rates:")
        for method in sorted(method_successes.keys(), key=lambda x: method_successes[x], reverse=True):
            success_rate = method_successes[method] / total_graphs * 100
            avg_score = np.mean(method_scores[method]) if method_scores[method] else 0
            avg_time = np.mean(method_times[method]) if method_times[method] else 0
            
            self.log(f"  {method:20s}: {success_rate:5.1f}% success, {avg_score:.3f} avg score, {avg_time:.3f}s")
        
        # Best performer by category
        self.log("\n🥇 Best Performers:")
        
        # Highest success rate
        best_success = max(method_successes.items(), key=lambda x: x[1])
        self.log(f"  Most Reliable: {best_success[0]} ({best_success[1]}/{total_graphs} successes)")
        
        # Highest average score
        if method_scores:
            best_score = max(method_scores.items(), key=lambda x: np.mean(x[1]) if x[1] else 0)
            self.log(f"  Highest Quality: {best_score[0]} ({np.mean(best_score[1]):.3f} avg score)")
        
        # Fastest
        if method_times:
            fastest = min(method_times.items(), key=lambda x: np.mean(x[1]) if x[1] else float('inf'))
            self.log(f"  Fastest: {fastest[0]} ({np.mean(fastest[1]):.3f}s avg time)")
        
        # Recommendations
        self.log("\n💡 FINAL RECOMMENDATIONS:")
        self.log("-" * 30)
        
        if method_successes:
            # Find most reliable method
            most_reliable = max(method_successes.items(), key=lambda x: x[1])[0]
            self.log(f"  🎯 For Production Use: {most_reliable}")
            
            # Find balanced performer
            balanced_scores = {}
            for method in method_successes.keys():
                if method_scores[method] and method_times[method]:
                    success_rate = method_successes[method] / total_graphs
                    avg_score = np.mean(method_scores[method])
                    avg_time = 1.0 / (np.mean(method_times[method]) + 0.001)  # Reciprocal for speed
                    
                    # Balanced score: success * quality * speed
                    balanced_scores[method] = success_rate * avg_score * (avg_time / 10)
            
            if balanced_scores:
                best_balanced = max(balanced_scores.items(), key=lambda x: x[1])[0]
                self.log(f"  ⚖️  For Balanced Performance: {best_balanced}")
        
        self.log("\n🔬 Ready for integration? Check individual test results above!")


def main():
    """Run the comprehensive test suite."""
    print("🧪 Comprehensive Adaptive Weighting Test Suite")
    print("This will test the adaptive weighting methods on 6 different graph types")
    print("with varying difficulty levels and structural properties.")
    print()
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Run comprehensive tests
    tester = ComprehensiveWeightingTester(verbose=True)
    results = tester.run_comprehensive_test()
    
    print("\n" + "="*65)
    print("🎯 Test suite completed!")
    print("📊 Check the detailed results above to decide on integration.")
    print("💡 Focus on methods with high success rates and good performance.")


if __name__ == "__main__":
    main()
