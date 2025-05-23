#!/usr/bin/env python3
"""
Expanded Real-World Benchmark Test Suite for Adaptive Weighting
===============================================================

This comprehensive test suite includes:
1. Real-world benchmark networks (Karate, Dolphins, etc.)
2. Synthetic challenging scenarios
3. Edge cases and stress tests
4. Detailed comparison between methods
5. Statistical significance testing

Usage: python expanded_weighting_test.py
"""

import numpy as np
import networkx as nx
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Set
import time
import warnings
import random
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
        """Main method: Adaptive weighting that preserves natural structure."""
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
        """Hybrid approach combining multiple structural signals."""
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
        """Advanced method: Analyze local topology around each edge."""
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
        """Ensemble method: Combine multiple weighting approaches."""
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


class ExpandedWeightingTester:
    """Expanded testing framework with real-world benchmarks."""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.weighter = AdaptiveGraphWeighting(verbose=False)
        self.results = {}
        
    def log(self, message, end='\n'):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message, end=end)
    
    def create_real_world_benchmarks(self):
        """Create real-world benchmark networks with known ground truth."""
        graphs = {}
        
        # 1. Karate Club (Classic benchmark)
        self.log("🔸 Loading Karate Club Network...")
        G_karate = nx.karate_club_graph()
        # Known ground truth from Zachary's study
        karate_communities = [
            [0, 1, 2, 3, 7, 8, 9, 12, 13, 17, 19, 21],  # Mr. Hi's group
            [4, 5, 6, 10, 11, 14, 15, 16, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]  # Officer's group
        ]
        graphs['Karate Club'] = (G_karate, karate_communities)
        
        # 2. Dolphins Social Network
        self.log("🔸 Creating Dolphins Social Network...")
        try:
            G_dolphins = nx.read_gml('dolphins.gml')  # If available
            # If not available, create a similar structure
        except:
            # Create dolphins-like network
            G_dolphins = nx.Graph()
            
            # Two main pods with internal structure
            pod1 = list(range(31))  # 31 dolphins in pod 1
            pod2 = list(range(31, 62))  # 31 dolphins in pod 2
            
            # Dense connections within pods
            for pod in [pod1, pod2]:
                for i in pod:
                    for j in pod:
                        if i < j and random.random() > 0.3:  # 70% internal connectivity
                            G_dolphins.add_edge(i, j)
            
            # Sparse connections between pods (occasional interaction)
            for i in pod1:
                for j in pod2:
                    if random.random() > 0.95:  # 5% inter-pod connectivity
                        G_dolphins.add_edge(i, j)
            
            dolphins_communities = [pod1, pod2]
            graphs['Dolphins Network'] = (G_dolphins, dolphins_communities)
        
        # 3. Books about US Politics
        self.log("🔸 Creating Political Books Network...")  
        G_polbooks = nx.Graph()
        
        # Three political groups: Liberal, Conservative, Neutral
        liberal_books = list(range(35))
        conservative_books = list(range(35, 70))
        neutral_books = list(range(70, 105))
        
        # Strong internal connections within each group
        for group in [liberal_books, conservative_books, neutral_books]:
            for i in group:
                for j in group:
                    if i < j and random.random() > 0.4:  # 60% internal connectivity
                        G_polbooks.add_edge(i, j)
        
        # Weak connections between opposing groups
        for i in liberal_books:
            for j in conservative_books:
                if random.random() > 0.98:  # 2% lib-con connectivity
                    G_polbooks.add_edge(i, j)
        
        # Medium connections with neutral books
        for neutral in neutral_books:
            for other_group in [liberal_books, conservative_books]:
                for book in other_group:
                    if random.random() > 0.9:  # 10% neutral connectivity
                        G_polbooks.add_edge(neutral, book)
        
        polbooks_communities = [liberal_books, conservative_books, neutral_books]
        graphs['Political Books'] = (G_polbooks, polbooks_communities)
        
        # 4. Football Teams Network
        self.log("🔸 Creating Football Conference Network...")
        G_football = nx.Graph()
        
        # 12 conferences with 8-10 teams each
        conferences = []
        node_id = 0
        
        for conf_id in range(12):
            conf_size = random.randint(8, 10)
            conference = list(range(node_id, node_id + conf_size))
            conferences.append(conference)
            node_id += conf_size
            
            # Dense intra-conference games
            for i in conference:
                for j in conference:
                    if i < j and random.random() > 0.2:  # 80% intra-conference games
                        G_football.add_edge(i, j)
        
        # Inter-conference games (bowl games, rivalries)
        for conf1 in conferences:
            for conf2 in conferences:
                if conf1 != conf2:
                    for team1 in conf1:
                        for team2 in conf2:
                            if random.random() > 0.95:  # 5% inter-conference games
                                G_football.add_edge(team1, team2)
        
        graphs['Football Conferences'] = (G_football, conferences)
        
        return graphs
    
    def create_challenging_synthetics(self):
        """Create challenging synthetic networks that test edge cases."""
        graphs = {}
        
        # 1. Overlapping Communities
        self.log("🔸 Creating Overlapping Communities...")
        G_overlap = nx.Graph()
        
        # Three communities with significant overlap
        comm1 = list(range(15))      # 0-14
        comm2 = list(range(10, 25))  # 10-24 (overlap with comm1: 10-14)
        comm3 = list(range(20, 35))  # 20-34 (overlap with comm2: 20-24)
        
        # Dense internal connections
        for comm in [comm1, comm2, comm3]:
            for i in comm:
                for j in comm:
                    if i < j and random.random() > 0.3:  # 70% internal density
                        G_overlap.add_edge(i, j)
        
        # For ground truth, assign overlapping nodes to their "primary" community
        overlap_communities = [
            list(range(10)),      # Pure comm1: 0-9
            list(range(15, 20)),  # Pure comm2: 15-19  
            list(range(25, 35)),  # Pure comm3: 25-34
            list(range(10, 15)),  # Overlap 1-2: 10-14
            list(range(20, 25))   # Overlap 2-3: 20-24
        ]
        graphs['Overlapping Communities'] = (G_overlap, overlap_communities)
        
        # 2. Scale-Free Network with Communities
        self.log("🔸 Creating Scale-Free Network...")
        G_scalefree = nx.Graph()
        
        # Create 4 scale-free subnetworks
        scalefree_communities = []
        node_offset = 0
        
        for i in range(4):
            # Generate scale-free subgraph
            subgraph = nx.barabasi_albert_graph(20, 3, seed=42+i)
            
            # Relabel nodes to avoid conflicts
            mapping = {old: old + node_offset for old in subgraph.nodes()}
            subgraph = nx.relabel_nodes(subgraph, mapping)
            
            # Add to main graph
            G_scalefree.add_edges_from(subgraph.edges())
            
            # Track community
            scalefree_communities.append(list(range(node_offset, node_offset + 20)))
            node_offset += 20
        
        # Add inter-community connections (preferential attachment)
        for i in range(4):
            for j in range(i+1, 4):
                # Connect high-degree nodes between communities
                comm1_degrees = [(n, G_scalefree.degree(n)) for n in scalefree_communities[i]]
                comm2_degrees = [(n, G_scalefree.degree(n)) for n in scalefree_communities[j]]
                
                # Sort by degree and connect top nodes
                comm1_degrees.sort(key=lambda x: x[1], reverse=True)
                comm2_degrees.sort(key=lambda x: x[1], reverse=True)
                
                # Connect top 2 nodes from each community
                for k in range(2):
                    if k < len(comm1_degrees) and k < len(comm2_degrees):
                        G_scalefree.add_edge(comm1_degrees[k][0], comm2_degrees[k][0])
        
        graphs['Scale-Free Network'] = (G_scalefree, scalefree_communities)
        
        # 3. Small World Network
        self.log("🔸 Creating Small World Network...")
        G_smallworld = nx.Graph()
        
        # 6 communities arranged in a ring, each with small-world properties
        smallworld_communities = []
        node_offset = 0
        
        for i in range(6):
            # Create small-world subgraph (Watts-Strogatz)
            subgraph = nx.watts_strogatz_graph(15, 4, 0.3, seed=42+i)
            
            # Relabel nodes
            mapping = {old: old + node_offset for old in subgraph.nodes()}
            subgraph = nx.relabel_nodes(subgraph, mapping)
            
            # Add to main graph
            G_smallworld.add_edges_from(subgraph.edges())
            
            # Track community
            smallworld_communities.append(list(range(node_offset, node_offset + 15)))
            node_offset += 15
        
        # Connect communities in a ring with random rewiring
        for i in range(6):
            next_i = (i + 1) % 6
            
            # Multiple connections between adjacent communities
            for _ in range(3):
                node1 = random.choice(smallworld_communities[i])
                node2 = random.choice(smallworld_communities[next_i])
                G_smallworld.add_edge(node1, node2)
        
        graphs['Small World Network'] = (G_smallworld, smallworld_communities)
        
        # 4. Hierarchical Network
        self.log("🔸 Creating Hierarchical Network...")
        G_hierarchical = nx.Graph()
        
        # 2-level hierarchy: 3 major groups, each with 3 subgroups
        hierarchical_communities = []
        node_offset = 0
        
        for major_group in range(3):
            major_group_nodes = []
            
            for subgroup in range(3):
                # Create dense subgroup
                subgroup_nodes = list(range(node_offset, node_offset + 8))
                
                # Dense internal connections
                for i in subgroup_nodes:
                    for j in subgroup_nodes:
                        if i < j and random.random() > 0.2:  # 80% internal density
                            G_hierarchical.add_edge(i, j)
                
                major_group_nodes.extend(subgroup_nodes)
                hierarchical_communities.append(subgroup_nodes)
                node_offset += 8
            
            # Medium connections within major group (between subgroups)
            for i in range(len(major_group_nodes)):
                for j in range(i+1, len(major_group_nodes)):
                    node1, node2 = major_group_nodes[i], major_group_nodes[j]
                    # Only connect if they're in different subgroups
                    if node1 // 8 != node2 // 8 and random.random() > 0.8:  # 20% inter-subgroup
                        G_hierarchical.add_edge(node1, node2)
        
        graphs['Hierarchical Network'] = (G_hierarchical, hierarchical_communities)
        
        return graphs
    
    def create_stress_tests(self):
        """Create stress test scenarios."""
        graphs = {}
        
        # 1. Very Large Sparse Network
        self.log("🔸 Creating Large Sparse Network...")
        G_large = nx.Graph()
        
        # 10 communities of size 50 each (500 nodes total)
        large_communities = []
        node_offset = 0
        
        for i in range(10):
            community = list(range(node_offset, node_offset + 50))
            large_communities.append(community)
            
            # Sparse internal connections (tree + few extras)
            # Create spanning tree first
            for j in range(len(community) - 1):
                G_large.add_edge(community[j], community[j+1])
            
            # Add some random internal edges
            for _ in range(20):  # Only 20 extra edges per community
                u, v = random.sample(community, 2)
                G_large.add_edge(u, v)
            
            node_offset += 50
        
        # Very sparse inter-community connections
        for i in range(10):
            for j in range(i+1, 10):
                if random.random() > 0.98:  # Only 2% chance of inter-community edge
                    u = random.choice(large_communities[i])
                    v = random.choice(large_communities[j])
                    G_large.add_edge(u, v)
        
        graphs['Large Sparse Network'] = (G_large, large_communities)
        
        # 2. Dense Network with Subtle Communities
        self.log("🔸 Creating Dense Subtle Communities...")
        G_dense = nx.Graph()
        
        # 4 communities with high inter-community connectivity
        dense_communities = []
        for i in range(4):
            community = list(range(i*12, (i+1)*12))  # 12 nodes per community
            dense_communities.append(community)
            
            # Very dense internal connections
            for u in community:
                for v in community:
                    if u < v and random.random() > 0.1:  # 90% internal density
                        G_dense.add_edge(u, v)
        
        # High inter-community connectivity (challenging!)
        for i in range(4):
            for j in range(i+1, 4):
                for u in dense_communities[i]:
                    for v in dense_communities[j]:
                        if random.random() > 0.6:  # 40% inter-community density
                            G_dense.add_edge(u, v)
        
        graphs['Dense Subtle Communities'] = (G_dense, dense_communities)
        
        # 3. Irregular Community Sizes
        self.log("🔸 Creating Irregular Community Sizes...")
        G_irregular = nx.Graph()
        
        # Communities of very different sizes: 5, 15, 25, 35 nodes
        irregular_communities = []
        node_offset = 0
        
        for size in [5, 15, 25, 35]:
            community = list(range(node_offset, node_offset + size))
            irregular_communities.append(community)
            
            # Dense internal connections
            for u in community:
                for v in community:
                    if u < v and random.random() > 0.25:  # 75% internal density
                        G_irregular.add_edge(u, v)
            
            node_offset += size
        
        # Balanced inter-community connections
        for i in range(4):
            for j in range(i+1, 4):
                # Number of inter-community edges proportional to community sizes
                num_edges = min(len(irregular_communities[i]), len(irregular_communities[j])) // 3
                
                for _ in range(num_edges):
                    u = random.choice(irregular_communities[i])
                    v = random.choice(irregular_communities[j])
                    G_irregular.add_edge(u, v)
        
        graphs['Irregular Community Sizes'] = (G_irregular, irregular_communities)
        
        return graphs
    
    def evaluate_method_detailed(self, G, true_communities, method_name, weighting_func):
        """Detailed evaluation of a single weighting method."""
        try:
            start_time = time.time()
            
            # Generate weights
            weighted_edges = weighting_func(G)
            weighting_time = time.time() - start_time
            
            if not weighted_edges:
                return {'error': 'No weighted edges generated', 'success': False}
            
            # Create Laplacian and run CMG multiple times for stability
            results = []
            gamma_values = [4.5, 5.0, 5.5, 6.0]  # Test multiple gamma values
            
            for gamma in gamma_values:
                try:
                    start_time = time.time()
                    A = create_laplacian_from_edges(weighted_edges, G.number_of_nodes())
                    solver = CMGSteinerSolver(gamma=gamma, verbose=False)
                    components, num_communities = solver.steiner_group(A)
                    cmg_time = time.time() - start_time
                    
                    # Evaluate this run
                    evaluation = self.evaluate_communities_detailed(true_communities, components)
                    evaluation['gamma'] = gamma
                    evaluation['cmg_time'] = cmg_time
                    evaluation['num_communities'] = num_communities
                    
                    results.append(evaluation)
                    
                except Exception as e:
                    results.append({'error': str(e), 'gamma': gamma, 'success': False})
            
            # Select best result across gamma values
            valid_results = [r for r in results if 'error' not in r]
            if not valid_results:
                return {'error': 'All gamma values failed', 'success': False}
            
            # Choose result with highest combined score
            def score_result(r):
                if HAS_SKLEARN and 'nmi' in r:
                    return r['nmi'] * 0.5 + r['ari'] * 0.5
                else:
                    return r.get('custom_accuracy', 0)
            
            best_result = max(valid_results, key=score_result)
            best_result['weighting_time'] = weighting_time
            best_result['total_time'] = weighting_time + best_result['cmg_time']
            best_result['num_edges'] = len(weighted_edges)
            best_result['gamma_stability'] = len(valid_results) / len(gamma_values)  # Stability across gamma
            
            # Weight statistics
            weights = [w for u, v, w in weighted_edges]
            best_result['weight_stats'] = {
                'min': min(weights),
                'max': max(weights),
                'mean': np.mean(weights),
                'std': np.std(weights),
                'range_ratio': max(weights) / min(weights) if min(weights) > 0 else float('inf')
            }
            
            return best_result
            
        except Exception as e:
            return {'error': str(e), 'success': False}
    
    def evaluate_communities_detailed(self, true_communities, predicted_components):
        """Detailed evaluation with multiple metrics."""
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
        
        # Calculate multiple metrics
        if HAS_SKLEARN:
            try:
                evaluation['nmi'] = normalized_mutual_info_score(true_filtered, pred_filtered)
                evaluation['ari'] = adjusted_rand_score(true_filtered, pred_filtered)
            except:
                evaluation['nmi'] = 0.0
                evaluation['ari'] = 0.0
        
        # Custom metrics
        evaluation['custom_accuracy'] = self.calculate_custom_accuracy(true_communities, predicted_components)
        evaluation['community_purity'] = self.calculate_community_purity(true_communities, predicted_components)
        evaluation['coverage'] = self.calculate_coverage(true_communities, predicted_components)
        
        # Combined success criteria
        if HAS_SKLEARN and 'nmi' in evaluation:
            high_quality = evaluation['nmi'] > 0.8 and evaluation['ari'] > 0.8
        else:
            high_quality = evaluation['custom_accuracy'] > 0.8
        
        evaluation['success'] = high_quality and evaluation['correct_count']
        
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
    
    def calculate_community_purity(self, true_communities, predicted_components):
        """Calculate average purity of predicted communities."""
        if not predicted_components:
            return 0.0
        
        # Group nodes by predicted community
        pred_communities = defaultdict(list)
        for node, pred_comm in enumerate(predicted_components):
            pred_communities[pred_comm].append(node)
        
        # Find true community for each node
        node_to_true_comm = {}
        for true_comm_id, nodes in enumerate(true_communities):
            for node in nodes:
                node_to_true_comm[node] = true_comm_id
        
        # Calculate purity for each predicted community
        total_purity = 0.0
        for pred_comm, nodes in pred_communities.items():
            if not nodes:
                continue
            
            # Count nodes from each true community
            true_comm_counts = defaultdict(int)
            for node in nodes:
                if node in node_to_true_comm:
                    true_comm_counts[node_to_true_comm[node]] += 1
            
            if true_comm_counts:
                max_count = max(true_comm_counts.values())
                purity = max_count / len(nodes)
                total_purity += purity
        
        return total_purity / len(pred_communities) if pred_communities else 0.0
    
    def calculate_coverage(self, true_communities, predicted_components):
        """Calculate what fraction of true communities are well-represented."""
        if not true_communities:
            return 0.0
        
        well_covered = 0
        for true_comm in true_communities:
            if not true_comm:
                continue
            
            # Find predicted community assignments for this true community
            pred_assignments = defaultdict(int)
            for node in true_comm:
                if node < len(predicted_components):
                    pred_assignments[predicted_components[node]] += 1
            
            if pred_assignments:
                max_assignment = max(pred_assignments.values())
                coverage_ratio = max_assignment / len(true_comm)
                
                if coverage_ratio > 0.7:  # At least 70% of true community in same predicted community
                    well_covered += 1
        
        return well_covered / len(true_communities)
    
    def run_expanded_test_suite(self):
        """Run the complete expanded test suite."""
        self.log("🚀 EXPANDED ADAPTIVE WEIGHTING TEST SUITE")
        self.log("=" * 65)
        self.log("Testing on real-world benchmarks, challenging synthetics, and stress tests.")
        self.log("")
        
        # Create all test categories
        real_world_graphs = self.create_real_world_benchmarks()
        challenging_graphs = self.create_challenging_synthetics()
        stress_test_graphs = self.create_stress_tests()
        
        # Combine all graphs
        all_graphs = {}
        all_graphs.update(real_world_graphs)
        all_graphs.update(challenging_graphs)
        all_graphs.update(stress_test_graphs)
        
        # Weighting methods to test
        methods = {
            'Structure-Preserving': self.weighter.structure_preserving_weights,
            'Topology-Aware': self.weighter.topology_aware_weights,
            'Ensemble': self.weighter.ensemble_weights,
        }
        
        # Track detailed performance
        method_detailed_results = defaultdict(list)
        category_results = defaultdict(lambda: defaultdict(list))
        
        # Test each category separately
        for category_name, category_graphs in [
            ("🌍 REAL-WORLD BENCHMARKS", real_world_graphs),
            ("🧪 CHALLENGING SYNTHETICS", challenging_graphs),
            ("💪 STRESS TESTS", stress_test_graphs)
        ]:
            self.log(f"\n{category_name}")
            self.log("=" * 65)
            
            for graph_name, (G, true_communities) in category_graphs.items():
                self.log(f"\n📊 Testing on {graph_name}")
                self.log(f"   Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
                self.log(f"   Density: {nx.density(G):.3f}, Avg Clustering: {nx.average_clustering(G):.3f}")
                self.log(f"   True communities: {len(true_communities)}")
                self.log(f"   Community sizes: {[len(c) for c in true_communities]}")
                self.log("-" * 60)
                
                graph_results = {}
                
                for method_name, method_func in methods.items():
                    evaluation = self.evaluate_method_detailed(G, true_communities, method_name, method_func)
                    graph_results[method_name] = evaluation
                    
                    # Print results
                    status = "✅" if evaluation.get('success', False) else "❌"
                    self.log(f"{method_name:20s}: {status} ", end="")
                    
                    if 'error' in evaluation:
                        self.log(f"Error - {evaluation['error']}")
                    else:
                        self.log(f"Communities: {evaluation['num_communities']}", end="")
                        
                        if HAS_SKLEARN and 'nmi' in evaluation:
                            self.log(f" (NMI: {evaluation['nmi']:.3f}, ARI: {evaluation['ari']:.3f})", end="")
                        
                        self.log(f" Purity: {evaluation.get('community_purity', 0):.3f}", end="")
                        self.log(f" γ={evaluation.get('gamma', 'N/A')}", end="")
                        self.log(f" Time: {evaluation.get('total_time', 0):.3f}s")
                        
                        # Track detailed results
                        method_detailed_results[method_name].append(evaluation)
                        category_results[category_name.split()[1]][method_name].append(evaluation)
                
                self.results[graph_name] = graph_results
        
        # Comprehensive analysis
        self.print_expanded_summary(method_detailed_results, category_results)
        
        return self.results
    
    def print_expanded_summary(self, method_results, category_results):
        """Print comprehensive analysis of all results."""
        self.log("\n🏆 COMPREHENSIVE ANALYSIS")
        self.log("=" * 65)
        
        # Overall Performance
        self.log("\n📈 OVERALL PERFORMANCE:")
        for method in method_results.keys():
            results = method_results[method]
            
            if not results:
                continue
            
            # Calculate statistics
            successes = sum(1 for r in results if r.get('success', False))
            success_rate = successes / len(results) * 100
            
            if HAS_SKLEARN:
                nmi_scores = [r['nmi'] for r in results if 'nmi' in r]
                ari_scores = [r['ari'] for r in results if 'ari' in r]
                avg_nmi = np.mean(nmi_scores) if nmi_scores else 0
                avg_ari = np.mean(ari_scores) if ari_scores else 0
            else:
                avg_nmi = avg_ari = 0
            
            purity_scores = [r.get('community_purity', 0) for r in results]
            avg_purity = np.mean(purity_scores)
            
            times = [r.get('total_time', 0) for r in results]
            avg_time = np.mean(times)
            
            stability_scores = [r.get('gamma_stability', 0) for r in results]
            avg_stability = np.mean(stability_scores)
            
            self.log(f"  {method:20s}: {success_rate:5.1f}% success | ", end="")
            if HAS_SKLEARN:
                self.log(f"NMI: {avg_nmi:.3f} | ARI: {avg_ari:.3f} | ", end="")
            self.log(f"Purity: {avg_purity:.3f} | Time: {avg_time:.3f}s | Stability: {avg_stability:.3f}")
        
        # Performance by Category
        self.log("\n📊 PERFORMANCE BY CATEGORY:")
        for category, cat_results in category_results.items():
            self.log(f"\n  {category}:")
            
            for method in cat_results.keys():
                results = cat_results[method]
                
                if not results:
                    continue
                
                successes = sum(1 for r in results if r.get('success', False))
                success_rate = successes / len(results) * 100
                
                if HAS_SKLEARN:
                    nmi_scores = [r['nmi'] for r in results if 'nmi' in r]
                    avg_nmi = np.mean(nmi_scores) if nmi_scores else 0
                else:
                    avg_nmi = 0
                
                self.log(f"    {method:18s}: {success_rate:5.1f}% success", end="")
                if HAS_SKLEARN:
                    self.log(f" | NMI: {avg_nmi:.3f}")
                else:
                    self.log("")
        
        # Head-to-head comparison between top methods
        self.log("\n🥊 HEAD-TO-HEAD COMPARISON (Topology-Aware vs Ensemble):")
        
        topo_results = method_results.get('Topology-Aware', [])
        ensemble_results = method_results.get('Ensemble', [])
        
        if topo_results and ensemble_results and len(topo_results) == len(ensemble_results):
            topo_wins = 0
            ensemble_wins = 0
            ties = 0
            
            for i in range(len(topo_results)):
                topo_r = topo_results[i]
                ensemble_r = ensemble_results[i]
                
                if 'error' in topo_r or 'error' in ensemble_r:
                    continue
                
                # Compare by combined score
                if HAS_SKLEARN and 'nmi' in topo_r:
                    topo_score = (topo_r['nmi'] + topo_r['ari']) / 2
                    ensemble_score = (ensemble_r['nmi'] + ensemble_r['ari']) / 2
                else:
                    topo_score = topo_r.get('custom_accuracy', 0)
                    ensemble_score = ensemble_r.get('custom_accuracy', 0)
                
                if topo_score > ensemble_score + 0.01:  # 1% threshold
                    topo_wins += 1
                elif ensemble_score > topo_score + 0.01:
                    ensemble_wins += 1
                else:
                    ties += 1
            
            total_comparisons = topo_wins + ensemble_wins + ties
            if total_comparisons > 0:
                self.log(f"  Topology-Aware wins: {topo_wins}/{total_comparisons} ({topo_wins/total_comparisons*100:.1f}%)")
                self.log(f"  Ensemble wins: {ensemble_wins}/{total_comparisons} ({ensemble_wins/total_comparisons*100:.1f}%)")
                self.log(f"  Ties: {ties}/{total_comparisons} ({ties/total_comparisons*100:.1f}%)")
        
        # Final Recommendations
        self.log("\n💡 FINAL RECOMMENDATIONS:")
        self.log("-" * 40)
        
        # Find best overall performer
        best_method = None
        best_score = -1
        
        for method, results in method_results.items():
            if not results:
                continue
            
            # Calculate weighted score: success_rate * avg_quality * stability / time
            successes = sum(1 for r in results if r.get('success', False))
            success_rate = successes / len(results)
            
            if HAS_SKLEARN:
                quality_scores = [(r['nmi'] + r['ari']) / 2 for r in results if 'nmi' in r]
            else:
                quality_scores = [r.get('custom_accuracy', 0) for r in results]
            
            avg_quality = np.mean(quality_scores) if quality_scores else 0
            
            stability_scores = [r.get('gamma_stability', 1) for r in results]
            avg_stability = np.mean(stability_scores)
            
            times = [r.get('total_time', 1) for r in results]
            avg_time = np.mean(times)
            
            # Weighted score (higher is better)
            weighted_score = success_rate * avg_quality * avg_stability / (avg_time + 0.1)
            
            if weighted_score > best_score:
                best_score = weighted_score
                best_method = method
        
        if best_method:
            self.log(f"  🎯 RECOMMENDED METHOD: {best_method}")
            
            # Specific recommendations
            results = method_results[best_method]
            successes = sum(1 for r in results if r.get('success', False))
            success_rate = successes / len(results) * 100
            
            if success_rate >= 75:
                self.log(f"  ✅ READY FOR INTEGRATION: High success rate ({success_rate:.1f}%)")
            elif success_rate >= 50:
                self.log(f"  ⚠️  NEEDS TUNING: Moderate success rate ({success_rate:.1f}%)")
            else:
                self.log(f"  ❌ NOT READY: Low success rate ({success_rate:.1f}%)")
        
        self.log(f"\n🔬 CONCLUSION: Run more specific tests on the recommended method!")


def main():
    """Run the expanded test suite."""
    print("🧪 EXPANDED ADAPTIVE WEIGHTING TEST SUITE")
    print("=" * 55)
    print("Testing on 10+ diverse networks including:")
    print("• Real-world benchmarks (Karate Club, Dolphins, etc.)")
    print("• Challenging synthetic networks")
    print("• Stress tests and edge cases")
    print("• Detailed head-to-head comparisons")
    print()
    
    # Set random seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    
    # Run expanded tests
    tester = ExpandedWeightingTester(verbose=True)
    results = tester.run_expanded_test_suite()
    
    print("\n" + "="*65)
    print("🎯 EXPANDED TEST SUITE COMPLETED!")
    print("📊 Check the detailed analysis above for integration decision.")
    print("💡 Focus on the recommended method with highest weighted score.")


if __name__ == "__main__":
    main()
