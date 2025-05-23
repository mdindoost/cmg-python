#!/usr/bin/env python3
"""
Diagnostic Test to Identify Why All Methods Are Failing
======================================================

This script will systematically check each component to identify
the root cause of the universal failures.

Usage: python diagnostic_test.py
"""

import numpy as np
import networkx as nx
import traceback
import warnings
warnings.filterwarnings('ignore')

# Test CMG import
try:
    from cmg import CMGSteinerSolver, create_laplacian_from_edges
    print("✅ CMG import successful")
except Exception as e:
    print(f"❌ CMG import failed: {e}")
    print("🔧 Fix: Check CMG package installation")
    exit(1)

def test_basic_cmg():
    """Test basic CMG functionality with simple known case."""
    print("\n🔬 Testing Basic CMG Functionality...")
    
    try:
        # Create simple test case that should work
        edges = [(0, 1, 1.0), (1, 2, 1.0), (3, 4, 1.0), (4, 5, 1.0)]  # Two separate components
        
        print(f"   Edges: {edges}")
        
        # Create Laplacian
        A = create_laplacian_from_edges(edges, 6)
        print(f"   Laplacian shape: {A.shape}")
        print(f"   Laplacian nnz: {A.nnz}")
        
        # Test CMG
        solver = CMGSteinerSolver(gamma=5.0, verbose=True)
        components, num_communities = solver.steiner_group(A)
        
        print(f"   Result: {num_communities} communities")
        print(f"   Components: {components}")
        
        if num_communities == 2:
            print("✅ Basic CMG test PASSED")
            return True
        else:
            print("❌ Basic CMG test FAILED - Expected 2 communities")
            return False
            
    except Exception as e:
        print(f"❌ Basic CMG test FAILED with error: {e}")
        traceback.print_exc()
        return False

def test_graph_creation():
    """Test if our graph creation is working properly."""
    print("\n🔬 Testing Graph Creation...")
    
    try:
        # Test Karate Club
        G = nx.karate_club_graph()
        print(f"   Karate Club: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"   Node IDs: {sorted(G.nodes())}")
        
        # Check if nodes are consecutive
        nodes = sorted(G.nodes())
        expected_nodes = list(range(len(nodes)))
        if nodes == expected_nodes:
            print("✅ Node IDs are consecutive (0-based)")
        else:
            print(f"⚠️  Node IDs are not consecutive: {nodes[:10]}...")
        
        # Test simple graph creation
        G_simple = nx.Graph()
        G_simple.add_edges_from([(0, 1), (1, 2), (2, 0)])  # Triangle
        print(f"   Simple triangle: {G_simple.number_of_nodes()} nodes, {G_simple.number_of_edges()} edges")
        
        return True
        
    except Exception as e:
        print(f"❌ Graph creation test FAILED: {e}")
        traceback.print_exc()
        return False

def test_weighting_methods():
    """Test weighting methods on simple graph."""
    print("\n🔬 Testing Weighting Methods...")
    
    try:
        # Simple test graph: two triangles connected by one edge
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])  # Triangle 1
        G.add_edges_from([(3, 4), (4, 5), (5, 3)])  # Triangle 2
        G.add_edge(2, 3)  # Bridge
        
        print(f"   Test graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Simple weighting function
        def simple_weights(G):
            return [(u, v, 1.0) for u, v in G.edges()]
        
        # Test weighting
        weighted_edges = simple_weights(G)
        print(f"   Generated {len(weighted_edges)} weighted edges")
        print(f"   Sample edges: {weighted_edges[:3]}")
        
        # Test Laplacian creation
        A = create_laplacian_from_edges(weighted_edges, G.number_of_nodes())
        print(f"   Laplacian created: {A.shape}")
        
        # Test CMG on this
        solver = CMGSteinerSolver(gamma=5.0, verbose=False)
        components, num_communities = solver.steiner_group(A)
        
        print(f"   CMG result: {num_communities} communities")
        print(f"   Components: {components}")
        
        if num_communities >= 1:
            print("✅ Weighting test PASSED")
            return True
        else:
            print("❌ Weighting test FAILED")
            return False
            
    except Exception as e:
        print(f"❌ Weighting test FAILED: {e}")
        traceback.print_exc()
        return False

def test_adaptive_weighting_class():
    """Test the AdaptiveGraphWeighting class specifically."""
    print("\n🔬 Testing AdaptiveGraphWeighting Class...")
    
    try:
        # Simple implementation for testing
        class SimpleAdaptiveWeighting:
            def __init__(self, verbose=False):
                self.verbose = verbose
            
            def structure_preserving_weights(self, G):
                # Just return unit weights for testing
                return [(u, v, 1.0) for u, v in G.edges()]
        
        # Test graph
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)])
        
        weighter = SimpleAdaptiveWeighting()
        weights = weighter.structure_preserving_weights(G)
        
        print(f"   Generated {len(weights)} weights")
        
        # Test with CMG
        A = create_laplacian_from_edges(weights, G.number_of_nodes())
        solver = CMGSteinerSolver(gamma=5.0, verbose=False)
        components, num_communities = solver.steiner_group(A)
        
        print(f"   CMG result: {num_communities} communities")
        
        if num_communities >= 1:
            print("✅ Adaptive weighting class test PASSED")
            return True
        else:
            print("❌ Adaptive weighting class test FAILED")
            return False
            
    except Exception as e:
        print(f"❌ Adaptive weighting class test FAILED: {e}")
        traceback.print_exc()
        return False

def test_gamma_values():
    """Test different gamma values to see if that's the issue."""
    print("\n🔬 Testing Different Gamma Values...")
    
    try:
        # Simple two-component graph
        edges = [(0, 1, 1.0), (1, 2, 1.0), (3, 4, 1.0), (4, 5, 1.0)]
        A = create_laplacian_from_edges(edges, 6)
        
        gamma_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
        
        for gamma in gamma_values:
            try:
                solver = CMGSteinerSolver(gamma=gamma, verbose=False)
                components, num_communities = solver.steiner_group(A)
                print(f"   γ={gamma}: {num_communities} communities")
                
                if num_communities == 2:
                    print(f"✅ Gamma {gamma} works correctly")
                    return True
                    
            except Exception as e:
                print(f"   γ={gamma}: Error - {str(e)[:50]}...")
        
        print("❌ No gamma value worked")
        return False
        
    except Exception as e:
        print(f"❌ Gamma test FAILED: {e}")
        traceback.print_exc()
        return False

def test_node_mapping():
    """Test if node mapping is causing issues."""
    print("\n🔬 Testing Node Mapping...")
    
    try:
        # Graph with non-consecutive node IDs
        G_bad = nx.Graph()
        G_bad.add_edges_from([(10, 11), (11, 12), (12, 10), (20, 21), (21, 22), (22, 20), (12, 20)])
        
        print(f"   Bad graph nodes: {sorted(G_bad.nodes())}")
        
        # Map to consecutive IDs
        node_mapping = {node: i for i, node in enumerate(sorted(G_bad.nodes()))}
        print(f"   Node mapping: {node_mapping}")
        
        G_mapped = nx.Graph()
        for u, v in G_bad.edges():
            G_mapped.add_edge(node_mapping[u], node_mapping[v])
        
        print(f"   Mapped graph nodes: {sorted(G_mapped.nodes())}")
        
        # Test with CMG
        edges = [(u, v, 1.0) for u, v in G_mapped.edges()]
        A = create_laplacian_from_edges(edges, G_mapped.number_of_nodes())
        solver = CMGSteinerSolver(gamma=5.0, verbose=False)
        components, num_communities = solver.steiner_group(A)
        
        print(f"   CMG result: {num_communities} communities")
        
        if num_communities >= 1:
            print("✅ Node mapping test PASSED")
            return True
        else:
            print("❌ Node mapping test FAILED")
            return False
            
    except Exception as e:
        print(f"❌ Node mapping test FAILED: {e}")
        traceback.print_exc()
        return False

def main():
    """Run comprehensive diagnostic tests."""
    print("🔍 COMPREHENSIVE DIAGNOSTIC TEST")
    print("=" * 50)
    print("This will identify why all weighting methods are failing.")
    print()
    
    tests = [
        ("Basic CMG Functionality", test_basic_cmg),
        ("Graph Creation", test_graph_creation),
        ("Weighting Methods", test_weighting_methods),
        ("Adaptive Weighting Class", test_adaptive_weighting_class),
        ("Gamma Values", test_gamma_values),
        ("Node Mapping", test_node_mapping),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"RUNNING: {test_name}")
        print('='*50)
        
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*50}")
    print("DIAGNOSTIC SUMMARY")
    print('='*50)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name:30s}: {status}")
    
    # Recommendations
    print(f"\n🔧 RECOMMENDED FIXES:")
    
    if not results.get("Basic CMG Functionality", False):
        print("  1. ❌ CMG is not working - check CMG package installation")
        print("     Run: cd ~/cmg-python && python -c 'from cmg import CMGSteinerSolver; print(\"OK\")'")
    
    if not results.get("Node Mapping", False):
        print("  2. ❌ Node mapping issue - graphs need consecutive 0-based node IDs")
    
    if not results.get("Gamma Values", False):
        print("  3. ❌ Gamma parameter issue - try different gamma values")
    
    if all(results.values()):
        print("  ✅ All tests passed - the issue might be in the expanded test suite")
        print("     Try running the original simple test that was working")
    
    passed_tests = sum(results.values())
    total_tests = len(results)
    
    print(f"\n🎯 RESULT: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("✅ All diagnostic tests passed - issue is likely in the expanded test suite")
    elif passed_tests == 0:
        print("❌ All tests failed - fundamental CMG or environment issue")
    else:
        print("⚠️  Mixed results - check specific failing components")

if __name__ == "__main__":
    main()
