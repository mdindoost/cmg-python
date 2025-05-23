#!/usr/bin/env python3
"""
Benchmark Tests for CMG-Python
==============================

Performance benchmarks and scaling tests.
Run with: python -m pytest tests/test_benchmarks.py -v -s
"""

import numpy as np
import time
import pytest
from cmg import CMGSteinerSolver, create_laplacian_from_edges
from cmg.utils.graph_utils import create_test_graphs


class TestCMGBenchmarks:
    """Benchmark tests for CMG performance."""
    
    @pytest.mark.parametrize("n", [10, 20, 50])
    def test_scaling_with_size(self, n):
        """Test how performance scales with graph size."""
        # Create random connected graph
        np.random.seed(42)
        edges = []
        
        # Ensure connectivity
        for i in range(n-1):
            edges.append((i, i+1, 1.0))
            
        # Add random edges (sparse graph)
        for _ in range(n // 2):
            i, j = np.random.choice(n, 2, replace=False)
            if i != j:
                edges.append((i, j, np.random.uniform(0.1, 2.0)))
                
        A = create_laplacian_from_edges(edges, n)
        solver = CMGSteinerSolver(verbose=False)
        
        # Time the algorithm
        start_time = time.time()
        components, num_comp = solver.steiner_group(A)
        elapsed = time.time() - start_time
        
        print(f"\nGraph size {n:3d}: {elapsed*1000:6.1f}ms, {num_comp} components")
        
        # Performance should be reasonable
        if n <= 20:
            assert elapsed < 0.1  # < 100ms for small graphs
        elif n <= 50:
            assert elapsed < 0.5  # < 500ms for medium graphs  
        else:
            assert elapsed < 2.0  # < 2s for larger graphs
            
        assert num_comp >= 1
        assert len(components) == n
        
    def test_weak_vs_strong_connections(self):
        """Benchmark weak vs strong connection detection."""
        solver = CMGSteinerSolver(verbose=False)
        
        # Test multiple weak connection scenarios
        weak_weights = [0.001, 0.01, 0.1]
        times = []
        
        for weak_weight in weak_weights:
            edges = [
                (0, 1, 1.0),
                (1, 2, weak_weight),  # Weak connection
                (2, 3, 1.0)
            ]
            A = create_laplacian_from_edges(edges, 4)
            
            start_time = time.time()
            components, num_comp = solver.steiner_group(A)
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            print(f"Weak weight {weak_weight:5.3f}: {elapsed*1000:5.1f}ms, {num_comp} components")
            
        # All should be fast
        for t in times:
            assert t < 0.1  # All under 100ms
            
    def test_different_graph_structures(self):
        """Benchmark different graph topologies."""
        solver = CMGSteinerSolver(verbose=False)
        n = 20
        
        structures = {
            'path': [(i, i+1, 1.0) for i in range(n-1)],
            'star': [(0, i, 1.0) for i in range(1, n)],
            'cycle': [(i, (i+1) % n, 1.0) for i in range(n)],
        }
        
        print(f"\nBenchmarking graph structures (n={n}):")
        for name, edges in structures.items():
            A = create_laplacian_from_edges(edges, n)
            
            start_time = time.time()
            components, num_comp = solver.steiner_group(A)
            elapsed = time.time() - start_time
            
            print(f"{name:10s}: {elapsed*1000:6.1f}ms, {num_comp:2d} components, {len(edges):3d} edges")
            
            # All should complete quickly
            assert elapsed < 0.5
            assert num_comp >= 1
            
    def test_gamma_parameter_performance(self):
        """Test performance with different gamma values."""
        graphs = create_test_graphs()
        A = create_laplacian_from_edges(graphs['two_triangles']['edges'], 
                                       graphs['two_triangles']['n'])
        
        gamma_values = [4.1, 5.0, 7.0, 10.0, 15.0]
        
        print(f"\nGamma parameter performance:")
        for gamma in gamma_values:
            solver = CMGSteinerSolver(gamma=gamma, verbose=False)
            
            start_time = time.time()
            components, num_comp = solver.steiner_group(A)
            elapsed = time.time() - start_time
            
            print(f"γ={gamma:4.1f}: {elapsed*1000:5.1f}ms, {num_comp} components")
            
            assert elapsed < 0.1  # Should all be fast
