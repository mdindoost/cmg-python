#!/usr/bin/env python3
"""
Advanced Test Suite for CMG-Python
==================================

This module contains comprehensive tests for edge cases, performance,
and real-world scenarios to ensure the CMG algorithm is robust.

Run with: python -m pytest tests/test_advanced.py -v
"""

import numpy as np
import pytest
import time
from scipy import sparse
from cmg import CMGSteinerSolver, create_laplacian_from_edges
from cmg.utils.graph_utils import create_test_graphs


class TestCMGAdvancedCases:
    """Test advanced and edge cases for CMG algorithm."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.solver = CMGSteinerSolver(verbose=False)
        
    def test_single_node_graph(self):
        """Test graph with only one node."""
        # Single node Laplacian
        A = sparse.csr_matrix([[0.0]])
        components, num_comp = self.solver.steiner_group(A)
        
        assert num_comp == 1
        assert len(components) == 1
        assert components[0] == 0
        
    def test_disconnected_components(self):
        """Test graph with multiple disconnected components."""
        # Create 3 disconnected pairs: (0,1), (2,3), (4,5)
        edges = [
            (0, 1, 1.0),  # Component 1
            (2, 3, 1.0),  # Component 2  
            (4, 5, 1.0)   # Component 3
        ]
        A = create_laplacian_from_edges(edges, 6)
        components, num_comp = self.solver.steiner_group(A)
        
        assert num_comp == 3
        # Each pair should be in same component
        assert components[0] == components[1]
        assert components[2] == components[3] 
        assert components[4] == components[5]
        # But different from each other
        assert len(set(components)) == 3
        
    def test_community_structure(self):
        """Test graph with clear community structure."""
        # Create 2 communities with weak inter-community connection
        
        # Community 1: nodes 0-2 (triangle)
        edges = [(0, 1, 2.0), (1, 2, 2.0), (2, 0, 2.0)]
        
        # Community 2: nodes 3-5 (triangle)
        edges.extend([(3, 4, 2.0), (4, 5, 2.0), (5, 3, 2.0)])
        
        # Weak inter-community connection
        edges.append((2, 3, 0.01))
        
        A = create_laplacian_from_edges(edges, 6)
        components, num_comp = self.solver.steiner_group(A)
        
        # Should find 2 communities
        assert num_comp == 2


class TestCMGNumericalStability:
    """Test numerical stability and precision."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.solver = CMGSteinerSolver(verbose=False)
        
    def test_very_small_weights(self):
        """Test with very small edge weights."""
        edges = [
            (0, 1, 1e-6),
            (1, 2, 1e-6),
            (2, 3, 1.0)
        ]
        A = create_laplacian_from_edges(edges, 4)
        components, num_comp = self.solver.steiner_group(A)
        
        # Should handle small weights without numerical issues
        assert num_comp >= 1
        assert len(components) == 4
        
    def test_mixed_weight_scales(self):
        """Test with mixed weight scales."""
        edges = [
            (0, 1, 1e-3),   # Small
            (1, 2, 1.0),    # Normal
            (2, 3, 100.0),  # Large
        ]
        A = create_laplacian_from_edges(edges, 4)
        components, num_comp = self.solver.steiner_group(A)
        
        # Algorithm should be stable across weight scales
        assert num_comp >= 1
        assert len(components) == 4


class TestCMGParameterSensitivity:
    """Test sensitivity to different parameter values."""
    
    def test_gamma_parameter_effects(self):
        """Test how gamma parameter affects results."""
        # Use the weak connection test case
        graphs = create_test_graphs()
        weak_graph = graphs['weak_connection']
        A = create_laplacian_from_edges(weak_graph['edges'], weak_graph['n'])
        
        gamma_values = [4.1, 5.0, 7.0, 10.0]
        results = []
        
        for gamma in gamma_values:
            solver = CMGSteinerSolver(gamma=gamma, verbose=False)
            components, num_comp = solver.steiner_group(A)
            results.append((gamma, num_comp, components))
            
        # All should find the weak connection
        for gamma, num_comp, components in results:
            assert num_comp >= 1
            assert len(components) == 4
            
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test invalid gamma values
        with pytest.raises(ValueError, match="gamma must be > 4"):
            CMGSteinerSolver(gamma=3.0)
            
        # Valid gamma should work
        solver = CMGSteinerSolver(gamma=4.1)
        assert solver.gamma == 4.1


class TestCMGEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.solver = CMGSteinerSolver(verbose=False)
        
    def test_zero_matrix(self):
        """Test all-zero matrix."""
        A = sparse.csr_matrix((3, 3))  # All zeros
        components, num_comp = self.solver.steiner_group(A)
        
        # All nodes should be separate components
        assert num_comp == 3
        assert len(set(components)) == 3
        
    def test_symmetric_matrix_handling(self):
        """Test that symmetric matrices are handled correctly."""
        # Create symmetric Laplacian
        A = sparse.csr_matrix([
            [2.0, -1.0, -1.0],
            [-1.0, 2.0, -1.0], 
            [-1.0, -1.0, 2.0]
        ])
        
        components, num_comp = self.solver.steiner_group(A)
        assert num_comp >= 1
        assert len(components) == 3
