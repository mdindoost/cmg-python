"""
Statistics and analysis utilities for CMG algorithms.

This module provides classes and functions for analyzing CMG decompositions
and computing various graph-theoretic metrics.
"""

import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Optional, Tuple, Any
import time
from collections import defaultdict


class CMGStatistics:
    """
    Comprehensive statistics collection and analysis for CMG decompositions.
    """
    
    def __init__(self):
        """Initialize statistics collector."""
        self.decomposition_history = []
        self.timing_data = {}
        
    def record_decomposition(self, 
                           solver_stats: Dict,
                           component_indices: np.ndarray,
                           A: sp.spmatrix,
                           additional_info: Optional[Dict] = None) -> None:
        """
        Record statistics from a CMG decomposition.
        
        Args:
            solver_stats: Statistics from CMGSteinerSolver
            component_indices: Component assignment array
            A: Original matrix
            additional_info: Additional metadata
        """
        timestamp = time.time()
        
        # Calculate additional metrics
        detailed_stats = self._calculate_detailed_metrics(component_indices, A)
        
        record = {
            'timestamp': timestamp,
            'solver_stats': solver_stats.copy(),
            'detailed_metrics': detailed_stats,
            'matrix_properties': self._analyze_matrix(A),
            'component_indices': component_indices.copy(),
            'additional_info': additional_info or {}
        }
        
        self.decomposition_history.append(record)
    
    def _calculate_detailed_metrics(self, component_indices: np.ndarray, A: sp.spmatrix) -> Dict:
        """Calculate detailed metrics for the decomposition."""
        n = len(component_indices)
        unique_components = np.unique(component_indices)
        num_components = len(unique_components)
        
        # Component size statistics
        component_sizes = []
        for comp_id in unique_components:
            size = np.sum(component_indices == comp_id)
            component_sizes.append(size)
        
        # Balance metrics
        size_std = np.std(component_sizes)
        size_cv = size_std / np.mean(component_sizes) if np.mean(component_sizes) > 0 else 0
        
        # Modularity calculation
        modularity = self._calculate_modularity(component_indices, A)
        
        # Cut metrics
        cut_metrics = self._calculate_cut_metrics(component_indices, A)
        
        return {
            'num_components': num_components,
            'component_sizes': component_sizes,
            'size_statistics': {
                'mean': np.mean(component_sizes),
                'std': size_std,
                'min': np.min(component_sizes),
                'max': np.max(component_sizes),
                'coefficient_of_variation': size_cv
            },
            'modularity': modularity,
            'cut_metrics': cut_metrics,
            'singleton_components': np.sum(np.array(component_sizes) == 1),
            'largest_component_fraction': np.max(component_sizes) / n
        }
    
    def _analyze_matrix(self, A: sp.spmatrix) -> Dict:
        """Analyze properties of the input matrix."""
        n = A.shape[0]
        nnz = A.nnz
        density = nnz / (n * n)
        
        # Weight statistics
        weights = np.abs(A.data)
        weight_stats = {
            'min': np.min(weights),
            'max': np.max(weights),
            'mean': np.mean(weights),
            'std': np.std(weights),
            'range_ratio': np.max(weights) / np.min(weights) if np.min(weights) > 0 else float('inf')
        }
        
        # Degree statistics (assuming Laplacian)
        degrees = np.array(A.diagonal()).flatten()
        degree_stats = {
            'min': np.min(degrees),
            'max': np.max(degrees),
            'mean': np.mean(degrees),
            'std': np.std(degrees)
        }
        
        return {
            'size': n,
            'nnz': nnz,
            'density': density,
            'weight_statistics': weight_stats,
            'degree_statistics': degree_stats,
            'is_symmetric': self._check_symmetry(A),
            'condition_estimate': self._estimate_condition_number(A)
        }
    
    def _calculate_modularity(self, component_indices: np.ndarray, A: sp.spmatrix) -> float:
        """
        Calculate modularity of the decomposition.
        
        Modularity measures how well the decomposition captures community structure.
        """
        # Convert to adjacency matrix if needed
        if np.all(A.diagonal() >= 0):
            # Looks like Laplacian, convert to adjacency
            A_adj = -A.copy()
            A_adj.setdiag(0)
            A_adj.eliminate_zeros()
            A_adj.data = np.abs(A_adj.data)
        else:
            A_adj = A.copy()
            A_adj.data = np.abs(A_adj.data)
        
        m = A_adj.sum() / 2.0  # Total edge weight
        if m == 0:
            return 0.0
        
        n = A_adj.shape[0]
        degrees = np.array(A_adj.sum(axis=1)).flatten()
        
        modularity = 0.0
        unique_components = np.unique(component_indices)
        
        for comp_id in unique_components:
            nodes_in_comp = np.where(component_indices == comp_id)[0]
            
            # Calculate edges within component
            internal_weight = 0.0
            degree_sum = 0.0
            
            for node in nodes_in_comp:
                degree_sum += degrees[node]
                
                # Count internal edges
                neighbors = A_adj[node].nonzero()[1]
                weights = A_adj[node, neighbors].toarray().flatten()
                
                for neighbor, weight in zip(neighbors, weights):
                    if neighbor in nodes_in_comp and node <= neighbor:  # Avoid double counting
                        internal_weight += weight
            
            # Modularity contribution
            modularity += internal_weight / m - (degree_sum / (2 * m)) ** 2
        
        return modularity
    
    def _calculate_cut_metrics(self, component_indices: np.ndarray, A: sp.spmatrix) -> Dict:
        """Calculate various cut-based metrics."""
        unique_components = np.unique(component_indices)
        
        total_cut_weight = 0.0
        total_internal_weight = 0.0
        max_cut_weight = 0.0
        
        # Convert to adjacency if needed
        if np.all(A.diagonal() >= 0):
            A_adj = -A.copy()
            A_adj.setdiag(0)
            A_adj.eliminate_zeros()
            A_adj.data = np.abs(A_adj.data)
        else:
            A_adj = A.copy()
            A_adj.data = np.abs(A_adj.data)
        
        for comp_id in unique_components:
            nodes_in_comp = set(np.where(component_indices == comp_id)[0])
            
            comp_cut_weight = 0.0
            comp_internal_weight = 0.0
            
            for node in nodes_in_comp:
                neighbors = A_adj[node].nonzero()[1]
                weights = A_adj[node, neighbors].toarray().flatten()
                
                for neighbor, weight in zip(neighbors, weights):
                    if neighbor in nodes_in_comp:
                        if node < neighbor:  # Avoid double counting
                            comp_internal_weight += weight
                    else:
                        comp_cut_weight += weight
            
            total_cut_weight += comp_cut_weight
            total_internal_weight += comp_internal_weight
            max_cut_weight = max(max_cut_weight, comp_cut_weight)
        
        # Normalize by total weight
        total_weight = total_cut_weight + total_internal_weight
        
        return {
            'total_cut_weight': total_cut_weight,
            'total_internal_weight': total_internal_weight,
            'cut_ratio': total_cut_weight / total_weight if total_weight > 0 else 0,
            'max_component_cut_weight': max_cut_weight,
            'average_cut_per_component': total_cut_weight / len(unique_components) if len(unique_components) > 0 else 0
        }
    
    def _check_symmetry(self, A: sp.spmatrix, tolerance: float = 1e-10) -> bool:
        """Check if matrix is symmetric."""
        if A.shape[0] != A.shape[1]:
            return False
        
        # For sparse matrices, check if A - A.T has only zero entries
        diff = A - A.T
        return np.max(np.abs(diff.data)) < tolerance if diff.nnz > 0 else True
    
    def _estimate_condition_number(self, A: sp.spmatrix) -> float:
        """Estimate condition number using power iteration."""
        if A.shape[0] > 1000:
            # For large matrices, skip condition number estimation
            return float('nan')
        
        try:
            # Use eigenvalue bounds for condition number estimation
            eigenvals = sp.linalg.eigsh(A, k=min(6, A.shape[0]-1), which='BE', return_eigenvectors=False)
            pos_eigenvals = eigenvals[eigenvals > 1e-12]
            
            if len(pos_eigenvals) > 1:
                return np.max(pos_eigenvals) / np.min(pos_eigenvals)
            else:
                return float('inf')
        except:
            return float('nan')
    
    def get_summary_statistics(self) -> Dict:
        """Get summary statistics across all recorded decompositions."""
        if not self.decomposition_history:
            return {}
        
        summary = {
            'total_decompositions': len(self.decomposition_history),
            'average_metrics': {},
            'trends': {}
        }
        
        # Collect metrics across all decompositions
        metrics_over_time = defaultdict(list)
        
        for record in self.decomposition_history:
            solver_stats = record['solver_stats']
            detailed_metrics = record['detailed_metrics']
            
            # Core metrics
            metrics_over_time['num_components'].append(solver_stats.get('num_components', 0))
            metrics_over_time['avg_component_size'].append(solver_stats.get('avg_component_size', 0))
            metrics_over_time['avg_conductance'].append(solver_stats.get('avg_conductance', float('inf')))
            metrics_over_time['modularity'].append(detailed_metrics.get('modularity', 0))
            metrics_over_time['cut_ratio'].append(detailed_metrics['cut_metrics'].get('cut_ratio', 0))
            
            # Performance metrics
            if 'decomposition_time' in record['solver_stats']:
                metrics_over_time['decomposition_time'].append(record['solver_stats']['decomposition_time'])
        
        # Calculate averages
        for metric, values in metrics_over_time.items():
            if values:
                finite_values = [v for v in values if np.isfinite(v)]
                if finite_values:
                    summary['average_metrics'][metric] = {
                        'mean': np.mean(finite_values),
                        'std': np.std(finite_values),
                        'min': np.min(finite_values),
                        'max': np.max(finite_values)
                    }
        
        return summary
    
    def compare_decompositions(self, indices1: List[int], indices2: List[int]) -> Dict:
        """
        Compare two sets of decompositions.
        
        Args:
            indices1: Indices of first set of decompositions
            indices2: Indices of second set of decompositions
            
        Returns:
            dict: Comparison results
        """
        if not indices1 or not indices2:
            return {}
        
        def get_metrics(indices):
            metrics = []
            for idx in indices:
                if 0 <= idx < len(self.decomposition_history):
                    record = self.decomposition_history[idx]
                    metrics.append({
                        'num_components': record['solver_stats'].get('num_components', 0),
                        'modularity': record['detailed_metrics'].get('modularity', 0),
                        'avg_conductance': record['solver_stats'].get('avg_conductance', float('inf'))
                    })
            return metrics
        
        metrics1 = get_metrics(indices1)
        metrics2 = get_metrics(indices2)
        
        comparison = {}
        
        for metric in ['num_components', 'modularity', 'avg_conductance']:
            values1 = [m[metric] for m in metrics1 if np.isfinite(m[metric])]
            values2 = [m[metric] for m in metrics2 if np.isfinite(m[metric])]
            
            if values1 and values2:
                comparison[metric] = {
                    'group1_mean': np.mean(values1),
                    'group2_mean': np.mean(values2),
                    'difference': np.mean(values2) - np.mean(values1),
                    'relative_difference': (np.mean(values2) - np.mean(values1)) / np.mean(values1) if np.mean(values1) != 0 else float('inf')
                }
        
        return comparison
    
    def export_statistics(self, filename: str, format: str = 'json') -> None:
        """
        Export statistics to file.
        
        Args:
            filename: Output filename
            format: Export format ('json' or 'csv')
        """
        if format == 'json':
            import json
            
            # Prepare data for JSON serialization
            export_data = {
                'summary': self.get_summary_statistics(),
                'decomposition_history': []
            }
            
            for record in self.decomposition_history:
                # Convert numpy arrays to lists for JSON serialization
                json_record = {}
                for key, value in record.items():
                    if isinstance(value, np.ndarray):
                        json_record[key] = value.tolist()
                    elif isinstance(value, dict):
                        json_record[key] = self._convert_for_json(value)
                    else:
                        json_record[key] = value
                
                export_data['decomposition_history'].append(json_record)
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        
        elif format == 'csv':
            import csv
            
            # Flatten data for CSV export
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write header
                header = ['timestamp', 'num_components', 'avg_component_size', 
                         'avg_conductance', 'modularity', 'cut_ratio', 'matrix_size']
                writer.writerow(header)
                
                # Write data rows
                for record in self.decomposition_history:
                    row = [
                        record['timestamp'],
                        record['solver_stats'].get('num_components', ''),
                        record['solver_stats'].get('avg_component_size', ''),
                        record['solver_stats'].get('avg_conductance', ''),
                        record['detailed_metrics'].get('modularity', ''),
                        record['detailed_metrics']['cut_metrics'].get('cut_ratio', ''),
                        record['matrix_properties'].get('size', '')
                    ]
                    writer.writerow(row)
    
    def _convert_for_json(self, obj):
        """Convert numpy objects to JSON-serializable types."""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj


def compute_decomposition_quality(component_indices: np.ndarray, 
                                A: sp.spmatrix,
                                metrics: List[str] = None) -> Dict[str, float]:
    """
    Compute various quality metrics for a graph decomposition.
    
    Args:
        component_indices: Array of component assignments
        A: Original matrix (Laplacian or adjacency)
        metrics: List of metrics to compute. If None, computes all.
        
    Returns:
        dict: Quality metrics
    """
    if metrics is None:
        metrics = ['modularity', 'conductance', 'cut_ratio', 'balance']
    
    stats_calculator = CMGStatistics()
    detailed_metrics = stats_calculator._calculate_detailed_metrics(component_indices, A)
    
    quality = {}
    
    if 'modularity' in metrics:
        quality['modularity'] = detailed_metrics['modularity']
    
    if 'conductance' in metrics:
        # Average conductance across components
        from ..algorithms.steiner import CMGSteinerSolver
        solver = CMGSteinerSolver(verbose=False)
        
        conductances = []
        unique_components = np.unique(component_indices)
        
        for comp_id in unique_components:
            nodes = np.where(component_indices == comp_id)[0].tolist()
            if len(nodes) > 1:
                cond = solver.conductance(A, nodes)
                if np.isfinite(cond):
                    conductances.append(cond)
        
        quality['avg_conductance'] = np.mean(conductances) if conductances else float('inf')
        quality['max_conductance'] = np.max(conductances) if conductances else float('inf')
    
    if 'cut_ratio' in metrics:
        quality['cut_ratio'] = detailed_metrics['cut_metrics']['cut_ratio']
    
    if 'balance' in metrics:
        # Measure how balanced the component sizes are
        component_sizes = detailed_metrics['component_sizes']
        if len(component_sizes) > 1:
            size_std = np.std(component_sizes)
            size_mean = np.mean(component_sizes)
            quality['balance'] = 1.0 / (1.0 + size_std / size_mean) if size_mean > 0 else 0.0
        else:
            quality['balance'] = 1.0
    
    return quality


def benchmark_algorithm_performance(graphs: Dict, 
                                  solver_configs: List[Dict],
                                  metrics: List[str] = None) -> Dict:
    """
    Benchmark CMG algorithm performance across multiple graphs and configurations.
    
    Args:
        graphs: Dictionary of graphs to test
        solver_configs: List of solver configuration dictionaries
        metrics: List of metrics to compute
        
    Returns:
        dict: Benchmark results
    """
    if metrics is None:
        metrics = ['decomposition_time', 'num_components', 'modularity', 'avg_conductance']
    
    from ..algorithms.steiner import CMGSteinerSolver
    from ..utils.graph_utils import create_laplacian_from_edges
    
    results = {
        'configurations': solver_configs,
        'graphs': list(graphs.keys()),
        'metrics': metrics,
        'results': {}
    }
    
    for graph_name, graph_data in graphs.items():
        results['results'][graph_name] = {}
        
        # Create Laplacian matrix
        A = create_laplacian_from_edges(graph_data['edges'], graph_data['n'])
        
        for i, config in enumerate(solver_configs):
            config_name = f"config_{i}"
            
            # Create solver with configuration
            solver = CMGSteinerSolver(**config)
            
            # Time the decomposition
            start_time = time.time()
            component_indices, num_components = solver.steiner_group(A)
            decomposition_time = time.time() - start_time
            
            # Compute quality metrics
            quality = compute_decomposition_quality(component_indices, A, metrics)
            
            # Store results
            result = {
                'decomposition_time': decomposition_time,
                'num_components': num_components,
                **quality,
                **solver.get_statistics()
            }
            
            results['results'][graph_name][config_name] = result
    
    return results


def analyze_parameter_sensitivity(graph_data: Dict,
                                 parameter_name: str,
                                 parameter_values: List,
                                 base_config: Dict = None) -> Dict:
    """
    Analyze sensitivity of CMG algorithm to parameter changes.
    
    Args:
        graph_data: Graph data dictionary
        parameter_name: Name of parameter to vary
        parameter_values: List of parameter values to test
        base_config: Base configuration for solver
        
    Returns:
        dict: Sensitivity analysis results
    """
    if base_config is None:
        base_config = {'verbose': False}
    
    from ..algorithms.steiner import CMGSteinerSolver
    from ..utils.graph_utils import create_laplacian_from_edges
    
    A = create_laplacian_from_edges(graph_data['edges'], graph_data['n'])
    
    results = {
        'parameter_name': parameter_name,
        'parameter_values': parameter_values,
        'results': []
    }
    
    for param_value in parameter_values:
        config = base_config.copy()
        config[parameter_name] = param_value
        
        try:
            solver = CMGSteinerSolver(**config)
            component_indices, num_components = solver.steiner_group(A)
            
            quality = compute_decomposition_quality(component_indices, A)
            
            result = {
                'parameter_value': param_value,
                'num_components': num_components,
                'success': True,
                **quality,
                **solver.get_statistics()
            }
        except Exception as e:
            result = {
                'parameter_value': param_value,
                'success': False,
                'error': str(e)
            }
        
        results['results'].append(result)
    
    return results
