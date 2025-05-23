#!/usr/bin/env python3
"""
Command-line interface for CMG-Python.

This module provides a command-line interface for running CMG decompositions
and demonstrations.
"""

import argparse
import sys
import json
import time
from pathlib import Path

from .algorithms.steiner import CMGSteinerSolver
from .utils.graph_utils import create_laplacian_from_edges, create_test_graphs
from .utils.statistics import compute_decomposition_quality


def run_demo():
    """Run a demonstration of the CMG algorithm."""
    print("CMG-Python Demonstration")
    print("=" * 50)
    
    # Get test graphs
    test_graphs = create_test_graphs()
    solver = CMGSteinerSolver(gamma=5.0, verbose=True)
    
    # Test weak connection graph
    print("Testing weak connection graph:")
    graph_data = test_graphs['weak_connection']
    A = create_laplacian_from_edges(graph_data['edges'], graph_data['n'])
    
    component_indices, num_components = solver.steiner_group(A)
    solver.visualize_components(component_indices)
    
    stats = solver.get_statistics()
    print(f"Computation time: {solver.last_decomposition_time:.4f} seconds")
    print(f"Average conductance: {stats['avg_conductance']:.6f}")


def run_decomposition(edges_file: str, output_file: str = None, 
                     gamma: float = 5.0, format: str = 'json'):
    """
    Run CMG decomposition on a graph from file.
    
    Args:
        edges_file: Path to file containing edge list
        output_file: Path to output file (optional)
        gamma: Gamma parameter for CMG
        format: Output format ('json' or 'txt')
    """
    print(f"Running CMG decomposition on {edges_file}")
    
    # Read edges from file
    edges = []
    max_node = -1
    
    try:
        with open(edges_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) < 2:
                    print(f"Warning: Invalid line {line_num}: {line}")
                    continue
                
                try:
                    u = int(parts[0])
                    v = int(parts[1])
                    weight = float(parts[2]) if len(parts) > 2 else 1.0
                    
                    edges.append((u, v, weight))
                    max_node = max(max_node, u, v)
                    
                except ValueError:
                    print(f"Warning: Invalid line {line_num}: {line}")
                    continue
        
        n = max_node + 1
        print(f"Read {len(edges)} edges, {n} nodes")
        
    except FileNotFoundError:
        print(f"Error: File {edges_file} not found")
        return False
    except Exception as e:
        print(f"Error reading file: {e}")
        return False
    
    # Create Laplacian matrix
    A = create_laplacian_from_edges(edges, n)
    
    # Run decomposition
    solver = CMGSteinerSolver(gamma=gamma, verbose=True)
    start_time = time.time()
    component_indices, num_components = solver.steiner_group(A)
    total_time = time.time() - start_time
    
    # Get quality metrics
    quality = compute_decomposition_quality(component_indices, A)
    stats = solver.get_statistics()
    
    # Prepare results
    results = {
        'input_file': edges_file,
        'graph_info': {
            'num_nodes': n,
            'num_edges': len(edges),
            'density': len(edges) / (n * (n - 1) / 2) if n > 1 else 0
        },
        'parameters': {
            'gamma': gamma
        },
        'results': {
            'num_components': num_components,
            'component_assignment': component_indices.tolist(),
            'computation_time': total_time
        },
        'quality_metrics': quality,
        'algorithm_statistics': stats
    }
    
    # Output results
    if output_file:
        if format == 'json':
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to {output_file}")
        else:
            with open(output_file, 'w') as f:
                f.write(f"CMG Decomposition Results\n")
                f.write(f"========================\n\n")
                f.write(f"Input: {edges_file}\n")
                f.write(f"Nodes: {n}, Edges: {len(edges)}\n")
                f.write(f"Gamma: {gamma}\n\n")
                f.write(f"Results:\n")
                f.write(f"  Components: {num_components}\n")
                f.write(f"  Time: {total_time:.4f} seconds\n")
                f.write(f"  Modularity: {quality.get('modularity', 'N/A')}\n")
                f.write(f"  Avg Conductance: {quality.get('avg_conductance', 'N/A')}\n\n")
                f.write(f"Component Assignment:\n")
                for i, comp in enumerate(component_indices):
                    f.write(f"  Node {i}: Component {comp}\n")
            print(f"Results saved to {output_file}")
    else:
        # Print to console
        print(f"\nResults:")
        print(f"  Components: {num_components}")
        print(f"  Time: {total_time:.4f} seconds")
        print(f"  Modularity: {quality.get('modularity', 'N/A')}")
        if 'avg_conductance' in quality:
            print(f"  Avg Conductance: {quality['avg_conductance']:.6f}")
    
    return True


def benchmark_performance(max_size: int = 1000):
    """Run performance benchmark."""
    print("CMG Performance Benchmark")
    print("=" * 30)
    
    from .utils.graph_utils import create_clustered_graph
    
    sizes = [50, 100, 200, 500]
    if max_size < 500:
        sizes = [s for s in sizes if s <= max_size]
    
    solver = CMGSteinerSolver(gamma=5.0, verbose=False)
    
    print(f"{'Size':<8} {'Edges':<8} {'Components':<12} {'Time (ms)':<12} {'Memory (MB)':<12}")
    print("-" * 60)
    
    for size in sizes:
        # Create clustered graph
        cluster_sizes = [size // 4] * 4
        edges, n = create_clustered_graph(cluster_sizes, seed=42)
        A = create_laplacian_from_edges(edges, n)
        
        # Measure performance
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        component_indices, num_components = solver.steiner_group(A)
        elapsed_time = (time.time() - start_time) * 1000  # ms
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        print(f"{size:<8} {len(edges):<8} {num_components:<12} {elapsed_time:<12.2f} {memory_used:<12.1f}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CMG-Python: Combinatorial Multigrid for Python",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cmg-demo                          # Run demonstration
  cmg-demo -f edges.txt             # Decompose graph from file
  cmg-demo -f edges.txt -o out.json # Save results to file
  cmg-demo --benchmark              # Run performance benchmark
  
Edge file format:
  Each line: node1 node2 [weight]
  Lines starting with # are ignored
  Example:
    0 1 1.0
    1 2 0.5
    2 0 2.0
        """
    )
    
    parser.add_argument('--demo', action='store_true',
                       help='Run CMG demonstration')
    
    parser.add_argument('-f', '--file', type=str,
                       help='Input file containing edge list')
    
    parser.add_argument('-o', '--output', type=str,
                       help='Output file for results')
    
    parser.add_argument('--format', choices=['json', 'txt'], default='json',
                       help='Output format (default: json)')
    
    parser.add_argument('--gamma', type=float, default=5.0,
                       help='Gamma parameter (default: 5.0)')
    
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark')
    
    parser.add_argument('--benchmark-size', type=int, default=1000,
                       help='Maximum size for benchmark (default: 1000)')
    
    parser.add_argument('--version', action='version', version='CMG-Python 0.1.0')
    
    args = parser.parse_args()
    
    # Handle different modes
    if args.benchmark:
        benchmark_performance(args.benchmark_size)
    elif args.file:
        success = run_decomposition(args.file, args.output, args.gamma, args.format)
        if not success:
            sys.exit(1)
    elif args.demo or len(sys.argv) == 1:
        run_demo()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
