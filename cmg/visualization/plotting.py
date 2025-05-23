"""
Visualization utilities for CMG algorithms.

This module provides plotting functions for visualizing graphs, decompositions,
and analysis results.
"""

import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Optional, Tuple, Any
import warnings

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import ListedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("Matplotlib not available. Visualization functions will not work.")

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    warnings.warn("NetworkX not available. Some visualization functions will not work.")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def _check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib is required for plotting. Install with: pip install matplotlib")


def _check_networkx():
    """Check if networkx is available."""
    if not HAS_NETWORKX:
        raise ImportError("NetworkX is required for graph visualization. Install with: pip install networkx")


def plot_graph_decomposition(A: sp.spmatrix, 
                           component_indices: np.ndarray,
                           title: str = "CMG Graph Decomposition",
                           figsize: Tuple[int, int] = (12, 8),
                           node_size: int = 300,
                           edge_width: float = 1.0,
                           layout: str = 'spring',
                           save_path: Optional[str] = None) -> None:
    """
    Plot a graph with its CMG decomposition.
    
    Args:
        A: Sparse matrix (Laplacian or adjacency)
        component_indices: Array of component assignments
        title: Plot title
        figsize: Figure size (width, height)
        node_size: Size of nodes in the plot
        edge_width: Width of edges
        layout: NetworkX layout algorithm ('spring', 'circular', 'kamada_kawai', etc.)
        save_path: Path to save the figure (optional)
    """
    _check_matplotlib()
    _check_networkx()
    
    # Convert to adjacency matrix if needed
    if np.all(A.diagonal() >= 0):
        A_adj = -A.copy()
        A_adj.setdiag(0)
        A_adj.eliminate_zeros()
        A_adj.data = np.abs(A_adj.data)
    else:
        A_adj = A.copy()
        A_adj.data = np.abs(A_adj.data)
    
    # Create NetworkX graph
    G = nx.from_scipy_sparse_array(A_adj)
    
    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Choose layout
    if layout == 'spring':
        pos = nx.spring_layout(G, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Plot 1: Original graph
    ax1.set_title("Original Graph")
    nx.draw(G, pos, ax=ax1, 
           node_color='lightblue',
           node_size=node_size,
           edge_color='gray',
           width=edge_width,
           with_labels=True,
           font_size=8)
    
    # Plot 2: Decomposition
    ax2.set_title("CMG Decomposition")
    
    # Create color map for components
    unique_components = np.unique(component_indices)
    num_components = len(unique_components)
    
    if HAS_SEABORN:
        colors = sns.color_palette("husl", num_components)
    else:
        # Fallback color scheme
        colors = plt.cm.Set3(np.linspace(0, 1, num_components))
    
    # Color nodes by component
    node_colors = [colors[component_indices[node]] for node in G.nodes()]
    
    nx.draw(G, pos, ax=ax2,
           node_color=node_colors,
           node_size=node_size,
           edge_color='gray',
           width=edge_width,
           with_labels=True,
           font_size=8)
    
    # Add legend for components
    legend_elements = []
    for i, comp_id in enumerate(unique_components):
        count = np.sum(component_indices == comp_id)
        legend_elements.append(
            patches.Patch(color=colors[i], label=f'Component {comp_id} ({count} nodes)')
        )
    
    ax2.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_conductance_analysis(A: sp.spmatrix,
                            component_indices: np.ndarray,
                            title: str = "Conductance Analysis",
                            figsize: Tuple[int, int] = (12, 6),
                            save_path: Optional[str] = None) -> None:
    """
    Plot conductance analysis for each component.
    
    Args:
        A: Sparse matrix
        component_indices: Component assignments
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
    """
    _check_matplotlib()
    
    from ..algorithms.steiner import CMGSteinerSolver
    solver = CMGSteinerSolver(verbose=False)
    
    unique_components = np.unique(component_indices)
    conductances = []
    component_sizes = []
    
    for comp_id in unique_components:
        nodes = np.where(component_indices == comp_id)[0].tolist()
        size = len(nodes)
        component_sizes.append(size)
        
        if size > 1:
            cond = solver.conductance(A, nodes)
            conductances.append(cond if np.isfinite(cond) else 0)
        else:
            conductances.append(0)  # Singleton has conductance 0
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Conductance by component
    ax1.bar(unique_components, conductances, alpha=0.7, color='skyblue')
    ax1.set_xlabel('Component ID')
    ax1.set_ylabel('Conductance')
    ax1.set_title('Conductance by Component')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (comp_id, cond) in enumerate(zip(unique_components, conductances)):
        ax1.text(comp_id, cond + 0.01 * max(conductances), f'{cond:.3f}', 
                ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Component size vs conductance
    ax2.scatter(component_sizes, conductances, alpha=0.7, s=60, color='coral')
    ax2.set_xlabel('Component Size')
    ax2.set_ylabel('Conductance')
    ax2.set_title('Component Size vs Conductance')
    ax2.grid(True, alpha=0.3)
    
    # Annotate points with component IDs
    for comp_id, size, cond in zip(unique_components, component_sizes, conductances):
        ax2.annotate(f'{comp_id}', (size, cond), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_parameter_sensitivity(sensitivity_results: Dict,
                             metric: str = 'num_components',
                             title: Optional[str] = None,
                             figsize: Tuple[int, int] = (10, 6),
                             save_path: Optional[str] = None) -> None:
    """
    Plot parameter sensitivity analysis results.
    
    Args:
        sensitivity_results: Results from analyze_parameter_sensitivity
        metric: Metric to plot
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
    """
    _check_matplotlib()
    
    param_name = sensitivity_results['parameter_name']
    param_values = sensitivity_results['parameter_values']
    results = sensitivity_results['results']
    
    # Extract metric values
    metric_values = []
    successful_params = []
    
    for param_val, result in zip(param_values, results):
        if result.get('success', False) and metric in result:
            metric_values.append(result[metric])
            successful_params.append(param_val)
    
    if not metric_values:
        print(f"No successful results found for metric '{metric}'")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(successful_params, metric_values, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel(param_name.replace('_', ' ').title())
    ax.set_ylabel(metric.replace('_', ' ').title())
    
    if title is None:
        title = f"{metric.replace('_', ' ').title()} vs {param_name.replace('_', ' ').title()}"
    
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for x, y in zip(successful_params, metric_values):
        ax.annotate(f'{y:.3f}', (x, y), xytext=(0, 10), 
                   textcoords='offset points', ha='center', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_benchmark_results(benchmark_results: Dict,
                          metric: str = 'decomposition_time',
                          title: Optional[str] = None,
                          figsize: Tuple[int, int] = (12, 8),
                          save_path: Optional[str] = None) -> None:
    """
    Plot benchmark results comparing different configurations and graphs.
    
    Args:
        benchmark_results: Results from benchmark_algorithm_performance
        metric: Metric to visualize
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
    """
    _check_matplotlib()
    
    graphs = benchmark_results['graphs']
    configs = benchmark_results['configurations']
    results = benchmark_results['results']
    
    # Prepare data for plotting
    data_matrix = np.zeros((len(graphs), len(configs)))
    
    for i, graph_name in enumerate(graphs):
        for j, config in enumerate(configs):
            config_name = f"config_{j}"
            if config_name in results[graph_name]:
                value = results[graph_name][config_name].get(metric, np.nan)
                data_matrix[i, j] = value if np.isfinite(value) else np.nan
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(data_matrix, cmap='viridis', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(configs)))
    ax.set_yticks(range(len(graphs)))
    ax.set_xticklabels([f"Config {i}" for i in range(len(configs))])
    ax.set_yticklabels(graphs)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric.replace('_', ' ').title())
    
    # Add text annotations
    for i in range(len(graphs)):
        for j in range(len(configs)):
            value = data_matrix[i, j]
            if not np.isnan(value):
                text = ax.text(j, i, f'{value:.3f}', ha="center", va="center", 
                             color="white" if value > np.nanmean(data_matrix) else "black",
                             fontsize=8)
    
    if title is None:
        title = f"Benchmark Results: {metric.replace('_', ' ').title()}"
    
    ax.set_title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_decomposition_statistics(statistics: Dict,
                                title: str = "Decomposition Statistics",
                                figsize: Tuple[int, int] = (15, 10),
                                save_path: Optional[str] = None) -> None:
    """
    Plot comprehensive statistics from CMG decompositions.
    
    Args:
        statistics: Statistics dictionary from CMGStatistics
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
    """
    _check_matplotlib()
    
    if not statistics:
        print("No statistics available to plot")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    avg_metrics = statistics.get('average_metrics', {})
    
    # Plot 1: Number of components
    if 'num_components' in avg_metrics:
        metric = avg_metrics['num_components']
        axes[0].bar(['Mean', 'Min', 'Max'], 
                   [metric['mean'], metric['min'], metric['max']], 
                   color=['blue', 'green', 'red'], alpha=0.7)
        axes[0].set_title('Number of Components')
        axes[0].set_ylabel('Count')
    
    # Plot 2: Average component size
    if 'avg_component_size' in avg_metrics:
        metric = avg_metrics['avg_component_size']
        axes[1].bar(['Mean', 'Min', 'Max'], 
                   [metric['mean'], metric['min'], metric['max']], 
                   color=['blue', 'green', 'red'], alpha=0.7)
        axes[1].set_title('Average Component Size')
        axes[1].set_ylabel('Size')
    
    # Plot 3: Average conductance
    if 'avg_conductance' in avg_metrics:
        metric = avg_metrics['avg_conductance']
        finite_values = [v for v in [metric['mean'], metric['min'], metric['max']] 
                        if np.isfinite(v)]
        if finite_values:
            axes[2].bar(['Mean', 'Min', 'Max'], 
                       [metric['mean'], metric['min'], metric['max']], 
                       color=['blue', 'green', 'red'], alpha=0.7)
        axes[2].set_title('Average Conductance')
        axes[2].set_ylabel('Conductance')
    
    # Plot 4: Modularity
    if 'modularity' in avg_metrics:
        metric = avg_metrics['modularity']
        axes[3].bar(['Mean', 'Min', 'Max'], 
                   [metric['mean'], metric['min'], metric['max']], 
                   color=['blue', 'green', 'red'], alpha=0.7)
        axes[3].set_title('Modularity')
        axes[3].set_ylabel('Modularity')
    
    # Plot 5: Cut ratio
    if 'cut_ratio' in avg_metrics:
        metric = avg_metrics['cut_ratio']
        axes[4].bar(['Mean', 'Min', 'Max'], 
                   [metric['mean'], metric['min'], metric['max']], 
                   color=['blue', 'green', 'red'], alpha=0.7)
        axes[4].set_title('Cut Ratio')
        axes[4].set_ylabel('Ratio')
    
    # Plot 6: Decomposition time
    if 'decomposition_time' in avg_metrics:
        metric = avg_metrics['decomposition_time']
        axes[5].bar(['Mean', 'Min', 'Max'], 
                   [metric['mean'], metric['min'], metric['max']], 
                   color=['blue', 'green', 'red'], alpha=0.7)
        axes[5].set_title('Decomposition Time')
        axes[5].set_ylabel('Time (seconds)')
    
    # Remove empty subplots
    for i in range(len(avg_metrics), len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_adjacency_matrix_plot(A: sp.spmatrix,
                               component_indices: Optional[np.ndarray] = None,
                               title: str = "Adjacency Matrix",
                               figsize: Tuple[int, int] = (10, 8),
                               save_path: Optional[str] = None) -> None:
    """
    Plot the adjacency matrix with optional component highlighting.
    
    Args:
        A: Sparse matrix
        component_indices: Optional component assignments
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
    """
    _check_matplotlib()
    
    # Convert to adjacency if needed
    if np.all(A.diagonal() >= 0):
        A_adj = -A.copy()
        A_adj.setdiag(0)
        A_adj.eliminate_zeros()
        A_adj.data = np.abs(A_adj.data)
    else:
        A_adj = A.copy()
        A_adj.data = np.abs(A_adj.data)
    
    # Convert to dense for visualization
    dense_matrix = A_adj.toarray()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot matrix
    im = ax.imshow(dense_matrix, cmap='Blues', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Edge Weight')
    
    # If components are provided, add grid lines to separate them
    if component_indices is not None:
        unique_components = np.unique(component_indices)
        boundaries = []
        
        for comp_id in unique_components[:-1]:  # Don't need boundary after last component
            # Find the last node in this component
            last_node = np.where(component_indices == comp_id)[0][-1]
            boundaries.append(last_node + 0.5)
        
        # Add vertical and horizontal lines
        for boundary in boundaries:
            ax.axhline(y=boundary, color='red', linewidth=2, alpha=0.7)
            ax.axvline(x=boundary, color='red', linewidth=2, alpha=0.7)
    
    ax.set_title(title)
    ax.set_xlabel('Node Index')
    ax.set_ylabel('Node Index')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_weight_distribution(A: sp.spmatrix,
                           title: str = "Edge Weight Distribution",
                           figsize: Tuple[int, int] = (10, 6),
                           bins: int = 50,
                           save_path: Optional[str] = None) -> None:
    """
    Plot the distribution of edge weights in the graph.
    
    Args:
        A: Sparse matrix
        title: Plot title
        figsize: Figure size
        bins: Number of histogram bins
        save_path: Path to save the figure
    """
    _check_matplotlib()
    
    # Extract edge weights (excluding diagonal)
    if np.all(A.diagonal() >= 0):
        # Laplacian matrix - convert to adjacency
        A_adj = -A.copy()
        A_adj.setdiag(0)
        A_adj.eliminate_zeros()
        weights = np.abs(A_adj.data)
    else:
        # Already adjacency matrix
        weights = np.abs(A.data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Histogram
    ax1.hist(weights, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Edge Weight')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Edge Weight Histogram')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log-scale histogram (if weights span multiple orders of magnitude)
    weight_range = np.max(weights) / np.min(weights) if np.min(weights) > 0 else 1
    if weight_range > 100:  # Large range, use log scale
        ax2.hist(weights, bins=bins, alpha=0.7, color='coral', edgecolor='black')
        ax2.set_xlabel('Edge Weight')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Edge Weight Histogram (Log Scale)')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
    else:
        # Box plot instead
        ax2.boxplot(weights, vert=True)
        ax2.set_ylabel('Edge Weight')
        ax2.set_title('Edge Weight Box Plot')
        ax2.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"""Statistics:
Min: {np.min(weights):.4f}
Max: {np.max(weights):.4f}
Mean: {np.mean(weights):.4f}
Std: {np.std(weights):.4f}
Range: {weight_range:.2f}x"""
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=9)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_comparison_plot(results_list: List[Dict],
                         labels: List[str],
                         metric: str = 'num_components',
                         title: Optional[str] = None,
                         figsize: Tuple[int, int] = (12, 8),
                         save_path: Optional[str] = None) -> None:
    """
    Create a comparison plot for multiple CMG results.
    
    Args:
        results_list: List of result dictionaries
        labels: Labels for each result set
        metric: Metric to compare
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
    """
    _check_matplotlib()
    
    if len(results_list) != len(labels):
        raise ValueError("Number of results and labels must match")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract metric values for each result set
    x_pos = np.arange(len(labels))
    values = []
    errors = []
    
    for results in results_list:
        if isinstance(results, dict) and 'average_metrics' in results:
            # Statistics format
            if metric in results['average_metrics']:
                metric_data = results['average_metrics'][metric]
                values.append(metric_data['mean'])
                errors.append(metric_data['std'])
            else:
                values.append(0)
                errors.append(0)
        elif isinstance(results, dict) and metric in results:
            # Direct result format
            values.append(results[metric])
            errors.append(0)
        else:
            values.append(0)
            errors.append(0)
    
    # Create bar plot with error bars
    bars = ax.bar(x_pos, values, yerr=errors, capsize=5, alpha=0.7, 
                  color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(labels)])
    
    # Customize plot
    ax.set_xlabel('Configuration')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    if title is None:
        title = f"Comparison: {metric.replace('_', ' ').title()}"
    ax.set_title(title)
    
    # Add value labels on bars
    for i, (bar, value, error) in enumerate(zip(bars, values, errors)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + error + 0.01*max(values),
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def save_all_plots(A: sp.spmatrix,
                  component_indices: np.ndarray,
                  solver_stats: Dict,
                  output_dir: str = "cmg_plots",
                  prefix: str = "cmg") -> None:
    """
    Generate and save all standard plots for a CMG decomposition.
    
    Args:
        A: Sparse matrix
        component_indices: Component assignments
        solver_stats: Statistics from solver
        output_dir: Directory to save plots
        prefix: Prefix for plot filenames
    """
    import os
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Graph decomposition plot
        plot_graph_decomposition(
            A, component_indices,
            title=f"{prefix.title()} Graph Decomposition",
            save_path=os.path.join(output_dir, f"{prefix}_decomposition.png")
        )
        
        # Conductance analysis
        plot_conductance_analysis(
            A, component_indices,
            title=f"{prefix.title()} Conductance Analysis",
            save_path=os.path.join(output_dir, f"{prefix}_conductance.png")
        )
        
        # Adjacency matrix
        create_adjacency_matrix_plot(
            A, component_indices,
            title=f"{prefix.title()} Adjacency Matrix",
            save_path=os.path.join(output_dir, f"{prefix}_adjacency.png")
        )
        
        # Weight distribution
        plot_weight_distribution(
            A,
            title=f"{prefix.title()} Weight Distribution",
            save_path=os.path.join(output_dir, f"{prefix}_weights.png")
        )
        
        print(f"All plots saved to {output_dir}/")
        
    except Exception as e:
        print(f"Error saving plots: {e}")


# Utility function for creating custom colormaps
def create_component_colormap(num_components: int) -> List:
    """
    Create a distinct colormap for visualizing components.
    
    Args:
        num_components: Number of components
        
    Returns:
        List of colors
    """
    if HAS_SEABORN:
        return sns.color_palette("husl", num_components)
    elif HAS_MATPLOTLIB:
        if num_components <= 10:
            return plt.cm.tab10(np.linspace(0, 1, num_components))
        else:
            return plt.cm.viridis(np.linspace(0, 1, num_components))
    else:
        # Fallback: generate simple colors
        colors = []
        for i in range(num_components):
            hue = i / num_components
            colors.append((hue, 0.7, 0.9))  # HSV format
        return colors
