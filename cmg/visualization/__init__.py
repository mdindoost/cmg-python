"""
CMG visualization package.
"""
try:
    from .plotting import (
        plot_graph_decomposition,
        plot_conductance_analysis,
        plot_parameter_sensitivity,
        create_adjacency_matrix_plot
    )
    __all__ = [
        'plot_graph_decomposition',
        'plot_conductance_analysis', 
        'plot_parameter_sensitivity',
        'create_adjacency_matrix_plot'
    ]
except ImportError:
    # Visualization dependencies not available
    __all__ = []
