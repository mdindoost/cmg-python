"""
CMG-Python: Combinatorial Multigrid for Python

A Python implementation of the Combinatorial Multigrid (CMG) Steiner Group algorithm
from "Combinatorial preconditioners and multilevel solvers for problems in computer 
vision and image processing" by Koutis, Miller, and Tolliver.
"""

__version__ = "0.1.0"
__author__ = "Mohammad Doostmohammadi"
__email__ = "your.email@example.com"
__license__ = "MIT"

# Core imports
from .algorithms.steiner import CMGSteinerSolver
from .utils.graph_utils import create_laplacian_from_edges, create_test_graphs
from .utils.statistics import CMGStatistics

# Optional imports (fail gracefully if dependencies missing)
try:
    from .visualization.plotting import plot_graph_decomposition, plot_conductance_analysis
    _HAS_VISUALIZATION = True
except ImportError:
    _HAS_VISUALIZATION = False

# Make key classes available at package level
__all__ = [
    'CMGSteinerSolver',
    'CMGStatistics', 
    'create_laplacian_from_edges',
    'create_test_graphs',
]

# Add visualization to exports if available
if _HAS_VISUALIZATION:
    __all__.extend(['plot_graph_decomposition', 'plot_conductance_analysis'])

def get_info():
    """Get package information."""
    info = {
        'version': __version__,
        'author': __author__,
        'license': __license__,
        'has_visualization': _HAS_VISUALIZATION,
    }
    return info

def run_demo():
    """Run a quick demonstration of the CMG algorithm."""
    print("CMG-Python Demo")
    print("=" * 50)
    
    # Create a simple test case
    from .utils.graph_utils import create_test_graphs
    
    solver = CMGSteinerSolver(verbose=True)
    test_graphs = create_test_graphs()
    
    # Test weak connection graph
    graph_data = test_graphs['weak_connection']
    A = create_laplacian_from_edges(graph_data['edges'], graph_data['n'])
    
    print(f"Testing: {graph_data['description']}")
    components, num_comp = solver.steiner_group(A)
    
    print(f"\nResult: {num_comp} components")
    solver.visualize_components(components)
    
    stats = solver.get_statistics()
    print(f"Average conductance: {stats['avg_conductance']:.6f}")
    print(f"Computation time: {solver.last_decomposition_time:.4f}s")
