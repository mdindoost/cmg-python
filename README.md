# CMG-Python: Combinatorial Multigrid for Python

[![Python Version](https://img.shields.io/pypi/pyversions/cmg-python)](https://pypi.org/project/cmg-python/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/yourusername/cmg-python/workflows/tests/badge.svg)](https://github.com/mdindoost/cmg-python/actions)

A Python implementation of the **Combinatorial Multigrid (CMG) Steiner Group algorithm** from the paper:

> *"Combinatorial preconditioners and multilevel solvers for problems in computer vision and image processing"*  
> by Ioannis Koutis, Gary L. Miller, and David Tolliver  
> Computer Vision and Image Understanding, Volume 115, Issue 12, 2011

## Overview

CMG (Combinatorial Multigrid) is a sophisticated algorithm for decomposing graphs into clusters with bounded conductance. Unlike simple graph partitioning methods, CMG creates hierarchical decompositions that are particularly effective for:

- **Linear system preconditioning**
- **Multigrid solver construction** 
- **Image processing applications**
- **Computer vision algorithms**
- **Scientific computing**

## Key Features

- ✅ **Faithful implementation** of the original CMG algorithm
- ✅ **High performance** with NumPy/SciPy optimization
- ✅ **Flexible input formats** (sparse matrices, edge lists, NetworkX graphs)
- ✅ **Comprehensive testing** and validation
- ✅ **Visualization tools** for understanding decompositions
- ✅ **Detailed documentation** and examples

## Installation

### From PyPI (recommended)
```bash
pip install cmg-python
```

### From source
```bash
git clone https://github.com/mdindoost/cmg-python.git
cd cmg-python
pip install -e .
```

### With optional dependencies
```bash
# For visualization
pip install cmg-python[visualization]

# For development
pip install cmg-python[dev]

# Everything
pip install cmg-python[all]
```

## Quick Start

```python
import numpy as np
import scipy.sparse as sp
from cmg import CMGSteinerSolver

# Create a graph with weak connections
edges = [(0, 1, 1.0), (1, 2, 0.01), (2, 3, 1.0)]  # Weak middle connection
n = 4

# Build Laplacian matrix
rows, cols, data = [], [], []
for u, v, weight in edges:
    rows.extend([u, v])
    cols.extend([v, u]) 
    data.extend([-weight, -weight])

# Add diagonal (degrees)
degrees = [0] * n
for u, v, weight in edges:
    degrees[u] += weight
    degrees[v] += weight

for i in range(n):
    rows.append(i)
    cols.append(i)
    data.append(degrees[i])

A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))

# Run CMG decomposition
solver = CMGSteinerSolver(gamma=5.0)
component_indices, num_components = solver.steiner_group(A)

print(f"Found {num_components} components:")
print(f"Component assignment: {component_indices}")
# Output: Found 2 components: [0 0 1 1]
```

## Examples

### Example 1: Weak Connection Detection

The algorithm detects and splits graphs at weak connections:

```python
from cmg.utils import create_laplacian_from_edges
from cmg import CMGSteinerSolver

# Graph: 1--2~~~3--4 (weak connection between 2 and 3)
edges = [(0, 1, 1.0), (1, 2, 0.001), (2, 3, 1.0)]
A = create_laplacian_from_edges(edges, 4)

solver = CMGSteinerSolver()
components, num_comp = solver.steiner_group(A)

# Result: splits into {1,2} and {3,4}
solver.visualize_components(components)
```

### Example 2: Multiple Clusters

```python
# Two triangles connected by a weak bridge
edges = [
    # Triangle 1
    (0, 1, 2.0), (1, 2, 2.0), (2, 0, 2.0),
    # Weak bridge
    (2, 3, 0.1),
    # Triangle 2  
    (3, 4, 2.0), (4, 5, 2.0), (5, 3, 2.0)
]

A = create_laplacian_from_edges(edges, 6)
components, num_comp = solver.steiner_group(A)

# Result: splits into two triangles
print(f"Components: {num_comp}")  # 2 components
```

### Example 3: Visualization

```python
from cmg.visualization import plot_graph_decomposition

# Visualize the decomposition
plot_graph_decomposition(A, components, title="CMG Decomposition")
```

## Algorithm Details

The CMG Steiner Group algorithm works by:

1. **Weighted Degree Calculation**: Computes `wd(v) = vol(v) / max_weight(v)` for each node
2. **Heaviest Edge Forest**: Builds a forest using the heaviest incident edge for each node
3. **High-Degree Node Filtering**: Removes problematic edges from high-degree nodes
4. **Component Extraction**: Finds connected components in the resulting forest

Unlike simple graph partitioning, this creates decompositions with bounded conductance, making them ideal for multigrid preconditioning.

## Performance

CMG-Python is optimized for performance:

- **Linear time complexity** for sparse graphs
- **NumPy vectorization** for core operations
- **Efficient sparse matrix handling**
- **Memory-conscious algorithms**

Typical performance on a modern laptop:
- **1K nodes**: < 1ms
- **10K nodes**: < 100ms  
- **100K nodes**: < 10s

## API Reference

### Core Classes

- **`CMGSteinerSolver`**: Main solver class
- **`CMGStatistics`**: Detailed algorithm statistics
- **`GraphUtils`**: Graph manipulation utilities

### Key Methods

- **`steiner_group(A)`**: Main decomposition algorithm
- **`weighted_degree(A, node)`**: Calculate weighted degree
- **`conductance(A, cluster)`**: Calculate cluster conductance
- **`visualize_components()`**: Display results

## Testing

Run the test suite:

```bash
# Basic tests
pytest tests/

# With coverage
pytest tests/ --cov=cmg --cov-report=html

# Performance benchmarks
pytest tests/test_performance.py -v
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch
3. Add tests for your changes
4. Ensure all tests pass
5. Submit a pull request

## Citation

If you use this library in your research, please cite:

```bibtex
@article{koutis2011combinatorial,
  title={Combinatorial preconditioners and multilevel solvers for problems in computer vision and image processing},
  author={Koutis, Ioannis and Miller, Gary L and Tolliver, David},
  journal={Computer Vision and Image Understanding},
  volume={115},
  number={12},
  pages={1638--1646},
  year={2011},
  publisher={Elsevier}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original CMG algorithm by Koutis, Miller, and Tolliver
- Inspired by the MATLAB implementation at [ikoutis/cmg-solver](https://github.com/ikoutis/cmg-solver)
- Julia implementation at [bodhi91/CombinatorialMultigrid.jl](https://github.com/bodhi91/CombinatorialMultigrid.jl)

## Links

- **Documentation**: [https://cmg-python.readthedocs.io/](https://cmg-python.readthedocs.io/)
- **PyPI**: [https://pypi.org/project/cmg-python/](https://pypi.org/project/cmg-python/)
- **Issues**: [https://github.com/yourusername/cmg-python/issues](https://github.com/yourusername/cmg-python/issues)
- **Original Paper**: [DOI:10.1016/j.cviu.2011.05.013](https://doi.org/10.1016/j.cviu.2011.05.013)
