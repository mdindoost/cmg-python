# CMG-Python Repository Setup Instructions

This guide will help you create a complete CMG-Python repository with all the files and structure needed for a professional Python package.

## Step 1: Create Repository Structure

Create the main directory and all subdirectories:

```bash
mkdir cmg-python
cd cmg-python

# Create package structure
mkdir -p cmg/algorithms
mkdir -p cmg/utils  
mkdir -p cmg/visualization
mkdir -p tests
mkdir -p examples
mkdir -p docs
mkdir -p .github/workflows

# Create __init__.py files
touch cmg/__init__.py
touch cmg/algorithms/__init__.py
touch cmg/utils/__init__.py
touch cmg/visualization/__init__.py
touch tests/__init__.py
```

## Step 2: Copy All Package Files

Copy the contents from the artifacts to create these files:

### Core Package Files
1. **`setup.py`** - Package setup and installation configuration
2. **`requirements.txt`** - Package dependencies
3. **`README.md`** - Main documentation and usage guide
4. **`Makefile`** - Development commands and workflows

### CMG Package Files
5. **`cmg/__init__.py`** - Main package initialization
6. **`cmg/algorithms/steiner.py`** - Core CMG algorithm implementation  
7. **`cmg/utils/graph_utils.py`** - Graph utilities and test cases
8. **`cmg/utils/statistics.py`** - Statistics and analysis tools
9. **`cmg/visualization/plotting.py`** - Visualization functions
10. **`cmg/cli.py`** - Command-line interface

### Initialize subpackages with the contents from `package_init_files.py`:
- **`cmg/algorithms/__init__.py`**
- **`cmg/utils/__init__.py`** 
- **`cmg/visualization/__init__.py`**

### Example and Test Files
11. **`examples/basic_usage.py`** - Basic usage examples
12. **`tests/test_steiner.py`** - Comprehensive unit tests

### CI/CD and Documentation
13. **`.github/workflows/tests.yml`** - GitHub Actions workflow

## Step 3: Additional Files to Create

Create these additional files manually:

### License File
```bash
# Create LICENSE file (MIT License example)
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2024 Mohammad Doostmohammadi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF
```

### Git Configuration
```bash
# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Documentation
docs/_build/
EOF
```

### Development Configuration
```bash
# Create pyproject.toml for modern Python packaging
cat > pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "cmg-python"
dynamic = ["version"]
description = "Python implementation of Combinatorial Multigrid (CMG) Steiner Group algorithm"
authors = [
    {name = "Mohammad Doostmohammadi", email = "your.email@example.com"}
]
license = {text = "MIT"}
readme = "README.md"
requires-python = ">=3.7"
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]
dependencies = [
    "numpy>=1.19.0",
    "scipy>=1.7.0",
]

[project.optional-dependencies]
visualization = [
    "matplotlib>=3.3.0",
    "networkx>=2.5",
    "seaborn>=0.11.0",
]
dev = [
    "pytest>=6.0.0",
    "pytest-cov>=2.0.0",
    "black>=21.0.0",
    "flake8>=3.8.0",
]

[project.scripts]
cmg-demo = "cmg.cli:main"

[project.urls]
Homepage = "https://github.com/yourusername/cmg-python"
Repository = "https://github.com/yourusername/cmg-python"
Documentation = "https://cmg-python.readthedocs.io/"
"Bug Reports" = "https://github.com/yourusername/cmg-python/issues"

[tool.setuptools.dynamic]
version = {attr = "cmg.__version__"}

[tool.black]
line-length = 100
target-version = ['py37']

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short"
EOF
```

## Step 4: Initialize Git Repository

```bash
# Initialize git repository
git init
git add .
git commit -m "Initial commit: CMG-Python package structure"

# Create main branch (if not already on main)
git branch -M main
```

## Step 5: Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e .[dev,visualization]

# Run tests to verify setup
python -m pytest tests/ -v

# Run examples
python examples/basic_usage.py

# Test CLI
python -m cmg.cli --demo
```

## Step 6: Verify Installation

Test that everything works:

```python
# Test basic import
python -c "import cmg; print('CMG-Python successfully installed!')"

# Test algorithm
python -c "
from cmg import CMGSteinerSolver, create_test_graphs, create_laplacian_from_edges
graphs = create_test_graphs()
A = create_laplacian_from_edges(graphs['weak_connection']['edges'], 4)
solver = CMGSteinerSolver()
components, num_comp = solver.steiner_group(A)
print(f'Found {num_comp} components: {components}')
"
```

## Step 7: GitHub Repository Setup

1. Create repository on GitHub
2. Update URLs in `setup.py` and `README.md` with your actual GitHub username
3. Push to GitHub:

```bash
git remote add origin https://github.com/yourusername/cmg-python.git
git push -u origin main
```

## Step 8: Optional Enhancements

### Documentation with Sphinx
```bash
# Install Sphinx
pip install sphinx sphinx-rtd-theme

# Initialize documentation
cd docs
sphinx-quickstart

# Build documentation  
make html
```

### PyPI Publishing
```bash
# Install publishing tools
pip install twine

# Build package
python setup.py sdist bdist_wheel

# Upload to PyPI (requires account)
twine upload dist/*
```

## Final Directory Structure

Your completed repository should look like this:

```
cmg-python/
├── .github/
│   └── workflows/
│       └── tests.yml
├── cmg/
│   ├── __init__.py
│   ├── cli.py
│   ├── algorithms/
│   │   ├── __init__.py
│   │   └── steiner.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── graph_utils.py
│   │   └── statistics.py
│   └── visualization/
│       ├── __init__.py
│       └── plotting.py
├── tests/
│   ├── __init__.py
│   └── test_steiner.py
├── examples/
│   └── basic_usage.py
├── docs/
├── .gitignore
├── LICENSE
├── Makefile
├── README.md
├── requirements.txt
├── setup.py
└── pyproject.toml
```

## Quick Start After Setup

```bash
# Run the demo
cmg-demo

# Run tests
make test

# Check code quality
make lint

# Format code
make format

# Build package
make build
```

You now have a complete, professional Python package for your CMG algorithm! 🎉
