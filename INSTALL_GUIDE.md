# CMG-Python Installation Guide

## Quick Start (Recommended)
```bash
pip install git+https://github.com/mdindoost/cmg-python.git
```
## Requirements

Python: 3.6 or higher
Core dependencies: NumPy, SciPy (installed automatically)
Optional: matplotlib (for visualization), networkx (for advanced features)

## Installation Methods
1. Direct pip install (Easiest)
```bash
pip install git+https://github.com/mdindoost/cmg-python.git
```
2. Clone and install (For development)
```bash
git clone https://github.com/mdindoost/cmg-python.git
cd cmg-python
pip install -e .
```
3. Virtual environment (Recommended for isolation)
```bash
# Create virtual environment
python -m venv cmg-env
source cmg-env/bin/activate  # Windows: cmg-env\Scripts\activate
```
# Install
```bash
pip install git+https://github.com/mdindoost/cmg-python.git
```

4. Conda environment
```bash
# Create conda environment
conda create -n cmg-python python=3.8 numpy scipy matplotlib
conda activate cmg-python
pip install git+https://github.com/mdindoost/cmg-python.git
```
# Verify Installation
```bash
# Test import
python -c "from cmg import CMGSteinerSolver; print('✅ CMG-Python ready!')"

# Run demo
python -m cmg.cli --demo

# Or use the command
cmg-demo
```
## Troubleshooting
Common Issues:

1. "No module named 'cmg'": Ensure installation completed successfully
2. Import errors: Check Python version (3.6+ required)
3. Performance issues: Install with pip install git+https://github.com/mdindoost/cmg-python.git[fast] for optimized dependencies

## Platform-specific notes:

Windows: Use python -m pip instead of just pip
Mac: May need brew install python for latest Python
Linux: Usually works out of the box with system Python

## What Gets Installed

Core CMG algorithm implementation
Command-line interface (cmg-demo)
Example graphs and test cases
Documentation and examples
Optional visualization tools

