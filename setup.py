#!/usr/bin/env python3
"""
Setup script for CMG (Combinatorial Multigrid) Python library.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="cmg-python",
    version="0.1.0",
    author="Mohammad Doostmohammadi",
    author_email="md724@NJIT.edu", 
    description="Python implementation of Combinatorial Multigrid (CMG) Steiner Group algorithm",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/mdindoost/cmg-python",  
    packages=find_packages(),
    classifiers=[
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
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements(),
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=21.0',
            'flake8>=3.8',
            'sphinx>=4.0',
            'sphinx-rtd-theme>=1.0',
        ],
        'visualization': [
            'matplotlib>=3.3',
            'networkx>=2.5',
            'seaborn>=0.11',
        ],
        'all': [
            'matplotlib>=3.3',
            'networkx>=2.5',
            'seaborn>=0.11',
            'pytest>=6.0',
            'pytest-cov>=2.0',
        ]
    },
    entry_points={
        'console_scripts': [
            'cmg-demo=cmg.cli:main',
        ],
    },
    keywords="multigrid, graph theory, linear algebra, steiner tree, preconditioner",
    project_urls={
        "Bug Reports": "https://github.com/mdindoost/cmg-python/issues",
        "Source": "https://github.com/mdindoost/cmg-python",
        "Documentation": "https://cmg-python.readthedocs.io/",
    },
)
