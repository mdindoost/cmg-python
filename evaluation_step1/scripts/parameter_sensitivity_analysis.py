#!/usr/bin/env python3
"""
Parameter Sensitivity Analysis for CMG
=====================================

This script performs detailed analysis of how CMG parameters affect performance
across different graph types.
"""

import sys
sys.path.append('../..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_parameter_sensitivity(results_file):
    """
    Analyze how gamma parameter affects CMG performance across graph types.
    """
    df = pd.read_csv(results_file)
    
    print("🔧 Parameter Sensitivity Analysis")
    print("=" * 50)
    
    # Analysis by graph type and gamma
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Number of communities vs gamma
    for graph_type in df['graph_type'].unique():
        subset = df[df['graph_type'] == graph_type]
        gamma_means = subset.groupby('gamma')['num_communities'].mean()
        axes[0,0].plot(gamma_means.index, gamma_means.values, 
                       marker='o', label=graph_type)
    
    axes[0,0].set_xlabel('Gamma Parameter')
    axes[0,0].set_ylabel('Average Number of Communities')
    axes[0,0].set_title('Communities vs Gamma by Graph Type')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Runtime vs gamma
    for graph_type in df['graph_type'].unique():
        subset = df[df['graph_type'] == graph_type]
        gamma_means = subset.groupby('gamma')['cmg_time'].mean()
        axes[0,1].plot(gamma_means.index, gamma_means.values, 
                       marker='s', label=graph_type)
    
    axes[0,1].set_xlabel('Gamma Parameter')
    axes[0,1].set_ylabel('Average Runtime (seconds)')
    axes[0,1].set_title('Runtime vs Gamma by Graph Type')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Conductance vs gamma (finite values only)
    df_finite = df[df['avg_conductance'] != float('inf')]
    if len(df_finite) > 0:
        for graph_type in df_finite['graph_type'].unique():
            subset = df_finite[df_finite['graph_type'] == graph_type]
            if len(subset) > 0:
                gamma_means = subset.groupby('gamma')['avg_conductance'].mean()
                axes[1,0].plot(gamma_means.index, gamma_means.values, 
                               marker='^', label=graph_type)
        
        axes[1,0].set_xlabel('Gamma Parameter')
        axes[1,0].set_ylabel('Average Conductance')
        axes[1,0].set_title('Conductance vs Gamma by Graph Type')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
    
    # 4. Success rate vs gamma
    df['is_successful'] = (
        (df['avg_conductance'] != float('inf')) & 
        (df['avg_conductance'] < 1.0) &
        (df['num_communities'] > 1) &
        (df['num_communities'] < df['n_nodes'] / 2)
    )
    
    success_by_gamma = df.groupby(['graph_type', 'gamma'])['is_successful'].mean().unstack()
    success_by_gamma.plot(kind='bar', ax=axes[1,1])
    axes[1,1].set_xlabel('Graph Type')
    axes[1,1].set_ylabel('Success Rate')
    axes[1,1].set_title('Success Rate by Graph Type and Gamma')
    axes[1,1].legend(title='Gamma', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Print recommendations
    print("\n🎯 PARAMETER RECOMMENDATIONS:")
    
    # Best gamma overall
    best_gamma_overall = df.groupby('gamma')['is_successful'].mean().idxmax()
    print(f"Best gamma overall: {best_gamma_overall}")
    
    # Best gamma by graph type
    print("\nBest gamma by graph type:")
    for graph_type in df['graph_type'].unique():
        subset = df[df['graph_type'] == graph_type]
        if len(subset) > 0:
            best_gamma = subset.groupby('gamma')['is_successful'].mean().idxmax()
            success_rate = subset.groupby('gamma')['is_successful'].mean().max()
            print(f"  {graph_type}: γ={best_gamma} (success rate: {success_rate:.2%})")

if __name__ == "__main__":
    # Find most recent results
    results_dir = Path('../results')
    csv_files = list(results_dir.glob('cmg_results_*.csv'))
    
    if csv_files:
        latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
        analyze_parameter_sensitivity(latest_file)
    else:
        print("No results files found. Run evaluate_cmg_systematic.py first.")
