#!/usr/bin/env python3
"""
Graph Property Analysis for CMG
==============================

This script analyzes which graph properties are most predictive of CMG success.
"""

import sys
sys.path.append('../..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def analyze_predictive_properties(results_file):
    """
    Analyze which graph properties best predict CMG success.
    """
    df = pd.read_csv(results_file)
    
    print("🔍 Graph Property Predictive Analysis")
    print("=" * 50)
    
    # Define success criteria
    df['is_successful'] = (
        (df['avg_conductance'] != float('inf')) & 
        (df['avg_conductance'] < 1.0) &
        (df['num_communities'] > 1) &
        (df['num_communities'] < df['n_nodes'] / 2)
    )
    
    # Prepare features for machine learning
    feature_cols = [
        'n_nodes', 'n_edges', 'density', 'avg_degree', 'max_degree', 
        'min_degree', 'degree_std', 'degree_skewness', 'gamma'
    ]
    
    # Only use rows with all features available
    df_ml = df.dropna(subset=feature_cols + ['is_successful'])
    
    if len(df_ml) < 20:
        print("Insufficient data for machine learning analysis")
        return
    
    X = df_ml[feature_cols]
    y = df_ml['is_successful']
    
    # Train random forest to identify important features
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🎯 FEATURE IMPORTANCE (Random Forest):")
    print(feature_importance.to_string(index=False))
    
    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Feature Importance for Predicting CMG Success')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.show()
    
    # Performance on test set
    y_pred = rf.predict(X_test)
    print(f"\n📊 MODEL PERFORMANCE:")
    print(f"Accuracy: {rf.score(X_test, y_test):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Analyze successful vs unsuccessful cases
    print(f"\n📈 SUCCESSFUL vs UNSUCCESSFUL ANALYSIS:")
    
    successful = df_ml[df_ml['is_successful']]
    unsuccessful = df_ml[~df_ml['is_successful']]
    
    print(f"Successful cases: {len(successful)}")
    print(f"Unsuccessful cases: {len(unsuccessful)}")
    
    if len(successful) > 0 and len(unsuccessful) > 0:
        print(f"\nProperty differences (successful vs unsuccessful):")
        for col in ['density', 'avg_degree', 'degree_std', 'n_nodes']:
            if col in df_ml.columns:
                succ_mean = successful[col].mean()
                unsucc_mean = unsuccessful[col].mean()
                print(f"  {col}: {succ_mean:.4f} vs {unsucc_mean:.4f}")
    
    # Correlation analysis
    plt.figure(figsize=(12, 8))
    
    # Select numeric columns for correlation
    numeric_cols = df_ml.select_dtypes(include=[np.number]).columns
    corr_with_success = df_ml[numeric_cols].corrwith(df_ml['is_successful'].astype(int))
    
    # Plot correlation with success
    corr_with_success = corr_with_success.drop('is_successful').sort_values(key=abs, ascending=False)
    
    plt.subplot(1, 2, 1)
    corr_with_success.plot(kind='barh')
    plt.title('Correlation with CMG Success')
    plt.xlabel('Correlation Coefficient')
    
    # Heatmap of all correlations
    plt.subplot(1, 2, 2)
    correlation_matrix = df_ml[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Property Correlation Matrix')
    
    plt.tight_layout()
    plt.show()
    
    return feature_importance, rf

if __name__ == "__main__":
    # Find most recent results
    results_dir = Path('../results')
    csv_files = list(results_dir.glob('cmg_results_*.csv'))
    
    if csv_files:
        latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
        analyze_predictive_properties(latest_file)
    else:
        print("No results files found. Run evaluate_cmg_systematic.py first.")
