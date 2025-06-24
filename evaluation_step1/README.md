# CMG Systematic Evaluation - Step 1

This directory contains a comprehensive evaluation framework to understand when and why CMG works well.

## 📁 Directory Structure

```
evaluation_step1/
├── datasets/           # Test graph datasets organized by type
├── benchmarks/         # Benchmark comparison scripts  
├── results/           # Evaluation results (CSV, pickle files)
├── analysis/          # Statistical analysis scripts
├── notebooks/         # Jupyter notebooks for interactive analysis
├── scripts/           # Utility scripts
├── evaluate_cmg_systematic.py  # Main evaluation script
└── README.md          # This file
```

## 🚀 Quick Start

1. **Install additional requirements:**
   ```bash
   pip install -r requirements_evaluation.txt
   ```

2. **Run systematic evaluation:**
   ```bash
   python evaluate_cmg_systematic.py
   ```

3. **Analyze results:**
   ```bash
   jupyter notebook notebooks/step1_analysis.ipynb
   ```

## 📊 What Gets Evaluated

### Graph Types Tested:
- **Grid graphs**: Regular lattice structures
- **Random graphs**: Erdős–Rényi graphs with different densities  
- **Clustered graphs**: Graphs with known community structure
- **Built-in test graphs**: From your existing test suite

### Metrics Collected:
- **Performance**: Runtime, scalability
- **Quality**: Conductance, community structure
- **Graph properties**: Density, degree distribution, spectral properties
- **Parameter sensitivity**: Different γ values

### Comparisons:
- **Spectral clustering**: Standard baseline
- **Parameter variations**: Different CMG settings
- **Graph property analysis**: What makes CMG work well

## 📈 Expected Outcomes

After running this evaluation, you'll have:

1. **Clear characterization** of when CMG works well vs poorly
2. **Data-driven insights** about optimal parameters
3. **Graph property predictors** for CMG success
4. **Baseline comparisons** showing CMG's strengths/weaknesses
5. **Foundation for Step 2** theoretical analysis

## 🔧 Customization

### Adding New Graph Types:
Edit `generate_test_graphs()` in `evaluate_cmg_systematic.py`

### Adding New Baselines:
Implement comparison methods in `CMGSystematicEvaluator`

### Modifying Metrics:
Update `evaluate_graph()` method to collect additional metrics

## 📋 Next Steps

1. Run the evaluation and review results
2. Use Jupyter notebook for deeper analysis
3. Identify CMG's "sweet spot" from the data
4. Proceed to Step 2 with data-driven insights

## 📝 Notes

- First run may take 10-30 minutes depending on your system
- Results are automatically saved with timestamps
- All graphs are generated deterministically (seed=42) for reproducibility
- Large graphs (>5000 nodes) skip expensive baseline comparisons
