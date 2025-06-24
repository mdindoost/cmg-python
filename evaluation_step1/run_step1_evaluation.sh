#!/bin/bash
# Complete Step 1 Evaluation Runner
# =================================

echo "🚀 Starting CMG Step 1 Systematic Evaluation"
echo "============================================="

# Check if we're in the right directory
if [ ! -f "evaluate_cmg_systematic.py" ]; then
    echo "❌ Error: evaluate_cmg_systematic.py not found!"
    echo "Please run this script from the evaluation_step1 directory"
    exit 1
fi

# Install requirements if needed
echo "📦 Checking requirements..."
pip install -r requirements_evaluation.txt

echo ""
echo "1️⃣ Running systematic evaluation..."
python evaluate_cmg_systematic.py

echo ""
echo "2️⃣ Running parameter sensitivity analysis..."
python scripts/parameter_sensitivity_analysis.py

echo ""
echo "3️⃣ Running graph property analysis..."
python scripts/graph_property_analysis.py

echo ""
echo "✅ Step 1 evaluation completed!"
echo ""
echo "📋 Next steps:"
echo "   1. Review the analysis outputs above"
echo "   2. Open Jupyter notebook for interactive analysis:"
echo "      jupyter notebook notebooks/step1_analysis.ipynb"
echo "   3. Check results/ directory for detailed data files"
echo "   4. Use insights to plan Step 2 (theoretical analysis)"
echo ""
echo "🎯 Key questions to answer from this data:"
echo "   - On which graph types does CMG work best?"
echo "   - What gamma values are optimal for different graphs?"
echo "   - Which graph properties predict CMG success?"
echo "   - Where does CMG fail and why?"
