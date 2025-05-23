#!/bin/bash
# CMG-Python Complete Environment Setup with Demo
# ==============================================

set -e  # Exit on any error

echo "🚀 Setting up CMG-Python environment..."
echo "======================================"

# Navigate to project directory
cd ~/cmg-python

# Activate virtual environment
if [ -d "venv" ]; then
    echo "📂 Activating virtual environment..."
    source venv/bin/activate
else
    echo "❌ Virtual environment not found! Please run:"
    echo "   python -m venv venv"
    exit 1
fi

# Check if CMG is installed
echo "🔍 Checking CMG installation..."
if python -c "from cmg import CMGSteinerSolver" 2>/dev/null; then
    echo "✅ CMG-Python core is installed"
else
    echo "❌ CMG-Python not found! Installing..."
    pip install -e .
fi

# Install/upgrade all packages
echo "📦 Installing all required packages..."
pip install --upgrade \
    numpy \
    scipy \
    matplotlib \
    networkx \
    jupyter \
    ipykernel \
    ipywidgets \
    pytest \
    pytest-cov \
    pandas \
    seaborn

# Verify installation
echo ""
echo "🔬 Verifying installation..."
echo "=========================="

# Test everything
python -c "
import numpy, scipy, matplotlib, networkx, jupyter, pytest, pandas, seaborn
from cmg import CMGSteinerSolver, create_laplacian_from_edges
print('✅ All packages imported successfully')

# Quick CMG test
edges = [(0,1,1.0), (1,2,0.1), (2,3,1.0)]
A = create_laplacian_from_edges(edges, 4)
solver = CMGSteinerSolver(verbose=False)
components, num_comp = solver.steiner_group(A)
print(f'✅ CMG test: {num_comp} communities detected')
"

echo "✅ All systems operational!"
echo ""

# Create demo files if they don't exist
echo "📋 Ensuring demo files are available..."

if [ ! -f "simple_two_cliques.py" ]; then
    echo "📝 Creating simple_two_cliques.py..."
    cat > simple_two_cliques.py << 'DEMO_EOF'
#!/usr/bin/env python3
from cmg import CMGSteinerSolver, create_laplacian_from_edges

def quick_demo():
    print("🔍 Quick CMG Demo: Two Cliques with Weak Bridge")
    print("=" * 45)
    
    # Two triangular cliques with weak bridge
    edges = [
        # Clique 1: strong triangle
        (0, 1, 2.0), (1, 2, 2.0), (2, 0, 2.0),
        # Clique 2: strong triangle  
        (3, 4, 2.0), (4, 5, 2.0), (5, 3, 2.0),
        # Weak bridge
        (2, 3, 0.01)
    ]
    
    A = create_laplacian_from_edges(edges, 6)
    solver = CMGSteinerSolver(gamma=5.0, verbose=False)
    components, num_comm = solver.steiner_group(A)
    
    print(f"Graph: 2 triangular cliques + weak bridge (weight=0.01)")
    print(f"Result: {num_comm} communities detected")
    print(f"Assignment: {components}")
    
    if num_comm == 2:
        print("✅ Perfect! CMG correctly found 2 communities")
    else:
        print("⚠️  Try different gamma values")
    
if __name__ == "__main__":
    quick_demo()
DEMO_EOF
    chmod +x simple_two_cliques.py
fi

echo ""
echo "🎯 Setup Complete! What would you like to do?"
echo "==========================================="
echo ""
echo "1️⃣  Run quick demo (simple_two_cliques.py)"
echo "2️⃣  Run full visualization demo (two_cliques_test.py)" 
echo "3️⃣  Start Jupyter notebook for interactive analysis"
echo "4️⃣  Run test suite to verify everything"
echo "5️⃣  Just show me the commands and exit"
echo ""

read -p "Choose an option (1-5): " -n 1 -r
echo ""
echo ""

case $REPLY in
    1)
        echo "🚀 Running quick demo..."
        python simple_two_cliques.py
        echo ""
        echo "💡 Demo complete! Try editing simple_two_cliques.py with your own graphs."
        ;;
    2)
        echo "🚀 Running full visualization demo..."
        if [ -f "two_cliques_test.py" ]; then
            python two_cliques_test.py
        else
            echo "⚠️  two_cliques_test.py not found. Running simple demo instead..."
            python simple_two_cliques.py
        fi
        ;;
    3)
        echo "📓 Starting Jupyter notebook..."
        echo "💡 Jupyter will open in your browser. Press Ctrl+C here to stop it."
        echo "💡 Try creating a new notebook with:"
        echo "   from cmg import CMGSteinerSolver, create_laplacian_from_edges"
        echo ""
        sleep 3
        jupyter notebook
        ;;
    4)
        echo "🧪 Running test suite..."
        python -m pytest tests/ -v
        ;;
    5)
        echo "📋 Available commands:"
        echo "   python simple_two_cliques.py        # Quick demo"
        echo "   python two_cliques_test.py          # Full visualization" 
        echo "   jupyter notebook                    # Interactive analysis"
        echo "   python -m pytest tests/ -v         # Run tests"
        echo "   python test_my_graph.py            # Your custom graphs"
        ;;
    *)
        echo "Invalid option. Here are the available commands:"
        echo "   python simple_two_cliques.py        # Quick demo"
        echo "   python two_cliques_test.py          # Full visualization"
        echo "   jupyter notebook                    # Interactive analysis"
        ;;
esac

echo ""
echo "🎉 CMG-Python is ready!"
echo "💡 Environment stays active in this terminal session."
echo "💡 To reactivate later: cd ~/cmg-python && source venv/bin/activate"
