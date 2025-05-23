#!/usr/bin/env python3
"""
CMG-Python Dependency Checker
============================
"""

import sys
import importlib

def check_package(package_name, description, required=True):
    """Check if a package is available."""
    try:
        importlib.import_module(package_name)
        print(f"✅ {package_name:12s} - {description}")
        return True
    except ImportError:
        status = "❌" if required else "⚠️ "
        req_text = "(required)" if required else "(optional)"
        print(f"{status} {package_name:12s} - {description} {req_text}")
        return False

def main():
    print("🔍 CMG-Python Dependency Check")
    print("=" * 35)
    
    all_good = True
    
    # Core requirements
    print("\n🔧 Core Requirements:")
    all_good &= check_package("numpy", "Numerical computing", True)
    all_good &= check_package("scipy", "Scientific computing", True)
    all_good &= check_package("cmg", "CMG algorithm", True)
    
    # Visualization
    print("\n📊 Visualization:")
    check_package("matplotlib", "Plotting", False)
    check_package("networkx", "Graph visualization", False)
    
    # Interactive
    print("\n📓 Interactive Analysis:")
    check_package("jupyter", "Jupyter notebooks", False)
    check_package("IPython", "Enhanced Python shell", False)
    
    # Testing
    print("\n🧪 Testing:")
    check_package("pytest", "Testing framework", False)
    
    # Optional
    print("\n🎨 Optional Enhancements:")
    check_package("pandas", "Data analysis", False)
    check_package("seaborn", "Statistical visualization", False)
    check_package("plotly", "Interactive plots", False)
    
    print("\n" + "=" * 35)
    
    if all_good:
        print("✅ All core requirements satisfied!")
        print("💡 Run visualization examples with confidence")
    else:
        print("❌ Missing core requirements")
        print("💡 Run: ./setup_cmg.sh to install everything")
        
    # Quick CMG test
    try:
        from cmg import CMGSteinerSolver, create_laplacian_from_edges
        edges = [(0,1,1.0)]
        A = create_laplacian_from_edges(edges, 2)
        solver = CMGSteinerSolver(verbose=False)
        components, num_comp = solver.steiner_group(A)
        print(f"✅ CMG functionality test: OK")
    except Exception as e:
        print(f"❌ CMG functionality test: {e}")
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())
