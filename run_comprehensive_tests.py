#!/usr/bin/env python3
"""
Comprehensive Test Runner for CMG-Python
========================================

Runs all tests with detailed reporting and performance metrics.
"""

import subprocess
import time
import sys


def run_test_suite(test_file, description):
    """Run a specific test suite and return results."""
    print(f"\n{'='*60}")
    print(f"Running {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    result = subprocess.run([
        sys.executable, '-m', 'pytest', 
        test_file, '-v', '-s', '--tb=short'
    ], capture_output=True, text=True)
    
    elapsed = time.time() - start_time
    
    # Print results
    print(result.stdout)
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
        
    success = result.returncode == 0
    
    print(f"\n{description} Results:")
    print(f"  Status: {'✅ PASSED' if success else '❌ FAILED'}")
    print(f"  Time: {elapsed:.2f} seconds")
    
    return success, elapsed


def main():
    """Run all test suites."""
    print("🧪 CMG-Python Comprehensive Test Suite")
    print("=" * 60)
    
    test_suites = [
        ('tests/test_steiner.py', 'Core Algorithm Tests'),
        ('tests/test_advanced.py', 'Advanced & Edge Case Tests'),
        ('tests/test_benchmarks.py', 'Performance Benchmarks')
    ]
    
    total_time = 0
    passed_suites = 0
    
    for test_file, description in test_suites:
        success, elapsed = run_test_suite(test_file, description)
        total_time += elapsed
        if success:
            passed_suites += 1
            
    # Final summary
    print(f"\n{'='*60}")
    print("🏆 FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Test Suites: {passed_suites}/{len(test_suites)} passed")
    print(f"Total Time: {total_time:.2f} seconds")
    
    if passed_suites == len(test_suites):
        print("🎉 ALL TESTS PASSED! Your CMG implementation is robust!")
        return 0
    else:
        print("❌ Some test suites failed. Check output above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
