#!/usr/bin/env python3
"""
Test runner for EMS pipeline tests.
Runs all tests and provides summary results.
"""

import unittest
import sys
import os
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "notebooks"))
sys.path.insert(0, str(project_root / "scripts"))
sys.path.insert(0, str(project_root / "tests"))

def run_all_tests():
    """Run all test suites and return results."""
    # Create test loader
    loader = unittest.TestLoader()
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test modules
    test_modules = [
        'tests.test_agent_invocations',
        'tests.test_smoke'
    ]
    
    for module in test_modules:
        try:
            suite.addTests(loader.loadTestsFromName(module))
            print(f"✓ Loaded tests from {module}")
        except ImportError as e:
            print(f"✗ Failed to load {module}: {e}")
            return False
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    # Print failures and errors
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Exception:')[-1].strip()}")
    
    return len(result.failures) == 0 and len(result.errors) == 0

def run_quick_smoke_test():
    """Run just the smoke tests for CI."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName('tests.test_smoke.TestSmokeTests')
    
    runner = unittest.TextTestRunner(verbosity=1, buffer=True)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_lint_checks():
    """Run lint checks for forbidden patterns."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName('tests.test_smoke.TestLintChecks')
    
    runner = unittest.TextTestRunner(verbosity=1, buffer=True)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    # Create output directories
    os.makedirs("results/plots", exist_ok=True)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--smoke':
            print("Running smoke tests only...")
            success = run_quick_smoke_test()
        elif sys.argv[1] == '--lint':
            print("Running lint checks only...")
            success = run_lint_checks()
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Usage: python run_tests.py [--smoke|--lint]")
            sys.exit(1)
    else:
        print("Running all tests...")
        success = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)