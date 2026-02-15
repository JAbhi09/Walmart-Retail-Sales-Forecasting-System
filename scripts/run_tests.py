#!/usr/bin/env python
"""
Test runner script for the Walmart Retail Forecasting System.
Runs all tests and generates coverage reports.
"""
import sys
import subprocess
from pathlib import Path


def run_tests(test_type="all", verbose=False, coverage=False):
    """
    Run tests based on specified type.
    
    Args:
        test_type: Type of tests to run (all, unit, integration, validation)
        verbose: Enable verbose output
        coverage: Generate coverage report
    """
    project_root = Path(__file__).parent.parent
    tests_dir = project_root / "tests"
    
    # Base pytest command
    cmd = ["pytest"]
    
    # Determine which tests to run
    if test_type == "all":
        cmd.append(str(tests_dir))
    elif test_type == "unit":
        cmd.extend([
            str(tests_dir / "test_database.py"),
            str(tests_dir / "test_feature_engineering.py"),
            str(tests_dir / "test_model.py"),
            str(tests_dir / "test_agents.py")
        ])
    elif test_type == "integration":
        cmd.append(str(tests_dir / "test_integration.py"))
    elif test_type == "validation":
        cmd.append(str(tests_dir / "test_validation.py"))
    else:
        print(f"Unknown test type: {test_type}")
        print("Valid types: all, unit, integration, validation")
        return 1
    
    # Add verbose flag
    if verbose:
        cmd.append("-v")
    
    # Add coverage flags
    if coverage:
        cmd.extend([
            "--cov=.",
            "--cov-report=html",
            "--cov-report=term"
        ])
    
    # Run tests
    print(f"Running {test_type} tests...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 80)
    
    result = subprocess.run(cmd, cwd=project_root)
    
    if result.returncode == 0:
        print("-" * 80)
        print("✅ All tests passed!")
        if coverage:
            print(f"📊 Coverage report generated: {project_root / 'htmlcov' / 'index.html'}")
    else:
        print("-" * 80)
        print("❌ Some tests failed!")
    
    return result.returncode


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run tests for Walmart Retail Forecasting System")
    parser.add_argument(
        "--type",
        choices=["all", "unit", "integration", "validation"],
        default="all",
        help="Type of tests to run (default: all)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "-c", "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    
    args = parser.parse_args()
    
    sys.exit(run_tests(args.type, args.verbose, args.coverage))


if __name__ == "__main__":
    main()
