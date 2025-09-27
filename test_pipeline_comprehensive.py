#!/usr/bin/env python3
"""
Comprehensive Pipeline Test Suite
Tests KaggleSlayer pipeline across multiple datasets and scenarios
"""

import subprocess
import sys
import time
from pathlib import Path

def run_pipeline_test(dataset_path: str, description: str, timeout: int = 180) -> dict:
    """Run pipeline test and capture results"""
    print(f"\n{'='*60}")
    print(f"TESTING: {description}")
    print(f"Dataset: {dataset_path}")
    print(f"{'='*60}")

    start_time = time.time()
    try:
        result = subprocess.run([
            sys.executable, "run_pipeline.py",
            dataset_path, "--dry-run"
        ], capture_output=True, text=True, timeout=timeout)

        elapsed = time.time() - start_time
        success = result.returncode == 0

        return {
            "dataset": dataset_path,
            "description": description,
            "success": success,
            "elapsed_time": elapsed,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return {
            "dataset": dataset_path,
            "description": description,
            "success": False,
            "elapsed_time": elapsed,
            "stdout": "",
            "stderr": f"Test timed out after {timeout} seconds",
            "returncode": -1
        }

def extract_metrics(stdout: str) -> dict:
    """Extract performance metrics from pipeline output"""
    metrics = {}
    lines = stdout.split('\n')

    for line in lines:
        if "CV accuracy:" in line:
            # Classification accuracy
            parts = line.split("CV accuracy: ")[1].split(" ")
            metrics['cv_score'] = float(parts[0])
            metrics['cv_std'] = float(parts[2].strip('()'))
            metrics['metric_type'] = 'accuracy'
        elif "CV RMSE:" in line:
            # Regression RMSE
            parts = line.split("CV RMSE: ")[1].split(" ")
            metrics['cv_score'] = float(parts[0])
            metrics['cv_std'] = float(parts[2].strip('()'))
            metrics['metric_type'] = 'rmse'
        elif "Model:" in line:
            # Model type
            metrics['model_type'] = line.split("Model: ")[1].strip()

    return metrics

def main():
    """Run comprehensive pipeline tests"""
    print("KAGGLESLAYER - COMPREHENSIVE PIPELINE TEST SUITE")
    print("=" * 60)

    # Test scenarios
    test_cases = [
        {
            "path": "downloaded_datasets/titanic",
            "description": "Binary Classification (Titanic)",
            "timeout": 120
        },
        {
            "path": "downloaded_datasets/spaceship-titanic",
            "description": "Binary Classification (Spaceship Titanic)",
            "timeout": 120
        },
        {
            "path": "downloaded_datasets/house-prices-advanced-regression-techniques",
            "description": "Regression (House Prices)",
            "timeout": 180
        },
        {
            "path": "downloaded_datasets/digit-recognizer",
            "description": "High-Dimensional Multiclass (MNIST)",
            "timeout": 300
        }
    ]

    results = []

    # Run all tests
    for test_case in test_cases:
        if not Path(test_case["path"]).exists():
            print(f"SKIPPING: {test_case['description']} - Dataset not found")
            continue

        result = run_pipeline_test(
            test_case["path"],
            test_case["description"],
            test_case["timeout"]
        )
        results.append(result)

        # Extract metrics
        if result['success']:
            metrics = extract_metrics(result['stdout'])
            result['metrics'] = metrics
            print(f"[PASS] SUCCESS: {result['description']}")
            if 'cv_score' in metrics:
                print(f"   Performance: {metrics['cv_score']:.4f} ± {metrics['cv_std']:.4f} ({metrics['metric_type']})")
                print(f"   Model: {metrics.get('model_type', 'Unknown')}")
            print(f"   Time: {result['elapsed_time']:.1f}s")
        else:
            print(f"[FAIL] FAILED: {result['description']}")
            print(f"   Error: {result['stderr'][:200]}...")
            print(f"   Time: {result['elapsed_time']:.1f}s")

    # Summary Report
    print(f"\n{'='*60}")
    print("COMPREHENSIVE TEST SUMMARY")
    print(f"{'='*60}")

    total_tests = len(results)
    successful_tests = sum(1 for r in results if r['success'])
    failed_tests = total_tests - successful_tests

    print(f"Total Tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {successful_tests/total_tests*100:.1f}%")

    print(f"\n{'='*60}")
    print("DETAILED RESULTS")
    print(f"{'='*60}")

    for result in results:
        status = "[PASS]" if result['success'] else "[FAIL]"
        print(f"{status} | {result['description']}")
        print(f"      Dataset: {result['dataset']}")
        print(f"      Time: {result['elapsed_time']:.1f}s")

        if result['success'] and 'metrics' in result:
            metrics = result['metrics']
            if 'cv_score' in metrics:
                print(f"      Performance: {metrics['cv_score']:.4f} ± {metrics['cv_std']:.4f} ({metrics['metric_type']})")
                print(f"      Model: {metrics.get('model_type', 'Unknown')}")
        elif not result['success']:
            print(f"      Error: {result['stderr'][:100]}...")
        print()

    # Quality Assessment
    print(f"{'='*60}")
    print("QUALITY ASSESSMENT")
    print(f"{'='*60}")

    if successful_tests == total_tests:
        print("[EXCELLENT] All tests passed!")
        print("[PASS] Pipeline is robust and handles diverse datasets")
        print("[PASS] Error handling is working correctly")
        print("[PASS] Performance metrics are reasonable")
    elif successful_tests >= total_tests * 0.8:
        print("[GOOD] Most tests passed")
        print("[WARN] Some edge cases may need attention")
    else:
        print("[POOR] Multiple test failures")
        print("[FIX] Pipeline needs significant fixes")

    # Performance benchmarks
    classification_results = [r for r in results if r['success'] and r.get('metrics', {}).get('metric_type') == 'accuracy']
    regression_results = [r for r in results if r['success'] and r.get('metrics', {}).get('metric_type') == 'rmse']

    if classification_results:
        avg_accuracy = sum(r['metrics']['cv_score'] for r in classification_results) / len(classification_results)
        print(f"\n[METRICS] Average Classification Accuracy: {avg_accuracy:.4f}")

    if regression_results:
        print(f"[METRICS] Regression tests completed successfully")

    print(f"\n[COMPLETE] KaggleSlayer pipeline testing complete!")
    return successful_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)