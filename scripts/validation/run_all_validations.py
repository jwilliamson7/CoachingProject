#!/usr/bin/env python
"""
Master Validation Runner for NFL Coach Tenure Prediction.

Runs all validation suites and produces a comprehensive report.

Usage:
    python scripts/validation/run_all_validations.py [--quick] [--verbose]

Options:
    --quick     Run only critical validations
    --verbose   Show detailed output
"""

import sys
import os
import argparse
import time
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from scripts.validation.validate_data import run_all_data_validations
from scripts.validation.validate_model import run_all_model_validations
from scripts.validation.validate_claims import run_all_claims_validations
from scripts.validation.validate_reproducibility import run_all_reproducibility_validations


def print_header():
    """Print validation header."""
    print("\n")
    print("=" * 80)
    print("NFL COACH TENURE PREDICTION - COMPREHENSIVE VALIDATION")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project root: {project_root}")
    print("=" * 80)
    print("\n")


def print_summary(all_results: dict, elapsed_time: float):
    """Print final summary."""
    print("\n")
    print("=" * 80)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 80)

    total_passed = 0
    total_failed = 0
    total_warnings = 0

    for suite_name, results in all_results.items():
        suite_passed = sum(1 for r in results.values() if r.passed)
        suite_failed = sum(1 for r in results.values() if not r.passed)
        suite_warnings = sum(
            len([m for m in r.messages if m.startswith("WARN")])
            for r in results.values()
        )

        total_passed += suite_passed
        total_failed += suite_failed
        total_warnings += suite_warnings

        status = "PASS" if suite_failed == 0 else "FAIL"
        print(f"  [{status}] {suite_name}: {suite_passed} passed, {suite_failed} failed, {suite_warnings} warnings")

    print("-" * 80)
    print(f"  TOTAL: {total_passed} passed, {total_failed} failed, {total_warnings} warnings")
    print(f"  Time elapsed: {elapsed_time:.2f} seconds")
    print("=" * 80)

    if total_failed == 0:
        print("\n  ALL VALIDATIONS PASSED")
    else:
        print(f"\n  {total_failed} VALIDATIONS FAILED - REVIEW REQUIRED")

    print("=" * 80)
    print("\n")

    return total_failed == 0


def run_critical_validations_only():
    """Run only the most critical validations (quick mode)."""
    print("\n")
    print("=" * 80)
    print("RUNNING CRITICAL VALIDATIONS ONLY (Quick Mode)")
    print("=" * 80)

    from scripts.validation.validate_data import (
        validate_dataset_size,
        validate_no_coach_overlap_train_test,
        validate_feature_ranges
    )
    from scripts.validation.validate_model import (
        test_probability_sum_to_one,
        test_probability_bounds,
        test_no_coach_leakage_in_cv
    )
    from scripts.validation.validate_claims import (
        validate_ordinal_metrics,
        validate_human_baseline
    )

    critical_tests = [
        ("Data: Dataset Size", validate_dataset_size),
        ("Data: No Coach Overlap", validate_no_coach_overlap_train_test),
        ("Data: Feature Ranges", validate_feature_ranges),
        ("Model: Probability Sum", test_probability_sum_to_one),
        ("Model: Probability Bounds", test_probability_bounds),
        ("Model: No CV Leakage", test_no_coach_leakage_in_cv),
        ("Claims: Ordinal Metrics", validate_ordinal_metrics),
        ("Claims: Human Baseline", validate_human_baseline),
    ]

    passed = 0
    failed = 0

    for name, test_func in critical_tests:
        try:
            result = test_func()
            status = "PASS" if result.passed else "FAIL"
            print(f"[{status}] {name}")

            if result.passed:
                passed += 1
            else:
                failed += 1
                for msg in result.messages:
                    if not msg.startswith("INFO"):
                        print(f"       {msg}")
        except Exception as e:
            print(f"[ERROR] {name}: {str(e)}")
            failed += 1

    print("-" * 80)
    print(f"Critical validations: {passed} passed, {failed} failed")
    print("=" * 80)

    return failed == 0


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive validations for NFL Coach Tenure Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--quick', '-q', action='store_true',
        help='Run only critical validations'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Show detailed output'
    )
    parser.add_argument(
        '--suite', '-s', type=str, choices=['data', 'model', 'claims', 'reproducibility'],
        help='Run only a specific validation suite'
    )

    args = parser.parse_args()

    start_time = time.time()

    if args.quick:
        all_passed = run_critical_validations_only()
        elapsed = time.time() - start_time
        print(f"\nQuick validation completed in {elapsed:.2f} seconds")
        sys.exit(0 if all_passed else 1)

    print_header()

    all_results = {}

    if args.suite:
        # Run specific suite
        suite_map = {
            'data': ('Data Pipeline', run_all_data_validations),
            'model': ('Model Correctness', run_all_model_validations),
            'claims': ('Statistical Claims', run_all_claims_validations),
            'reproducibility': ('Reproducibility', run_all_reproducibility_validations),
        }

        name, run_func = suite_map[args.suite]
        print(f"Running {name} validations only...\n")
        all_results[name] = run_func()

    else:
        # Run all suites
        validation_suites = [
            ("Data Pipeline", run_all_data_validations),
            ("Model Correctness", run_all_model_validations),
            ("Statistical Claims", run_all_claims_validations),
            ("Reproducibility", run_all_reproducibility_validations),
        ]

        for suite_name, run_func in validation_suites:
            print(f"\n{'#' * 80}")
            print(f"# Running: {suite_name}")
            print(f"{'#' * 80}\n")

            try:
                results = run_func()
                all_results[suite_name] = results
            except Exception as e:
                print(f"ERROR running {suite_name}: {str(e)}")
                import traceback
                traceback.print_exc()

    elapsed_time = time.time() - start_time
    all_passed = print_summary(all_results, elapsed_time)

    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
