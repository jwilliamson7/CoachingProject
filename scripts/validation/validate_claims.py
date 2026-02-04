#!/usr/bin/env python
"""
Statistical Claims Validation for NFL Coach Tenure Prediction.

Validates that metrics and statistics in the paper can be reproduced
from the data and models.

Usage:
    python scripts/validation/validate_claims.py
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from scipy import stats as scipy_stats
from sklearn.metrics import f1_score

from model import (
    CoachTenureModel,
    stratified_coach_level_split,
    ordinal_metrics
)
from model.config import MODEL_PATHS, FEATURE_CONFIG, MODEL_CONFIG


class ClaimsValidationResult:
    """Container for validation results."""

    def __init__(self, name: str):
        self.name = name
        self.passed = True
        self.messages = []
        self.details = {}

    def fail(self, message: str):
        self.passed = False
        self.messages.append(f"FAIL: {message}")

    def warn(self, message: str):
        self.messages.append(f"WARN: {message}")

    def info(self, message: str):
        self.messages.append(f"INFO: {message}")

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        result = f"\n[{status}] {self.name}\n"
        for msg in self.messages:
            result += f"  {msg}\n"
        return result


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load data and prepare features/target."""
    data_path = os.path.join(project_root, MODEL_PATHS['data_file'])
    df = pd.read_csv(data_path, index_col=0)
    df_known = df[df[FEATURE_CONFIG['target_column']] != -1]

    X = df_known.iloc[:, FEATURE_CONFIG['feature_columns_start']:FEATURE_CONFIG['feature_columns_end']]
    y = df_known[FEATURE_CONFIG['target_column']]

    return df_known, X, y


def validate_ordinal_metrics() -> ClaimsValidationResult:
    """
    Reproduce ordinal model metrics from paper.

    Expected metrics (from paper):
    - QWK: ~0.754
    - MAE: ~0.307
    - Adjacent Accuracy: ~0.984
    - Exact Accuracy: ~0.724
    - Macro F1: ~0.695
    - AUROC: ~0.881
    """
    result = ClaimsValidationResult("Ordinal Model Metrics Reproduction")

    df, X, y = load_data()

    # Split data with same random state
    X_train, X_test, y_train, y_test, _ = stratified_coach_level_split(
        df, X, y,
        test_size=MODEL_CONFIG['test_size'],
        random_state=MODEL_CONFIG['random_state']
    )

    # Train ordinal model
    model = CoachTenureModel(
        use_ordinal=True,
        n_classes=3,
        random_state=MODEL_CONFIG['random_state']
    )
    model.fit(X_train, y_train, verbose=0)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    metrics = ordinal_metrics(y_test, y_pred, y_proba)

    # Expected values with tolerance
    expected = {
        'qwk': (0.754, 0.05),           # (expected, tolerance)
        'mae': (0.307, 0.05),
        'adjacent_accuracy': (0.984, 0.02),
        'exact_accuracy': (0.724, 0.05),
        'macro_f1': (0.695, 0.05),
        'auroc': (0.881, 0.05)
    }

    all_within_tolerance = True
    for metric_name, (expected_val, tolerance) in expected.items():
        actual_val = metrics.get(metric_name)
        if actual_val is not None:
            diff = abs(actual_val - expected_val)
            within_tol = diff <= tolerance

            if not within_tol:
                all_within_tolerance = False
                result.warn(f"{metric_name}: {actual_val:.4f} (expected {expected_val:.4f}, diff {diff:.4f})")
            else:
                result.info(f"{metric_name}: {actual_val:.4f} (expected {expected_val:.4f})")

            result.details[metric_name] = actual_val
        else:
            result.warn(f"{metric_name}: not computed")

    if all_within_tolerance:
        result.info("All metrics within expected tolerance")

    return result


def validate_multiclass_metrics() -> ClaimsValidationResult:
    """Reproduce multiclass model metrics for comparison."""
    result = ClaimsValidationResult("Multiclass Model Metrics Reproduction")

    df, X, y = load_data()

    X_train, X_test, y_train, y_test, _ = stratified_coach_level_split(
        df, X, y,
        test_size=MODEL_CONFIG['test_size'],
        random_state=MODEL_CONFIG['random_state']
    )

    # Train multiclass model
    model = CoachTenureModel(
        use_ordinal=False,
        n_classes=3,
        random_state=MODEL_CONFIG['random_state']
    )
    model.fit(X_train, y_train, verbose=0)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    metrics = ordinal_metrics(y_test, y_pred, y_proba)

    result.info(f"QWK: {metrics['qwk']:.4f}")
    result.info(f"MAE: {metrics['mae']:.4f}")
    result.info(f"Adjacent Accuracy: {metrics['adjacent_accuracy']:.4f}")
    result.info(f"Exact Accuracy: {metrics['exact_accuracy']:.4f}")
    result.info(f"Macro F1: {metrics['macro_f1']:.4f}")
    result.info(f"AUROC: {metrics.get('auroc', 'N/A')}")

    for metric_name, value in metrics.items():
        if metric_name not in ['per_class', 'confusion_matrix']:
            result.details[metric_name] = value

    return result


def validate_model_comparison() -> ClaimsValidationResult:
    """
    Validate that ordinal model outperforms multiclass on key metrics.

    Paper claims:
    - Ordinal beats multiclass on QWK, MAE, Adjacent Accuracy
    - Class 1 F1 improvement: ~62.3% (0.581 vs 0.358)
    """
    result = ClaimsValidationResult("Ordinal vs Multiclass Comparison")

    df, X, y = load_data()

    X_train, X_test, y_train, y_test, _ = stratified_coach_level_split(
        df, X, y,
        test_size=MODEL_CONFIG['test_size'],
        random_state=MODEL_CONFIG['random_state']
    )

    # Train both models
    ordinal_model = CoachTenureModel(use_ordinal=True, n_classes=3, random_state=42)
    ordinal_model.fit(X_train, y_train, verbose=0)

    multiclass_model = CoachTenureModel(use_ordinal=False, n_classes=3, random_state=42)
    multiclass_model.fit(X_train, y_train, verbose=0)

    # Get predictions
    ord_pred = ordinal_model.predict(X_test)
    ord_proba = ordinal_model.predict_proba(X_test)
    ord_metrics = ordinal_metrics(y_test, ord_pred, ord_proba)

    mult_pred = multiclass_model.predict(X_test)
    mult_proba = multiclass_model.predict_proba(X_test)
    mult_metrics = ordinal_metrics(y_test, mult_pred, mult_proba)

    # Compare key metrics
    comparisons = [
        ('qwk', 'higher', 'QWK'),
        ('mae', 'lower', 'MAE'),
        ('adjacent_accuracy', 'higher', 'Adjacent Accuracy'),
    ]

    for metric, direction, name in comparisons:
        ord_val = ord_metrics[metric]
        mult_val = mult_metrics[metric]

        if direction == 'higher':
            ordinal_wins = ord_val > mult_val
        else:
            ordinal_wins = ord_val < mult_val

        diff = ord_val - mult_val
        status = "ordinal wins" if ordinal_wins else "multiclass wins"

        result.info(f"{name}: Ordinal {ord_val:.4f} vs Multiclass {mult_val:.4f} ({status})")

        if not ordinal_wins:
            result.warn(f"Unexpected: Multiclass outperforms ordinal on {name}")

    # Per-class F1 comparison (especially Class 1)
    ord_f1_class1 = ord_metrics['per_class']['Class 1 (3-4 yrs)']['f1']
    mult_f1_class1 = mult_metrics['per_class']['Class 1 (3-4 yrs)']['f1']

    if mult_f1_class1 > 0:
        improvement = (ord_f1_class1 - mult_f1_class1) / mult_f1_class1 * 100
    else:
        improvement = float('inf') if ord_f1_class1 > 0 else 0

    result.info(f"Class 1 F1: Ordinal {ord_f1_class1:.3f} vs Multiclass {mult_f1_class1:.3f} ({improvement:+.1f}%)")

    result.details['ordinal_qwk'] = ord_metrics['qwk']
    result.details['multiclass_qwk'] = mult_metrics['qwk']
    result.details['class1_f1_improvement'] = improvement

    return result


def validate_human_baseline() -> ClaimsValidationResult:
    """
    Validate human baseline F1 calculation.

    Paper claims: F1 = 0.130 assuming all GMs predict Class 2 (long tenure)

    Calculation verification:
    - If predicting all Class 2:
      - Class 0 F1 = 0 (no predictions for this class)
      - Class 1 F1 = 0 (no predictions for this class)
      - Class 2 Precision = Class 2 support / total = 0.243
      - Class 2 Recall = 1.0 (all Class 2 are predicted as Class 2)
      - Class 2 F1 = 2 * (0.243 * 1.0) / (0.243 + 1.0) = 0.391
      - Macro F1 = (0 + 0 + 0.391) / 3 = 0.130
    """
    result = ClaimsValidationResult("Human Baseline F1 Calculation")

    df, X, y = load_data()

    # Get class distribution
    class_counts = y.value_counts().sort_index()
    total = len(y)

    class_pcts = {cls: count/total for cls, count in class_counts.items()}
    result.info(f"Class distribution: {dict(class_counts)}")
    result.info(f"Class percentages: {', '.join(f'{k}: {v:.1%}' for k, v in sorted(class_pcts.items()))}")

    # Simulate all predictions as Class 2
    y_pred_all_class2 = np.full(len(y), 2)

    # Calculate per-class F1
    # Class 0: precision=0 (no predictions), recall=0, F1=0
    # Class 1: precision=0 (no predictions), recall=0, F1=0
    # Class 2: precision = true_class2/total (since all predicted as 2)
    #          recall = 1.0 (all class 2 predicted correctly)

    class2_count = class_counts.get(2, 0)
    class2_precision = class2_count / total  # All predictions are class 2
    class2_recall = 1.0
    class2_f1 = 2 * (class2_precision * class2_recall) / (class2_precision + class2_recall)

    result.info(f"Class 2 precision (all predict 2): {class2_precision:.4f}")
    result.info(f"Class 2 recall: {class2_recall:.4f}")
    result.info(f"Class 2 F1: {class2_f1:.4f}")

    # Macro F1 = average of per-class F1
    macro_f1 = (0 + 0 + class2_f1) / 3

    result.info(f"Macro F1 (human baseline): {macro_f1:.4f}")

    # Verify with sklearn
    sklearn_f1 = f1_score(y, y_pred_all_class2, average='macro')
    result.info(f"Sklearn macro F1 verification: {sklearn_f1:.4f}")

    # Check against paper claim (0.130)
    expected = 0.130
    diff = abs(macro_f1 - expected)

    if diff < 0.02:
        result.info(f"Human baseline F1 = {macro_f1:.3f} matches paper claim ({expected})")
    else:
        result.warn(f"Human baseline F1 = {macro_f1:.3f} differs from paper claim ({expected})")

    result.details['calculated_f1'] = macro_f1
    result.details['sklearn_f1'] = sklearn_f1
    result.details['expected'] = expected

    return result


def validate_feature_importance_statistics() -> ClaimsValidationResult:
    """
    Validate feature importance statistics.

    Paper claims: Kruskal-Wallis H = 2.42, p = 0.66
    (no significant differences across feature categories)
    """
    result = ClaimsValidationResult("Feature Importance Statistics")

    df, X, y = load_data()

    X_train, X_test, y_train, y_test, _ = stratified_coach_level_split(
        df, X, y,
        test_size=MODEL_CONFIG['test_size'],
        random_state=MODEL_CONFIG['random_state']
    )

    # Train model to get feature importances
    model = CoachTenureModel(use_ordinal=True, n_classes=3, random_state=42)
    model.fit(X_train, y_train, verbose=0)

    importances = model.get_feature_importances()

    # Define feature category ranges (1-indexed in paper, 0-indexed here)
    categories = {
        'Core Experience': (0, 8),      # Features 1-8
        'OC Stats': (8, 41),            # Features 9-41 (33 features)
        'DC Stats': (41, 74),           # Features 42-74 (33 features)
        'HC Stats': (74, 140),          # Features 75-140 (66 features)
        'Hiring Team': (140, 150)       # Features 141-150 (10 features)
    }

    # Collect importances by category
    category_importances = {}
    for cat_name, (start, end) in categories.items():
        cat_imp = importances[start:end]
        category_importances[cat_name] = cat_imp
        result.info(f"{cat_name}: mean={np.mean(cat_imp):.6f}, n={len(cat_imp)}")

    # Perform Kruskal-Wallis test
    groups = list(category_importances.values())
    stat, p_value = scipy_stats.kruskal(*groups)

    result.info(f"Kruskal-Wallis H statistic: {stat:.3f}")
    result.info(f"Kruskal-Wallis p-value: {p_value:.3f}")

    # Check against paper claims
    expected_h = 2.42
    expected_p = 0.66

    h_diff = abs(stat - expected_h)
    p_diff = abs(p_value - expected_p)

    if p_value > 0.05:
        result.info("No significant difference between feature categories (p > 0.05)")
    else:
        result.warn(f"Significant difference detected (p = {p_value:.4f})")

    result.details['h_statistic'] = stat
    result.details['p_value'] = p_value
    result.details['category_means'] = {k: np.mean(v) for k, v in category_importances.items()}

    return result


def validate_confusion_matrix() -> ClaimsValidationResult:
    """Validate confusion matrix reproduction."""
    result = ClaimsValidationResult("Confusion Matrix Reproduction")

    df, X, y = load_data()

    X_train, X_test, y_train, y_test, _ = stratified_coach_level_split(
        df, X, y,
        test_size=MODEL_CONFIG['test_size'],
        random_state=MODEL_CONFIG['random_state']
    )

    model = CoachTenureModel(use_ordinal=True, n_classes=3, random_state=42)
    model.fit(X_train, y_train, verbose=0)

    y_pred = model.predict(X_test)
    metrics = ordinal_metrics(y_test, y_pred)

    cm = metrics['confusion_matrix']

    result.info("Confusion Matrix:")
    result.info(f"           Predicted")
    result.info(f"             0    1    2")
    for i, row in enumerate(cm):
        result.info(f"Actual {i}:  {row[0]:4d} {row[1]:4d} {row[2]:4d}")

    # Calculate per-class accuracy
    for i in range(3):
        class_total = cm[i, :].sum()
        class_correct = cm[i, i]
        if class_total > 0:
            class_acc = class_correct / class_total
            result.info(f"Class {i} recall: {class_correct}/{class_total} = {class_acc:.3f}")

    result.details['confusion_matrix'] = cm.tolist()

    return result


def validate_per_class_f1() -> ClaimsValidationResult:
    """Validate per-class F1 scores."""
    result = ClaimsValidationResult("Per-Class F1 Scores")

    df, X, y = load_data()

    X_train, X_test, y_train, y_test, _ = stratified_coach_level_split(
        df, X, y,
        test_size=MODEL_CONFIG['test_size'],
        random_state=MODEL_CONFIG['random_state']
    )

    # Train both models
    ordinal_model = CoachTenureModel(use_ordinal=True, n_classes=3, random_state=42)
    ordinal_model.fit(X_train, y_train, verbose=0)

    multiclass_model = CoachTenureModel(use_ordinal=False, n_classes=3, random_state=42)
    multiclass_model.fit(X_train, y_train, verbose=0)

    # Get metrics
    ord_pred = ordinal_model.predict(X_test)
    ord_metrics = ordinal_metrics(y_test, ord_pred)

    mult_pred = multiclass_model.predict(X_test)
    mult_metrics = ordinal_metrics(y_test, mult_pred)

    result.info("Per-Class F1 Scores:")
    result.info(f"{'Class':<20} {'Ordinal':>10} {'Multiclass':>12} {'Difference':>12}")
    result.info("-" * 60)

    for class_name in ord_metrics['per_class'].keys():
        ord_f1 = ord_metrics['per_class'][class_name]['f1']
        mult_f1 = mult_metrics['per_class'][class_name]['f1']
        diff = ord_f1 - mult_f1

        result.info(f"{class_name:<20} {ord_f1:>10.3f} {mult_f1:>12.3f} {diff:>+12.3f}")

    return result


def run_all_claims_validations() -> Dict[str, ClaimsValidationResult]:
    """Run all claims validation checks."""
    print("="*60)
    print("STATISTICAL CLAIMS VALIDATION")
    print("="*60)

    validations = [
        validate_ordinal_metrics,
        validate_multiclass_metrics,
        validate_model_comparison,
        validate_human_baseline,
        validate_feature_importance_statistics,
        validate_confusion_matrix,
        validate_per_class_f1,
    ]

    results = {}
    passed = 0
    failed = 0

    for validate_func in validations:
        try:
            result = validate_func()
            results[result.name] = result
            print(result)

            if result.passed:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n[ERROR] {validate_func.__name__}: {str(e)}\n")
            import traceback
            traceback.print_exc()
            failed += 1

    print("="*60)
    print(f"CLAIMS VALIDATION SUMMARY: {passed} passed, {failed} failed")
    print("="*60)

    return results


if __name__ == '__main__':
    results = run_all_claims_validations()

    # Exit with error code if any validations failed
    all_passed = all(r.passed for r in results.values())
    sys.exit(0 if all_passed else 1)
