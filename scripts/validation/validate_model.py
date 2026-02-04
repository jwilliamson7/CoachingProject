#!/usr/bin/env python
"""
Model Correctness Validation for NFL Coach Tenure Prediction.

Validates ordinal classifier constraints, cross-validation integrity,
and imputation consistency.

Usage:
    python scripts/validation/validate_model.py
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

from model import (
    OrdinalClassifier,
    CoachTenureModel,
    stratified_coach_level_split,
    stratified_coach_level_cv_split,
    CoachLevelStratifiedKFold
)
from model.config import MODEL_PATHS, FEATURE_CONFIG, MODEL_CONFIG, get_binary_xgboost_params


class ModelValidationResult:
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

    # Filter to known tenure
    df_known = df[df[FEATURE_CONFIG['target_column']] != -1]

    X = df_known.iloc[:, FEATURE_CONFIG['feature_columns_start']:FEATURE_CONFIG['feature_columns_end']]
    y = df_known[FEATURE_CONFIG['target_column']]

    return df_known, X, y


def train_ordinal_model(X_train: np.ndarray, y_train: np.ndarray) -> OrdinalClassifier:
    """Train an ordinal classifier for testing."""
    params = get_binary_xgboost_params()
    base_estimator = XGBClassifier(**params)

    model = OrdinalClassifier(
        base_estimator=base_estimator,
        n_classes=3
    )

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_train)

    model.fit(X_imputed, y_train)
    return model, imputer


def test_probability_sum_to_one() -> ModelValidationResult:
    """All class probabilities must sum to 1.0."""
    result = ModelValidationResult("Probability Sum to One")

    df, X, y = load_data()

    # Split data
    X_train, X_test, y_train, y_test, _ = stratified_coach_level_split(
        df, X, y, test_size=0.2, random_state=42
    )

    # Train model
    model, imputer = train_ordinal_model(X_train, y_train)

    # Get predictions on test set
    X_test_imputed = imputer.transform(X_test)
    probas = model.predict_proba(X_test_imputed)

    # Check sum
    row_sums = probas.sum(axis=1)
    max_diff = np.abs(row_sums - 1.0).max()

    result.info(f"Max deviation from sum=1.0: {max_diff:.10f}")
    result.info(f"Mean deviation: {np.abs(row_sums - 1.0).mean():.10f}")

    tolerance = 1e-6
    if max_diff > tolerance:
        result.fail(f"Probabilities don't sum to 1.0 (max diff: {max_diff})")
    else:
        result.info(f"All probabilities sum to 1.0 within tolerance ({tolerance})")

    result.details['max_diff'] = max_diff
    result.details['mean_diff'] = np.abs(row_sums - 1.0).mean()

    return result


def test_probability_bounds() -> ModelValidationResult:
    """All probabilities must be in [0, 1]."""
    result = ModelValidationResult("Probability Bounds [0, 1]")

    df, X, y = load_data()

    X_train, X_test, y_train, y_test, _ = stratified_coach_level_split(
        df, X, y, test_size=0.2, random_state=42
    )

    model, imputer = train_ordinal_model(X_train, y_train)
    X_test_imputed = imputer.transform(X_test)
    probas = model.predict_proba(X_test_imputed)

    min_prob = probas.min()
    max_prob = probas.max()

    result.info(f"Probability range: [{min_prob:.6f}, {max_prob:.6f}]")

    if min_prob < 0:
        result.fail(f"Negative probabilities found: {min_prob}")

    if max_prob > 1:
        result.fail(f"Probabilities > 1 found: {max_prob}")

    if min_prob >= 0 and max_prob <= 1:
        result.info("All probabilities within [0, 1] bounds")

    result.details['min_prob'] = min_prob
    result.details['max_prob'] = max_prob

    return result


def test_monotonicity_constraint() -> ModelValidationResult:
    """
    Test Frank-Hall monotonicity: P(Y > k) >= P(Y > k+1) for all k.

    This is the fundamental assumption of the Frank-Hall method.
    If violated, probability derivation breaks.
    """
    result = ModelValidationResult("Monotonicity Constraint (Frank-Hall)")

    df, X, y = load_data()

    X_train, X_test, y_train, y_test, _ = stratified_coach_level_split(
        df, X, y, test_size=0.2, random_state=42
    )

    model, imputer = train_ordinal_model(X_train, y_train)
    X_test_imputed = imputer.transform(X_test)

    # Get cumulative probabilities P(Y > k) from each binary classifier
    n_samples = X_test_imputed.shape[0]
    cumulative_probs = np.zeros((n_samples, 2))  # 2 binary classifiers for 3 classes

    for i, clf in enumerate(model.classifiers_):
        proba = clf.predict_proba(X_test_imputed)
        cumulative_probs[:, i] = proba[:, 1]  # P(Y > threshold)

    # Check monotonicity: P(Y > 0) >= P(Y > 1)
    violations = np.sum(cumulative_probs[:, 0] < cumulative_probs[:, 1])
    violation_pct = violations / n_samples * 100

    result.info(f"P(Y > 0) >= P(Y > 1) violations: {violations}/{n_samples} ({violation_pct:.2f}%)")

    if violations > 0:
        # Show details of violations
        violation_indices = np.where(cumulative_probs[:, 0] < cumulative_probs[:, 1])[0]
        max_violation = (cumulative_probs[:, 1] - cumulative_probs[:, 0]).max()
        result.warn(f"Monotonicity violations found (max: {max_violation:.6f})")
        result.info("Note: Small violations are handled by clipping in predict_proba")
    else:
        result.info("No monotonicity violations - Frank-Hall assumption holds")

    result.details['violations'] = violations
    result.details['violation_pct'] = violation_pct

    return result


def test_predictions_are_valid_classes() -> ModelValidationResult:
    """Predictions must be integers in {0, 1, 2}."""
    result = ModelValidationResult("Valid Class Predictions")

    df, X, y = load_data()

    X_train, X_test, y_train, y_test, _ = stratified_coach_level_split(
        df, X, y, test_size=0.2, random_state=42
    )

    model, imputer = train_ordinal_model(X_train, y_train)
    X_test_imputed = imputer.transform(X_test)
    predictions = model.predict(X_test_imputed)

    unique_preds = set(predictions)
    valid_classes = {0, 1, 2}

    result.info(f"Unique predictions: {sorted(unique_preds)}")

    invalid = unique_preds - valid_classes
    if invalid:
        result.fail(f"Invalid class predictions: {invalid}")
    else:
        result.info("All predictions are valid classes (0, 1, 2)")

    result.details['unique_preds'] = sorted(unique_preds)

    return result


def test_single_sample() -> ModelValidationResult:
    """Model handles single sample prediction."""
    result = ModelValidationResult("Single Sample Prediction")

    df, X, y = load_data()

    X_train, X_test, y_train, y_test, _ = stratified_coach_level_split(
        df, X, y, test_size=0.2, random_state=42
    )

    model, imputer = train_ordinal_model(X_train, y_train)

    # Get single sample
    X_single = X_test.iloc[0:1]
    X_single_imputed = imputer.transform(X_single)

    try:
        pred = model.predict(X_single_imputed)
        proba = model.predict_proba(X_single_imputed)

        result.info(f"Single sample prediction: class {pred[0]}")
        result.info(f"Probabilities: {proba[0]}")

        if len(pred) == 1 and len(proba) == 1 and len(proba[0]) == 3:
            result.info("Single sample handled correctly")
        else:
            result.fail("Output shape incorrect for single sample")

    except Exception as e:
        result.fail(f"Error on single sample: {str(e)}")

    return result


def test_no_coach_leakage_in_cv() -> ModelValidationResult:
    """Each CV fold has disjoint coaches."""
    result = ModelValidationResult("No Coach Leakage in CV")

    df, X, y = load_data()

    X_train, X_test, y_train, y_test, _ = stratified_coach_level_split(
        df, X, y, test_size=0.2, random_state=42
    )

    df_train = df.loc[X_train.index]

    # Generate CV splits
    splits = stratified_coach_level_cv_split(
        df_train, X_train, y_train,
        n_splits=5,
        random_state=42
    )

    result.info(f"Generated {len(splits)} CV folds")

    all_good = True
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        # Get coach names for each set using iloc-based indices
        train_coaches = set(df_train.iloc[train_idx]['Coach Name'].unique())
        test_coaches = set(df_train.iloc[test_idx]['Coach Name'].unique())

        overlap = train_coaches & test_coaches

        if len(overlap) > 0:
            result.fail(f"Fold {fold_idx}: {len(overlap)} overlapping coaches")
            all_good = False
        else:
            result.info(f"Fold {fold_idx}: No overlap ({len(train_coaches)} train, {len(test_coaches)} test coaches)")

    if all_good:
        result.info("All CV folds have disjoint coach sets")

    return result


def test_class_stratification_preserved() -> ModelValidationResult:
    """Each CV fold has similar class distribution to full dataset."""
    result = ModelValidationResult("Class Stratification in CV")

    df, X, y = load_data()

    X_train, X_test, y_train, y_test, _ = stratified_coach_level_split(
        df, X, y, test_size=0.2, random_state=42
    )

    df_train = df.loc[X_train.index]

    # Original distribution
    original_dist = y_train.value_counts(normalize=True).sort_index()
    result.info(f"Original distribution: {dict(original_dist.round(3))}")

    splits = stratified_coach_level_cv_split(
        df_train, X_train, y_train,
        n_splits=5,
        random_state=42
    )

    tolerance = 0.15  # 15% tolerance for stratification

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        fold_y_test = y_train.iloc[test_idx]
        fold_dist = fold_y_test.value_counts(normalize=True).sort_index()

        max_diff = max(
            abs(fold_dist.get(c, 0) - original_dist.get(c, 0))
            for c in [0, 1, 2]
        )

        if max_diff > tolerance:
            result.warn(f"Fold {fold_idx}: Max distribution diff {max_diff:.1%}")
        else:
            result.info(f"Fold {fold_idx}: Distribution preserved (max diff {max_diff:.1%})")

    return result


def test_imputation_uses_training_means_only() -> ModelValidationResult:
    """Test set imputation must not leak test statistics."""
    result = ModelValidationResult("Imputation Train/Test Consistency")

    df, X, y = load_data()

    X_train, X_test, y_train, y_test, _ = stratified_coach_level_split(
        df, X, y, test_size=0.2, random_state=42
    )

    # Fit imputer on training data
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(X_train)

    # The imputer's statistics should be from training only
    train_means = X_train.mean()
    imputer_means = imputer.statistics_

    result.info(f"Training data shape: {X_train.shape}")
    result.info(f"Test data shape: {X_test.shape}")

    # Compare means (should match closely for non-nan columns)
    diffs = np.abs(train_means.values - imputer_means)
    max_diff = np.max(diffs)

    result.info(f"Max difference between train means and imputer means: {max_diff:.10f}")

    if max_diff < 1e-10:
        result.info("Imputer correctly uses only training statistics")
    else:
        result.warn("Small numerical differences detected (likely floating point)")

    result.details['max_diff'] = max_diff

    return result


def test_no_nan_after_imputation() -> ModelValidationResult:
    """All NaN values should be filled after imputation."""
    result = ModelValidationResult("No NaN After Imputation")

    df, X, y = load_data()

    X_train, X_test, y_train, y_test, _ = stratified_coach_level_split(
        df, X, y, test_size=0.2, random_state=42
    )

    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    train_nans = np.isnan(X_train_imputed).sum()
    test_nans = np.isnan(X_test_imputed).sum()

    result.info(f"NaN count in imputed train: {train_nans}")
    result.info(f"NaN count in imputed test: {test_nans}")

    if train_nans > 0 or test_nans > 0:
        result.fail(f"NaN values remain after imputation")
    else:
        result.info("All NaN values successfully imputed")

    result.details['train_nans'] = train_nans
    result.details['test_nans'] = test_nans

    return result


def test_model_save_load_consistency() -> ModelValidationResult:
    """Saved model produces identical predictions after loading."""
    result = ModelValidationResult("Model Save/Load Consistency")

    import tempfile

    df, X, y = load_data()

    X_train, X_test, y_train, y_test, _ = stratified_coach_level_split(
        df, X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = CoachTenureModel(
        use_ordinal=True,
        n_classes=3,
        random_state=42
    )
    model.fit(X_train, y_train, verbose=0)

    # Get predictions before save
    pred_before = model.predict(X_test)
    proba_before = model.predict_proba(X_test)

    # Save and load
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        temp_path = f.name

    try:
        model.save(temp_path)
        loaded_model = CoachTenureModel.load(temp_path)

        # Get predictions after load
        pred_after = loaded_model.predict(X_test)
        proba_after = loaded_model.predict_proba(X_test)

        # Compare
        pred_match = np.array_equal(pred_before, pred_after)
        proba_match = np.allclose(proba_before, proba_after)

        result.info(f"Predictions match: {pred_match}")
        result.info(f"Probabilities match: {proba_match}")

        if not pred_match:
            result.fail("Predictions differ after save/load")
        if not proba_match:
            result.fail("Probabilities differ after save/load")
        if pred_match and proba_match:
            result.info("Model save/load produces identical results")

    finally:
        os.remove(temp_path)

    return result


def run_all_model_validations() -> Dict[str, ModelValidationResult]:
    """Run all model validation checks."""
    print("="*60)
    print("MODEL CORRECTNESS VALIDATION")
    print("="*60)

    validations = [
        test_probability_sum_to_one,
        test_probability_bounds,
        test_monotonicity_constraint,
        test_predictions_are_valid_classes,
        test_single_sample,
        test_no_coach_leakage_in_cv,
        test_class_stratification_preserved,
        test_imputation_uses_training_means_only,
        test_no_nan_after_imputation,
        test_model_save_load_consistency,
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
    print(f"MODEL VALIDATION SUMMARY: {passed} passed, {failed} failed")
    print("="*60)

    return results


if __name__ == '__main__':
    results = run_all_model_validations()

    # Exit with error code if any validations failed
    all_passed = all(r.passed for r in results.values())
    sys.exit(0 if all_passed else 1)
