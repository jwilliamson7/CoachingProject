#!/usr/bin/env python
"""
Reproducibility Validation for NFL Coach Tenure Prediction.

Validates end-to-end pipeline reproducibility and prediction consistency.

Usage:
    python scripts/validation/validate_reproducibility.py
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import tempfile

from model import (
    CoachTenureModel,
    stratified_coach_level_split,
    ordinal_metrics
)
from model.config import MODEL_PATHS, FEATURE_CONFIG, MODEL_CONFIG


class ReproducibilityResult:
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


def load_data() -> pd.DataFrame:
    """Load the full dataset."""
    data_path = os.path.join(project_root, MODEL_PATHS['data_file'])
    return pd.read_csv(data_path, index_col=0)


def reproduce_training_pipeline() -> ReproducibilityResult:
    """
    Reproduce the full training pipeline.

    Steps:
    1. Load svd_imputed_master_data.csv
    2. Filter tenure_class != -1
    3. Run stratified_coach_level_split (random_state=42)
    4. Train ordinal model with optimized params
    5. Evaluate on test set
    """
    result = ReproducibilityResult("Training Pipeline Reproduction")

    # Step 1: Load data
    data_path = os.path.join(project_root, MODEL_PATHS['data_file'])
    df = pd.read_csv(data_path, index_col=0)
    result.info(f"Step 1: Loaded {len(df)} total rows from CSV")

    # Step 2: Filter to known tenure
    df_known = df[df[FEATURE_CONFIG['target_column']] != -1]
    result.info(f"Step 2: Filtered to {len(df_known)} instances with known tenure")

    # Prepare features
    X = df_known.iloc[:, FEATURE_CONFIG['feature_columns_start']:FEATURE_CONFIG['feature_columns_end']]
    y = df_known[FEATURE_CONFIG['target_column']]

    result.info(f"Features shape: {X.shape}")
    result.info(f"Target distribution: {dict(y.value_counts().sort_index())}")

    # Step 3: Train/test split
    X_train, X_test, y_train, y_test, test_coaches = stratified_coach_level_split(
        df_known, X, y,
        test_size=MODEL_CONFIG['test_size'],
        random_state=MODEL_CONFIG['random_state']
    )

    result.info(f"Step 3: Split - Train {len(X_train)}, Test {len(X_test)}")

    # Step 4: Train model
    model = CoachTenureModel(
        use_ordinal=True,
        n_classes=3,
        random_state=MODEL_CONFIG['random_state']
    )
    model.fit(X_train, y_train, verbose=0)
    result.info("Step 4: Model trained successfully")

    # Step 5: Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    metrics = ordinal_metrics(y_test, y_pred, y_proba)

    result.info(f"Step 5: Evaluation complete")
    result.info(f"  QWK: {metrics['qwk']:.4f}")
    result.info(f"  MAE: {metrics['mae']:.4f}")
    result.info(f"  Accuracy: {metrics['exact_accuracy']:.4f}")

    result.details['train_size'] = len(X_train)
    result.details['test_size'] = len(X_test)
    result.details['metrics'] = {k: v for k, v in metrics.items()
                                  if k not in ['per_class', 'confusion_matrix']}

    return result


def reproduce_predictions() -> ReproducibilityResult:
    """
    Reproduce predictions for recent hires.

    Validates specific predictions match expected values.
    """
    result = ReproducibilityResult("Prediction Reproduction")

    # Load full data including recent hires
    df = load_data()

    # Get known tenure for training
    df_known = df[df[FEATURE_CONFIG['target_column']] != -1]
    X_known = df_known.iloc[:, FEATURE_CONFIG['feature_columns_start']:FEATURE_CONFIG['feature_columns_end']]
    y_known = df_known[FEATURE_CONFIG['target_column']]

    # Get recent hires (class -1)
    df_recent = df[df[FEATURE_CONFIG['target_column']] == -1]

    if len(df_recent) == 0:
        result.info("No recent hires found (tenure_class == -1)")
        return result

    result.info(f"Found {len(df_recent)} recent hires")

    # Train model on all known data
    model = CoachTenureModel(use_ordinal=True, n_classes=3, random_state=42)
    model.fit(X_known, y_known, verbose=0)

    # Prepare features for recent hires
    X_recent = df_recent.iloc[:, FEATURE_CONFIG['feature_columns_start']:FEATURE_CONFIG['feature_columns_end']]

    # Get predictions
    predictions = model.predict(X_recent)
    probas = model.predict_proba(X_recent)

    # Show predictions for recent hires
    result.info("\nRecent hire predictions:")
    result.info(f"{'Coach Name':<25} {'Year':>4} {'Pred':>4} {'P(0)':>6} {'P(1)':>6} {'P(2)':>6}")
    result.info("-" * 60)

    for i, idx in enumerate(df_recent.index):
        coach = df_recent.loc[idx, 'Coach Name']
        year = df_recent.loc[idx, 'Year']
        pred = predictions[i]
        p0, p1, p2 = probas[i]

        result.info(f"{coach:<25} {year:>4} {pred:>4} {p0:>6.3f} {p1:>6.3f} {p2:>6.3f}")

    result.details['num_recent'] = len(df_recent)
    result.details['prediction_distribution'] = {
        int(c): int(count) for c, count in
        pd.Series(predictions).value_counts().sort_index().items()
    }

    return result


def test_deterministic_training() -> ReproducibilityResult:
    """Verify training is deterministic with same random_state."""
    result = ReproducibilityResult("Deterministic Training")

    df = load_data()
    df_known = df[df[FEATURE_CONFIG['target_column']] != -1]

    X = df_known.iloc[:, FEATURE_CONFIG['feature_columns_start']:FEATURE_CONFIG['feature_columns_end']]
    y = df_known[FEATURE_CONFIG['target_column']]

    # Train model twice with same random state
    model1 = CoachTenureModel(use_ordinal=True, n_classes=3, random_state=42)
    model1.fit(X, y, verbose=0)
    pred1 = model1.predict(X)
    proba1 = model1.predict_proba(X)

    model2 = CoachTenureModel(use_ordinal=True, n_classes=3, random_state=42)
    model2.fit(X, y, verbose=0)
    pred2 = model2.predict(X)
    proba2 = model2.predict_proba(X)

    # Compare
    pred_match = np.array_equal(pred1, pred2)
    proba_match = np.allclose(proba1, proba2)

    result.info(f"Predictions match: {pred_match}")
    result.info(f"Probabilities match: {proba_match}")

    if not pred_match:
        diff_count = np.sum(pred1 != pred2)
        result.fail(f"Predictions differ in {diff_count} instances")

    if not proba_match:
        max_diff = np.abs(proba1 - proba2).max()
        result.fail(f"Probabilities differ (max diff: {max_diff})")

    if pred_match and proba_match:
        result.info("Training is deterministic with same random_state")

    return result


def test_split_determinism() -> ReproducibilityResult:
    """Verify train/test split is deterministic."""
    result = ReproducibilityResult("Split Determinism")

    df = load_data()
    df_known = df[df[FEATURE_CONFIG['target_column']] != -1]

    X = df_known.iloc[:, FEATURE_CONFIG['feature_columns_start']:FEATURE_CONFIG['feature_columns_end']]
    y = df_known[FEATURE_CONFIG['target_column']]

    # Split twice with same random state
    X_train1, X_test1, y_train1, y_test1, coaches1 = stratified_coach_level_split(
        df_known, X, y, test_size=0.2, random_state=42
    )

    X_train2, X_test2, y_train2, y_test2, coaches2 = stratified_coach_level_split(
        df_known, X, y, test_size=0.2, random_state=42
    )

    # Compare
    train_idx_match = set(X_train1.index) == set(X_train2.index)
    test_idx_match = set(X_test1.index) == set(X_test2.index)
    coaches_match = set(coaches1) == set(coaches2)

    result.info(f"Train indices match: {train_idx_match}")
    result.info(f"Test indices match: {test_idx_match}")
    result.info(f"Test coaches match: {coaches_match}")

    if train_idx_match and test_idx_match and coaches_match:
        result.info("Split is deterministic with same random_state")
    else:
        result.fail("Split is not deterministic")

    # Now try with different random state
    X_train3, X_test3, _, _, coaches3 = stratified_coach_level_split(
        df_known, X, y, test_size=0.2, random_state=123
    )

    different_with_new_seed = set(coaches1) != set(coaches3)
    result.info(f"Different split with new seed: {different_with_new_seed}")

    return result


def test_cross_run_consistency() -> ReproducibilityResult:
    """
    Test that metrics are consistent across independent runs.

    This tests the full pipeline reproducibility.
    """
    result = ReproducibilityResult("Cross-Run Consistency")

    df = load_data()
    df_known = df[df[FEATURE_CONFIG['target_column']] != -1]

    X = df_known.iloc[:, FEATURE_CONFIG['feature_columns_start']:FEATURE_CONFIG['feature_columns_end']]
    y = df_known[FEATURE_CONFIG['target_column']]

    metrics_runs = []

    # Run pipeline 3 times
    for run in range(3):
        X_train, X_test, y_train, y_test, _ = stratified_coach_level_split(
            df_known, X, y, test_size=0.2, random_state=42
        )

        model = CoachTenureModel(use_ordinal=True, n_classes=3, random_state=42)
        model.fit(X_train, y_train, verbose=0)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        metrics = ordinal_metrics(y_test, y_pred, y_proba)

        metrics_runs.append(metrics)

    # Compare metrics across runs
    qwk_values = [m['qwk'] for m in metrics_runs]
    mae_values = [m['mae'] for m in metrics_runs]

    qwk_consistent = len(set(qwk_values)) == 1
    mae_consistent = len(set(mae_values)) == 1

    result.info(f"QWK across runs: {qwk_values}")
    result.info(f"MAE across runs: {mae_values}")

    if qwk_consistent and mae_consistent:
        result.info("Metrics are identical across all runs")
    else:
        result.fail("Metrics differ across runs")

    return result


def test_save_load_reproducibility() -> ReproducibilityResult:
    """Test that saved/loaded models produce same results."""
    result = ReproducibilityResult("Save/Load Reproducibility")

    df = load_data()
    df_known = df[df[FEATURE_CONFIG['target_column']] != -1]

    X = df_known.iloc[:, FEATURE_CONFIG['feature_columns_start']:FEATURE_CONFIG['feature_columns_end']]
    y = df_known[FEATURE_CONFIG['target_column']]

    X_train, X_test, y_train, y_test, _ = stratified_coach_level_split(
        df_known, X, y, test_size=0.2, random_state=42
    )

    # Train and evaluate
    model = CoachTenureModel(use_ordinal=True, n_classes=3, random_state=42)
    model.fit(X_train, y_train, verbose=0)

    pred_original = model.predict(X_test)
    proba_original = model.predict_proba(X_test)
    metrics_original = ordinal_metrics(y_test, pred_original, proba_original)

    # Save and load
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        temp_path = f.name

    try:
        model.save(temp_path)
        loaded_model = CoachTenureModel.load(temp_path)

        pred_loaded = loaded_model.predict(X_test)
        proba_loaded = loaded_model.predict_proba(X_test)
        metrics_loaded = ordinal_metrics(y_test, pred_loaded, proba_loaded)

        # Compare
        pred_match = np.array_equal(pred_original, pred_loaded)
        proba_match = np.allclose(proba_original, proba_loaded)
        qwk_match = metrics_original['qwk'] == metrics_loaded['qwk']

        result.info(f"Predictions match: {pred_match}")
        result.info(f"Probabilities match: {proba_match}")
        result.info(f"QWK match: {qwk_match} ({metrics_original['qwk']:.4f} vs {metrics_loaded['qwk']:.4f})")

        if pred_match and proba_match and qwk_match:
            result.info("Saved/loaded model produces identical results")
        else:
            result.fail("Results differ after save/load")

    finally:
        os.remove(temp_path)

    return result


def test_existing_model_if_available() -> ReproducibilityResult:
    """Test against existing saved model if available."""
    result = ReproducibilityResult("Existing Model Validation")

    model_path = os.path.join(project_root, MODEL_PATHS['ordinal_model_output'])

    if not os.path.exists(model_path):
        result.info(f"No existing model found at {model_path}")
        result.info("Skipping existing model validation")
        return result

    # Load existing model
    existing_model = CoachTenureModel.load(model_path)

    # Load data and evaluate
    df = load_data()
    df_known = df[df[FEATURE_CONFIG['target_column']] != -1]

    X = df_known.iloc[:, FEATURE_CONFIG['feature_columns_start']:FEATURE_CONFIG['feature_columns_end']]
    y = df_known[FEATURE_CONFIG['target_column']]

    X_train, X_test, y_train, y_test, _ = stratified_coach_level_split(
        df_known, X, y, test_size=0.2, random_state=42
    )

    # Evaluate existing model
    pred = existing_model.predict(X_test)
    proba = existing_model.predict_proba(X_test)
    metrics = ordinal_metrics(y_test, pred, proba)

    result.info(f"Existing model evaluation:")
    result.info(f"  QWK: {metrics['qwk']:.4f}")
    result.info(f"  MAE: {metrics['mae']:.4f}")
    result.info(f"  Accuracy: {metrics['exact_accuracy']:.4f}")

    result.details['existing_model_metrics'] = {
        k: v for k, v in metrics.items()
        if k not in ['per_class', 'confusion_matrix']
    }

    return result


def run_all_reproducibility_validations() -> Dict[str, ReproducibilityResult]:
    """Run all reproducibility validation checks."""
    print("="*60)
    print("REPRODUCIBILITY VALIDATION")
    print("="*60)

    validations = [
        reproduce_training_pipeline,
        reproduce_predictions,
        test_deterministic_training,
        test_split_determinism,
        test_cross_run_consistency,
        test_save_load_reproducibility,
        test_existing_model_if_available,
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
    print(f"REPRODUCIBILITY VALIDATION SUMMARY: {passed} passed, {failed} failed")
    print("="*60)

    return results


if __name__ == '__main__':
    results = run_all_reproducibility_validations()

    # Exit with error code if any validations failed
    all_passed = all(r.passed for r in results.values())
    sys.exit(0 if all_passed else 1)
