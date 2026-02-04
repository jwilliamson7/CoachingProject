#!/usr/bin/env python
"""
Evaluation script for NFL Coach Tenure Prediction Model.

Usage:
    python scripts/evaluate.py --model model.pkl [--data test_data.csv]

Examples:
    # Evaluate saved model on default test split
    python scripts/evaluate.py --model coach_tenure_ordinal_model.pkl

    # Evaluate on specific data file
    python scripts/evaluate.py --model coach_tenure_model.pkl --data test_data.csv

    # Compare ordinal vs multiclass models
    python scripts/evaluate.py --compare
"""

import argparse
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd

from model import (
    CoachTenureModel,
    stratified_coach_level_split,
    ordinal_metrics,
    format_metrics_report
)
from model.config import MODEL_PATHS, FEATURE_CONFIG, MODEL_CONFIG


def load_model(model_path: str) -> CoachTenureModel:
    """Load a trained model."""
    print(f"Loading model from {model_path}...")
    return CoachTenureModel.load(model_path)


def load_data(data_path: str = None) -> pd.DataFrame:
    """Load evaluation data."""
    if data_path is None:
        data_path = os.path.join(project_root, MODEL_PATHS['data_file'])

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, index_col=0)

    # Filter out coaches with unknown tenure (class -1)
    df = df[df[FEATURE_CONFIG['target_column']] != -1]

    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """Extract features and target from dataframe."""
    X = df.iloc[:, FEATURE_CONFIG['feature_columns_start']:FEATURE_CONFIG['feature_columns_end']]
    y = df[FEATURE_CONFIG['target_column']]
    return X, y


def evaluate_model(
    model: CoachTenureModel,
    X: pd.DataFrame,
    y: pd.Series,
    title: str = "Model Evaluation"
) -> dict:
    """Evaluate model and return metrics."""
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")

    metrics = model.evaluate(X, y, print_report=True)
    return metrics


def print_feature_importances(model: CoachTenureModel, top_n: int = 20):
    """Print top feature importances."""
    print(f"\n{'='*60}")
    print(f"TOP {top_n} FEATURE IMPORTANCES")
    print(f"{'='*60}")

    importances = model.get_feature_importances()

    # Create feature names
    feature_names = [f"Feature {i+1}" for i in range(len(importances))]

    # Sort by importance
    sorted_indices = np.argsort(importances)[::-1]

    print(f"{'Rank':<6} {'Feature':<15} {'Importance':<12}")
    print("-"*40)

    for rank, idx in enumerate(sorted_indices[:top_n], 1):
        print(f"{rank:<6} {feature_names[idx]:<15} {importances[idx]:.4f}")


def compare_models(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series):
    """Compare ordinal vs multiclass models."""
    ordinal_path = os.path.join(project_root, MODEL_PATHS['ordinal_model_output'])
    multiclass_path = os.path.join(project_root, MODEL_PATHS['multiclass_model_output'])

    # Check if models exist
    if not os.path.exists(ordinal_path):
        print(f"Ordinal model not found at {ordinal_path}")
        print("Run: python scripts/train.py --ordinal")
        return

    if not os.path.exists(multiclass_path):
        print(f"Multiclass model not found at {multiclass_path}")
        print("Run: python scripts/train.py --multiclass")
        return

    # Load models
    ordinal_model = CoachTenureModel.load(ordinal_path)
    multiclass_model = CoachTenureModel.load(multiclass_path)

    # Get test split
    X_train, X_test, y_train, y_test, _ = stratified_coach_level_split(
        df, X, y,
        test_size=MODEL_CONFIG['test_size'],
        random_state=MODEL_CONFIG['random_state']
    )

    # Evaluate both
    print("\n" + "="*80)
    print("MODEL COMPARISON ON HELD-OUT TEST SET")
    print("="*80)
    print(f"Test set size: {len(X_test)} instances")

    ordinal_metrics_result = evaluate_model(
        ordinal_model, X_test, y_test,
        title="ORDINAL MODEL (Frank-Hall)"
    )

    multiclass_metrics_result = evaluate_model(
        multiclass_model, X_test, y_test,
        title="MULTICLASS MODEL (Standard XGBoost)"
    )

    # Comparison summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Metric':<25} {'Ordinal':>15} {'Multiclass':>15} {'Winner':>15}")
    print("-"*80)

    metrics_to_compare = [
        ('mae', 'lower'),
        ('qwk', 'higher'),
        ('adjacent_accuracy', 'higher'),
        ('exact_accuracy', 'higher'),
        ('macro_f1', 'higher'),
        ('auroc', 'higher')
    ]

    for metric, better_direction in metrics_to_compare:
        ord_val = ordinal_metrics_result[metric]
        mult_val = multiclass_metrics_result[metric]

        if better_direction == 'lower':
            winner = "Ordinal" if ord_val < mult_val else "Multiclass"
        else:
            winner = "Ordinal" if ord_val > mult_val else "Multiclass"

        print(f"{metric:<25} {ord_val:>15.4f} {mult_val:>15.4f} {winner:>15}")

    # Class 1 (middle class) comparison - key metric
    print("\n" + "-"*80)
    print("KEY METRIC: Class 1 F1 (Middle Class - Most Difficult)")
    print("-"*80)
    ord_class1_f1 = ordinal_metrics_result['per_class']['Class 1 (3-4 yrs)']['f1']
    mult_class1_f1 = multiclass_metrics_result['per_class']['Class 1 (3-4 yrs)']['f1']
    improvement = (ord_class1_f1 - mult_class1_f1) / mult_class1_f1 * 100 if mult_class1_f1 > 0 else 0

    print(f"Ordinal Class 1 F1:    {ord_class1_f1:.3f}")
    print(f"Multiclass Class 1 F1: {mult_class1_f1:.3f}")
    print(f"Improvement:           {improvement:+.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate NFL Coach Tenure Prediction Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--model', '-m', type=str, default=None,
        help='Path to trained model file'
    )
    parser.add_argument(
        '--data', '-d', type=str, default=None,
        help='Path to evaluation data CSV file'
    )
    parser.add_argument(
        '--compare', action='store_true',
        help='Compare ordinal vs multiclass models'
    )
    parser.add_argument(
        '--features', action='store_true',
        help='Print feature importances'
    )
    parser.add_argument(
        '--top-n', type=int, default=20,
        help='Number of top features to display (default: 20)'
    )

    args = parser.parse_args()

    # Load data
    df = load_data(args.data)
    X, y = prepare_features(df)

    print(f"\nDataset: {len(df)} instances, {X.shape[1]} features")

    if args.compare:
        compare_models(df, X, y)
        return

    # Single model evaluation
    if args.model is None:
        # Try default ordinal model
        model_path = os.path.join(project_root, MODEL_PATHS['ordinal_model_output'])
        if not os.path.exists(model_path):
            model_path = os.path.join(project_root, MODEL_PATHS['multiclass_model_output'])
            if not os.path.exists(model_path):
                print("No trained model found. Run training first:")
                print("  python scripts/train.py")
                return
    else:
        model_path = args.model

    model = load_model(model_path)

    # Get test split
    X_train, X_test, y_train, y_test, test_coaches = stratified_coach_level_split(
        df, X, y,
        test_size=MODEL_CONFIG['test_size'],
        random_state=MODEL_CONFIG['random_state']
    )

    print(f"Test set: {len(X_test)} instances from {len(test_coaches)} coaches")

    # Evaluate
    model_type = "Ordinal" if model.use_ordinal else "Multiclass"
    metrics = evaluate_model(
        model, X_test, y_test,
        title=f"{model_type} Model Evaluation"
    )

    # Feature importances
    if args.features:
        print_feature_importances(model, top_n=args.top_n)

    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
