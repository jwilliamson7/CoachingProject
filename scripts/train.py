#!/usr/bin/env python
"""
Training script for NFL Coach Tenure Prediction Model.

Usage:
    python scripts/train.py [--ordinal] [--multiclass] [--n-iter 500] [--output model.pkl]

Examples:
    # Train ordinal model (default)
    python scripts/train.py

    # Train multiclass model for comparison
    python scripts/train.py --multiclass

    # Train with hyperparameter tuning
    python scripts/train.py --tune --n-iter 1000

    # Compare both models
    python scripts/train.py --compare
"""

import argparse
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from model import (
    CoachTenureModel,
    stratified_coach_level_split,
    stratified_coach_level_cv_split,
    ordinal_metrics,
    format_metrics_report
)
from model.config import MODEL_CONFIG, MODEL_PATHS, FEATURE_CONFIG


def load_data(data_path: str = None) -> pd.DataFrame:
    """Load the training data."""
    if data_path is None:
        data_path = os.path.join(project_root, MODEL_PATHS['data_file'])

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, index_col=0)

    # Filter out coaches with unknown tenure (class -1)
    df = df[df[FEATURE_CONFIG['target_column']] != -1]

    print(f"Loaded {len(df)} instances with known tenure classifications")
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """Extract features and target from dataframe."""
    X = df.iloc[:, FEATURE_CONFIG['feature_columns_start']:FEATURE_CONFIG['feature_columns_end']]
    y = df[FEATURE_CONFIG['target_column']]
    return X, y


def train_model(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    use_ordinal: bool = True,
    tune: bool = False,
    n_iter: int = 500,
    verbose: int = 1
) -> CoachTenureModel:
    """Train a coach tenure model."""
    model_type = "ordinal" if use_ordinal else "multiclass"
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()} model")
    print(f"{'='*60}")

    # Split data
    X_train, X_test, y_train, y_test, test_coaches = stratified_coach_level_split(
        df, X, y,
        test_size=MODEL_CONFIG['test_size'],
        random_state=MODEL_CONFIG['random_state']
    )

    print(f"Train set: {len(X_train)} instances")
    print(f"Test set: {len(X_test)} instances")
    print(f"No coach overlap between train/test: {len(set(df.loc[X_train.index, 'Coach Name']) & set(test_coaches)) == 0}")

    # Create and train model
    model = CoachTenureModel(
        use_ordinal=use_ordinal,
        n_classes=3,
        random_state=MODEL_CONFIG['random_state']
    )

    if tune:
        # Generate CV splits for tuning
        df_train = df.loc[X_train.index]
        cv_splits = stratified_coach_level_cv_split(
            df_train, X_train, y_train,
            n_splits=MODEL_CONFIG['cv_folds'],
            random_state=MODEL_CONFIG['random_state']
        )
        print(f"Using {len(cv_splits)}-fold coach-level CV for hyperparameter tuning")

        model.fit(
            X_train, y_train,
            tune_hyperparameters=True,
            n_iter=n_iter,
            cv_splits=cv_splits,
            verbose=verbose
        )
    else:
        model.fit(X_train, y_train, verbose=verbose)

    # Evaluate on test set
    print("\nHold-out Test Set Performance:")
    metrics = model.evaluate(X_test, y_test, print_report=True)

    return model, X_train, X_test, y_train, y_test, metrics


def compare_models(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, tune: bool = False, n_iter: int = 500):
    """Train and compare ordinal vs multiclass models."""
    print("\n" + "="*80)
    print("MODEL COMPARISON: Ordinal vs Multiclass")
    print("="*80)

    # Train both models
    ordinal_model, X_train, X_test, y_train, y_test, ordinal_metrics = train_model(
        df, X, y, use_ordinal=True, tune=tune, n_iter=n_iter
    )

    multiclass_model, _, _, _, _, multiclass_metrics = train_model(
        df, X, y, use_ordinal=False, tune=tune, n_iter=n_iter
    )

    # Comparison summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Metric':<25} {'Ordinal':>15} {'Multiclass':>15} {'Difference':>15}")
    print("-"*80)

    metrics_to_compare = ['mae', 'qwk', 'adjacent_accuracy', 'exact_accuracy', 'macro_f1', 'auroc']

    for metric in metrics_to_compare:
        ord_val = ordinal_metrics.get(metric)
        mult_val = multiclass_metrics.get(metric)

        if ord_val is None or mult_val is None:
            continue

        diff = ord_val - mult_val

        # For MAE, lower is better; for others, higher is better
        if metric == 'mae':
            better = "ordinal" if ord_val < mult_val else "multiclass"
        else:
            better = "ordinal" if ord_val > mult_val else "multiclass"

        sign = "+" if diff > 0 else ""
        print(f"{metric:<25} {ord_val:>15.4f} {mult_val:>15.4f} {sign}{diff:>14.4f} ({better})")

    # Per-class F1 comparison
    print("\nPer-Class F1 Scores:")
    print(f"{'Class':<25} {'Ordinal':>15} {'Multiclass':>15} {'Difference':>15}")
    print("-"*80)

    for class_name in ordinal_metrics['per_class'].keys():
        ord_f1 = ordinal_metrics['per_class'][class_name]['f1']
        mult_f1 = multiclass_metrics['per_class'][class_name]['f1']
        diff = ord_f1 - mult_f1
        sign = "+" if diff > 0 else ""
        print(f"{class_name:<25} {ord_f1:>15.3f} {mult_f1:>15.3f} {sign}{diff:>14.3f}")

    return ordinal_model, multiclass_model


def main():
    parser = argparse.ArgumentParser(
        description="Train NFL Coach Tenure Prediction Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--ordinal', action='store_true', default=True,
        help='Use ordinal classification (default)'
    )
    parser.add_argument(
        '--multiclass', action='store_true',
        help='Use multiclass classification instead of ordinal'
    )
    parser.add_argument(
        '--compare', action='store_true',
        help='Train and compare both ordinal and multiclass models'
    )
    parser.add_argument(
        '--tune', action='store_true',
        help='Perform hyperparameter tuning with RandomizedSearchCV'
    )
    parser.add_argument(
        '--n-iter', type=int, default=500,
        help='Number of iterations for RandomizedSearchCV (default: 500)'
    )
    parser.add_argument(
        '--data', type=str, default=None,
        help='Path to training data CSV file'
    )
    parser.add_argument(
        '--output', '-o', type=str, default=None,
        help='Output path for trained model'
    )
    parser.add_argument(
        '--verbose', '-v', type=int, default=1,
        help='Verbosity level (0=silent, 1=normal, 2=detailed)'
    )

    args = parser.parse_args()

    # Ensure models directory exists
    models_dir = os.path.join(project_root, MODEL_PATHS.get('models_dir', 'models'))
    os.makedirs(models_dir, exist_ok=True)

    # Load data
    df = load_data(args.data)
    X, y = prepare_features(df)

    print(f"\nDataset summary:")
    print(f"  Total instances: {len(df)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Class distribution:")
    for cls, count in y.value_counts().sort_index().items():
        print(f"    Class {cls}: {count} ({count/len(y)*100:.1f}%)")

    if args.compare:
        ordinal_model, multiclass_model = compare_models(
            df, X, y, tune=args.tune, n_iter=args.n_iter
        )

        # Save both models
        if args.output:
            base, ext = os.path.splitext(args.output)
            ordinal_path = f"{base}_ordinal{ext}"
            multiclass_path = f"{base}_multiclass{ext}"
        else:
            ordinal_path = os.path.join(project_root, MODEL_PATHS['ordinal_model_output'])
            multiclass_path = os.path.join(project_root, MODEL_PATHS['multiclass_model_output'])

        ordinal_model.save(ordinal_path)
        multiclass_model.save(multiclass_path)

    else:
        # Train single model
        use_ordinal = not args.multiclass

        model, X_train, X_test, y_train, y_test, metrics = train_model(
            df, X, y,
            use_ordinal=use_ordinal,
            tune=args.tune,
            n_iter=args.n_iter,
            verbose=args.verbose
        )

        # Save model
        if args.output:
            output_path = args.output
        else:
            if use_ordinal:
                output_path = os.path.join(project_root, MODEL_PATHS['ordinal_model_output'])
            else:
                output_path = os.path.join(project_root, MODEL_PATHS['multiclass_model_output'])

        model.save(output_path)

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
