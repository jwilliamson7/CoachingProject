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

    # Train on modern era only (1970+)
    python scripts/train.py --min-year 1970

    # Compare eras (full dataset vs modern era)
    python scripts/train.py --compare-eras
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


def load_data(data_path: str = None, min_year: int = None, max_year: int = None) -> pd.DataFrame:
    """
    Load the training data with optional year filtering.

    Parameters
    ----------
    data_path : str, optional
        Path to data file. If None, uses default.
    min_year : int, optional
        Minimum year to include (inclusive). E.g., 1970 for modern era.
    max_year : int, optional
        Maximum year to include (inclusive).

    Returns
    -------
    df : DataFrame
        Filtered training data.
    """
    if data_path is None:
        data_path = os.path.join(project_root, MODEL_PATHS['data_file'])

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, index_col=0)

    # Filter out coaches with unknown tenure (class -1)
    df = df[df[FEATURE_CONFIG['target_column']] != -1]

    initial_count = len(df)

    # Apply year filters
    if min_year is not None:
        df = df[df[FEATURE_CONFIG['year_column']] >= min_year]
        print(f"Filtered to years >= {min_year}")

    if max_year is not None:
        df = df[df[FEATURE_CONFIG['year_column']] <= max_year]
        print(f"Filtered to years <= {max_year}")

    if min_year is not None or max_year is not None:
        year_range = df[FEATURE_CONFIG['year_column']]
        print(f"Year range: {year_range.min()} - {year_range.max()}")
        print(f"Filtered from {initial_count} to {len(df)} instances")
    else:
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


def compare_eras(
    data_path: str = None,
    modern_era_start: int = 1970,
    tune: bool = False,
    n_iter: int = 500
):
    """
    Compare model performance between full dataset and modern era.

    Parameters
    ----------
    data_path : str, optional
        Path to data file.
    modern_era_start : int, default=1970
        Start year for modern era (NFL-AFL merger).
    tune : bool, default=False
        Whether to tune hyperparameters.
    n_iter : int, default=500
        Number of tuning iterations.
    """
    print("\n" + "=" * 80)
    print("ERA COMPARISON: Full Dataset vs Modern Era")
    print("=" * 80)

    results = {}

    # Train on full dataset
    print("\n" + "#" * 80)
    print("# FULL DATASET (All Years)")
    print("#" * 80)

    df_full = load_data(data_path)
    X_full, y_full = prepare_features(df_full)

    print(f"\nFull dataset summary:")
    print(f"  Instances: {len(df_full)}")
    print(f"  Year range: {df_full[FEATURE_CONFIG['year_column']].min()} - {df_full[FEATURE_CONFIG['year_column']].max()}")
    print(f"  Class distribution:")
    for cls, count in y_full.value_counts().sort_index().items():
        print(f"    Class {cls}: {count} ({count/len(y_full)*100:.1f}%)")

    _, _, _, _, _, full_metrics = train_model(
        df_full, X_full, y_full,
        use_ordinal=True, tune=tune, n_iter=n_iter
    )
    results['full'] = full_metrics

    # Train on modern era
    print("\n" + "#" * 80)
    print(f"# MODERN ERA ({modern_era_start}+)")
    print("#" * 80)

    df_modern = load_data(data_path, min_year=modern_era_start)
    X_modern, y_modern = prepare_features(df_modern)

    print(f"\nModern era summary:")
    print(f"  Instances: {len(df_modern)}")
    print(f"  Year range: {df_modern[FEATURE_CONFIG['year_column']].min()} - {df_modern[FEATURE_CONFIG['year_column']].max()}")
    print(f"  Class distribution:")
    for cls, count in y_modern.value_counts().sort_index().items():
        print(f"    Class {cls}: {count} ({count/len(y_modern)*100:.1f}%)")

    _, _, _, _, _, modern_metrics = train_model(
        df_modern, X_modern, y_modern,
        use_ordinal=True, tune=tune, n_iter=n_iter
    )
    results['modern'] = modern_metrics

    # Print comparison
    print("\n" + "=" * 80)
    print("ERA COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<25} {'Full Dataset':>15} {'Modern Era':>15} {'Difference':>15}")
    print("-" * 80)

    metrics_to_compare = ['mae', 'qwk', 'adjacent_accuracy', 'exact_accuracy', 'macro_f1', 'auroc']

    for metric in metrics_to_compare:
        full_val = full_metrics.get(metric)
        modern_val = modern_metrics.get(metric)

        if full_val is None or modern_val is None:
            continue

        diff = modern_val - full_val

        # For MAE, lower is better; for others, higher is better
        if metric == 'mae':
            better = "modern" if modern_val < full_val else "full"
        else:
            better = "modern" if modern_val > full_val else "full"

        sign = "+" if diff > 0 else ""
        print(f"{metric:<25} {full_val:>15.4f} {modern_val:>15.4f} {sign}{diff:>14.4f} ({better})")

    # Per-class F1 comparison
    print("\nPer-Class F1 Scores:")
    print(f"{'Class':<25} {'Full Dataset':>15} {'Modern Era':>15} {'Difference':>15}")
    print("-" * 80)

    for class_name in full_metrics['per_class'].keys():
        full_f1 = full_metrics['per_class'][class_name]['f1']
        modern_f1 = modern_metrics['per_class'][class_name]['f1']
        diff = modern_f1 - full_f1
        sign = "+" if diff > 0 else ""
        better = "modern" if modern_f1 > full_f1 else "full"
        print(f"{class_name:<25} {full_f1:>15.3f} {modern_f1:>15.3f} {sign}{diff:>14.3f} ({better})")

    # Dataset size comparison
    print("\n" + "-" * 80)
    print("Dataset Size Impact:")
    excluded_count = len(df_full) - len(df_modern)
    excluded_pct = excluded_count / len(df_full) * 100
    print(f"  Pre-{modern_era_start} instances excluded: {excluded_count} ({excluded_pct:.1f}%)")
    print(f"  Training data reduction: {len(df_full)} -> {len(df_modern)} instances")
    print("=" * 80)

    return results


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
    parser.add_argument(
        '--min-year', type=int, default=None,
        help='Minimum year to include in training data (e.g., 1970 for modern era)'
    )
    parser.add_argument(
        '--max-year', type=int, default=None,
        help='Maximum year to include in training data'
    )
    parser.add_argument(
        '--compare-eras', action='store_true',
        help='Compare model performance between full dataset and modern era (1970+)'
    )
    parser.add_argument(
        '--modern-era-start', type=int, default=1970,
        help='Start year for modern era comparison (default: 1970, NFL-AFL merger)'
    )

    args = parser.parse_args()

    # Ensure models directory exists
    models_dir = os.path.join(project_root, MODEL_PATHS.get('models_dir', 'models'))
    os.makedirs(models_dir, exist_ok=True)

    # Handle era comparison mode
    if args.compare_eras:
        compare_eras(
            data_path=args.data,
            modern_era_start=args.modern_era_start,
            tune=args.tune,
            n_iter=args.n_iter
        )
        print("\nEra comparison complete!")
        return

    # Load data with optional year filtering
    df = load_data(args.data, min_year=args.min_year, max_year=args.max_year)
    X, y = prepare_features(df)

    print(f"\nDataset summary:")
    print(f"  Total instances: {len(df)}")
    print(f"  Year range: {df[FEATURE_CONFIG['year_column']].min()} - {df[FEATURE_CONFIG['year_column']].max()}")
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
