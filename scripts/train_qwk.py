#!/usr/bin/env python
"""
Training script with QWK-based hyperparameter tuning.

This script uses Quadratic Weighted Kappa (QWK) as the optimization metric,
which is more appropriate for ordinal classification than AUROC or accuracy.

QWK penalizes distant misclassifications more heavily than adjacent ones,
making it ideal for ordinal problems like tenure prediction.

Usage:
    python scripts/train_qwk.py [--n-iter 500] [--cv 5] [--output model.pkl]

Examples:
    # Train with QWK tuning (default 500 iterations)
    python scripts/train_qwk.py

    # Train with more iterations
    python scripts/train_qwk.py --n-iter 1000

    # Compare QWK vs other metrics
    python scripts/train_qwk.py --compare-metrics
"""

import argparse
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from model import (
    CoachTenureModel,
    stratified_coach_level_split,
    stratified_coach_level_cv_split,
    ordinal_metrics,
    format_metrics_report
)
from model.tuning import (
    tune_ordinal_model_qwk,
    compare_tuning_metrics,
    TunableOrdinalClassifier,
    QWK_SCORER
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


def train_with_qwk_tuning(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 500,
    cv_folds: int = 5,
    verbose: int = 1
) -> tuple:
    """
    Train ordinal model with QWK-based hyperparameter tuning.

    Parameters
    ----------
    df : DataFrame
        Full dataframe (needed for coach-level splitting)
    X : DataFrame
        Features
    y : Series
        Target
    n_iter : int
        Number of random search iterations
    cv_folds : int
        Number of cross-validation folds
    verbose : int
        Verbosity level

    Returns
    -------
    model : CoachTenureModel
        Trained model with best hyperparameters
    metrics : dict
        Test set metrics
    best_params : dict
        Best hyperparameters found
    """
    print(f"\n{'='*60}")
    print("Training ORDINAL model with QWK-based tuning")
    print(f"{'='*60}")

    # Split data (coach-level stratified)
    X_train, X_test, y_train, y_test, test_coaches = stratified_coach_level_split(
        df, X, y,
        test_size=MODEL_CONFIG['test_size'],
        random_state=MODEL_CONFIG['random_state']
    )

    print(f"Train set: {len(X_train)} instances")
    print(f"Test set: {len(X_test)} instances")
    print(f"No coach overlap between train/test: {len(set(df.loc[X_train.index, 'Coach Name']) & set(test_coaches)) == 0}")

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Generate coach-level CV splits for tuning
    df_train = df.loc[X_train.index]
    cv_splits = stratified_coach_level_cv_split(
        df_train, X_train, y_train,
        n_splits=cv_folds,
        random_state=MODEL_CONFIG['random_state']
    )

    print(f"\nUsing {cv_folds}-fold coach-level CV for hyperparameter tuning")
    print(f"Optimization metric: Quadratic Weighted Kappa (QWK)")

    # Tune with QWK
    best_tunable_model, best_params, cv_results = tune_ordinal_model_qwk(
        X_train_imputed,
        y_train.values,
        n_iter=n_iter,
        cv=cv_splits,
        n_classes=3,
        random_state=MODEL_CONFIG['random_state'],
        verbose=verbose
    )

    # Create a CoachTenureModel with the best parameters
    model = CoachTenureModel(
        use_ordinal=True,
        n_classes=3,
        xgboost_params=best_params,
        random_state=MODEL_CONFIG['random_state']
    )

    # Fit the model (it will use the optimized params)
    model.fit(X_train, y_train, verbose=0)
    model.best_params_ = best_params

    # Evaluate on test set
    print("\n" + "="*60)
    print("Hold-out Test Set Performance")
    print("="*60)
    metrics = model.evaluate(X_test, y_test, print_report=True)

    return model, X_train, X_test, y_train, y_test, metrics, best_params


def compare_metrics_experiment(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 100,
    cv_folds: int = 5
):
    """
    Compare models tuned with different metrics (QWK, F1, Accuracy).

    This helps understand whether QWK-based tuning produces better
    ordinal classification results.
    """
    print("\n" + "="*80)
    print("METRIC COMPARISON EXPERIMENT")
    print("Comparing hyperparameter tuning with different scoring metrics")
    print("="*80)

    # Split data
    X_train, X_test, y_train, y_test, _ = stratified_coach_level_split(
        df, X, y,
        test_size=MODEL_CONFIG['test_size'],
        random_state=MODEL_CONFIG['random_state']
    )

    # Impute
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Get CV splits
    df_train = df.loc[X_train.index]
    cv_splits = stratified_coach_level_cv_split(
        df_train, X_train, y_train,
        n_splits=cv_folds,
        random_state=MODEL_CONFIG['random_state']
    )

    print(f"\nTuning with {n_iter} iterations per metric, {cv_folds}-fold CV")
    print("This may take a while...\n")

    # Compare metrics
    results = compare_tuning_metrics(
        X_train_imputed,
        y_train.values,
        n_iter=n_iter,
        cv=cv_splits,
        random_state=MODEL_CONFIG['random_state'],
        verbose=1
    )

    # Evaluate each model on test set
    print("\n" + "="*80)
    print("TEST SET COMPARISON")
    print("="*80)

    comparison_metrics = ['qwk', 'mae', 'adjacent_accuracy', 'exact_accuracy', 'macro_f1']

    print(f"\n{'Tuning Metric':<15} {'QWK':>10} {'MAE':>10} {'Adj Acc':>10} {'Exact Acc':>10} {'Macro F1':>10}")
    print("-"*70)

    for metric_name, result in results.items():
        model = result['best_estimator']
        y_pred = model.predict(X_test_imputed)
        y_proba = model.predict_proba(X_test_imputed)

        test_metrics = ordinal_metrics(y_test, y_pred, y_proba)

        print(f"{metric_name.upper():<15} "
              f"{test_metrics['qwk']:>10.4f} "
              f"{test_metrics['mae']:>10.4f} "
              f"{test_metrics['adjacent_accuracy']:>10.4f} "
              f"{test_metrics['exact_accuracy']:>10.4f} "
              f"{test_metrics['macro_f1']:>10.4f}")

    print("\nNote: QWK-tuned model should generally perform best on QWK and ordinal metrics")
    print("      (MAE, adjacent accuracy) while other metrics may vary.")


def main():
    parser = argparse.ArgumentParser(
        description="Train NFL Coach Tenure Model with QWK-based Hyperparameter Tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--n-iter', type=int, default=500,
        help='Number of iterations for RandomizedSearchCV (default: 500)'
    )
    parser.add_argument(
        '--cv', type=int, default=5,
        help='Number of cross-validation folds (default: 5)'
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
        '--compare-metrics', action='store_true',
        help='Compare QWK vs F1 vs Accuracy tuning (runs 3 separate tuning experiments)'
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

    if args.compare_metrics:
        # Run comparison experiment
        compare_metrics_experiment(
            df, X, y,
            n_iter=min(args.n_iter, 100),  # Use fewer iterations for comparison
            cv_folds=args.cv
        )
    else:
        # Train with QWK tuning
        model, X_train, X_test, y_train, y_test, metrics, best_params = train_with_qwk_tuning(
            df, X, y,
            n_iter=args.n_iter,
            cv_folds=args.cv,
            verbose=args.verbose
        )

        # Save model
        if args.output:
            output_path = args.output
        else:
            output_path = os.path.join(project_root, 'data/models/coach_tenure_ordinal_qwk_model.pkl')

        model.save(output_path)

        # Print summary
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Optimization metric: Quadratic Weighted Kappa (QWK)")
        print(f"Best CV QWK score: (see above)")
        print(f"Test set QWK: {metrics['qwk']:.4f}")
        print(f"Test set MAE: {metrics['mae']:.4f}")
        print(f"Test set Adjacent Accuracy: {metrics['adjacent_accuracy']:.4f}")
        print(f"\nBest hyperparameters:")
        for param, value in sorted(best_params.items()):
            print(f"  {param}: {value}")
        print(f"\nModel saved to: {output_path}")

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
