"""
Train WAR (Wins Above Replacement) regression model for NFL coaches.

This script trains an XGBoost regression model to predict average WAR
per season during a coach's tenure.

Usage:
    python scripts/train_war.py
    python scripts/train_war.py --tune --n-iter 500
    python scripts/train_war.py --min-year 1980 --max-year 2020
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.war_regressor import WARRegressor
from model.config import MODEL_PATHS, WAR_CONFIG, MODEL_CONFIG
from model.cross_validation import stratified_coach_level_cv_split


def load_data(
    data_path: str,
    min_year: int = None,
    max_year: int = None
) -> pd.DataFrame:
    """
    Load WAR prediction dataset.

    Parameters
    ----------
    data_path : str
        Path to war_prediction_data.csv
    min_year : int, optional
        Minimum hire year to include
    max_year : int, optional
        Maximum hire year to include

    Returns
    -------
    pd.DataFrame
        Loaded and optionally filtered dataset
    """
    df = pd.read_csv(data_path)

    if min_year is not None:
        df = df[df['Year'] >= min_year]
    if max_year is not None:
        df = df[df['Year'] <= max_year]

    print(f"Loaded {len(df)} coaching hires")
    if min_year or max_year:
        year_range = f"{min_year or 'start'} to {max_year or 'end'}"
        print(f"  Filtered to years: {year_range}")

    return df


def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features and target from WAR dataset.

    Parameters
    ----------
    df : pd.DataFrame
        WAR prediction dataset

    Returns
    -------
    X : np.ndarray
        Feature matrix (140 features)
    y : np.ndarray
        Target values (avg_war_per_season)
    """
    # Features 1-140 are in columns 2:142 (0-indexed)
    feature_cols = [f'Feature {i}' for i in range(1, 141)]
    X = df[feature_cols].values
    y = df[WAR_CONFIG['target_column']].values

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target range: [{y.min():.4f}, {y.max():.4f}]")

    return X, y


def coach_level_split(
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Split data ensuring no coach appears in both train and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with Coach Name column
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target values
    test_size : float
        Proportion for test set
    random_state : int
        Random seed

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
        Split feature and target arrays
    split_info : dict
        Information about the split
    """
    # Get unique coaches
    coaches = df['Coach Name'].values
    unique_coaches = df['Coach Name'].unique()

    # Split at coach level
    np.random.seed(random_state)
    n_test = int(len(unique_coaches) * test_size)
    test_coaches = set(np.random.choice(unique_coaches, n_test, replace=False))
    train_coaches = set(unique_coaches) - test_coaches

    # Get indices
    train_idx = np.array([i for i, c in enumerate(coaches) if c in train_coaches])
    test_idx = np.array([i for i, c in enumerate(coaches) if c in test_coaches])

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    split_info = {
        'n_train_coaches': len(train_coaches),
        'n_test_coaches': len(test_coaches),
        'n_train_samples': len(train_idx),
        'n_test_samples': len(test_idx),
        'train_coaches': train_coaches,
        'test_coaches': test_coaches
    }

    print(f"\nCoach-level split:")
    print(f"  Train: {len(train_coaches)} coaches, {len(train_idx)} samples")
    print(f"  Test:  {len(test_coaches)} coaches, {len(test_idx)} samples")
    print(f"  Train target mean: {y_train.mean():.4f}")
    print(f"  Test target mean:  {y_test.mean():.4f}")

    return X_train, X_test, y_train, y_test, split_info


def generate_coach_level_cv_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42
):
    """
    Generate coach-level cross-validation splits.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with Coach Name column
    n_splits : int
        Number of CV folds
    random_state : int
        Random seed

    Returns
    -------
    list of (train_idx, val_idx) tuples
    """
    coaches = df['Coach Name'].values
    unique_coaches = np.array(df['Coach Name'].unique())

    np.random.seed(random_state)
    np.random.shuffle(unique_coaches)

    # Split coaches into folds
    fold_size = len(unique_coaches) // n_splits
    folds = []

    for i in range(n_splits):
        start = i * fold_size
        end = start + fold_size if i < n_splits - 1 else len(unique_coaches)
        folds.append(set(unique_coaches[start:end]))

    # Generate train/val index pairs
    splits = []
    for i in range(n_splits):
        val_coaches = folds[i]
        train_coaches = set(unique_coaches) - val_coaches

        train_idx = np.array([j for j, c in enumerate(coaches) if c in train_coaches])
        val_idx = np.array([j for j, c in enumerate(coaches) if c in val_coaches])

        splits.append((train_idx, val_idx))

    return splits


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    df_train: pd.DataFrame = None,
    tune: bool = False,
    n_iter: int = 500,
    output_path: str = None
) -> Tuple[WARRegressor, Dict]:
    """
    Train WAR regression model.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training data
    X_test, y_test : np.ndarray
        Test data
    df_train : pd.DataFrame, optional
        Training dataframe for coach-level CV splits
    tune : bool
        Whether to tune hyperparameters
    n_iter : int
        Number of RandomizedSearchCV iterations
    output_path : str, optional
        Path to save trained model

    Returns
    -------
    model : WARRegressor
        Trained model
    metrics : dict
        Test set metrics
    """
    model = WARRegressor()

    # Generate coach-level CV splits if tuning
    cv_splits = None
    if tune and df_train is not None:
        print("\nGenerating coach-level CV splits for tuning...")
        cv_splits = generate_coach_level_cv_splits(
            df_train,
            n_splits=MODEL_CONFIG['cv_folds'],
            random_state=MODEL_CONFIG['random_state']
        )

    # Train
    print("\n" + "=" * 60)
    print("TRAINING WAR REGRESSION MODEL")
    print("=" * 60)

    model.fit(
        X_train, y_train,
        tune_hyperparameters=tune,
        n_iter=n_iter,
        cv_splits=cv_splits
    )

    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION ON TEST SET")
    print("=" * 60)
    metrics = model.evaluate(X_test, y_test)

    # Compare to baseline (predict mean)
    baseline_pred = np.full_like(y_test, y_train.mean())
    baseline_mse = np.mean((y_test - baseline_pred) ** 2)
    model_mse = metrics['mse']
    improvement = (baseline_mse - model_mse) / baseline_mse * 100

    print(f"\nBaseline (predict mean) MSE: {baseline_mse:.6f}")
    print(f"Model MSE improvement: {improvement:.1f}%")

    # Save if requested
    if output_path:
        model.save(output_path)

    return model, metrics


def analyze_feature_importance(
    model: WARRegressor,
    top_n: int = 20
):
    """
    Print top feature importances.

    Parameters
    ----------
    model : WARRegressor
        Trained model
    top_n : int
        Number of top features to display
    """
    importances = model.get_feature_importances()
    feature_names = [f'Feature {i}' for i in range(1, 141)]

    # Sort by importance
    indices = np.argsort(importances)[::-1]

    print("\n" + "=" * 60)
    print("TOP FEATURE IMPORTANCES")
    print("=" * 60)
    print(f"{'Rank':<6} {'Feature':<20} {'Importance':>12}")
    print("-" * 40)

    for rank, idx in enumerate(indices[:top_n], 1):
        print(f"{rank:<6} {feature_names[idx]:<20} {importances[idx]:>12.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Train WAR regression model for NFL coaches'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default=MODEL_PATHS['war_data_file'],
        help='Path to WAR prediction data'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=MODEL_PATHS['war_model_output'],
        help='Path to save trained model'
    )
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Perform hyperparameter tuning'
    )
    parser.add_argument(
        '--n-iter',
        type=int,
        default=500,
        help='Number of RandomizedSearchCV iterations'
    )
    parser.add_argument(
        '--min-year',
        type=int,
        default=None,
        help='Minimum hire year to include'
    )
    parser.add_argument(
        '--max-year',
        type=int,
        default=None,
        help='Maximum hire year to include'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set proportion'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save the trained model'
    )

    args = parser.parse_args()

    # Load data
    print("=" * 60)
    print("WAR REGRESSION MODEL TRAINING")
    print("=" * 60)

    df = load_data(args.data_path, args.min_year, args.max_year)
    X, y = prepare_features(df)

    # Split data
    X_train, X_test, y_train, y_test, split_info = coach_level_split(
        df, X, y,
        test_size=args.test_size,
        random_state=MODEL_CONFIG['random_state']
    )

    # Create train dataframe for CV splits
    train_coaches = split_info['train_coaches']
    df_train = df[df['Coach Name'].isin(train_coaches)].copy()

    # Train model
    output_path = None if args.no_save else args.output
    model, metrics = train_model(
        X_train, y_train,
        X_test, y_test,
        df_train=df_train,
        tune=args.tune,
        n_iter=args.n_iter,
        output_path=output_path
    )

    # Feature importance
    analyze_feature_importance(model)

    print("\nTraining complete!")

    return model, metrics


if __name__ == '__main__':
    main()
