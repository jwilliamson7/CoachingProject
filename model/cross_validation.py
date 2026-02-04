"""
Coach-level stratified cross-validation utilities.

This module provides cross-validation splitting strategies that ensure
no coach appears in both training and test sets, preventing data leakage
when coaches have multiple hiring instances.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
from typing import List, Tuple, Generator, Optional


def stratified_coach_level_split(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Stratified train/test split at the coach level.

    Ensures no coach appears in both train and test sets while attempting
    to preserve class distribution. Coaches with multiple instances are
    kept together in either train or test.

    Parameters
    ----------
    df : DataFrame
        Full dataframe containing 'Coach Name' column and target.

    X : DataFrame
        Feature matrix (subset of df).

    y : Series
        Target variable.

    test_size : float, default=0.2
        Proportion of data to include in test set.

    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    X_train : DataFrame
        Training features.

    X_test : DataFrame
        Test features.

    y_train : Series
        Training labels.

    y_test : Series
        Test labels.

    test_coaches : list
        List of coach names in the test set.
    """
    np.random.seed(random_state)

    # Group coaches by their most common class
    coach_classes = {}
    for coach in df['Coach Name'].unique():
        coach_instances = df[df['Coach Name'] == coach]
        # Use mode (most common) class for coaches with multiple instances
        coach_class = coach_instances.iloc[:, -1].mode()[0]
        coach_classes[coach] = coach_class

    # Separate coaches by class
    coaches_by_class = {0: [], 1: [], 2: []}
    for coach, cls in coach_classes.items():
        if cls in coaches_by_class:
            coaches_by_class[cls].append(coach)

    # Shuffle coaches within each class
    for cls in coaches_by_class:
        np.random.shuffle(coaches_by_class[cls])

    # Calculate target test instances per class
    class_counts = y.value_counts()
    target_test_per_class = {
        cls: int(count * test_size)
        for cls, count in class_counts.items()
    }

    # Select test coaches from each class
    test_coaches = []
    for cls in [0, 1, 2]:
        current_test_instances = 0
        for coach in coaches_by_class.get(cls, []):
            coach_count = len(df[df['Coach Name'] == coach])
            if current_test_instances < target_test_per_class.get(cls, 0):
                test_coaches.append(coach)
                current_test_instances += coach_count
            else:
                break

    # Create train/test masks
    test_mask = df['Coach Name'].isin(test_coaches)
    train_mask = ~test_mask

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    return X_train, X_test, y_train, y_test, test_coaches


def stratified_coach_level_cv_split(
    df_train: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    random_state: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate stratified coach-level cross-validation splits.

    Parameters
    ----------
    df_train : DataFrame
        Training dataframe containing 'Coach Name' column and target.

    X_train : DataFrame
        Training feature matrix.

    y_train : Series
        Training target variable.

    n_splits : int, default=5
        Number of CV folds.

    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    splits : list of (train_indices, test_indices) tuples
        Each tuple contains arrays of integer indices for train and test sets.
    """
    np.random.seed(random_state)

    # Group coaches by their most common class
    coach_classes = {}
    for coach in df_train['Coach Name'].unique():
        coach_instances = df_train[df_train['Coach Name'] == coach]
        coach_class = coach_instances.iloc[:, -1].mode()[0]
        coach_classes[coach] = coach_class

    # Separate coaches by class
    coaches_by_class = {0: [], 1: [], 2: []}
    for coach, cls in coach_classes.items():
        if cls in coaches_by_class:
            coaches_by_class[cls].append(coach)

    # Shuffle coaches within each class
    for cls in coaches_by_class:
        np.random.shuffle(coaches_by_class[cls])

    # Create folds maintaining class distribution
    splits = []
    for fold_idx in range(n_splits):
        test_coaches_fold = []

        # Select test coaches from each class for this fold
        for cls in [0, 1, 2]:
            coaches_in_class = coaches_by_class.get(cls, [])
            if len(coaches_in_class) == 0:
                continue

            fold_size = max(1, len(coaches_in_class) // n_splits)
            start_idx = fold_idx * fold_size
            end_idx = min(start_idx + fold_size, len(coaches_in_class))

            # For the last fold, include remaining coaches
            if fold_idx == n_splits - 1:
                end_idx = len(coaches_in_class)

            test_coaches_fold.extend(coaches_in_class[start_idx:end_idx])

        # Skip empty folds
        if not test_coaches_fold:
            continue

        # Create masks
        test_mask = df_train['Coach Name'].isin(test_coaches_fold)
        train_mask = ~test_mask

        # Get indices that exist in X_train
        train_indices = []
        test_indices = []

        for idx in X_train.index:
            if train_mask[idx]:
                train_indices.append(X_train.index.get_loc(idx))
            elif test_mask[idx]:
                test_indices.append(X_train.index.get_loc(idx))

        # Only add split if both train and test have data
        if len(train_indices) > 0 and len(test_indices) > 0:
            splits.append((np.array(train_indices), np.array(test_indices)))

    return splits


class CoachLevelStratifiedKFold(BaseCrossValidator):
    """
    Stratified K-Fold cross-validator at the coach level.

    Provides train/test indices to split data while ensuring no coach
    appears in both train and test sets within the same fold.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.

    random_state : int, default=42
        Random seed for shuffling coaches.

    Attributes
    ----------
    n_splits : int
        Number of folds.

    Examples
    --------
    >>> cv = CoachLevelStratifiedKFold(n_splits=5)
    >>> for train_idx, test_idx in cv.split(X, y, groups=coach_names):
    ...     X_train, X_test = X[train_idx], X[test_idx]
    ...     y_train, y_test = y[train_idx], y[test_idx]
    """

    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices to split data into training and test sets.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target variable.

        groups : array-like of shape (n_samples,)
            Coach names for each sample. Required.

        Yields
        ------
        train : ndarray
            Training set indices for the current fold.

        test : ndarray
            Test set indices for the current fold.
        """
        if groups is None:
            raise ValueError("groups (coach names) must be provided")

        X = np.asarray(X)
        y = np.asarray(y)
        groups = np.asarray(groups)

        np.random.seed(self.random_state)

        # Group coaches by their most common class
        unique_coaches = np.unique(groups)
        coach_classes = {}

        for coach in unique_coaches:
            coach_mask = groups == coach
            coach_labels = y[coach_mask]
            # Mode: most frequent label
            values, counts = np.unique(coach_labels, return_counts=True)
            coach_classes[coach] = values[np.argmax(counts)]

        # Separate coaches by class
        n_classes = len(np.unique(y))
        coaches_by_class = {i: [] for i in range(n_classes)}

        for coach, cls in coach_classes.items():
            if cls in coaches_by_class:
                coaches_by_class[cls].append(coach)

        # Shuffle coaches within each class
        for cls in coaches_by_class:
            np.random.shuffle(coaches_by_class[cls])

        # Create folds
        for fold_idx in range(self.n_splits):
            test_coaches = []

            # Select test coaches from each class for this fold
            for cls in range(n_classes):
                coaches_in_class = coaches_by_class.get(cls, [])
                if len(coaches_in_class) == 0:
                    continue

                fold_size = max(1, len(coaches_in_class) // self.n_splits)
                start_idx = fold_idx * fold_size
                end_idx = min(start_idx + fold_size, len(coaches_in_class))

                # For the last fold, include remaining coaches
                if fold_idx == self.n_splits - 1:
                    end_idx = len(coaches_in_class)

                test_coaches.extend(coaches_in_class[start_idx:end_idx])

            # Skip empty folds
            if not test_coaches:
                continue

            # Create index arrays
            test_mask = np.isin(groups, test_coaches)
            train_mask = ~test_mask

            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]

            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices

    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> int:
        """Returns the number of splitting iterations."""
        return self.n_splits
