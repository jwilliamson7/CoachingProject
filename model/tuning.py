"""
Hyperparameter tuning utilities for NFL Coach Tenure Prediction Model.

This module provides QWK-based hyperparameter tuning for ordinal classification,
which is more appropriate than AUROC for ordinal problems.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from typing import Dict, Any, Optional, List, Tuple

from .ordinal_classifier import OrdinalClassifier
from .evaluation import quadratic_weighted_kappa
from .config import (
    XGBOOST_PARAM_DISTRIBUTIONS,
    DEFAULT_XGBOOST_BINARY_PARAMS,
    MODEL_CONFIG
)


def qwk_scorer(y_true, y_pred):
    """
    Quadratic Weighted Kappa scorer for use with sklearn.

    This is appropriate for ordinal classification as it penalizes
    distant misclassifications more heavily than adjacent ones.
    """
    return quadratic_weighted_kappa(y_true, y_pred)


# Create sklearn-compatible scorer
QWK_SCORER = make_scorer(qwk_scorer, greater_is_better=True)


class TunableOrdinalClassifier(BaseEstimator, ClassifierMixin):
    """
    Ordinal classifier wrapper that exposes XGBoost hyperparameters for tuning.

    This wrapper allows RandomizedSearchCV to tune the underlying XGBoost
    parameters while maintaining the Frank-Hall ordinal classification structure.

    Parameters
    ----------
    n_classes : int, default=3
        Number of ordinal classes.

    n_estimators : int, default=100
        Number of boosting rounds.

    learning_rate : float, default=0.25
        Boosting learning rate.

    max_depth : int, default=2
        Maximum tree depth.

    gamma : float, default=0
        Minimum loss reduction for split.

    reg_lambda : float, default=0.1
        L2 regularization term.

    reg_alpha : float, default=0
        L1 regularization term.

    subsample : float, default=0.9
        Subsample ratio of training instances.

    colsample_bytree : float, default=0.9
        Subsample ratio of columns when constructing each tree.

    min_child_weight : int, default=1
        Minimum sum of instance weight needed in a child.

    random_state : int, default=42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_classes: int = 3,
        n_estimators: int = 100,
        learning_rate: float = 0.25,
        max_depth: int = 2,
        gamma: float = 0,
        reg_lambda: float = 0.1,
        reg_alpha: float = 0,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        min_child_weight: int = 1,
        random_state: int = 42
    ):
        self.n_classes = n_classes
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.gamma = gamma
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.random_state = random_state

        self.ordinal_classifier_ = None

    def _create_base_estimator(self) -> XGBClassifier:
        """Create XGBoost base estimator with current parameters."""
        params = DEFAULT_XGBOOST_BINARY_PARAMS.copy()
        params.update({
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'gamma': self.gamma,
            'reg_lambda': self.reg_lambda,
            'reg_alpha': self.reg_alpha,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'min_child_weight': self.min_child_weight,
            'random_state': self.random_state
        })
        return XGBClassifier(**params)

    def fit(self, X, y):
        """Fit the ordinal classifier."""
        base_estimator = self._create_base_estimator()
        self.ordinal_classifier_ = OrdinalClassifier(
            base_estimator=base_estimator,
            n_classes=self.n_classes
        )
        self.ordinal_classifier_.fit(X, y)
        self.classes_ = np.arange(self.n_classes)
        return self

    def predict(self, X):
        """Predict class labels."""
        return self.ordinal_classifier_.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities."""
        return self.ordinal_classifier_.predict_proba(X)

    def get_feature_importances(self, aggregation='mean'):
        """Get aggregated feature importances from binary classifiers."""
        return self.ordinal_classifier_.get_feature_importances(aggregation)


def tune_ordinal_model_qwk(
    X: np.ndarray,
    y: np.ndarray,
    n_iter: int = 500,
    cv: int = 5,
    n_classes: int = 3,
    random_state: int = 42,
    verbose: int = 1,
    n_jobs: int = -1
) -> Tuple[TunableOrdinalClassifier, Dict[str, Any], Any]:
    """
    Tune ordinal classifier hyperparameters using QWK as the scoring metric.

    This is the recommended approach for ordinal classification as QWK
    properly accounts for the ordinal structure of the classes.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training features.

    y : array-like of shape (n_samples,)
        Training labels (ordinal classes 0, 1, 2, ...).

    n_iter : int, default=500
        Number of parameter settings to sample.

    cv : int or cross-validation generator, default=5
        Cross-validation splitting strategy.

    n_classes : int, default=3
        Number of ordinal classes.

    random_state : int, default=42
        Random seed for reproducibility.

    verbose : int, default=1
        Verbosity level.

    n_jobs : int, default=-1
        Number of jobs to run in parallel.

    Returns
    -------
    best_model : TunableOrdinalClassifier
        The best model found during search.

    best_params : dict
        Best hyperparameters found.

    cv_results : dict
        Full cross-validation results.
    """
    # Create base model for tuning
    base_model = TunableOrdinalClassifier(
        n_classes=n_classes,
        random_state=random_state
    )

    # Parameter distributions for tuning
    param_distributions = XGBOOST_PARAM_DISTRIBUTIONS.copy()

    if verbose > 0:
        print(f"Tuning ordinal model with QWK scoring...")
        print(f"  - {n_iter} iterations")
        print(f"  - {cv}-fold cross-validation")
        print(f"  - Optimizing: Quadratic Weighted Kappa (QWK)")

    # Run randomized search with QWK scorer
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=QWK_SCORER,
        cv=cv,
        verbose=verbose,
        random_state=random_state,
        n_jobs=n_jobs,
        return_train_score=True
    )

    search.fit(X, y)

    if verbose > 0:
        print(f"\nBest QWK score: {search.best_score_:.4f}")
        print(f"Best parameters: {search.best_params_}")

    return search.best_estimator_, search.best_params_, search.cv_results_


def compare_tuning_metrics(
    X: np.ndarray,
    y: np.ndarray,
    n_iter: int = 100,
    cv: int = 5,
    random_state: int = 42,
    verbose: int = 1
) -> Dict[str, Dict[str, Any]]:
    """
    Compare hyperparameter tuning using different scoring metrics.

    This function tunes the model using QWK, Macro F1, and Accuracy,
    then compares the resulting models.

    Parameters
    ----------
    X : array-like
        Training features.

    y : array-like
        Training labels.

    n_iter : int, default=100
        Number of iterations per metric (use fewer for comparison).

    cv : int, default=5
        Cross-validation folds.

    random_state : int, default=42
        Random seed.

    verbose : int, default=1
        Verbosity level.

    Returns
    -------
    results : dict
        Dictionary containing best params and scores for each metric.
    """
    from sklearn.metrics import make_scorer, f1_score, accuracy_score

    metrics = {
        'qwk': QWK_SCORER,
        'macro_f1': make_scorer(f1_score, average='macro'),
        'accuracy': make_scorer(accuracy_score)
    }

    results = {}

    for metric_name, scorer in metrics.items():
        if verbose > 0:
            print(f"\n{'='*60}")
            print(f"Tuning with {metric_name.upper()} scoring...")
            print(f"{'='*60}")

        base_model = TunableOrdinalClassifier(n_classes=3, random_state=random_state)

        search = RandomizedSearchCV(
            base_model,
            param_distributions=XGBOOST_PARAM_DISTRIBUTIONS,
            n_iter=n_iter,
            scoring=scorer,
            cv=cv,
            verbose=max(0, verbose - 1),
            random_state=random_state,
            n_jobs=-1
        )

        search.fit(X, y)

        results[metric_name] = {
            'best_score': search.best_score_,
            'best_params': search.best_params_,
            'best_estimator': search.best_estimator_
        }

        if verbose > 0:
            print(f"Best {metric_name} score: {search.best_score_:.4f}")

    return results
