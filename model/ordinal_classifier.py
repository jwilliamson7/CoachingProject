"""
Ordinal Classification using the Frank-Hall method.

This module implements ordinal classification by decomposing the problem
into binary classification tasks. For K ordinal classes (0, 1, ..., K-1),
we train K-1 binary classifiers:
    - Classifier i: P(Y > i) for i = 0, 1, ..., K-2

Class probabilities are then derived as:
    - P(Y = 0) = 1 - P(Y > 0)
    - P(Y = k) = P(Y > k-1) - P(Y > k) for 0 < k < K-1
    - P(Y = K-1) = P(Y > K-2)
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from xgboost import XGBClassifier
from typing import Optional, List, Dict, Any


class OrdinalClassifier(BaseEstimator, ClassifierMixin):
    """
    Ordinal classifier using Frank-Hall binary decomposition.

    For ordinal classes 0 < 1 < 2, trains two binary classifiers:
        1. P(Y > 0): Distinguishes class 0 from classes 1+2
        2. P(Y > 1): Distinguishes classes 0+1 from class 2

    Parameters
    ----------
    base_estimator : estimator object, default=None
        The base classifier to use for each binary classification task.
        If None, uses XGBClassifier with default parameters.

    n_classes : int, default=3
        Number of ordinal classes. Must be >= 2.

    Attributes
    ----------
    classifiers_ : list of estimators
        The fitted binary classifiers. Length is n_classes - 1.

    classes_ : ndarray of shape (n_classes,)
        The class labels.

    n_features_in_ : int
        Number of features seen during fit.

    Examples
    --------
    >>> from xgboost import XGBClassifier
    >>> clf = OrdinalClassifier(base_estimator=XGBClassifier())
    >>> clf.fit(X_train, y_train)
    >>> probas = clf.predict_proba(X_test)
    >>> predictions = clf.predict(X_test)
    """

    def __init__(
        self,
        base_estimator: Optional[BaseEstimator] = None,
        n_classes: int = 3
    ):
        self.base_estimator = base_estimator
        self.n_classes = n_classes

    def fit(self, X, y, **fit_params):
        """
        Fit the ordinal classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values. Must be ordinal integers 0, 1, ..., n_classes-1.

        **fit_params : dict
            Additional parameters passed to the base estimator's fit method.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = check_X_y(X, y)

        self.classes_ = np.arange(self.n_classes)
        self.n_features_in_ = X.shape[1]

        # Validate target values
        unique_classes = np.unique(y)
        if not np.all(np.isin(unique_classes, self.classes_)):
            raise ValueError(
                f"y contains classes {unique_classes} not in expected "
                f"range [0, {self.n_classes - 1}]"
            )

        # Initialize base estimator
        if self.base_estimator is None:
            base = XGBClassifier(
                objective='binary:logistic',
                n_jobs=-1,
                random_state=42
            )
        else:
            base = self.base_estimator

        # Train K-1 binary classifiers
        self.classifiers_ = []

        for threshold in range(self.n_classes - 1):
            # Create binary target: 1 if y > threshold, 0 otherwise
            y_binary = (y > threshold).astype(int)

            # Clone the base estimator for this threshold
            clf = clone(base)

            # Ensure binary objective for XGBoost
            if isinstance(clf, XGBClassifier):
                clf.set_params(objective='binary:logistic')

            # Fit the binary classifier
            clf.fit(X, y_binary, **fit_params)
            self.classifiers_.append(clf)

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities. Rows sum to 1.
        """
        check_is_fitted(self)
        X = check_array(X)

        n_samples = X.shape[0]

        # Get P(Y > k) for each threshold k
        cumulative_probs = np.zeros((n_samples, self.n_classes - 1))

        for i, clf in enumerate(self.classifiers_):
            # Get probability of positive class (Y > threshold)
            proba = clf.predict_proba(X)
            cumulative_probs[:, i] = proba[:, 1]

        # Derive class probabilities
        class_probs = np.zeros((n_samples, self.n_classes))

        # P(Y = 0) = 1 - P(Y > 0)
        class_probs[:, 0] = 1 - cumulative_probs[:, 0]

        # P(Y = k) = P(Y > k-1) - P(Y > k) for 0 < k < K-1
        for k in range(1, self.n_classes - 1):
            class_probs[:, k] = cumulative_probs[:, k-1] - cumulative_probs[:, k]

        # P(Y = K-1) = P(Y > K-2)
        class_probs[:, -1] = cumulative_probs[:, -1]

        # Handle numerical issues: clip to [0, 1] and renormalize
        class_probs = np.clip(class_probs, 0, 1)
        row_sums = class_probs.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums = np.where(row_sums == 0, 1, row_sums)
        class_probs = class_probs / row_sums

        return class_probs

    def predict(self, X):
        """
        Predict class labels for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def get_feature_importances(self, aggregation: str = 'mean') -> np.ndarray:
        """
        Get aggregated feature importances across all binary classifiers.

        Parameters
        ----------
        aggregation : str, default='mean'
            How to aggregate importances: 'mean', 'max', or 'sum'.

        Returns
        -------
        importances : ndarray of shape (n_features,)
            Aggregated feature importances.
        """
        check_is_fitted(self)

        # Collect importances from each classifier
        all_importances = []
        for clf in self.classifiers_:
            if hasattr(clf, 'feature_importances_'):
                all_importances.append(clf.feature_importances_)

        if not all_importances:
            raise AttributeError(
                "Base estimator does not have feature_importances_ attribute"
            )

        importances = np.array(all_importances)

        if aggregation == 'mean':
            return importances.mean(axis=0)
        elif aggregation == 'max':
            return importances.max(axis=0)
        elif aggregation == 'sum':
            return importances.sum(axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        params = {
            'base_estimator': self.base_estimator,
            'n_classes': self.n_classes
        }
        if deep and self.base_estimator is not None:
            base_params = self.base_estimator.get_params(deep=True)
            params.update({f'base_estimator__{k}': v for k, v in base_params.items()})
        return params

    def set_params(self, **params) -> 'OrdinalClassifier':
        """Set parameters for this estimator."""
        # Handle base_estimator parameters
        base_params = {}
        own_params = {}

        for key, value in params.items():
            if key.startswith('base_estimator__'):
                base_params[key[16:]] = value
            else:
                own_params[key] = value

        # Set own parameters
        for key, value in own_params.items():
            setattr(self, key, value)

        # Set base estimator parameters
        if base_params and self.base_estimator is not None:
            self.base_estimator.set_params(**base_params)

        return self
