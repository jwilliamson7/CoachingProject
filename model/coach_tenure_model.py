"""
Main NFL Coach Tenure Prediction Model.

This module provides the CoachTenureModel class, which wraps ordinal
and multiclass classification approaches for predicting NFL head coach tenure.
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from typing import Dict, Any, Optional, Tuple, Union

from .ordinal_classifier import OrdinalClassifier
from .cross_validation import (
    stratified_coach_level_split,
    stratified_coach_level_cv_split,
    CoachLevelStratifiedKFold
)
from .evaluation import ordinal_metrics, format_metrics_report, quadratic_weighted_kappa
from .config import (
    XGBOOST_PARAM_DISTRIBUTIONS,
    DEFAULT_XGBOOST_PARAMS,
    DEFAULT_XGBOOST_BINARY_PARAMS,
    OPTIMIZED_XGBOOST_PARAMS,
    ORDINAL_CONFIG,
    MODEL_CONFIG,
    FEATURE_CONFIG,
    CONFIDENCE_THRESHOLDS,
    get_combined_xgboost_params,
    get_binary_xgboost_params
)


class CoachTenureModel:
    """
    NFL Coach Tenure Prediction Model.

    Supports both ordinal classification (Frank-Hall method) and standard
    multiclass classification using XGBoost.

    Parameters
    ----------
    use_ordinal : bool, default=True
        If True, use ordinal classification. If False, use standard multiclass.

    n_classes : int, default=3
        Number of tenure classes.

    xgboost_params : dict, optional
        Custom XGBoost parameters. If None, uses optimized defaults.

    random_state : int, default=42
        Random seed for reproducibility.

    Attributes
    ----------
    model_ : OrdinalClassifier or XGBClassifier
        The fitted model.

    imputer_ : SimpleImputer
        The fitted imputer for handling missing values.

    best_params_ : dict
        Best hyperparameters found during training (if tuning was performed).

    cv_results_ : dict
        Cross-validation results (if CV was performed).
    """

    def __init__(
        self,
        use_ordinal: bool = True,
        n_classes: int = 3,
        xgboost_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42
    ):
        self.use_ordinal = use_ordinal
        self.n_classes = n_classes
        self.xgboost_params = xgboost_params
        self.random_state = random_state

        self.model_ = None
        self.imputer_ = None
        self.best_params_ = None
        self.cv_results_ = None
        self._is_fitted = False

    def _create_base_estimator(self) -> XGBClassifier:
        """Create the base XGBoost estimator with appropriate parameters."""
        if self.use_ordinal:
            params = get_binary_xgboost_params(self.xgboost_params)
        else:
            params = get_combined_xgboost_params(self.xgboost_params)
            params['num_class'] = self.n_classes

        return XGBClassifier(**params)

    def _create_model(self) -> Union[OrdinalClassifier, XGBClassifier]:
        """Create the classifier based on configuration."""
        base_estimator = self._create_base_estimator()

        if self.use_ordinal:
            return OrdinalClassifier(
                base_estimator=base_estimator,
                n_classes=self.n_classes
            )
        else:
            return base_estimator

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        tune_hyperparameters: bool = False,
        n_iter: int = 500,
        cv_splits: Optional[list] = None,
        verbose: int = 1
    ) -> 'CoachTenureModel':
        """
        Fit the model to training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.

        y : array-like of shape (n_samples,)
            Training labels.

        tune_hyperparameters : bool, default=False
            If True, perform hyperparameter tuning with RandomizedSearchCV.

        n_iter : int, default=500
            Number of parameter settings sampled for RandomizedSearchCV.

        cv_splits : list, optional
            Pre-computed CV splits. If None, uses 3-fold CV for tuning.

        verbose : int, default=1
            Verbosity level.

        Returns
        -------
        self : CoachTenureModel
            Fitted model.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Impute missing values
        self.imputer_ = SimpleImputer(strategy='mean')
        X_imputed = self.imputer_.fit_transform(X)

        if tune_hyperparameters:
            self._fit_with_tuning(X_imputed, y, n_iter, cv_splits, verbose)
        else:
            self._fit_direct(X_imputed, y, verbose)

        self._is_fitted = True
        return self

    def _fit_direct(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: int = 1
    ) -> None:
        """Fit model directly without hyperparameter tuning."""
        if verbose > 0:
            model_type = "ordinal" if self.use_ordinal else "multiclass"
            print(f"Training {model_type} model on {X.shape[0]} instances...")

        self.model_ = self._create_model()
        self.model_.fit(X, y)

        if verbose > 0:
            print("Training complete.")

    def _fit_with_tuning(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_iter: int,
        cv_splits: Optional[list],
        verbose: int
    ) -> None:
        """Fit model with hyperparameter tuning."""
        if verbose > 0:
            model_type = "ordinal" if self.use_ordinal else "multiclass"
            print(f"Tuning {model_type} model with {n_iter} iterations...")

        # For ordinal classification, tune the full ordinal model on QWK
        if self.use_ordinal:
            base_params = get_binary_xgboost_params()
            base_estimator = XGBClassifier(**base_params)

            ordinal_estimator = OrdinalClassifier(
                base_estimator=base_estimator,
                n_classes=self.n_classes,
            )

            # Prefix param keys for nested estimator
            ordinal_param_distributions = {
                f'base_estimator__{k}': v
                for k, v in XGBOOST_PARAM_DISTRIBUTIONS.items()
            }

            cv = cv_splits if cv_splits else 3
            qwk_scorer = make_scorer(quadratic_weighted_kappa)

            search = RandomizedSearchCV(
                ordinal_estimator,
                param_distributions=ordinal_param_distributions,
                n_iter=n_iter,
                scoring=qwk_scorer,
                n_jobs=-1,
                cv=cv,
                verbose=verbose,
                random_state=self.random_state
            )

            search.fit(X, y)

            # Extract the base estimator params (strip prefix for storage)
            self.best_params_ = {
                k.replace('base_estimator__', ''): v
                for k, v in search.best_params_.items()
            }
            self.model_ = search.best_estimator_

        else:
            # Multiclass tuning
            base_params = get_combined_xgboost_params()
            base_params['num_class'] = self.n_classes
            base_estimator = XGBClassifier(**base_params)

            cv = cv_splits if cv_splits else 3

            search = RandomizedSearchCV(
                base_estimator,
                param_distributions=XGBOOST_PARAM_DISTRIBUTIONS,
                n_iter=n_iter,
                scoring='roc_auc_ovr',
                n_jobs=-1,
                cv=cv,
                verbose=verbose,
                random_state=self.random_state
            )

            search.fit(X, y)
            self.best_params_ = search.best_params_
            self.model_ = search.best_estimator_

        if verbose > 0:
            print(f"Best parameters: {self.best_params_}")

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class labels for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        self._check_is_fitted()
        X = np.asarray(X)
        X_imputed = self.imputer_.transform(X)
        return self.model_.predict(X_imputed)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        self._check_is_fitted()
        X = np.asarray(X)
        X_imputed = self.imputer_.transform(X)
        return self.model_.predict_proba(X_imputed)

    def evaluate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        print_report: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test features.

        y : array-like of shape (n_samples,)
            True labels.

        print_report : bool, default=True
            If True, print formatted metrics report.

        Returns
        -------
        metrics : dict
            Dictionary containing all evaluation metrics.
        """
        self._check_is_fitted()

        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)

        metrics = ordinal_metrics(
            y_true=y,
            y_pred=y_pred,
            y_proba=y_proba,
            class_names=ORDINAL_CONFIG['class_names']
        )

        if print_report:
            model_type = "Ordinal" if self.use_ordinal else "Multiclass"
            title = f"NFL Coach Tenure Model ({model_type})"
            print(format_metrics_report(metrics, title))

        return metrics

    def get_feature_importances(self) -> np.ndarray:
        """
        Get feature importances from the model.

        Returns
        -------
        importances : ndarray of shape (n_features,)
            Feature importances.
        """
        self._check_is_fitted()

        if self.use_ordinal:
            return self.model_.get_feature_importances(aggregation='mean')
        else:
            return self.model_.feature_importances_

    def predict_with_confidence(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with confidence levels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Predicted class labels.

        max_probas : ndarray of shape (n_samples,)
            Maximum probability for each prediction.

        confidence_levels : ndarray of shape (n_samples,)
            Confidence level strings ('HIGH', 'MED', 'LOW').
        """
        self._check_is_fitted()

        predictions = self.predict(X)
        probas = self.predict_proba(X)
        max_probas = probas.max(axis=1)

        confidence_levels = np.array([
            'HIGH' if p >= CONFIDENCE_THRESHOLDS['high']
            else 'MED' if p >= CONFIDENCE_THRESHOLDS['medium']
            else 'LOW'
            for p in max_probas
        ])

        return predictions, max_probas, confidence_levels

    def save(self, filepath: str) -> None:
        """
        Save the model to a file.

        Parameters
        ----------
        filepath : str
            Path to save the model.
        """
        self._check_is_fitted()

        model_data = {
            'use_ordinal': self.use_ordinal,
            'n_classes': self.n_classes,
            'xgboost_params': self.xgboost_params,
            'random_state': self.random_state,
            'model': self.model_,
            'imputer': self.imputer_,
            'best_params': self.best_params_,
            'cv_results': self.cv_results_
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'CoachTenureModel':
        """
        Load a model from a file.

        Parameters
        ----------
        filepath : str
            Path to the saved model.

        Returns
        -------
        model : CoachTenureModel
            Loaded model.
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        model = cls(
            use_ordinal=model_data['use_ordinal'],
            n_classes=model_data['n_classes'],
            xgboost_params=model_data['xgboost_params'],
            random_state=model_data['random_state']
        )

        model.model_ = model_data['model']
        model.imputer_ = model_data['imputer']
        model.best_params_ = model_data['best_params']
        model.cv_results_ = model_data['cv_results']
        model._is_fitted = True

        print(f"Model loaded from {filepath}")
        return model

    def _check_is_fitted(self) -> None:
        """Check if the model is fitted."""
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

    def __repr__(self) -> str:
        model_type = "ordinal" if self.use_ordinal else "multiclass"
        fitted_str = "fitted" if self._is_fitted else "not fitted"
        return f"CoachTenureModel(type={model_type}, n_classes={self.n_classes}, {fitted_str})"
