"""
WAR (Wins Above Replacement) Regression Model for NFL Coaches.

This module provides an XGBoost-based regression model for predicting
average WAR per season for NFL head coaches.
"""

import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb

from .config import (
    DEFAULT_XGBOOST_REGRESSION_PARAMS,
    OPTIMIZED_XGBOOST_PARAMS,
    XGBOOST_PARAM_DISTRIBUTIONS,
    get_regression_xgboost_params,
    WAR_CONFIG,
    MODEL_CONFIG
)
from .evaluation import regression_metrics, format_regression_report


class WARRegressor:
    """
    XGBoost regression model for predicting coach WAR.

    This model predicts average WAR per season during a coach's tenure
    using 140 features (core experience + coordinator stats, excluding
    hiring team factors).

    Parameters
    ----------
    **xgb_params : dict
        Custom XGBoost parameters to override defaults.

    Attributes
    ----------
    model_ : xgb.XGBRegressor
        Fitted XGBoost model.
    imputer_ : SimpleImputer
        Fitted imputer for handling missing values.
    best_params_ : dict
        Best parameters found during hyperparameter tuning.
    cv_results_ : dict
        Cross-validation results from RandomizedSearchCV.

    Examples
    --------
    >>> from model.war_regressor import WARRegressor
    >>> model = WARRegressor()
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    >>> metrics = model.evaluate(X_test, y_test)
    """

    def __init__(self, **xgb_params):
        self.xgb_params = xgb_params
        self.model_: Optional[xgb.XGBRegressor] = None
        self.imputer_: Optional[SimpleImputer] = None
        self.best_params_: Optional[Dict[str, Any]] = None
        self.cv_results_: Optional[Dict[str, Any]] = None
        self._is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        tune_hyperparameters: bool = False,
        n_iter: int = 500,
        cv_splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        verbose: bool = True
    ) -> 'WARRegressor':
        """
        Fit the WAR regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.

        y : array-like of shape (n_samples,)
            Training target values (avg WAR per season).

        tune_hyperparameters : bool, default=False
            Whether to perform hyperparameter tuning with RandomizedSearchCV.

        n_iter : int, default=500
            Number of iterations for RandomizedSearchCV.

        cv_splits : list of (train_idx, val_idx) tuples, optional
            Pre-computed CV splits for coach-level cross-validation.
            If None and tune_hyperparameters=True, uses standard k-fold.

        verbose : bool, default=True
            Whether to print progress information.

        Returns
        -------
        self : WARRegressor
            Fitted model instance.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Handle missing values
        self.imputer_ = SimpleImputer(strategy='mean')
        X_imputed = self.imputer_.fit_transform(X)

        # Get XGBoost parameters
        params = get_regression_xgboost_params(self.xgb_params)

        if tune_hyperparameters:
            self._fit_with_tuning(
                X_imputed, y, params, n_iter, cv_splits, verbose
            )
        else:
            self._fit_direct(X_imputed, y, params, verbose)

        self._is_fitted = True
        return self

    def _fit_direct(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: Dict[str, Any],
        verbose: bool
    ):
        """Fit model directly with specified parameters."""
        if verbose:
            print("Training WAR regression model...")

        self.model_ = xgb.XGBRegressor(**params)
        self.model_.fit(X, y)
        self.best_params_ = params

        if verbose:
            print("Training complete.")

    def _fit_with_tuning(
        self,
        X: np.ndarray,
        y: np.ndarray,
        base_params: Dict[str, Any],
        n_iter: int,
        cv_splits: Optional[List],
        verbose: bool
    ):
        """Fit model with hyperparameter tuning."""
        if verbose:
            print(f"Starting hyperparameter tuning with {n_iter} iterations...")

        # Create base estimator
        estimator = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_jobs=-1,
            tree_method='hist',
            random_state=MODEL_CONFIG['random_state']
        )

        # Prepare CV
        cv = cv_splits if cv_splits is not None else MODEL_CONFIG['cv_folds']

        # Run RandomizedSearchCV
        search = RandomizedSearchCV(
            estimator,
            param_distributions=XGBOOST_PARAM_DISTRIBUTIONS,
            n_iter=n_iter,
            scoring='neg_mean_squared_error',
            cv=cv,
            n_jobs=-1,
            verbose=2 if verbose else 0,
            random_state=MODEL_CONFIG['random_state']
        )

        search.fit(X, y)

        self.model_ = search.best_estimator_
        self.best_params_ = search.best_params_
        self.cv_results_ = search.cv_results_

        if verbose:
            print(f"\nBest parameters: {self.best_params_}")
            print(f"Best CV score (neg MSE): {search.best_score_:.6f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict WAR values for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted WAR values.
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")

        X = np.asarray(X)
        X_imputed = self.imputer_.transform(X)
        return self.model_.predict(X_imputed)

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test feature matrix.

        y : array-like of shape (n_samples,)
            True target values.

        verbose : bool, default=True
            Whether to print evaluation report.

        Returns
        -------
        metrics : dict
            Dictionary containing MAE, MSE, RMSE, R², and correlation.
        """
        y_pred = self.predict(X)
        metrics = regression_metrics(y, y_pred)

        if verbose:
            report = format_regression_report(
                metrics,
                title="WAR Prediction Evaluation"
            )
            print(report)

        return metrics

    def get_feature_importances(self) -> np.ndarray:
        """
        Get feature importances from the fitted model.

        Returns
        -------
        importances : array of shape (n_features,)
            Feature importance scores (gain-based).
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")

        return self.model_.feature_importances_

    def save(self, filepath: str):
        """
        Save the fitted model to disk.

        Parameters
        ----------
        filepath : str
            Path to save the model file.
        """
        if not self._is_fitted:
            raise ValueError("Model has not been fitted. Call fit() first.")

        model_data = {
            'model': self.model_,
            'imputer': self.imputer_,
            'best_params': self.best_params_,
            'cv_results': self.cv_results_,
            'xgb_params': self.xgb_params
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'WARRegressor':
        """
        Load a fitted model from disk.

        Parameters
        ----------
        filepath : str
            Path to the saved model file.

        Returns
        -------
        model : WARRegressor
            Loaded model instance.
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        model = cls(**model_data.get('xgb_params', {}))
        model.model_ = model_data['model']
        model.imputer_ = model_data['imputer']
        model.best_params_ = model_data['best_params']
        model.cv_results_ = model_data.get('cv_results')
        model._is_fitted = True

        return model
