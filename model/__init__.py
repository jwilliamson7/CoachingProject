"""
NFL Coach Tenure Prediction Model Package.

This package provides:
1. Ordinal classification models for predicting NFL head coach tenure
   using the Frank-Hall binary decomposition method.
2. Regression models for predicting coach WAR (Wins Above Replacement).
"""

from .ordinal_classifier import OrdinalClassifier
from .coach_tenure_model import CoachTenureModel
from .war_regressor import WARRegressor
from .cross_validation import (
    stratified_coach_level_split,
    stratified_coach_level_cv_split,
    CoachLevelStratifiedKFold
)
from .evaluation import (
    ordinal_metrics,
    quadratic_weighted_kappa,
    mean_absolute_error_ordinal,
    adjacent_accuracy,
    format_metrics_report,
    regression_metrics,
    format_regression_report
)
from .config import (
    XGBOOST_PARAM_DISTRIBUTIONS,
    DEFAULT_XGBOOST_PARAMS,
    OPTIMIZED_XGBOOST_PARAMS,
    OPTIMIZED_WAR_PARAMS,
    ORDINAL_CONFIG,
    MODEL_CONFIG,
    WAR_CONFIG
)
from .tuning import (
    tune_ordinal_model_qwk,
    TunableOrdinalClassifier,
    QWK_SCORER,
    compare_tuning_metrics
)

__all__ = [
    # Classification models
    'OrdinalClassifier',
    'CoachTenureModel',
    # Regression models
    'WARRegressor',
    # Cross-validation utilities
    'stratified_coach_level_split',
    'stratified_coach_level_cv_split',
    'CoachLevelStratifiedKFold',
    # Classification metrics
    'ordinal_metrics',
    'quadratic_weighted_kappa',
    'mean_absolute_error_ordinal',
    'adjacent_accuracy',
    'format_metrics_report',
    # Regression metrics
    'regression_metrics',
    'format_regression_report',
    # Configuration
    'XGBOOST_PARAM_DISTRIBUTIONS',
    'DEFAULT_XGBOOST_PARAMS',
    'OPTIMIZED_XGBOOST_PARAMS',
    'OPTIMIZED_WAR_PARAMS',
    'ORDINAL_CONFIG',
    'MODEL_CONFIG',
    'WAR_CONFIG',
    # Tuning
    'tune_ordinal_model_qwk',
    'TunableOrdinalClassifier',
    'QWK_SCORER',
    'compare_tuning_metrics'
]
