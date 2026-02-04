"""
NFL Coach Tenure Prediction Model Package.

This package provides ordinal classification models for predicting
NFL head coach tenure using the Frank-Hall binary decomposition method.
"""

from .ordinal_classifier import OrdinalClassifier
from .coach_tenure_model import CoachTenureModel
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
    format_metrics_report
)
from .config import (
    XGBOOST_PARAM_DISTRIBUTIONS,
    DEFAULT_XGBOOST_PARAMS,
    ORDINAL_CONFIG,
    MODEL_CONFIG
)

__all__ = [
    'OrdinalClassifier',
    'CoachTenureModel',
    'stratified_coach_level_split',
    'stratified_coach_level_cv_split',
    'CoachLevelStratifiedKFold',
    'ordinal_metrics',
    'quadratic_weighted_kappa',
    'mean_absolute_error_ordinal',
    'adjacent_accuracy',
    'format_metrics_report',
    'XGBOOST_PARAM_DISTRIBUTIONS',
    'DEFAULT_XGBOOST_PARAMS',
    'ORDINAL_CONFIG',
    'MODEL_CONFIG'
]
