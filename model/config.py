"""
Configuration and hyperparameters for NFL coach tenure prediction models.

This module contains hyperparameter distributions for model tuning,
default model parameters, and configuration settings.
"""

from typing import Dict, List, Any

# Hyperparameter distributions for RandomizedSearchCV
# Based on optimal ranges discovered in v5 notebook experiments
XGBOOST_PARAM_DISTRIBUTIONS: Dict[str, List] = {
    "n_estimators": [25, 50, 100, 200],
    "learning_rate": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
    "max_depth": [2, 3, 4],
    "gamma": [0, 0.01, 0.05, 0.1],
    "reg_lambda": [0, 0.01, 0.1, 0.5],
    "reg_alpha": [0, 0.01, 0.1],
    "subsample": [0.8, 0.85, 0.9, 0.95, 1.0],
    "colsample_bytree": [0.8, 0.85, 0.9, 0.95, 1.0],
    "min_child_weight": [1, 2, 3, 4, 5]
}

# Default XGBoost parameters for multiclass classification
DEFAULT_XGBOOST_PARAMS: Dict[str, Any] = {
    'verbosity': 1,
    'objective': 'multi:softprob',
    'n_jobs': -1,
    'tree_method': 'hist',
    'max_bin': 256,
    'random_state': 42
}

# Default XGBoost parameters for binary classification (ordinal sub-classifiers)
DEFAULT_XGBOOST_BINARY_PARAMS: Dict[str, Any] = {
    'verbosity': 1,
    'objective': 'binary:logistic',
    'n_jobs': -1,
    'tree_method': 'hist',
    'max_bin': 256,
    'random_state': 42
}

# Default XGBoost parameters for regression (WAR prediction)
DEFAULT_XGBOOST_REGRESSION_PARAMS: Dict[str, Any] = {
    'verbosity': 1,
    'objective': 'reg:squarederror',
    'n_jobs': -1,
    'tree_method': 'hist',
    'max_bin': 256,
    'random_state': 42
}

# Optimal parameters discovered from 1000-iteration RandomizedSearchCV
# with 5-fold coach-level cross-validation, QWK-scored.
# Re-tuned LEAKAGE-FREE (Jun 2026): imputation is fit on the training partition
# only (SVD imputer, feature columns only) before tuning, so these parameters
# are optimized for the honest problem rather than the leaked one. The previous
# values (n_estimators=200, lr=0.25, reg_lambda=0.1, reg_alpha=0.01,
# min_child_weight=3) were tuned on the leaked, target-contaminated matrix.
OPTIMIZED_XGBOOST_PARAMS: Dict[str, Any] = {
    # Final config (Jun 2026): re-tuned leakage-free on the modern-era (1970+),
    # 171-feature dataset with relocation/partial-season-corrected tenure labels,
    # QWK-scored via 1000-iteration RandomizedSearchCV with 10-fold COACH-LEVEL
    # cross-validation (imputation fit on each training partition only). The
    # deeper 10-fold CV selects a regularized optimum (shallow depth-2 trees,
    # 50 rounds), appropriate for the modest sample (N=373). Best CV QWK 0.339.
    # (The QWK landscape is flat, so the optimum is only weakly identified; this
    # set is locked for reproducibility.) Previous 5-fold tune yielded a less
    # regularized set (max_depth=4, subsample=1.0) that generalized slightly worse.
    'n_estimators': 50,
    'learning_rate': 0.1,
    'max_depth': 2,
    'gamma': 0.05,
    'reg_lambda': 0,
    'reg_alpha': 0.01,
    'subsample': 0.85,
    'colsample_bytree': 0.9,
    'min_child_weight': 4
}

# Optimal parameters for WAR regression from 1000-iteration RandomizedSearchCV
# with 5-fold coach-level cross-validation, optimized for MSE (Feb 2026)
# Model trained EXCLUDING recent hires (tenure class -1) to avoid data leakage
# Test set: R² = 0.276, MAE = 0.051, Correlation = 0.525
OPTIMIZED_WAR_PARAMS: Dict[str, Any] = {
    'n_estimators': 200,
    'learning_rate': 0.1,
    'max_depth': 3,
    'gamma': 0,
    'reg_lambda': 0,
    'reg_alpha': 0,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'min_child_weight': 1
}

# Ordinal classification configuration
ORDINAL_CONFIG: Dict[str, Any] = {
    'n_classes': 3,
    'class_names': [
        'Class 0 (1-2 yrs)',
        'Class 1 (3-4 yrs)',
        'Class 2 (5+ yrs)'
    ],
    'class_thresholds': {
        0: (1, 2),   # 1-2 years tenure
        1: (3, 4),   # 3-4 years tenure
        2: (5, None) # 5+ years tenure
    }
}

# WAR regression model configuration
WAR_CONFIG: Dict[str, Any] = {
    'n_features': 140,  # Excludes hiring team factors (Features 141-150)
    'feature_columns_start': 2,  # Features start after Coach Name, Year
    'feature_columns_end': 142,  # Features 1-140 (0-indexed: columns 2-141)
    'target_column': 'avg_war_per_season',
    'coach_name_column': 'Coach Name',
    'year_column': 'Year'
}

# Model training configuration
MODEL_CONFIG: Dict[str, Any] = {
    'cv_folds': 5,
    'test_size': 0.2,
    'random_state': 42,
    'n_iter_random_search': 500,
    'scoring': 'roc_auc_ovr',
    'internal_cv_folds': 3  # CV folds within RandomizedSearchCV
}

# File paths (relative to project root)
MODEL_PATHS: Dict[str, str] = {
    'data_dir': 'data',
    # DEPRECATED (Jun 2026): this pre-imputed file leaked the imputation across
    # the train/test split (and the target sat in the SVD matrix). It has been
    # deleted; the corrected pipeline imputes per split from 'raw_data_file'.
    # The path is kept only so any unported legacy script fails loudly rather
    # than silently reading stale/leaked data.
    'data_file': 'data/svd_imputed_master_data.csv',
    # Canonical modeling input (Jun 2026): modern-era (1970+) raw features with the
    # Tier 1+2 engineered career-path/rank features appended (150 + 21 = 171),
    # un-imputed (imputation is fit per train split). Built by
    # scripts/data/engineer_career_features.py. The original all-era 150-feature
    # raw file remains at data/master_data.csv.
    'raw_data_file': 'data/master_data_extended.csv',
    'raw_data_file_original': 'data/master_data.csv',
    'models_dir': 'data/models',
    'default_model_output': 'data/models/coach_tenure_model.pkl',
    'ordinal_model_output': 'data/models/coach_tenure_ordinal_model.pkl',
    'multiclass_model_output': 'data/models/coach_tenure_multiclass_model.pkl',
    'war_data_file': 'data/war_prediction_data.csv',
    'war_trajectories_file': 'data/coach_war_trajectories_with_team.csv',
    'war_model_output': 'data/models/coach_war_model.pkl'
}

# Feature configuration
FEATURE_CONFIG: Dict[str, Any] = {
    'n_features': 150,
    'feature_columns_start': 2,  # Features start at column 2 (after Coach Name, Year)
    'feature_columns_end': -2,   # Features end at column -2 (before Win Pct, Tenure Class)
    'coach_name_column': 'Coach Name',
    'year_column': 'Year',
    'target_column': 'Coach Tenure Class',
    'win_pct_column': 'Avg 2Y Win Pct'
}

# Confidence thresholds for predictions
CONFIDENCE_THRESHOLDS: Dict[str, float] = {
    'high': 0.7,
    'medium': 0.5,
    'low': 0.0
}


def get_total_param_combinations() -> int:
    """Calculate total number of hyperparameter combinations."""
    total = 1
    for values in XGBOOST_PARAM_DISTRIBUTIONS.values():
        total *= len(values)
    return total


def get_combined_xgboost_params(custom_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get combined XGBoost parameters (default + optimized + custom).

    Parameters
    ----------
    custom_params : dict, optional
        Custom parameters to override defaults.

    Returns
    -------
    params : dict
        Combined parameter dictionary.
    """
    params = DEFAULT_XGBOOST_PARAMS.copy()
    params.update(OPTIMIZED_XGBOOST_PARAMS)

    if custom_params:
        params.update(custom_params)

    return params


def get_binary_xgboost_params(custom_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get XGBoost parameters for binary classification (ordinal sub-classifiers).

    Parameters
    ----------
    custom_params : dict, optional
        Custom parameters to override defaults.

    Returns
    -------
    params : dict
        Combined parameter dictionary for binary classification.
    """
    params = DEFAULT_XGBOOST_BINARY_PARAMS.copy()
    params.update(OPTIMIZED_XGBOOST_PARAMS)

    if custom_params:
        params.update(custom_params)

    return params


def get_regression_xgboost_params(custom_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get XGBoost parameters for regression (WAR prediction).

    Parameters
    ----------
    custom_params : dict, optional
        Custom parameters to override defaults.

    Returns
    -------
    params : dict
        Combined parameter dictionary for regression.
    """
    params = DEFAULT_XGBOOST_REGRESSION_PARAMS.copy()
    params.update(OPTIMIZED_WAR_PARAMS)

    if custom_params:
        params.update(custom_params)

    return params
