"""
Shared leakage-free modeling pipeline for NFL coach tenure prediction.

Every analysis (feature selection, confusion matrix, SHAP validation, the
regression baseline, the ordinal-vs-multiclass comparison, and recent-hire
prediction) uses the SAME core routine:

    coach-level split  ->  SVD imputation fit on the TRAIN partition only
                       ->  apply to both partitions
                       ->  select the top-K SHAP-ranked feature columns
                       ->  fit the model

This module is the single source of that routine so the scripts stay thin and
provably consistent. It also centralizes loading the modeling data, the cached
SHAP ranking, and the best-K read from the parsimony results.

Nothing here imputes across the split or lets the target enter the imputation
matrix (the two leaks that inflated the original results); imputation operates on
feature columns only and is re-fit for every split / training population.
"""

import os
import pickle
from collections import namedtuple

import numpy as np
import pandas as pd

from .coach_tenure_model import CoachTenureModel
from .cross_validation import stratified_coach_level_split
from .config import MODEL_CONFIG, MODEL_PATHS, FEATURE_CONFIG

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FSTART = FEATURE_CONFIG['feature_columns_start']
FEND = FEATURE_CONFIG['feature_columns_end']
TARGET = FEATURE_CONFIG['target_column']

# Imputed/selected train & test arrays plus the fitted imputer and the original
# row indices (so callers can align an alternative target, e.g. continuous tenure).
SplitData = namedtuple(
    'SplitData',
    ['X_train', 'X_test', 'y_train', 'y_test', 'imputer', 'train_index', 'test_index'],
)


# --------------------------------------------------------------------------- #
# Data / ranking loaders
# --------------------------------------------------------------------------- #
def load_modeling_data(known_only=True):
    """Load the canonical raw modeling data (171 features, un-imputed).

    Returns (df, X, y). When known_only, drops recent hires (tenure class -1).
    """
    df = pd.read_csv(os.path.join(PROJECT_ROOT, MODEL_PATHS['raw_data_file']), index_col=0)
    if known_only:
        df = df[df[TARGET] != -1].copy()
    X = df.iloc[:, FSTART:FEND]
    y = df[TARGET]
    return df, X, y


def load_recent_hires():
    """Load the recent-hire prediction set (tenure class == -1)."""
    df = pd.read_csv(os.path.join(PROJECT_ROOT, MODEL_PATHS['raw_data_file']), index_col=0)
    return df[df[TARGET] == -1].copy()


def shap_feature_ranking():
    """Feature indices ordered by mean |SHAP| from the cached leakage-free values."""
    with open(os.path.join(PROJECT_ROOT, 'data', 'shap_values_cache.pkl'), 'rb') as f:
        agg = pickle.load(f)['aggregated_shap']
    mean_abs = np.abs(agg.values).mean(axis=0)
    return np.argsort(mean_abs)[::-1]


def best_k(default=20):
    """Best feature count by mean QWK from the locked parsimony results."""
    path = os.path.join(PROJECT_ROOT, 'analysis', 'parsimony_results.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            res = pickle.load(f)['results']
        return max(res, key=lambda k: res[k]['qwk']['mean'])
    return default


def top_k_indices(k=None):
    """Top-K SHAP-ranked feature indices in importance order (unsorted)."""
    if k is None:
        k = best_k()
    return np.asarray(shap_feature_ranking()[:k])


# --------------------------------------------------------------------------- #
# Core leakage-free routine
# --------------------------------------------------------------------------- #
def _svd_imputer():
    # Lazy import keeps the model package importable without the scripts package.
    from scripts.data.matrix_factorization_imputation import SVDImputer
    return SVDImputer()


def leakage_free_split(df, X, y, seed, feature_indices=None, test_size=None):
    """One coach-level split with leakage-free imputation and feature selection.

    The SVD imputer is fit on the training partition only (feature columns,
    never the target), applied to both partitions, then ``feature_indices`` (if
    given) selects the modeled columns. Returns a :class:`SplitData`.
    """
    if test_size is None:
        test_size = MODEL_CONFIG['test_size']

    Xtr_df, Xte_df, ytr, yte, _ = stratified_coach_level_split(
        df, X, y, test_size=test_size, random_state=seed)

    imp = _svd_imputer().fit(np.asarray(Xtr_df))
    Xtr = imp.transform(np.asarray(Xtr_df))
    Xte = imp.transform(np.asarray(Xte_df))
    if feature_indices is not None:
        idx = np.asarray(feature_indices)
        Xtr, Xte = Xtr[:, idx], Xte[:, idx]

    return SplitData(Xtr, Xte, np.asarray(ytr), np.asarray(yte),
                     imp, Xtr_df.index, Xte_df.index)


def fit_on_full(X_df, y, feature_indices=None, model_factory=None):
    """Fit imputer + model on a whole training population (no held-out split).

    Used for recent-hire prediction: imputer fit on all known coaches, model
    trained on the selected columns. Returns (model, imputer).
    """
    Xa = np.asarray(X_df)
    imp = _svd_imputer().fit(Xa)
    Xt = imp.transform(Xa)
    if feature_indices is not None:
        Xt = Xt[:, np.asarray(feature_indices)]
    model = (model_factory or ordinal_model)()
    model.fit(pd.DataFrame(Xt), pd.Series(np.asarray(y)), verbose=0)
    return model, imp


def transform_select(X_df, imputer, feature_indices=None):
    """Apply a fitted imputer to new rows and select the modeled columns."""
    Xt = imputer.transform(np.asarray(X_df))
    return Xt[:, np.asarray(feature_indices)] if feature_indices is not None else Xt


# --------------------------------------------------------------------------- #
# Model factories
# --------------------------------------------------------------------------- #
def ordinal_model(seed=42):
    return CoachTenureModel(use_ordinal=True, n_classes=3, random_state=seed)


def multiclass_model(seed=42):
    return CoachTenureModel(use_ordinal=False, n_classes=3, random_state=seed)


def fit_model(split, model_factory=ordinal_model, seed=42):
    """Fit a model on a SplitData's training partition and return it."""
    model = model_factory(seed)
    model.fit(pd.DataFrame(split.X_train), pd.Series(split.y_train), verbose=0)
    return model


# --------------------------------------------------------------------------- #
# Deployed (production) model: trained once on all known coaches, saved to disk
# --------------------------------------------------------------------------- #
# Unlike the per-split evaluation, the deployed model IS a single artifact: the
# imputer is fit on the whole known-tenure population, the top-K columns are
# selected, and one ordinal model is trained. We persist the imputer, the
# selected indices, and the model together so prediction is a pure load+apply.
PRODUCTION_MODEL_PATH = os.path.join(PROJECT_ROOT, MODEL_PATHS['models_dir'],
                                     'coach_tenure_production.pkl')


def build_production_model():
    """Fit the deployed bundle on the full known-tenure population."""
    df, X, y = load_modeling_data(known_only=True)
    idx = top_k_indices()
    model, imputer = fit_on_full(X, y, feature_indices=idx)
    return {'model': model, 'imputer': imputer, 'feature_indices': idx,
            'best_k': int(len(idx)), 'n_train': int(len(df))}


def save_production_model(path=None):
    """Build and persist the deployed model bundle. Returns (path, bundle)."""
    bundle = build_production_model()
    path = path or PRODUCTION_MODEL_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(bundle, f)
    return path, bundle


def load_production_model(path=None):
    """Load the deployed model bundle (model, imputer, feature_indices, ...)."""
    path = path or PRODUCTION_MODEL_PATH
    with open(path, 'rb') as f:
        return pickle.load(f)
