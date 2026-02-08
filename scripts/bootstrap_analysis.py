#!/usr/bin/env python
"""
Bootstrap Confidence Intervals and Uncertainty Analysis for NFL Coach Tenure Prediction.

Addresses JQAS reviewer feedback: "Be careful of just using point estimates."

This script produces:
1. Bootstrap CIs on held-out test set metrics (1000 resamples)
2. Cross-validation fold-level variance (5-fold coach-level CV)
3. Paired bootstrap comparison of ordinal vs multiclass models
4. SHAP category importance stability across CV folds
5. Split robustness: retrain with 20 different random seeds to show metrics
   are stable regardless of which coaches land in train vs test

Usage:
    python scripts/bootstrap_analysis.py
    python scripts/bootstrap_analysis.py --n-boot 2000
    python scripts/bootstrap_analysis.py --skip-shap  # faster, skip SHAP stability
    python scripts/bootstrap_analysis.py --n-seeds 30  # more split seeds
"""

import os
import sys
import argparse
import warnings
import pickle

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from scipy import stats

from model import (
    CoachTenureModel,
    stratified_coach_level_split,
    stratified_coach_level_cv_split,
    ordinal_metrics,
)
from model.config import MODEL_CONFIG, MODEL_PATHS, FEATURE_CONFIG, ORDINAL_CONFIG

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def load_data():
    """Load and prepare training data."""
    data_path = os.path.join(project_root, MODEL_PATHS['data_file'])
    df = pd.read_csv(data_path, index_col=0)
    df = df[df[FEATURE_CONFIG['target_column']] != -1].copy()

    X = df.iloc[:, FEATURE_CONFIG['feature_columns_start']:FEATURE_CONFIG['feature_columns_end']]
    y = df[FEATURE_CONFIG['target_column']]
    return df, X, y


def get_shap_feature_ranking():
    """Load cached SHAP values and return feature ranking by mean |SHAP|."""
    cache_path = os.path.join(project_root, 'data', 'shap_values_cache.pkl')
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    aggregated = cache['aggregated_shap']
    mean_abs = np.abs(aggregated.values).mean(axis=0)
    ranking = np.argsort(mean_abs)[::-1]
    return ranking


def subset_features(X, top_k):
    """Subset X to top-K SHAP-ranked features. Returns (X_subset, selected_indices)."""
    ranking = get_shap_feature_ranking()
    selected = sorted(ranking[:top_k])  # Sort to preserve relative ordering
    X_subset = X.iloc[:, selected]
    return X_subset, np.array(selected)


def build_category_mapping(selected_indices):
    """
    Build category ranges for the subsetted feature space.

    Maps each selected feature (by its new 0-based position) back to its
    original category, producing new (start, end) tuples for each category.

    Returns:
        categories_mapped: dict of {cat_name: (start, end)} for the new indices
        off_def_mapped: dict of {'Offensive Metrics': [...], 'Defensive Metrics': [...], ...}
    """
    from scripts.shap_analysis import get_feature_categories
    from scripts.shap_analysis_by_background import get_offense_defense_categories

    orig_categories = get_feature_categories()
    orig_off_def = get_offense_defense_categories()

    # Map each new index to its original category
    def orig_cat_for(orig_idx):
        for cat_name, (start, end) in orig_categories.items():
            if start <= orig_idx < end:
                return cat_name
        return None

    def orig_off_def_for(orig_idx):
        for group_name, ranges in orig_off_def.items():
            for (start, end) in ranges:
                if start <= orig_idx < end:
                    return group_name
        return None

    # Build new category ranges: group consecutive new-indices by original category
    # Since features are sorted by original index, category blocks are contiguous
    categories_mapped = {}
    for cat_name in orig_categories:
        indices_in_cat = [new_i for new_i, orig_i in enumerate(selected_indices)
                          if orig_cat_for(orig_i) == cat_name]
        if indices_in_cat:
            categories_mapped[cat_name] = (min(indices_in_cat), max(indices_in_cat) + 1)

    off_def_mapped = {}
    for group_name in orig_off_def:
        ranges = []
        indices_in_group = [new_i for new_i, orig_i in enumerate(selected_indices)
                            if orig_off_def_for(orig_i) == group_name]
        if indices_in_group:
            # Build contiguous ranges
            start = indices_in_group[0]
            prev = start
            for idx in indices_in_group[1:]:
                if idx == prev + 1:
                    prev = idx
                else:
                    ranges.append((start, prev + 1))
                    start = idx
                    prev = idx
            ranges.append((start, prev + 1))
        off_def_mapped[group_name] = ranges

    return categories_mapped, off_def_mapped


def compute_metrics_from_arrays(y_true, y_pred, y_proba=None):
    """Compute all ordinal metrics from arrays."""
    m = ordinal_metrics(y_true, y_pred, y_proba, ORDINAL_CONFIG['class_names'])
    result = {
        'mae': m['mae'],
        'qwk': m['qwk'],
        'adjacent_accuracy': m['adjacent_accuracy'],
        'exact_accuracy': m['exact_accuracy'],
        'macro_f1': m['macro_f1'],
    }
    if 'auroc' in m and m['auroc'] is not None:
        result['auroc'] = m['auroc']

    for class_name, cm in m['per_class'].items():
        short = class_name.split('(')[0].strip().replace(' ', '_').lower()
        result[f'{short}_f1'] = cm['f1']

    return result


def bootstrap_test_set_metrics(model, X_test, y_test, n_boot=1000, random_state=42):
    """
    Bootstrap resample the test set and compute CIs for all metrics.

    Returns dict with point estimate, mean, std, and 95% CI for each metric.
    """
    print(f"\n{'='*70}")
    print(f"BOOTSTRAP CONFIDENCE INTERVALS (n={n_boot} resamples)")
    print(f"{'='*70}")

    rng = np.random.RandomState(random_state)
    X_test_arr = np.asarray(X_test)
    y_test_arr = np.asarray(y_test)
    n = len(y_test_arr)

    # Point estimates
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    point_metrics = compute_metrics_from_arrays(y_test_arr, y_pred, y_proba)

    # Bootstrap
    boot_results = {k: [] for k in point_metrics}

    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        y_true_b = y_test_arr[idx]
        y_pred_b = y_pred[idx]
        y_proba_b = y_proba[idx] if y_proba is not None else None

        # Skip resamples where not all classes are represented
        if len(np.unique(y_true_b)) < 3:
            continue

        try:
            m = compute_metrics_from_arrays(y_true_b, y_pred_b, y_proba_b)
            for k in boot_results:
                boot_results[k].append(m.get(k, np.nan))
        except Exception:
            continue

    # Compute CIs
    results = {}
    print(f"\n{'Metric':<25} {'Point':>8} {'Mean':>8} {'Std':>8} {'95% CI Low':>12} {'95% CI High':>12}")
    print('-' * 75)

    for k in point_metrics:
        vals = np.array(boot_results[k])
        vals = vals[~np.isnan(vals)]
        ci_low = np.percentile(vals, 2.5)
        ci_high = np.percentile(vals, 97.5)
        results[k] = {
            'point': point_metrics[k],
            'mean': np.mean(vals),
            'std': np.std(vals),
            'ci_low': ci_low,
            'ci_high': ci_high,
            'n_valid': len(vals),
        }
        print(f"{k:<25} {point_metrics[k]:>8.4f} {np.mean(vals):>8.4f} "
              f"{np.std(vals):>8.4f} {ci_low:>12.4f} {ci_high:>12.4f}")

    print(f"\nValid bootstrap samples: {len(boot_results[list(boot_results.keys())[0]])}/{n_boot}")
    return results


def cross_validation_variance(df, X, y, n_folds=5, random_state=42):
    """
    Train fresh models on each CV fold and report per-fold metrics.

    This shows how metrics vary depending on which data is held out.
    """
    print(f"\n{'='*70}")
    print(f"CROSS-VALIDATION FOLD-LEVEL VARIANCE ({n_folds}-fold)")
    print(f"{'='*70}")

    # First do the standard train/test split to get training data only
    X_train, X_test, y_train, y_test, _ = stratified_coach_level_split(
        df, X, y,
        test_size=MODEL_CONFIG['test_size'],
        random_state=random_state,
    )
    df_train = df.loc[X_train.index]

    # Generate CV splits on training data
    cv_splits = stratified_coach_level_cv_split(
        df_train, X_train, y_train,
        n_splits=n_folds,
        random_state=random_state,
    )

    fold_metrics = []
    X_train_arr = np.asarray(X_train)
    y_train_arr = np.asarray(y_train)

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        print(f"\n  Fold {fold_idx + 1}/{len(cv_splits)}: "
              f"train={len(train_idx)}, val={len(val_idx)}")

        X_fold_train = X_train_arr[train_idx]
        X_fold_val = X_train_arr[val_idx]
        y_fold_train = y_train_arr[train_idx]
        y_fold_val = y_train_arr[val_idx]

        model = CoachTenureModel(
            use_ordinal=True,
            n_classes=3,
            random_state=random_state,
        )
        model.fit(
            pd.DataFrame(X_fold_train),
            pd.Series(y_fold_train),
            verbose=0,
        )

        y_pred = model.predict(pd.DataFrame(X_fold_val))
        y_proba = model.predict_proba(pd.DataFrame(X_fold_val))

        m = compute_metrics_from_arrays(y_fold_val, y_pred, y_proba)
        fold_metrics.append(m)
        print(f"    QWK={m['qwk']:.4f}  MAE={m['mae']:.4f}  "
              f"Adj.Acc={m['adjacent_accuracy']:.4f}  F1={m['macro_f1']:.4f}")

    # Summary
    print(f"\n{'Metric':<25} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print('-' * 60)

    summary = {}
    all_keys = fold_metrics[0].keys()
    for k in all_keys:
        vals = [fm[k] for fm in fold_metrics]
        summary[k] = {
            'mean': np.mean(vals),
            'std': np.std(vals),
            'min': np.min(vals),
            'max': np.max(vals),
            'per_fold': vals,
        }
        print(f"{k:<25} {np.mean(vals):>8.4f} {np.std(vals):>8.4f} "
              f"{np.min(vals):>8.4f} {np.max(vals):>8.4f}")

    return summary


def paired_bootstrap_comparison(
    ordinal_model, multiclass_model, X_test, y_test, n_boot=1000, random_state=42
):
    """
    Paired bootstrap test: are ordinal metrics significantly better than multiclass?

    Uses the same bootstrap resamples for both models to compute paired differences.
    """
    print(f"\n{'='*70}")
    print(f"PAIRED BOOTSTRAP: ORDINAL vs MULTICLASS (n={n_boot})")
    print(f"{'='*70}")

    rng = np.random.RandomState(random_state)
    X_test_arr = np.asarray(X_test)
    y_test_arr = np.asarray(y_test)
    n = len(y_test_arr)

    # Point predictions for both models
    ord_pred = ordinal_model.predict(X_test)
    ord_proba = ordinal_model.predict_proba(X_test)
    mc_pred = multiclass_model.predict(X_test)
    mc_proba = multiclass_model.predict_proba(X_test)

    ord_point = compute_metrics_from_arrays(y_test_arr, ord_pred, ord_proba)
    mc_point = compute_metrics_from_arrays(y_test_arr, mc_pred, mc_proba)

    # Paired bootstrap differences
    metrics_to_compare = ['mae', 'qwk', 'adjacent_accuracy', 'exact_accuracy', 'macro_f1', 'auroc']
    diff_results = {k: [] for k in metrics_to_compare}

    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        y_true_b = y_test_arr[idx]

        if len(np.unique(y_true_b)) < 3:
            continue

        try:
            ord_m = compute_metrics_from_arrays(y_true_b, ord_pred[idx], ord_proba[idx])
            mc_m = compute_metrics_from_arrays(y_true_b, mc_pred[idx], mc_proba[idx])

            for k in metrics_to_compare:
                if k in ord_m and k in mc_m:
                    diff_results[k].append(ord_m[k] - mc_m[k])
        except Exception:
            continue

    print(f"\n{'Metric':<20} {'Ord':>8} {'MC':>8} {'Diff':>8} {'95% CI':>20} {'p-value':>10}")
    print('-' * 80)

    comparison = {}
    for k in metrics_to_compare:
        diffs = np.array(diff_results[k])
        if len(diffs) == 0:
            continue

        ci_low = np.percentile(diffs, 2.5)
        ci_high = np.percentile(diffs, 97.5)
        mean_diff = np.mean(diffs)

        # For MAE, ordinal is better if diff < 0 (lower MAE)
        # For others, ordinal is better if diff > 0 (higher metric)
        if k == 'mae':
            p_value = np.mean(diffs >= 0)  # proportion where MC is better or equal
        else:
            p_value = np.mean(diffs <= 0)  # proportion where MC is better or equal

        ord_val = ord_point.get(k, np.nan)
        mc_val = mc_point.get(k, np.nan)

        comparison[k] = {
            'ordinal': ord_val,
            'multiclass': mc_val,
            'mean_diff': mean_diff,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'p_value': p_value,
        }

        sig = '*' if p_value < 0.05 else ''
        print(f"{k:<20} {ord_val:>8.4f} {mc_val:>8.4f} {mean_diff:>+8.4f} "
              f"[{ci_low:>+7.4f}, {ci_high:>+7.4f}] {p_value:>9.4f} {sig}")

    print("\n* = significant at p < 0.05")
    return comparison


def shap_stability_across_folds(df, X, y, n_folds=5, random_state=42):
    """
    Compute SHAP values on each CV fold's test set and report category stability.
    """
    print(f"\n{'='*70}")
    print(f"SHAP CATEGORY IMPORTANCE STABILITY ({n_folds}-fold CV)")
    print(f"{'='*70}")

    from scripts.shap_analysis import compute_shap_values, compute_aggregated_shap, get_feature_names, get_feature_categories
    from scripts.shap_analysis_by_background import get_offense_defense_categories

    feature_names = get_feature_names()
    categories = get_feature_categories()
    off_def = get_offense_defense_categories()

    X_train, _, y_train, _, _ = stratified_coach_level_split(
        df, X, y,
        test_size=MODEL_CONFIG['test_size'],
        random_state=random_state,
    )
    df_train = df.loc[X_train.index]

    cv_splits = stratified_coach_level_cv_split(
        df_train, X_train, y_train,
        n_splits=n_folds,
        random_state=random_state,
    )

    X_train_arr = np.asarray(X_train)
    y_train_arr = np.asarray(y_train)

    fold_category_shap = []
    fold_def_off_ratios = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        print(f"\n  Fold {fold_idx + 1}/{len(cv_splits)}: computing SHAP values...")

        X_fold_train = X_train_arr[train_idx]
        X_fold_val = X_train_arr[val_idx]
        y_fold_train = y_train_arr[train_idx]

        model = CoachTenureModel(use_ordinal=True, n_classes=3, random_state=random_state)
        model.fit(pd.DataFrame(X_fold_train), pd.Series(y_fold_train), verbose=0)

        shap_dict = compute_shap_values(model, X_fold_val, feature_names, n_background=50)
        aggregated = compute_aggregated_shap(shap_dict, feature_names)

        mean_abs = np.abs(aggregated.values).mean(axis=0)

        cat_importance = {}
        for cat_name, (start, end) in categories.items():
            cat_importance[cat_name] = {
                'total': mean_abs[start:end].sum(),
                'avg': mean_abs[start:end].mean(),
            }
        fold_category_shap.append(cat_importance)

        # Compute offense vs defense ratio
        off_shap = sum(mean_abs[s:e].sum() for s, e in off_def['Offensive Metrics'])
        def_shap = sum(mean_abs[s:e].sum() for s, e in off_def['Defensive Metrics'])
        n_off = sum(e - s for s, e in off_def['Offensive Metrics'])
        n_def = sum(e - s for s, e in off_def['Defensive Metrics'])
        off_avg = off_shap / n_off
        def_avg = def_shap / n_def

        ratio = def_avg / off_avg if off_avg > 0 else np.nan
        fold_def_off_ratios.append(ratio)
        print(f"    Def/Off ratio: {ratio:.3f}")

    # Summary
    print(f"\n{'Category':<25} {'Mean Total':>12} {'Std Total':>12} {'Mean Avg':>12} {'Std Avg':>12}")
    print('-' * 75)

    cat_summary = {}
    for cat_name in categories:
        totals = [fc[cat_name]['total'] for fc in fold_category_shap]
        avgs = [fc[cat_name]['avg'] for fc in fold_category_shap]
        cat_summary[cat_name] = {
            'total_mean': np.mean(totals),
            'total_std': np.std(totals),
            'avg_mean': np.mean(avgs),
            'avg_std': np.std(avgs),
        }
        print(f"{cat_name:<25} {np.mean(totals):>12.4f} {np.std(totals):>12.4f} "
              f"{np.mean(avgs):>12.4f} {np.std(avgs):>12.4f}")

    print(f"\nDefensive/Offensive SHAP Ratio Across Folds:")
    print(f"  Mean: {np.mean(fold_def_off_ratios):.3f}")
    print(f"  Std:  {np.std(fold_def_off_ratios):.3f}")
    print(f"  Min:  {np.min(fold_def_off_ratios):.3f}")
    print(f"  Max:  {np.max(fold_def_off_ratios):.3f}")
    print(f"  Per-fold: {[f'{r:.3f}' for r in fold_def_off_ratios]}")

    return cat_summary, fold_def_off_ratios


def split_robustness(df, X, y, n_seeds=20, compute_shap=False, compare_multiclass=False,
                     selected_indices=None):
    """
    Retrain the ordinal model with different train/test splits and report
    metric distributions. This directly shows whether results are robust to
    which coaches land in train vs test.

    Each seed produces a different coach-level stratified 80/20 split, a
    freshly trained model, and a full evaluation on its held-out test set.
    If compute_shap=True, also computes SHAP category importance per seed.
    If compare_multiclass=True, also trains a multiclass model per seed.

    Parameters
    ----------
    selected_indices : array-like, optional
        Original feature indices (into the full 150-feature space) for the
        subset in X. Used to map SHAP values back to categories.
    """
    print(f"\n{'='*70}")
    print(f"SPLIT ROBUSTNESS: RETRAIN ACROSS {n_seeds} RANDOM SEEDS")
    if compute_shap:
        print("  (with SHAP analysis per seed)")
    if compare_multiclass:
        print("  (with ordinal vs multiclass comparison per seed)")
    print(f"{'='*70}")

    if compute_shap:
        from scripts.shap_analysis import compute_shap_values, compute_aggregated_shap, get_feature_names, get_feature_categories
        from scripts.shap_analysis_by_background import get_offense_defense_categories

        if selected_indices is not None:
            # Use subset feature names and remapped categories
            all_feature_names = get_feature_names()
            feature_names = [all_feature_names[i] for i in selected_indices]
            categories, off_def = build_category_mapping(selected_indices)
        else:
            feature_names = get_feature_names()
            categories = get_feature_categories()
            off_def = get_offense_defense_categories()

    seed_metrics = []
    seed_mc_metrics = []
    seed_diffs = []
    seed_shap_ratios = []
    seed_category_shap = []
    seed_per_feature_shap = []

    # Always include seed 42 plus seeds 0..n_seeds-1
    seeds = sorted(set(range(n_seeds)) | {42})

    for i, seed in enumerate(seeds):
        X_train, X_test, y_train, y_test, test_coaches = stratified_coach_level_split(
            df, X, y,
            test_size=MODEL_CONFIG['test_size'],
            random_state=seed,
        )

        model = CoachTenureModel(
            use_ordinal=True,
            n_classes=3,
            random_state=seed,
        )
        model.fit(X_train, y_train, verbose=0)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        m = compute_metrics_from_arrays(np.asarray(y_test), y_pred, y_proba)
        m['seed'] = seed
        m['n_test'] = len(y_test)
        seed_metrics.append(m)

        # Multiclass comparison
        mc_info = ""
        if compare_multiclass:
            mc_model = CoachTenureModel(
                use_ordinal=False,
                n_classes=3,
                random_state=seed,
            )
            mc_model.fit(X_train, y_train, verbose=0)
            mc_pred = mc_model.predict(X_test)
            mc_proba = mc_model.predict_proba(X_test)
            mc_m = compute_metrics_from_arrays(np.asarray(y_test), mc_pred, mc_proba)
            mc_m['seed'] = seed
            seed_mc_metrics.append(mc_m)

            diff = {k: m[k] - mc_m[k] for k in m if k not in {'seed', 'n_test'}}
            diff['seed'] = seed
            seed_diffs.append(diff)
            mc_info = f"  MC_QWK={mc_m['qwk']:.4f}"

        # SHAP
        shap_info = ""
        if compute_shap:
            shap_dict = compute_shap_values(model, np.asarray(X_test), feature_names, n_background=50)
            aggregated = compute_aggregated_shap(shap_dict, feature_names)
            mean_abs = np.abs(aggregated.values).mean(axis=0)

            cat_importance = {}
            for cat_name, (start, end) in categories.items():
                cat_importance[cat_name] = {
                    'total': mean_abs[start:end].sum(),
                    'avg': mean_abs[start:end].mean(),
                }
            seed_category_shap.append(cat_importance)

            # Sanity check: if any SHAP total exceeds 10, the seed is degenerate
            total_shap = mean_abs.sum()
            if total_shap > 10:
                print(f"    WARNING: Degenerate SHAP values (total={total_shap:.2f}), skipping this seed's SHAP")
                seed_category_shap.pop()  # Remove the degenerate entry we just appended
                seed_shap_ratios.append(np.nan)
                shap_info = "  SHAP=DEGENERATE"
            else:
                seed_per_feature_shap.append(mean_abs.copy())
                off_shap = sum(mean_abs[s:e].sum() for s, e in off_def['Offensive Metrics'])
                def_shap = sum(mean_abs[s:e].sum() for s, e in off_def['Defensive Metrics'])
                n_off = sum(e - s for s, e in off_def['Offensive Metrics'])
                n_def = sum(e - s for s, e in off_def['Defensive Metrics'])
                ratio = (def_shap / n_def) / (off_shap / n_off) if off_shap > 1e-8 else np.nan
                seed_shap_ratios.append(ratio)
                shap_info = f"  Def/Off={ratio:.3f}"

        print(f"  [{i+1}/{len(seeds)}] Seed {seed:>3}: QWK={m['qwk']:.4f}  MAE={m['mae']:.4f}  "
              f"F1={m['macro_f1']:.4f}{mc_info}{shap_info}")

    # Summary statistics
    print(f"\n{'Metric':<25} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'95% CI':>22}")
    print('-' * 80)

    summary = {}
    skip_keys = {'seed', 'n_test'}
    metric_keys = [k for k in seed_metrics[0] if k not in skip_keys]

    for k in metric_keys:
        vals = np.array([sm[k] for sm in seed_metrics])
        ci_low = np.percentile(vals, 2.5)
        ci_high = np.percentile(vals, 97.5)
        summary[k] = {
            'mean': np.mean(vals),
            'std': np.std(vals),
            'min': np.min(vals),
            'max': np.max(vals),
            'ci_low': ci_low,
            'ci_high': ci_high,
            'per_seed': vals.tolist(),
        }
        print(f"{k:<25} {np.mean(vals):>8.4f} {np.std(vals):>8.4f} "
              f"{np.min(vals):>8.4f} {np.max(vals):>8.4f} "
              f"[{ci_low:.4f}, {ci_high:.4f}]")

    # Multiclass comparison summary
    comparison_summary = None
    if compare_multiclass and seed_diffs:
        from scipy import stats as sp_stats
        print(f"\nOrdinal vs Multiclass Across {len(seeds)} Seeds:")
        print(f"{'Metric':<25} {'Mean Diff':>10} {'Std':>8} {'Ord Wins':>10} {'p (paired t)':>12}")
        print('-' * 70)

        comparison_summary = {}
        compare_keys = [k for k in seed_diffs[0] if k != 'seed']
        for k in compare_keys:
            diffs = np.array([d[k] for d in seed_diffs])
            # For MAE, negative diff = ordinal better; for others, positive = ordinal better
            if k == 'mae':
                ord_wins = np.sum(diffs < 0)
                # One-sided paired t-test: H0: mean_diff >= 0, H1: mean_diff < 0
                t_stat, two_sided_p = sp_stats.ttest_1samp(diffs, 0)
                p_val = two_sided_p / 2 if t_stat < 0 else 1 - two_sided_p / 2
            else:
                ord_wins = np.sum(diffs > 0)
                # One-sided paired t-test: H0: mean_diff <= 0, H1: mean_diff > 0
                t_stat, two_sided_p = sp_stats.ttest_1samp(diffs, 0)
                p_val = two_sided_p / 2 if t_stat > 0 else 1 - two_sided_p / 2

            comparison_summary[k] = {
                'mean_diff': np.mean(diffs),
                'std_diff': np.std(diffs),
                'ord_wins': int(ord_wins),
                'total': len(diffs),
                'p_value': p_val,
                'per_seed_diffs': diffs.tolist(),
            }
            sig = '*' if p_val < 0.05 else ''
            print(f"{k:<25} {np.mean(diffs):>+10.4f} {np.std(diffs):>8.4f} "
                  f"{ord_wins:>5}/{len(diffs):<4} {p_val:>11.4f} {sig}")

    # SHAP summary
    shap_summary = None
    if compute_shap and seed_category_shap:
        print(f"\nSHAP Category Importance Across {len(seeds)} Seeds:")
        print(f"{'Category':<25} {'Mean Total':>12} {'Std Total':>12}")
        print('-' * 50)

        shap_summary = {}
        for cat_name in categories:
            totals = [sc[cat_name]['total'] for sc in seed_category_shap]
            avgs = [sc[cat_name]['avg'] for sc in seed_category_shap]
            shap_summary[cat_name] = {
                'total_mean': np.mean(totals),
                'total_std': np.std(totals),
                'avg_mean': np.mean(avgs),
                'avg_std': np.std(avgs),
            }
            print(f"{cat_name:<25} {np.mean(totals):>12.4f} {np.std(totals):>12.4f}")

        valid_ratios = [r for r in seed_shap_ratios if not np.isnan(r)]
        n_valid = len(valid_ratios)
        n_total = len(seed_shap_ratios)
        print(f"\nDef/Off SHAP Ratio ({n_valid}/{n_total} valid seeds): "
              f"{np.mean(valid_ratios):.3f} +/- {np.std(valid_ratios):.3f}")
        print(f"  Range: [{np.min(valid_ratios):.3f}, {np.max(valid_ratios):.3f}]")

    # Build per-feature SHAP summary if available
    per_feature_shap_summary = None
    if seed_per_feature_shap:
        shap_matrix = np.array(seed_per_feature_shap)  # (n_seeds, n_features)
        per_feature_shap_summary = {
            'feature_names': feature_names,
            'mean': shap_matrix.mean(axis=0),
            'std': shap_matrix.std(axis=0),
            'ci_low': np.percentile(shap_matrix, 2.5, axis=0),
            'ci_high': np.percentile(shap_matrix, 97.5, axis=0),
            'per_seed': shap_matrix,
        }

    return summary, seed_metrics, shap_summary, seed_shap_ratios, comparison_summary, per_feature_shap_summary


def save_results(boot_results, cv_summary, comparison, shap_summary=None, shap_ratios=None,
                 split_summary=None, per_feature_shap=None):
    """Save all results to analysis/ directory."""
    analysis_dir = os.path.join(project_root, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    report_path = os.path.join(analysis_dir, 'bootstrap_analysis_report.txt')

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("UNCERTAINTY ANALYSIS REPORT\n")
        f.write("NFL Coach Tenure Prediction - Ordinal Classification\n")
        f.write("=" * 80 + "\n\n")

        # Bootstrap CIs
        f.write("1. BOOTSTRAP CONFIDENCE INTERVALS (Test Set)\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Metric':<25} {'Point':>8} {'95% CI':>25}\n")
        f.write("-" * 60 + "\n")
        for k, v in boot_results.items():
            f.write(f"{k:<25} {v['point']:>8.4f} [{v['ci_low']:.4f}, {v['ci_high']:.4f}]\n")
        f.write("\n")

        # CV fold variance
        f.write("\n2. CROSS-VALIDATION FOLD VARIANCE\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Metric':<25} {'Mean +/- Std':>20}\n")
        f.write("-" * 60 + "\n")
        for k, v in cv_summary.items():
            f.write(f"{k:<25} {v['mean']:.4f} +/- {v['std']:.4f}\n")
        f.write("\n")

        # Ordinal vs Multiclass comparison across seeds
        if comparison:
            f.write("\n3. ORDINAL vs MULTICLASS COMPARISON (Across Seeds)\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Metric':<20} {'Mean Diff':>10} {'Std':>8} {'Ord Wins':>10} {'p':>8}\n")
            f.write("-" * 60 + "\n")
            for k, v in comparison.items():
                f.write(f"{k:<20} {v['mean_diff']:>+10.4f} {v['std_diff']:>8.4f} "
                        f"{v['ord_wins']:>5}/{v['total']:<4} "
                        f"{v['p_value']:>8.4f}\n")
            f.write("\n")

        # SHAP stability
        if shap_summary:
            f.write("\n4. SHAP CATEGORY IMPORTANCE STABILITY\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Category':<25} {'Mean +/- Std (Total)':>25}\n")
            f.write("-" * 60 + "\n")
            for k, v in shap_summary.items():
                f.write(f"{k:<25} {v['total_mean']:.4f} +/- {v['total_std']:.4f}\n")

            if shap_ratios:
                valid_ratios = [r for r in shap_ratios if not np.isnan(r)]
                if valid_ratios:
                    f.write(f"\nDef/Off Ratio ({len(valid_ratios)}/{len(shap_ratios)} valid seeds): "
                            f"{np.mean(valid_ratios):.3f} +/- {np.std(valid_ratios):.3f}\n")

        # Split robustness
        if split_summary:
            f.write("\n5. SPLIT ROBUSTNESS (Varying Random Seed)\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Metric':<25} {'Mean +/- Std':>20} {'95% Range':>25}\n")
            f.write("-" * 60 + "\n")
            for k, v in split_summary.items():
                f.write(f"{k:<25} {v['mean']:.4f} +/- {v['std']:.4f} "
                        f"[{v['ci_low']:.4f}, {v['ci_high']:.4f}]\n")
            f.write("\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"\nResults saved to: {report_path}")

    # Also save as pickle for programmatic access
    pickle_path = os.path.join(analysis_dir, 'bootstrap_analysis_results.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump({
            'bootstrap_cis': boot_results,
            'cv_variance': cv_summary,
            'paired_comparison': comparison,
            'shap_stability': shap_summary,
            'shap_def_off_ratios': shap_ratios,
            'split_robustness': split_summary,
            'per_feature_shap': per_feature_shap,
        }, f)
    print(f"Pickle saved to: {pickle_path}")


def main():
    parser = argparse.ArgumentParser(description='Bootstrap CI and uncertainty analysis')
    parser.add_argument('--n-boot', type=int, default=1000, help='Number of bootstrap resamples')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--skip-shap', action='store_true', help='Skip SHAP stability (faster)')
    parser.add_argument('--skip-comparison', action='store_true', help='Skip ordinal vs multiclass comparison')
    parser.add_argument('--n-seeds', type=int, default=20, help='Number of random seeds for split robustness')
    parser.add_argument('--skip-split-robustness', action='store_true', help='Skip split robustness analysis')
    parser.add_argument('--top-k', type=int, default=None,
                        help='Use only the top-K SHAP-ranked features (default: all)')
    args = parser.parse_args()

    print("=" * 70)
    print("UNCERTAINTY ANALYSIS FOR NFL COACH TENURE PREDICTION")
    print("=" * 70)

    # Load data
    df, X, y = load_data()
    print(f"Loaded {len(df)} instances with {X.shape[1]} features")

    # Optional feature subsetting
    selected_indices = None
    if args.top_k:
        X, selected_indices = subset_features(X, args.top_k)
        # Update df to have the subsetted X columns for split functions
        # (df is used by stratified_coach_level_split to get coach names)
        print(f"Selected top {args.top_k} SHAP-ranked features ({X.shape[1]} columns)")

    # Get train/test split (same as training)
    X_train, X_test, y_train, y_test, test_coaches = stratified_coach_level_split(
        df, X, y,
        test_size=MODEL_CONFIG['test_size'],
        random_state=MODEL_CONFIG['random_state'],
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Train a fresh model on this feature set (no pre-trained model for subsets)
    print("Training fresh ordinal model...")
    ordinal_model = CoachTenureModel(use_ordinal=True, n_classes=3, random_state=42)
    ordinal_model.fit(X_train, y_train, verbose=1)

    # 1. Bootstrap CIs on test set
    boot_results = bootstrap_test_set_metrics(
        ordinal_model, X_test, y_test, n_boot=args.n_boot
    )

    # 2. CV fold-level variance
    cv_summary = cross_validation_variance(df, X, y, n_folds=args.n_folds)

    # 3. Split robustness (varying random seed), with optional SHAP and comparison
    split_summary = None
    shap_summary = None
    shap_ratios = None
    comparison = {}
    per_feature_shap = None
    if not args.skip_split_robustness:
        split_summary, _, shap_summary, shap_ratios, comparison, per_feature_shap = split_robustness(
            df, X, y, n_seeds=args.n_seeds,
            compute_shap=not args.skip_shap,
            compare_multiclass=not args.skip_comparison,
            selected_indices=selected_indices,
        )

    # Save all results
    save_results(boot_results, cv_summary, comparison, shap_summary, shap_ratios,
                 split_summary, per_feature_shap)

    print(f"\n{'='*70}")
    print("UNCERTAINTY ANALYSIS COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
