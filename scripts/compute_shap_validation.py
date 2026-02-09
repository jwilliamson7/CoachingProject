#!/usr/bin/env python
"""
Compute SHAP values across 50 seeds for the 40-feature ordinal model.

Uses the same seeds (0-49) and feature subsetting as parsimonious_model.py
so results are consistent with the parsimony/Table 2 data.

Saves per-seed per-feature SHAP values, category totals, and def/off ratios
to analysis/shap_validation_data.pkl for paper verification.
"""

import os
import sys
import warnings
import pickle

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from scipy import stats

from model import CoachTenureModel
from model.config import MODEL_CONFIG, MODEL_PATHS, FEATURE_CONFIG
from model.cross_validation import stratified_coach_level_split
from scripts.shap_analysis import (
    compute_shap_values, compute_aggregated_shap, get_feature_names
)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def get_shap_feature_ranking():
    """Load cached SHAP values and return feature ranking by mean |SHAP|."""
    cache_path = os.path.join(project_root, 'data', 'shap_values_cache.pkl')
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    aggregated = cache['aggregated_shap']
    mean_abs = np.abs(aggregated.values).mean(axis=0)
    ranking = np.argsort(mean_abs)[::-1]
    return ranking


def main():
    n_seeds = 50
    top_k = 40

    # Load data
    data_path = os.path.join(project_root, MODEL_PATHS['data_file'])
    df = pd.read_csv(data_path, index_col=0)
    df = df[df[FEATURE_CONFIG['target_column']] != -1].copy()
    X = df.iloc[:, FEATURE_CONFIG['feature_columns_start']:FEATURE_CONFIG['feature_columns_end']]
    y = df[FEATURE_CONFIG['target_column']]

    # Get feature ranking and select top-K (sorted by original index, matching bootstrap_analysis.py)
    ranking = get_shap_feature_ranking()
    selected_indices = sorted(ranking[:top_k])
    X = X.iloc[:, selected_indices]

    all_feature_names = get_feature_names()
    feature_names = [all_feature_names[i] for i in selected_indices]

    # Category mapping for the 40-feature subset
    # Original ranges: Core(0-7), OC(8-40), DC(41-73), HC Team(74-106), HC Opp(107-139), Hiring(140-149)
    orig_categories = {
        'Core Experience': (0, 8),
        'OC Stats': (8, 41),
        'DC Stats': (41, 74),
        'HC Team Stats': (74, 107),
        'HC Opp Stats': (107, 140),
        'Hiring Team': (140, 150),
    }

    def get_orig_cat(orig_idx):
        for cat, (s, e) in orig_categories.items():
            if s <= orig_idx < e:
                return cat
        return None

    # Map each position in the 40-feature subset to its original category
    feature_categories = [get_orig_cat(orig_idx) for orig_idx in selected_indices]

    # Offensive = OC Stats + HC Team Stats; Defensive = DC Stats + HC Opp Stats
    off_cats = {'OC Stats', 'HC Team Stats'}
    def_cats = {'DC Stats', 'HC Opp Stats'}

    print(f"Computing SHAP values across {n_seeds} seeds for top-{top_k} features")
    print(f"Features: {X.shape[1]} columns")
    print(f"Category breakdown: {dict(pd.Series(feature_categories).value_counts())}")
    print()

    seed_per_feature_shap = []  # (n_seeds, 40) - mean |SHAP| per feature per seed
    seed_def_off_ratios = []
    seed_category_totals = []  # per-seed category total SHAP

    for seed in range(n_seeds):
        X_train, X_test, y_train, y_test, _ = stratified_coach_level_split(
            df, X, y,
            test_size=MODEL_CONFIG['test_size'],
            random_state=seed,
        )

        model = CoachTenureModel(use_ordinal=True, n_classes=3, random_state=seed)
        model.fit(X_train, y_train, verbose=0)

        # Compute SHAP
        shap_dict = compute_shap_values(
            model, np.asarray(X_test), feature_names, n_background=50
        )
        aggregated = compute_aggregated_shap(shap_dict, feature_names)
        mean_abs = np.abs(aggregated.values).mean(axis=0)  # (40,)

        # Sanity check
        total_shap = mean_abs.sum()
        if total_shap > 10:
            print(f"  Seed {seed}: DEGENERATE (total={total_shap:.2f}), skipping")
            continue

        seed_per_feature_shap.append(mean_abs.copy())

        # Category totals
        cat_totals = {}
        for cat_name in orig_categories:
            mask = [fc == cat_name for fc in feature_categories]
            n_in_cat = sum(mask)
            if n_in_cat > 0:
                cat_total = mean_abs[mask].sum()
                cat_avg = mean_abs[mask].mean()
                cat_totals[cat_name] = {'total': cat_total, 'avg': cat_avg, 'count': n_in_cat}
        seed_category_totals.append(cat_totals)

        # Def/off ratio
        off_mask = [fc in off_cats for fc in feature_categories]
        def_mask = [fc in def_cats for fc in feature_categories]
        off_shap = mean_abs[off_mask].sum()
        def_shap = mean_abs[def_mask].sum()
        n_off = sum(off_mask)
        n_def = sum(def_mask)
        off_avg = off_shap / n_off if n_off > 0 else 0
        def_avg = def_shap / n_def if n_def > 0 else 0
        ratio = def_avg / off_avg if off_avg > 1e-8 else np.nan
        seed_def_off_ratios.append(ratio)

        print(f"  Seed {seed:>2}/{n_seeds}: total_SHAP={total_shap:.4f}, Def/Off={ratio:.3f}")

    # Convert to arrays
    shap_matrix = np.array(seed_per_feature_shap)  # (n_valid_seeds, 40)
    ratios = np.array(seed_def_off_ratios)
    n_valid = len(seed_per_feature_shap)

    print(f"\n{'='*70}")
    print(f"RESULTS ({n_valid} valid seeds out of {n_seeds})")
    print(f"{'='*70}")

    t_crit = stats.t.ppf(0.975, df=n_valid - 1)

    # Per-feature importance (normalized to sum to 1, per seed)
    norm_matrix = shap_matrix / shap_matrix.sum(axis=1, keepdims=True)
    feature_means = norm_matrix.mean(axis=0)
    feature_stds = norm_matrix.std(axis=0, ddof=1)
    feature_se = feature_stds / np.sqrt(n_valid)
    feature_ci_lo = feature_means - t_crit * feature_se
    feature_ci_hi = feature_means + t_crit * feature_se

    # Sort by mean importance
    sort_order = np.argsort(feature_means)[::-1]

    print(f"\nPer-feature normalized importance (top 40):")
    print(f"{'Rank':<5} {'Feature':<30} {'Imp':>6} {'CI_lo':>7} {'CI_hi':>7}")
    print('-' * 60)
    for rank, idx in enumerate(sort_order):
        print(f"{rank+1:<5} {feature_names[idx]:<30} {feature_means[idx]:.3f}  "
              f"[{feature_ci_lo[idx]:.3f}, {feature_ci_hi[idx]:.3f}]")

    # Category summary (unnormalized SHAP)
    # Combine HC Team + HC Opp into "HC Stats" for paper table
    print(f"\nCategory importance (unnormalized mean |SHAP|):")
    print(f"{'Category':<30} {'# Feat':>7} {'Total':>8} {'Avg':>8}")
    print('-' * 60)

    cat_summary = {}
    for cat_name in ['HC Team Stats', 'HC Opp Stats', 'DC Stats', 'OC Stats', 'Hiring Team', 'Core Experience']:
        totals = [ct[cat_name]['total'] for ct in seed_category_totals if cat_name in ct]
        avgs = [ct[cat_name]['avg'] for ct in seed_category_totals if cat_name in ct]
        counts = [ct[cat_name]['count'] for ct in seed_category_totals if cat_name in ct]
        if totals:
            cat_summary[cat_name] = {
                'count': counts[0],
                'total_mean': np.mean(totals),
                'total_std': np.std(totals, ddof=1),
                'avg_mean': np.mean(avgs),
                'avg_std': np.std(avgs, ddof=1),
                'all_totals': totals,
                'all_avgs': avgs,
            }
            print(f"{cat_name:<30} {counts[0]:>7} {np.mean(totals):>8.4f} {np.mean(avgs):>8.4f}")

    # HC Stats combined
    hc_team_totals = np.array(cat_summary['HC Team Stats']['all_totals'])
    hc_opp_totals = np.array(cat_summary['HC Opp Stats']['all_totals'])
    hc_combined_totals = hc_team_totals + hc_opp_totals
    hc_count = cat_summary['HC Team Stats']['count'] + cat_summary['HC Opp Stats']['count']
    hc_combined_avgs = hc_combined_totals / hc_count
    print(f"{'HC Stats (combined)':<30} {hc_count:>7} {hc_combined_totals.mean():>8.4f} {hc_combined_avgs.mean():>8.4f}")

    # Def/Off ratio
    mean_ratio = ratios.mean()
    std_ratio = ratios.std(ddof=1)
    se_ratio = std_ratio / np.sqrt(n_valid)
    ci_lo_ratio = mean_ratio - t_crit * se_ratio
    ci_hi_ratio = mean_ratio + t_crit * se_ratio
    print(f"\nDefensive/Offensive SHAP Ratio:")
    print(f"  Mean: {mean_ratio:.4f}")
    print(f"  Std:  {std_ratio:.4f}")
    print(f"  95% CI: [{ci_lo_ratio:.4f}, {ci_hi_ratio:.4f}]")
    print(f"  Rounded: {mean_ratio:.2f}, CI [{ci_lo_ratio:.2f}, {ci_hi_ratio:.2f}]")

    # Save everything
    output = {
        'n_seeds': n_valid,
        'selected_indices': selected_indices,
        'feature_names': feature_names,
        'feature_categories': feature_categories,
        'shap_matrix': shap_matrix,  # (n_seeds, 40) unnormalized
        'norm_matrix': norm_matrix,  # (n_seeds, 40) normalized per seed
        'feature_means': feature_means,  # normalized means
        'feature_ci_lo': feature_ci_lo,
        'feature_ci_hi': feature_ci_hi,
        'sort_order': sort_order,
        'category_summary': cat_summary,
        'def_off_ratios': ratios,
        'def_off_ratio_mean': mean_ratio,
        'def_off_ratio_ci': (ci_lo_ratio, ci_hi_ratio),
        'seed_category_totals': seed_category_totals,
    }

    output_path = os.path.join(project_root, 'analysis', 'shap_validation_data.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(output, f)
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()
