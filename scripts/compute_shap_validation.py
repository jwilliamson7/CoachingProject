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

from model.pipeline import (
    load_modeling_data, shap_feature_ranking, best_k, leakage_free_split, fit_model, ordinal_model,
)
from scripts.shap_analysis import (
    compute_shap_values, compute_aggregated_shap, get_feature_names
)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def main():
    n_seeds = 50

    # best-K from the locked parsimony results (single source of truth)
    top_k = best_k()

    # Load RAW modeling data (171 features); imputation is fit per train split.
    df, X, y = load_modeling_data()
    n_total_feats = X.shape[1]

    # Leakage-free SHAP ranking; top-K in importance order to match
    # parsimonious_model.py exactly (unsorted ranking[:k]). Do NOT subset X
    # up front: imputation is fit on the full feature set per split, then the
    # top-K columns are selected, identical to the parsimony protocol.
    ranking = shap_feature_ranking()
    selected_indices = list(ranking[:top_k])

    # Full readable names = 150 base names + engineered (cf_/rf_) column names
    base_names = get_feature_names()
    feature_names_full = base_names + list(X.columns[len(base_names):])
    feature_names = [feature_names_full[i] for i in selected_indices]

    # Category of each ORIGINAL feature index. Base ranges cover features 0-149;
    # the engineered features (indices 150+) are classified by column prefix
    # (cf_ = career path, rf_ = rank/recency).
    base_ranges = {
        'Core Experience': (0, 8),
        'OC Stats': (8, 41),
        'DC Stats': (41, 74),
        'HC Team Stats': (74, 107),
        'HC Opp Stats': (107, 140),
        'Hiring Team': (140, 150),
    }

    def get_orig_cat(orig_idx):
        col = str(X.columns[orig_idx])
        if col.startswith('cf_'):
            return 'Career Path'
        if col.startswith('rf_'):
            return 'Rank/Recency'
        for cat, (s, e) in base_ranges.items():
            if s <= orig_idx < e:
                return cat
        return None

    all_categories = list(base_ranges.keys()) + ['Career Path', 'Rank/Recency']
    feature_categories = [get_orig_cat(orig_idx) for orig_idx in selected_indices]

    # Offensive = OC Stats + HC Team Stats; Defensive = DC Stats + HC Opp Stats.
    # Engineered features are neither (excluded from the def/off ratio).
    off_cats = {'OC Stats', 'HC Team Stats'}
    def_cats = {'DC Stats', 'HC Opp Stats'}

    print(f"Computing SHAP values across {n_seeds} seeds for top-{top_k} features "
          f"(of {n_total_feats}); leakage-free per-split imputation")
    print(f"Category breakdown: {dict(pd.Series(feature_categories).value_counts())}")
    print()

    seed_per_feature_shap = []  # (n_seeds, 40) - mean |SHAP| per feature per seed
    seed_def_off_ratios = []
    seed_category_totals = []  # per-seed category total SHAP

    for seed in range(n_seeds):
        # Shared leakage-free split (train-only imputation, top-K selection)
        split = leakage_free_split(df, X, y, seed, feature_indices=selected_indices)
        model = fit_model(split, ordinal_model, seed)

        # Compute SHAP on the held-out test partition (already imputed)
        shap_dict = compute_shap_values(model, split.X_test, feature_names)
        aggregated = compute_aggregated_shap(shap_dict, feature_names)
        mean_abs = np.abs(aggregated.values).mean(axis=0)

        # Sanity check
        total_shap = mean_abs.sum()
        if total_shap > 10:
            print(f"  Seed {seed}: DEGENERATE (total={total_shap:.2f}), skipping")
            continue

        seed_per_feature_shap.append(mean_abs.copy())

        # Category totals
        cat_totals = {}
        for cat_name in all_categories:
            mask = np.array([fc == cat_name for fc in feature_categories])
            n_in_cat = int(mask.sum())
            if n_in_cat > 0:
                cat_total = mean_abs[mask].sum()
                cat_avg = mean_abs[mask].mean()
                cat_totals[cat_name] = {'total': cat_total, 'avg': cat_avg, 'count': n_in_cat}
        seed_category_totals.append(cat_totals)

        # Def/off ratio (per-feature averages, engineered features excluded)
        off_mask = np.array([fc in off_cats for fc in feature_categories])
        def_mask = np.array([fc in def_cats for fc in feature_categories])
        off_avg = mean_abs[off_mask].mean() if off_mask.any() else 0.0
        def_avg = mean_abs[def_mask].mean() if def_mask.any() else 0.0
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

    print(f"\nPer-feature normalized importance (top {top_k}):")
    print(f"{'Rank':<5} {'Feature':<30} {'Imp':>6} {'CI_lo':>7} {'CI_hi':>7}")
    print('-' * 60)
    for rank, idx in enumerate(sort_order):
        print(f"{rank+1:<5} {feature_names[idx]:<30} {feature_means[idx]:.3f}  "
              f"[{feature_ci_lo[idx]:.3f}, {feature_ci_hi[idx]:.3f}]")

    # Category summary (unnormalized SHAP). Only categories actually present in
    # the top-K subset are reported (e.g. Hiring Team may not survive selection).
    print(f"\nCategory importance (unnormalized mean |SHAP|):")
    print(f"{'Category':<30} {'# Feat':>7} {'Total':>8} {'Avg':>8}")
    print('-' * 60)

    cat_summary = {}
    cat_print_order = ['HC Team Stats', 'HC Opp Stats', 'DC Stats', 'OC Stats',
                       'Core Experience', 'Hiring Team', 'Career Path', 'Rank/Recency']
    for cat_name in cat_print_order:
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

    # HC Stats combined (only if both HC sub-categories survived selection)
    if 'HC Team Stats' in cat_summary and 'HC Opp Stats' in cat_summary:
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
