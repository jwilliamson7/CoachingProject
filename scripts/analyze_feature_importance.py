#!/usr/bin/env python
"""
Analyze feature importance by category with statistical tests.

Performs Kruskal-Wallis test across categories and pairwise Mann-Whitney U tests
to determine if differences in average feature importance are statistically significant.
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
from scipy import stats
from itertools import combinations

from model import CoachTenureModel
from model.config import MODEL_PATHS


def main():
    # Load model
    model_path = os.path.join(project_root, MODEL_PATHS['ordinal_model_output'])
    print(f"Loading model from {model_path}...")
    model = CoachTenureModel.load(model_path)

    # Get feature importances
    importances = model.get_feature_importances()
    print(f"Total features: {len(importances)}")
    print(f"Sum of importances: {importances.sum():.4f}")

    # Define categories (1-indexed feature numbers)
    categories = {
        'Core Experience': (1, 8),
        'OC Stats': (9, 41),
        'DC Stats': (42, 74),
        'HC Stats': (75, 140),  # Combined HC team + opponent
        'Hiring Team': (141, 150)
    }

    # Extract importances by category
    category_importances = {}
    for cat_name, (start, end) in categories.items():
        # Convert to 0-indexed
        cat_imp = importances[start-1:end]
        category_importances[cat_name] = cat_imp

    # Print summary statistics
    print("\n" + "="*70)
    print("CATEGORY SUMMARY STATISTICS")
    print("="*70)
    print(f"{'Category':<20} {'Count':>6} {'Total':>10} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-"*70)

    for cat_name, imp in category_importances.items():
        print(f"{cat_name:<20} {len(imp):>6} {imp.sum():>10.4f} {imp.mean():>10.4f} "
              f"{imp.std():>10.4f} {imp.min():>10.4f} {imp.max():>10.4f}")

    # Kruskal-Wallis test (non-parametric ANOVA)
    print("\n" + "="*70)
    print("KRUSKAL-WALLIS TEST (Non-parametric test for differences across groups)")
    print("="*70)

    groups = list(category_importances.values())
    h_stat, p_value = stats.kruskal(*groups)

    print(f"H-statistic: {h_stat:.4f}")
    print(f"p-value: {p_value:.6f}")

    if p_value < 0.05:
        print("Result: SIGNIFICANT difference exists among categories (p < 0.05)")
    else:
        print("Result: NO significant difference among categories (p >= 0.05)")

    # Pairwise Mann-Whitney U tests
    print("\n" + "="*70)
    print("PAIRWISE MANN-WHITNEY U TESTS")
    print("="*70)

    cat_names = list(category_importances.keys())
    n_comparisons = len(list(combinations(cat_names, 2)))
    bonferroni_alpha = 0.05 / n_comparisons

    print(f"Number of comparisons: {n_comparisons}")
    print(f"Bonferroni-corrected alpha: {bonferroni_alpha:.4f}")
    print()

    print(f"{'Comparison':<35} {'U-stat':>10} {'p-value':>12} {'Significant':>12}")
    print("-"*70)

    results = []
    for cat1, cat2 in combinations(cat_names, 2):
        imp1 = category_importances[cat1]
        imp2 = category_importances[cat2]

        u_stat, p_val = stats.mannwhitneyu(imp1, imp2, alternative='two-sided')

        significant = "YES" if p_val < bonferroni_alpha else "no"
        comparison = f"{cat1} vs {cat2}"

        print(f"{comparison:<35} {u_stat:>10.1f} {p_val:>12.6f} {significant:>12}")

        results.append({
            'cat1': cat1,
            'cat2': cat2,
            'mean1': imp1.mean(),
            'mean2': imp2.mean(),
            'u_stat': u_stat,
            'p_value': p_val,
            'significant': p_val < bonferroni_alpha
        })

    # Summary of significant differences
    print("\n" + "="*70)
    print("SIGNIFICANT PAIRWISE DIFFERENCES (after Bonferroni correction)")
    print("="*70)

    sig_results = [r for r in results if r['significant']]
    if sig_results:
        for r in sig_results:
            higher = r['cat1'] if r['mean1'] > r['mean2'] else r['cat2']
            lower = r['cat2'] if r['mean1'] > r['mean2'] else r['cat1']
            higher_mean = max(r['mean1'], r['mean2'])
            lower_mean = min(r['mean1'], r['mean2'])
            print(f"{higher} (mean={higher_mean:.4f}) > {lower} (mean={lower_mean:.4f}), p={r['p_value']:.6f}")
    else:
        print("No pairwise comparisons reached significance after Bonferroni correction.")

    # Effect sizes (rank-biserial correlation)
    print("\n" + "="*70)
    print("EFFECT SIZES (Rank-biserial correlation)")
    print("="*70)
    print("Note: |r| < 0.1 = negligible, 0.1-0.3 = small, 0.3-0.5 = medium, > 0.5 = large")
    print()

    print(f"{'Comparison':<35} {'r':>10} {'Effect Size':>15}")
    print("-"*60)

    for cat1, cat2 in combinations(cat_names, 2):
        imp1 = category_importances[cat1]
        imp2 = category_importances[cat2]

        u_stat, _ = stats.mannwhitneyu(imp1, imp2, alternative='two-sided')
        n1, n2 = len(imp1), len(imp2)

        # Rank-biserial correlation: r = 1 - (2U)/(n1*n2)
        r = 1 - (2 * u_stat) / (n1 * n2)

        if abs(r) < 0.1:
            effect = "negligible"
        elif abs(r) < 0.3:
            effect = "small"
        elif abs(r) < 0.5:
            effect = "medium"
        else:
            effect = "large"

        comparison = f"{cat1} vs {cat2}"
        print(f"{comparison:<35} {r:>10.4f} {effect:>15}")


def test_oc_vs_non_oc(importances):
    """Test if OC Stats have different average importance than non-OC features."""
    print("\n" + "="*70)
    print("OC STATS vs NON-OC STATS COMPARISON")
    print("="*70)

    # OC Stats: features 9-41 (0-indexed: 8-40)
    oc_importances = importances[8:41]

    # Non-OC: everything else
    non_oc_importances = np.concatenate([
        importances[0:8],      # Core Experience
        importances[41:150]    # DC Stats, HC Stats, Hiring Team
    ])

    print(f"OC Stats: n={len(oc_importances)}, mean={oc_importances.mean():.4f}, std={oc_importances.std():.4f}")
    print(f"Non-OC:   n={len(non_oc_importances)}, mean={non_oc_importances.mean():.4f}, std={non_oc_importances.std():.4f}")

    # Mann-Whitney U test
    u_stat, p_value = stats.mannwhitneyu(oc_importances, non_oc_importances, alternative='two-sided')

    print(f"\nMann-Whitney U test:")
    print(f"  U-statistic: {u_stat:.1f}")
    print(f"  p-value: {p_value:.6f}")

    if p_value < 0.05:
        print(f"  Result: SIGNIFICANT difference (p < 0.05)")
        if oc_importances.mean() < non_oc_importances.mean():
            print(f"  OC Stats have LOWER average importance than non-OC features")
        else:
            print(f"  OC Stats have HIGHER average importance than non-OC features")
    else:
        print(f"  Result: NO significant difference (p >= 0.05)")

    # Effect size
    n1, n2 = len(oc_importances), len(non_oc_importances)
    r = 1 - (2 * u_stat) / (n1 * n2)

    if abs(r) < 0.1:
        effect = "negligible"
    elif abs(r) < 0.3:
        effect = "small"
    elif abs(r) < 0.5:
        effect = "medium"
    else:
        effect = "large"

    print(f"\nEffect size (rank-biserial r): {r:.4f} ({effect})")


if __name__ == '__main__':
    main()

    # Also run the OC vs non-OC test
    model_path = os.path.join(project_root, MODEL_PATHS['ordinal_model_output'])
    model = CoachTenureModel.load(model_path)
    importances = model.get_feature_importances()
    test_oc_vs_non_oc(importances)
