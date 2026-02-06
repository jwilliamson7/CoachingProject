#!/usr/bin/env python
"""
Enhanced SHAP Analysis: Feature Importance by Coach Background

Analyzes whether defensive vs offensive metrics matter differently based on
the coach's career background (offensive coordinator vs defensive coordinator).

Key questions:
1. Are defensive metrics more predictive than offensive metrics overall?
2. Do offensive-background coaches benefit more from strong defensive metrics?
3. Do defensive-background coaches benefit more from strong offensive metrics?
"""

import os
import sys
import pickle
import warnings

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from scipy import stats
from typing import List, Dict, Tuple

from model import CoachTenureModel
from model.config import MODEL_PATHS
from data_constants import get_all_feature_names, HIRING_TEAM_FEATURES, BASE_TEAM_STATISTICS

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def get_refined_feature_categories() -> Dict[str, Tuple[int, int]]:
    """
    Get refined feature categories splitting HC into offense/defense.

    Feature layout (0-indexed):
    - 0-7: Core Experience (8 features)
    - 8-40: OC Stats (33 features) - offensive performance as OC
    - 41-73: DC Stats (33 features) - defensive performance as DC
    - 74-106: HC Team Stats (33 features) - team's offensive performance as HC
    - 107-139: HC Opponent Stats (33 features) - opponent's performance (your defense) as HC
    - 140-149: Hiring Team Context (10 features)
    """
    return {
        'Core Experience': (0, 8),
        'OC Stats (Offense)': (8, 41),
        'DC Stats (Defense)': (41, 74),
        'HC Team Stats (Offense)': (74, 107),
        'HC Opp Stats (Defense)': (107, 140),
        'Hiring Team': (140, 150)
    }


def get_offense_defense_categories() -> Dict[str, List[Tuple[int, int]]]:
    """Group categories into offensive vs defensive."""
    return {
        'Offensive Metrics': [(8, 41), (74, 107)],  # OC Stats + HC Team Stats
        'Defensive Metrics': [(41, 74), (107, 140)],  # DC Stats + HC Opp Stats
        'Other': [(0, 8), (140, 150)]  # Core Experience + Hiring Team
    }


def load_coach_backgrounds() -> pd.DataFrame:
    """
    Load or compute coach background classifications.
    Returns DataFrame with Coach_Name and Background columns.
    """
    # Try to load pre-computed backgrounds
    cache_paths = [
        os.path.join(project_root, 'data', 'coach_backgrounds.csv'),
        os.path.join(project_root, 'analysis', 'coach_backgrounds_from_history.csv'),
    ]

    for cache_path in cache_paths:
        if os.path.exists(cache_path):
            print(f"Loading coach backgrounds from {cache_path}...")
            return pd.read_csv(cache_path)

    # If no cache, compute from coaching history files
    print("Computing coach backgrounds from coaching history files...")
    return compute_coach_backgrounds_from_history()


def compute_coach_backgrounds_from_history() -> pd.DataFrame:
    """
    Compute coach backgrounds from coaching history files in Coaches/ directory.
    Adapted from coach_background_from_history.py
    """
    import glob

    coaches_dir = os.path.join(project_root, 'Coaches')
    coach_dirs = glob.glob(os.path.join(coaches_dir, '*'))

    print(f"Found {len(coach_dirs)} coach directories")

    coach_backgrounds = []

    # Offensive role patterns
    offensive_patterns = [
        'Offensive Coordinator', 'Off. Coordinator',
        'Quarterbacks', 'QB Coach', 'Quarterback',
        'Running Backs', 'RB Coach', 'Running Back',
        'Wide Receivers', 'WR Coach', 'Wide Receiver', 'Receivers',
        'Tight Ends', 'TE Coach', 'Tight End',
        'Offensive Line', 'OL Coach', 'O-Line',
        'Offensive Quality Control', 'Offensive Assistant',
        'Passing Game Coordinator', 'Run Game Coordinator',
    ]
    offensive_pattern = '|'.join(offensive_patterns)

    # Defensive role patterns
    defensive_patterns = [
        'Defensive Coordinator', 'Def. Coordinator',
        'Linebackers', 'LB Coach', 'Linebacker',
        'Defensive Backs', 'DB Coach', 'Secondary',
        'Cornerbacks', 'CB Coach', 'Cornerback',
        'Safeties', 'Safety',
        'Defensive Line', 'DL Coach', 'D-Line',
        'Defensive Ends', 'DE Coach',
        'Defensive Tackles', 'DT Coach',
        'Defensive Quality Control', 'Defensive Assistant',
    ]
    defensive_pattern = '|'.join(defensive_patterns)

    for coach_dir in coach_dirs:
        coach_name = os.path.basename(coach_dir)
        history_file = os.path.join(coach_dir, 'all_coaching_history.csv')

        if not os.path.exists(history_file):
            continue

        try:
            history_df = pd.read_csv(history_file)

            if 'Role' not in history_df.columns:
                continue

            # Filter out head coach roles
            non_hc_roles = history_df[~history_df['Role'].str.contains('Head Coach', case=False, na=False)]

            # Count offensive and defensive experience
            offensive_exp = non_hc_roles[non_hc_roles['Role'].str.contains(
                offensive_pattern, case=False, na=False)]
            defensive_exp = non_hc_roles[non_hc_roles['Role'].str.contains(
                defensive_pattern, case=False, na=False)]

            offensive_years = len(offensive_exp)
            defensive_years = len(defensive_exp)

            # Classify
            if offensive_years > 2 and defensive_years > 2:
                background = 'Both'
            elif offensive_years > defensive_years:
                background = 'Offensive'
            elif defensive_years > offensive_years:
                background = 'Defensive'
            elif offensive_years > 0:
                background = 'Offensive'
            else:
                background = 'Other'

            coach_backgrounds.append({
                'Coach_Name': coach_name,
                'Background': background,
                'Offensive_Years': offensive_years,
                'Defensive_Years': defensive_years
            })

        except Exception as e:
            continue

    backgrounds_df = pd.DataFrame(coach_backgrounds)

    # Print distribution
    if not backgrounds_df.empty:
        print(f"\nCoach Background Distribution:")
        print(backgrounds_df['Background'].value_counts())

    # Save cache
    cache_path = os.path.join(project_root, 'data', 'coach_backgrounds.csv')
    backgrounds_df.to_csv(cache_path, index=False)
    print(f"Saved coach backgrounds to {cache_path}")

    return backgrounds_df


def compute_coach_backgrounds_from_data() -> pd.DataFrame:
    """
    Compute coach backgrounds from the master data file.
    Uses the experience features to classify coaches.
    """
    data_path = os.path.join(project_root, MODEL_PATHS['data_file'])
    df = pd.read_csv(data_path)

    # Get unique coaches
    if 'Coach Name' in df.columns:
        coach_col = 'Coach Name'
    elif 'Unnamed: 1' in df.columns:
        coach_col = 'Unnamed: 1'
    else:
        coach_col = df.columns[1]

    coaches = df[coach_col].unique()

    # Feature indices for coordinator experience (0-indexed in feature columns)
    # num_yr_nfl_coor is feature 7 (index 6 in 0-indexed features)
    # We need to look at the actual OC/DC stats to infer background

    backgrounds = []
    for coach in coaches:
        coach_data = df[df[coach_col] == coach].iloc[0]

        # Get OC and DC stat columns (features 9-41 and 42-74 in 1-indexed)
        # In the dataframe, these start at column index 2 (after Coach Name, Year)
        feature_start = 2

        # OC stats: columns 2+8 to 2+40 (features 9-41)
        oc_cols = range(feature_start + 8, feature_start + 41)
        # DC stats: columns 2+41 to 2+73 (features 42-74)
        dc_cols = range(feature_start + 41, feature_start + 74)

        # Count non-zero/non-nan values as indicator of experience
        oc_experience = sum(1 for i in oc_cols if i < len(coach_data) and pd.notna(coach_data.iloc[i]) and coach_data.iloc[i] != 0)
        dc_experience = sum(1 for i in dc_cols if i < len(coach_data) and pd.notna(coach_data.iloc[i]) and coach_data.iloc[i] != 0)

        if oc_experience > dc_experience * 1.5:
            background = 'Offensive'
        elif dc_experience > oc_experience * 1.5:
            background = 'Defensive'
        elif oc_experience > 0 and dc_experience > 0:
            background = 'Both'
        else:
            background = 'Other'

        backgrounds.append({
            'Coach_Name': coach,
            'Background': background,
            'OC_Experience': oc_experience,
            'DC_Experience': dc_experience
        })

    return pd.DataFrame(backgrounds)


def load_data_and_shap() -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, Dict, shap.Explanation]:
    """Load data, model, and cached SHAP values."""
    # Load model
    model_path = os.path.join(project_root, MODEL_PATHS['ordinal_model_output'])
    print(f"Loading model from {model_path}...")
    model = CoachTenureModel.load(model_path)

    # Load data
    data_path = os.path.join(project_root, MODEL_PATHS['data_file'])
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df = df[df['Coach Tenure Class'] != -1].copy()

    # Handle column structure
    feature_cols = df.columns[2:-2]
    if len(feature_cols) != 150:
        feature_cols = df.columns[2:152]

    X = df[feature_cols].values
    y = df['Coach Tenure Class'].values

    # Get coach names
    if 'Coach Name' in df.columns:
        coach_names = df['Coach Name'].values
    else:
        coach_names = df.iloc[:, 1].values

    # Load cached SHAP values
    shap_cache_path = os.path.join(project_root, 'data', 'shap_values_cache.pkl')
    if os.path.exists(shap_cache_path):
        print(f"Loading cached SHAP values...")
        with open(shap_cache_path, 'rb') as f:
            cache = pickle.load(f)
            shap_values_dict = cache['shap_values_dict']
            aggregated_shap = cache['aggregated_shap']
    else:
        raise FileNotFoundError("SHAP cache not found. Run shap_analysis.py --save-shap first.")

    return X, y, df, shap_values_dict, aggregated_shap, coach_names


def statistical_tests_offense_vs_defense(aggregated_shap: shap.Explanation, output_dir: str):
    """
    Perform statistical tests comparing offensive vs defensive SHAP values.

    Tests:
    1. Mann-Whitney U test: Compare distributions of |SHAP| values
    2. Permutation test: Test if defensive dominance is significant
    3. Bootstrap confidence intervals for the ratio
    4. Effect size (rank-biserial correlation)
    """
    print("\n" + "=" * 70)
    print("STATISTICAL TESTS: OFFENSIVE vs DEFENSIVE METRICS")
    print("=" * 70)

    mean_abs_shap = np.abs(aggregated_shap.values).mean(axis=0)
    off_def = get_offense_defense_categories()

    # Extract SHAP values for offensive and defensive features
    off_indices = []
    for start, end in off_def['Offensive Metrics']:
        off_indices.extend(range(start, end))

    def_indices = []
    for start, end in off_def['Defensive Metrics']:
        def_indices.extend(range(start, end))

    off_shap = mean_abs_shap[off_indices]
    def_shap = mean_abs_shap[def_indices]

    print(f"\nSample sizes:")
    print(f"  Offensive features: {len(off_shap)}")
    print(f"  Defensive features: {len(def_shap)}")

    print(f"\nDescriptive Statistics:")
    print(f"  Offensive: mean={off_shap.mean():.4f}, std={off_shap.std():.4f}, median={np.median(off_shap):.4f}")
    print(f"  Defensive: mean={def_shap.mean():.4f}, std={def_shap.std():.4f}, median={np.median(def_shap):.4f}")

    # 1. Mann-Whitney U Test (non-parametric)
    print("\n" + "-" * 50)
    print("1. MANN-WHITNEY U TEST")
    print("-" * 50)
    u_stat, p_value_mw = stats.mannwhitneyu(def_shap, off_shap, alternative='greater')

    # Effect size: rank-biserial correlation
    n1, n2 = len(def_shap), len(off_shap)
    r_rb = 1 - (2 * u_stat) / (n1 * n2)

    print(f"  H0: Defensive SHAP values <= Offensive SHAP values")
    print(f"  H1: Defensive SHAP values > Offensive SHAP values")
    print(f"  U-statistic: {u_stat:.1f}")
    print(f"  p-value (one-tailed): {p_value_mw:.6f}")
    print(f"  Rank-biserial correlation (effect size): r = {r_rb:.3f}")

    if abs(r_rb) < 0.1:
        effect_interp = "negligible"
    elif abs(r_rb) < 0.3:
        effect_interp = "small"
    elif abs(r_rb) < 0.5:
        effect_interp = "medium"
    else:
        effect_interp = "large"
    print(f"  Effect size interpretation: {effect_interp}")

    if p_value_mw < 0.001:
        print(f"  Result: HIGHLY SIGNIFICANT (p < 0.001)")
    elif p_value_mw < 0.01:
        print(f"  Result: VERY SIGNIFICANT (p < 0.01)")
    elif p_value_mw < 0.05:
        print(f"  Result: SIGNIFICANT (p < 0.05)")
    else:
        print(f"  Result: NOT SIGNIFICANT (p >= 0.05)")

    # 2. Bootstrap Confidence Interval for Ratio
    print("\n" + "-" * 50)
    print("2. BOOTSTRAP CONFIDENCE INTERVAL FOR RATIO")
    print("-" * 50)

    n_bootstrap = 10000
    np.random.seed(42)

    bootstrap_ratios = []
    for _ in range(n_bootstrap):
        off_boot = np.random.choice(off_shap, size=len(off_shap), replace=True)
        def_boot = np.random.choice(def_shap, size=len(def_shap), replace=True)
        ratio = def_boot.mean() / off_boot.mean() if off_boot.mean() > 0 else np.nan
        bootstrap_ratios.append(ratio)

    bootstrap_ratios = np.array([r for r in bootstrap_ratios if not np.isnan(r)])

    ci_lower = np.percentile(bootstrap_ratios, 2.5)
    ci_upper = np.percentile(bootstrap_ratios, 97.5)
    observed_ratio = def_shap.mean() / off_shap.mean()

    print(f"  Observed ratio (Def/Off): {observed_ratio:.3f}")
    print(f"  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
    print(f"  Bootstrap SE: {np.std(bootstrap_ratios):.3f}")

    if ci_lower > 1.0:
        print(f"  Result: Defensive metrics significantly higher (CI excludes 1.0)")
    else:
        print(f"  Result: CI includes 1.0, ratio not significantly different from 1")

    # 3. Permutation Test
    print("\n" + "-" * 50)
    print("3. PERMUTATION TEST")
    print("-" * 50)

    observed_diff = def_shap.mean() - off_shap.mean()

    combined = np.concatenate([off_shap, def_shap])
    n_off = len(off_shap)

    n_permutations = 10000
    perm_diffs = []

    np.random.seed(42)
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_off = combined[:n_off]
        perm_def = combined[n_off:]
        perm_diffs.append(perm_def.mean() - perm_off.mean())

    perm_diffs = np.array(perm_diffs)
    p_value_perm = (perm_diffs >= observed_diff).mean()

    print(f"  Observed difference (Def - Off): {observed_diff:.4f}")
    print(f"  Permutation p-value (one-tailed): {p_value_perm:.6f}")

    if p_value_perm < 0.001:
        print(f"  Result: HIGHLY SIGNIFICANT (p < 0.001)")
    elif p_value_perm < 0.01:
        print(f"  Result: VERY SIGNIFICANT (p < 0.01)")
    elif p_value_perm < 0.05:
        print(f"  Result: SIGNIFICANT (p < 0.05)")
    else:
        print(f"  Result: NOT SIGNIFICANT (p >= 0.05)")

    # 4. Kolmogorov-Smirnov Test (distribution comparison)
    print("\n" + "-" * 50)
    print("4. KOLMOGOROV-SMIRNOV TEST (Distribution Comparison)")
    print("-" * 50)

    ks_stat, p_value_ks = stats.ks_2samp(def_shap, off_shap)

    print(f"  H0: Distributions are identical")
    print(f"  KS statistic: {ks_stat:.4f}")
    print(f"  p-value: {p_value_ks:.6f}")

    if p_value_ks < 0.05:
        print(f"  Result: Distributions are SIGNIFICANTLY DIFFERENT")
    else:
        print(f"  Result: Cannot reject that distributions are identical")

    # Summary
    print("\n" + "=" * 70)
    print("STATISTICAL SUMMARY")
    print("=" * 70)
    print(f"\nDefensive metrics have {observed_ratio:.2f}x higher average |SHAP| than offensive metrics")
    print(f"  - 95% Bootstrap CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
    print(f"  - Mann-Whitney p-value: {p_value_mw:.4f}")
    print(f"  - Permutation p-value: {p_value_perm:.4f}")
    print(f"  - Effect size (rank-biserial r): {r_rb:.3f} ({effect_interp})")

    return {
        'observed_ratio': observed_ratio,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'mann_whitney_p': p_value_mw,
        'permutation_p': p_value_perm,
        'effect_size_r': r_rb,
        'ks_p': p_value_ks
    }


def statistical_tests_by_era(df: pd.DataFrame, aggregated_shap: shap.Explanation):
    """
    Test if defensive bias differs significantly across eras.
    Uses Kruskal-Wallis test and pairwise comparisons.
    """
    print("\n" + "=" * 70)
    print("STATISTICAL TESTS: ERA COMPARISON")
    print("=" * 70)

    # Get years
    if 'Year' in df.columns:
        years = df['Year'].values
    else:
        years = df.iloc[:, 2].values

    eras = {
        'Pre-1970': years < 1970,
        '1970-1999': (years >= 1970) & (years < 2000),
        '2000+': years >= 2000
    }

    shap_values = aggregated_shap.values
    off_def = get_offense_defense_categories()

    # For each sample, compute the defensive/offensive ratio
    def compute_sample_ratio(sample_shap):
        off_sum = sum(np.abs(sample_shap[start:end]).mean() for start, end in off_def['Offensive Metrics'])
        def_sum = sum(np.abs(sample_shap[start:end]).mean() for start, end in off_def['Defensive Metrics'])
        return def_sum / off_sum if off_sum > 0 else np.nan

    era_ratios = {}
    for era_name, mask in eras.items():
        idx = np.where(mask)[0]
        if len(idx) >= 10:
            ratios = [compute_sample_ratio(shap_values[i]) for i in idx]
            ratios = [r for r in ratios if not np.isnan(r)]
            era_ratios[era_name] = ratios

    print("\nPer-sample Def/Off ratios by era:")
    for era, ratios in era_ratios.items():
        print(f"  {era}: n={len(ratios)}, mean={np.mean(ratios):.3f}, std={np.std(ratios):.3f}")

    # Kruskal-Wallis test
    if len(era_ratios) >= 2:
        print("\n" + "-" * 50)
        print("KRUSKAL-WALLIS TEST: Do eras differ in defensive bias?")
        print("-" * 50)

        era_groups = list(era_ratios.values())
        h_stat, p_value_kw = stats.kruskal(*era_groups)

        print(f"  H-statistic: {h_stat:.4f}")
        print(f"  p-value: {p_value_kw:.4f}")

        if p_value_kw < 0.05:
            print(f"  Result: SIGNIFICANT difference across eras")
        else:
            print(f"  Result: NO significant difference across eras")
            print(f"  Interpretation: Defensive bias is CONSISTENT across all eras")

    return era_ratios


def analyze_refined_categories(aggregated_shap: shap.Explanation, feature_names: List[str]):
    """Analyze SHAP importance with refined offensive/defensive split."""

    categories = get_refined_feature_categories()
    mean_abs_shap = np.abs(aggregated_shap.values).mean(axis=0)

    print("\n" + "=" * 70)
    print("REFINED CATEGORY ANALYSIS: OFFENSIVE vs DEFENSIVE METRICS")
    print("=" * 70)

    results = {}
    for cat_name, (start, end) in categories.items():
        cat_shap = mean_abs_shap[start:end]
        results[cat_name] = {
            'total': cat_shap.sum(),
            'avg': cat_shap.mean(),
            'n_features': end - start
        }

    # Print table
    print(f"\n{'Category':<25} {'Total SHAP':>12} {'Avg SHAP':>12} {'# Features':>12}")
    print("-" * 65)

    # Sort by average importance
    sorted_cats = sorted(results.items(), key=lambda x: x[1]['avg'], reverse=True)
    for cat_name, stats in sorted_cats:
        print(f"{cat_name:<25} {stats['total']:>12.4f} {stats['avg']:>12.4f} {stats['n_features']:>12}")

    # Aggregate into Offensive vs Defensive
    print("\n" + "-" * 65)
    print("AGGREGATED: OFFENSIVE vs DEFENSIVE")
    print("-" * 65)

    off_def = get_offense_defense_categories()
    agg_results = {}

    for group_name, ranges in off_def.items():
        total_shap = 0
        total_features = 0
        for start, end in ranges:
            total_shap += mean_abs_shap[start:end].sum()
            total_features += (end - start)
        agg_results[group_name] = {
            'total': total_shap,
            'avg': total_shap / total_features if total_features > 0 else 0,
            'n_features': total_features
        }

    for group_name, stats in agg_results.items():
        print(f"{group_name:<25} {stats['total']:>12.4f} {stats['avg']:>12.4f} {stats['n_features']:>12}")

    return results, agg_results


def analyze_by_coach_background(X: np.ndarray, aggregated_shap: shap.Explanation,
                                 coach_names: np.ndarray, backgrounds_df: pd.DataFrame,
                                 feature_names: List[str], output_dir: str):
    """
    Analyze SHAP values segmented by coach background.

    Key hypothesis: Are coaches evaluated based on their area of expertise?
    - Do defensive metrics matter more for DEFENSIVE-background coaches?
    - Do offensive metrics matter more for OFFENSIVE-background coaches?
    """
    print("\n" + "=" * 70)
    print("SHAP ANALYSIS BY COACH BACKGROUND")
    print("=" * 70)
    print("\nHypothesis: Coaches are evaluated based on their area of expertise")
    print("  - Defensive metrics should matter more for defensive-background coaches")
    print("  - Offensive metrics should matter more for offensive-background coaches")

    # Match coaches with backgrounds
    background_map = dict(zip(backgrounds_df['Coach_Name'], backgrounds_df['Background']))

    # Get indices for each background type
    offensive_idx = []
    defensive_idx = []
    other_idx = []

    for i, coach in enumerate(coach_names):
        bg = background_map.get(coach, 'Other')
        if bg == 'Offensive':
            offensive_idx.append(i)
        elif bg == 'Defensive':
            defensive_idx.append(i)
        else:
            other_idx.append(i)

    print(f"\nCoach distribution in SHAP sample:")
    print(f"  Offensive background: {len(offensive_idx)}")
    print(f"  Defensive background: {len(defensive_idx)}")
    print(f"  Other/Both: {len(other_idx)}")

    # Get SHAP values for each group
    shap_values = aggregated_shap.values

    off_def = get_offense_defense_categories()

    # Store results for comparison
    bg_results = {}

    # Analyze by background
    for bg_name, idx_list in [('Offensive', offensive_idx),
                               ('Defensive', defensive_idx)]:
        if len(idx_list) < 10:
            print(f"\n{bg_name} Background: Insufficient samples ({len(idx_list)})")
            continue

        print(f"\n{bg_name} Background Coaches (n={len(idx_list)}):")
        print("-" * 60)

        bg_shap = shap_values[idx_list]
        mean_abs_bg = np.abs(bg_shap).mean(axis=0)

        bg_results[bg_name] = {'indices': idx_list}

        # Offensive vs Defensive importance for this background
        for group_name, ranges in off_def.items():
            if group_name == 'Other':
                continue
            total_shap = sum(mean_abs_bg[start:end].sum() for start, end in ranges)
            n_features = sum(end - start for start, end in ranges)
            avg_shap = total_shap / n_features

            bg_results[bg_name][group_name] = {'total': total_shap, 'avg': avg_shap, 'n': n_features}
            print(f"  {group_name:<25} Total: {total_shap:.4f}, Avg: {avg_shap:.4f}")

    # Test hypothesis: Do coaches get evaluated on their expertise?
    if 'Offensive' in bg_results and 'Defensive' in bg_results:
        print("\n" + "-" * 60)
        print("HYPOTHESIS TEST: Are coaches evaluated on their expertise?")
        print("-" * 60)

        # For offensive coaches: compare offensive vs defensive metric importance
        off_coach_off_metrics = bg_results['Offensive']['Offensive Metrics']['avg']
        off_coach_def_metrics = bg_results['Offensive']['Defensive Metrics']['avg']

        # For defensive coaches: compare offensive vs defensive metric importance
        def_coach_off_metrics = bg_results['Defensive']['Offensive Metrics']['avg']
        def_coach_def_metrics = bg_results['Defensive']['Defensive Metrics']['avg']

        print(f"\nOffensive-background coaches:")
        print(f"  Offensive metrics avg: {off_coach_off_metrics:.4f}")
        print(f"  Defensive metrics avg: {off_coach_def_metrics:.4f}")
        if off_coach_off_metrics > off_coach_def_metrics:
            print(f"  -> Offensive metrics are {((off_coach_off_metrics/off_coach_def_metrics)-1)*100:.1f}% more important (supports hypothesis)")
        else:
            print(f"  -> Defensive metrics are {((off_coach_def_metrics/off_coach_off_metrics)-1)*100:.1f}% more important (contradicts hypothesis)")

        print(f"\nDefensive-background coaches:")
        print(f"  Offensive metrics avg: {def_coach_off_metrics:.4f}")
        print(f"  Defensive metrics avg: {def_coach_def_metrics:.4f}")
        if def_coach_def_metrics > def_coach_off_metrics:
            print(f"  -> Defensive metrics are {((def_coach_def_metrics/def_coach_off_metrics)-1)*100:.1f}% more important (supports hypothesis)")
        else:
            print(f"  -> Offensive metrics are {((def_coach_off_metrics/def_coach_def_metrics)-1)*100:.1f}% more important (contradicts hypothesis)")

        # Create visualization
        plot_background_comparison(bg_results, output_dir)

        # Run statistical tests
        statistical_tests_by_background(shap_values, bg_results, off_def)

    return bg_results


def statistical_tests_by_background(shap_values: np.ndarray, bg_results: Dict, off_def: Dict):
    """
    Statistical tests for coach background analysis.

    Tests:
    1. Within each background: Is defensive > offensive significant?
    2. Between backgrounds: Does the def/off ratio differ?
    """
    print("\n" + "=" * 70)
    print("STATISTICAL TESTS: COACH BACKGROUND ANALYSIS")
    print("=" * 70)

    # Get feature indices
    off_indices = []
    for start, end in off_def['Offensive Metrics']:
        off_indices.extend(range(start, end))

    def_indices = []
    for start, end in off_def['Defensive Metrics']:
        def_indices.extend(range(start, end))

    # For each background, compute per-sample def/off ratios
    def compute_sample_ratios(indices):
        """Compute def/off ratio for each sample."""
        ratios = []
        for i in indices:
            sample_shap = np.abs(shap_values[i])
            off_avg = sample_shap[off_indices].mean()
            def_avg = sample_shap[def_indices].mean()
            if off_avg > 0:
                ratios.append(def_avg / off_avg)
        return np.array(ratios)

    off_bg_ratios = compute_sample_ratios(bg_results['Offensive']['indices'])
    def_bg_ratios = compute_sample_ratios(bg_results['Defensive']['indices'])

    # 1. Test: Within offensive-background coaches, is defensive > offensive?
    print("\n" + "-" * 50)
    print("1. OFFENSIVE-BACKGROUND COACHES: Def vs Off importance")
    print("-" * 50)

    # One-sample Wilcoxon test: Is median ratio > 1?
    stat_off, p_off = stats.wilcoxon(off_bg_ratios - 1, alternative='greater')
    median_off = np.median(off_bg_ratios)

    print(f"  Median Def/Off ratio: {median_off:.3f}")
    print(f"  Wilcoxon signed-rank test (H1: ratio > 1)")
    print(f"  Statistic: {stat_off:.1f}, p-value: {p_off:.4f}")

    if p_off < 0.05:
        print(f"  Result: SIGNIFICANT - Defensive metrics are significantly more important")
    else:
        print(f"  Result: Not significant")

    # 2. Test: Within defensive-background coaches, is defensive > offensive?
    print("\n" + "-" * 50)
    print("2. DEFENSIVE-BACKGROUND COACHES: Def vs Off importance")
    print("-" * 50)

    stat_def, p_def = stats.wilcoxon(def_bg_ratios - 1, alternative='greater')
    median_def = np.median(def_bg_ratios)

    print(f"  Median Def/Off ratio: {median_def:.3f}")
    print(f"  Wilcoxon signed-rank test (H1: ratio > 1)")
    print(f"  Statistic: {stat_def:.1f}, p-value: {p_def:.4f}")

    if p_def < 0.05:
        print(f"  Result: SIGNIFICANT - Defensive metrics are significantly more important")
    else:
        print(f"  Result: Not significant")

    # 3. Test: Does the ratio differ between offensive and defensive background coaches?
    print("\n" + "-" * 50)
    print("3. COMPARISON: Does defensive bias differ by coach background?")
    print("-" * 50)

    u_stat, p_between = stats.mannwhitneyu(off_bg_ratios, def_bg_ratios, alternative='two-sided')

    print(f"  Offensive-background median ratio: {median_off:.3f}")
    print(f"  Defensive-background median ratio: {median_def:.3f}")
    print(f"  Mann-Whitney U test (two-tailed)")
    print(f"  U-statistic: {u_stat:.1f}, p-value: {p_between:.4f}")

    if p_between < 0.05:
        print(f"  Result: SIGNIFICANT - Defensive bias differs by coach background")
    else:
        print(f"  Result: NOT SIGNIFICANT - Defensive bias is similar regardless of background")

    # 4. Bootstrap CI for difference in ratios
    print("\n" + "-" * 50)
    print("4. BOOTSTRAP CI: Difference in Def/Off ratios between backgrounds")
    print("-" * 50)

    n_bootstrap = 10000
    np.random.seed(42)

    diff_bootstraps = []
    for _ in range(n_bootstrap):
        off_boot = np.random.choice(off_bg_ratios, size=len(off_bg_ratios), replace=True)
        def_boot = np.random.choice(def_bg_ratios, size=len(def_bg_ratios), replace=True)
        diff_bootstraps.append(np.median(def_boot) - np.median(off_boot))

    diff_bootstraps = np.array(diff_bootstraps)
    ci_lower = np.percentile(diff_bootstraps, 2.5)
    ci_upper = np.percentile(diff_bootstraps, 97.5)
    observed_diff = median_def - median_off

    print(f"  Observed difference (Def-bg minus Off-bg): {observed_diff:.3f}")
    print(f"  95% Bootstrap CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

    if ci_lower > 0:
        print(f"  Result: Defensive-background coaches show significantly MORE defensive bias")
    elif ci_upper < 0:
        print(f"  Result: Offensive-background coaches show significantly MORE defensive bias")
    else:
        print(f"  Result: No significant difference in defensive bias between backgrounds")

    # Summary
    print("\n" + "=" * 70)
    print("BACKGROUND ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"\n1. Offensive-background coaches: Defensive metrics {median_off:.2f}x more important (p={p_off:.4f})")
    print(f"2. Defensive-background coaches: Defensive metrics {median_def:.2f}x more important (p={p_def:.4f})")
    print(f"3. Difference between backgrounds: p={p_between:.4f}")

    if p_between >= 0.05:
        print(f"\nConclusion: Defensive metrics dominate for ALL coaches regardless of background")
    else:
        if median_def > median_off:
            print(f"\nConclusion: Defensive-background coaches show stronger defensive bias")


def plot_background_comparison(bg_results: Dict, output_dir: str):
    """Create visualization comparing metric importance by coach background."""

    fig, ax = plt.subplots(figsize=(10, 6))

    backgrounds = ['Offensive', 'Defensive']
    metric_types = ['Offensive Metrics', 'Defensive Metrics']

    x = np.arange(len(backgrounds))
    width = 0.35

    off_values = [bg_results[bg]['Offensive Metrics']['avg'] for bg in backgrounds]
    def_values = [bg_results[bg]['Defensive Metrics']['avg'] for bg in backgrounds]

    bars1 = ax.bar(x - width/2, off_values, width, label='Offensive Metrics', color='#FF6B35', alpha=0.8)
    bars2 = ax.bar(x + width/2, def_values, width, label='Defensive Metrics', color='#004E89', alpha=0.8)

    ax.set_ylabel('Average |SHAP| per Feature', fontsize=12)
    ax.set_title('Feature Importance by Coach Background\n(Testing: Are coaches evaluated on their expertise?)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{bg}\nBackground' for bg in backgrounds], fontsize=11)
    ax.legend(fontsize=11)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.0001,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    # Add hypothesis indicators
    # For offensive coaches, offensive metrics should be higher
    # For defensive coaches, defensive metrics should be higher

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'shap_by_coach_background.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}")


def plot_refined_category_importance(aggregated_shap: shap.Explanation,
                                      output_dir: str):
    """Create visualization with refined offensive/defensive categories."""

    categories = get_refined_feature_categories()
    mean_abs_shap = np.abs(aggregated_shap.values).mean(axis=0)

    # Calculate stats
    cat_stats = {}
    for cat_name, (start, end) in categories.items():
        cat_shap = mean_abs_shap[start:end]
        cat_stats[cat_name] = {
            'total': cat_shap.sum(),
            'avg': cat_shap.mean()
        }

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Color coding: blue for defensive, orange for offensive, gray for other
    colors = {
        'Core Experience': '#808080',
        'OC Stats (Offense)': '#FF6B35',
        'DC Stats (Defense)': '#004E89',
        'HC Team Stats (Offense)': '#FF9F1C',
        'HC Opp Stats (Defense)': '#2EC4B6',
        'Hiring Team': '#808080'
    }

    cats = list(cat_stats.keys())
    totals = [cat_stats[c]['total'] for c in cats]
    avgs = [cat_stats[c]['avg'] for c in cats]
    bar_colors = [colors[c] for c in cats]

    # Total importance
    bars1 = ax1.barh(cats, totals, color=bar_colors)
    ax1.set_xlabel('Total Mean |SHAP|', fontsize=12)
    ax1.set_title('Total SHAP Importance by Category', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    # Extend x-axis to accommodate labels
    ax1.set_xlim(0, max(totals) * 1.25)

    for bar, val in zip(bars1, totals):
        ax1.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontsize=10)

    # Average importance
    bars2 = ax2.barh(cats, avgs, color=bar_colors)
    ax2.set_xlabel('Average Mean |SHAP| per Feature', fontsize=12)
    ax2.set_title('Average SHAP Importance by Category', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    # Extend x-axis to accommodate labels
    ax2.set_xlim(0, max(avgs) * 1.25)

    for bar, val in zip(bars2, avgs):
        ax2.text(val + 0.0002, bar.get_y() + bar.get_height()/2,
                 f'{val:.4f}', va='center', fontsize=10)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B35', label='Offensive'),
        Patch(facecolor='#004E89', label='Defensive'),
        Patch(facecolor='#808080', label='Other')
    ]
    ax2.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'shap_offense_defense_split.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}")

    return cat_stats


def plot_offense_vs_defense_summary(aggregated_shap: shap.Explanation, output_dir: str):
    """Create a simple offense vs defense comparison chart."""

    mean_abs_shap = np.abs(aggregated_shap.values).mean(axis=0)
    off_def = get_offense_defense_categories()

    # Calculate totals and averages
    results = {}
    for group_name, ranges in off_def.items():
        total = sum(mean_abs_shap[start:end].sum() for start, end in ranges)
        n_features = sum(end - start for start, end in ranges)
        results[group_name] = {
            'total': total,
            'avg': total / n_features,
            'n': n_features
        }

    # Create simple bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    groups = ['Offensive Metrics', 'Defensive Metrics']
    avgs = [results[g]['avg'] for g in groups]
    totals = [results[g]['total'] for g in groups]
    colors = ['#FF6B35', '#004E89']

    x = np.arange(len(groups))
    width = 0.35

    bars1 = ax.bar(x - width/2, avgs, width, label='Avg per Feature', color=colors, alpha=0.7)

    ax.set_ylabel('Mean |SHAP| Value', fontsize=12)
    ax.set_title('Feature Importance: Offensive vs Defensive Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=12)

    # Add value labels
    for bar, val in zip(bars1, avgs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11)

    # Add feature counts
    for i, (g, r) in enumerate([(g, results[g]) for g in groups]):
        ax.text(i, -0.0003, f'({r["n"]} features)', ha='center', fontsize=10)

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'shap_offense_vs_defense.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    return results


def get_feature_names() -> List[str]:
    """Get human-readable feature names."""
    base_names = get_all_feature_names()
    all_names = base_names + HIRING_TEAM_FEATURES

    readable_names = []
    for name in all_names:
        name = name.replace('__oc', ' (OC)')
        name = name.replace('__dc', ' (DC)')
        name = name.replace('__opp__hc', ' (HC Opp)')
        name = name.replace('__hc', ' (HC)')
        name = name.replace('hiring_team_', 'Hiring: ')
        name = name.replace('num_yr_', 'Yrs ')
        name = name.replace('num_times_', '# ')
        name = name.replace('_', ' ')
        readable_names.append(name)

    return readable_names


def analyze_by_era(df: pd.DataFrame, aggregated_shap: shap.Explanation, output_dir: str):
    """
    Analyze SHAP values segmented by era to see if defensive emphasis changed over time.

    Eras:
    - Pre-1970: Pre-merger, limited data
    - 1970-1999: Post-merger, defense-dominant era
    - 2000+: Modern passing era (especially post-2004 rule changes)
    """
    print("\n" + "=" * 70)
    print("SHAP ANALYSIS BY ERA")
    print("=" * 70)
    print("\nQuestion: Has the importance of defensive metrics changed over time?")
    print("  - Pre-merger era (before 1970): Limited data")
    print("  - Post-merger era (1970-1999): Defense-dominant philosophy")
    print("  - Modern era (2000+): Passing-friendly rules, offensive emphasis")

    # Get years from dataframe
    if 'Year' in df.columns:
        years = df['Year'].values
    else:
        years = df.iloc[:, 2].values  # Assume Year is third column

    # Define eras
    eras = {
        'Pre-1970': years < 1970,
        '1970-1999': (years >= 1970) & (years < 2000),
        '2000+': years >= 2000
    }

    shap_values = aggregated_shap.values
    off_def = get_offense_defense_categories()

    era_results = {}

    print(f"\n{'Era':<15} {'N':>6} {'Off Avg':>10} {'Def Avg':>10} {'Def/Off':>10}")
    print("-" * 55)

    for era_name, mask in eras.items():
        idx = np.where(mask)[0]

        if len(idx) < 10:
            print(f"{era_name:<15} {len(idx):>6}   (insufficient samples)")
            continue

        era_shap = shap_values[idx]
        mean_abs_era = np.abs(era_shap).mean(axis=0)

        # Calculate offensive vs defensive
        off_total = sum(mean_abs_era[start:end].sum() for start, end in off_def['Offensive Metrics'])
        off_n = sum(end - start for start, end in off_def['Offensive Metrics'])
        off_avg = off_total / off_n

        def_total = sum(mean_abs_era[start:end].sum() for start, end in off_def['Defensive Metrics'])
        def_n = sum(end - start for start, end in off_def['Defensive Metrics'])
        def_avg = def_total / def_n

        ratio = def_avg / off_avg if off_avg > 0 else 0

        era_results[era_name] = {
            'n': len(idx),
            'off_avg': off_avg,
            'def_avg': def_avg,
            'ratio': ratio
        }

        print(f"{era_name:<15} {len(idx):>6} {off_avg:>10.4f} {def_avg:>10.4f} {ratio:>10.2f}x")

    # Interpretation
    print("\n" + "-" * 55)
    print("INTERPRETATION:")

    if '1970-1999' in era_results and '2000+' in era_results:
        old_ratio = era_results['1970-1999']['ratio']
        new_ratio = era_results['2000+']['ratio']

        if new_ratio < old_ratio:
            pct_change = ((old_ratio - new_ratio) / old_ratio) * 100
            print(f"  Defensive emphasis has DECREASED by {pct_change:.1f}% in modern era")
            print(f"  - 1970-1999: Defensive metrics {old_ratio:.2f}x more important")
            print(f"  - 2000+: Defensive metrics {new_ratio:.2f}x more important")
        else:
            pct_change = ((new_ratio - old_ratio) / old_ratio) * 100
            print(f"  Defensive emphasis has INCREASED by {pct_change:.1f}% in modern era")
            print(f"  - 1970-1999: Defensive metrics {old_ratio:.2f}x more important")
            print(f"  - 2000+: Defensive metrics {new_ratio:.2f}x more important")

    # Create visualization
    plot_era_comparison(era_results, output_dir)

    return era_results


def plot_era_comparison(era_results: Dict, output_dir: str):
    """Create visualization comparing offensive vs defensive importance by era."""

    # Filter to eras with enough data
    eras = [e for e in ['Pre-1970', '1970-1999', '2000+'] if e in era_results and era_results[e]['n'] >= 10]

    if len(eras) < 2:
        print("Insufficient eras with data for visualization")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(eras))
    width = 0.35

    off_values = [era_results[e]['off_avg'] for e in eras]
    def_values = [era_results[e]['def_avg'] for e in eras]

    bars1 = ax.bar(x - width/2, off_values, width, label='Offensive Metrics', color='#FF6B35', alpha=0.8)
    bars2 = ax.bar(x + width/2, def_values, width, label='Defensive Metrics', color='#004E89', alpha=0.8)

    ax.set_ylabel('Average |SHAP| per Feature', fontsize=12)
    ax.set_title('Feature Importance by Era:\nHas Defensive Emphasis Changed Over Time?',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{e}\n(n={era_results[e]["n"]})' for e in eras], fontsize=11)
    ax.legend(fontsize=11)

    # Extend y-axis to accommodate labels and ratio annotations
    max_val = max(max(off_values), max(def_values))
    ax.set_ylim(0, max_val * 1.35)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.0001,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=9)

    # Add ratio annotations
    for i, era in enumerate(eras):
        ratio = era_results[era]['ratio']
        ax.annotate(f'{ratio:.1f}x', xy=(i, max(off_values[i], def_values[i]) + 0.0005),
                    ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'shap_by_era.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}")


def main():
    print("=" * 70)
    print("ENHANCED SHAP ANALYSIS: OFFENSIVE vs DEFENSIVE METRICS")
    print("=" * 70)

    # Load data and SHAP values
    X, y, df, shap_values_dict, aggregated_shap, coach_names = load_data_and_shap()
    feature_names = get_feature_names()

    print(f"\nData loaded: {X.shape[0]} samples, {X.shape[1]} features")

    # Output directories
    paper_dir = os.path.join(project_root, 'latex', 'figures')
    exploratory_dir = os.path.join(project_root, 'figures', 'tenure')
    os.makedirs(paper_dir, exist_ok=True)
    os.makedirs(exploratory_dir, exist_ok=True)

    # 1. Refined category analysis
    refined_results, agg_results = analyze_refined_categories(aggregated_shap, feature_names)

    # 2. Create visualizations
    print("\nGenerating visualizations...")
    plot_refined_category_importance(aggregated_shap, paper_dir)
    off_def_results = plot_offense_vs_defense_summary(aggregated_shap, exploratory_dir)

    # 3. Statistical tests for offense vs defense
    stat_results = statistical_tests_offense_vs_defense(aggregated_shap, exploratory_dir)

    # 4. Analyze by era
    era_results = analyze_by_era(df, aggregated_shap, paper_dir)

    # 5. Statistical tests for era comparison
    era_stat_results = statistical_tests_by_era(df, aggregated_shap)

    # 6. Load coach backgrounds and analyze by background
    bg_results = None
    try:
        backgrounds_df = load_coach_backgrounds()
        if not backgrounds_df.empty:
            bg_results = analyze_by_coach_background(X, aggregated_shap, coach_names,
                                                      backgrounds_df, feature_names, exploratory_dir)
    except Exception as e:
        print(f"\nCould not analyze by coach background: {e}")
        print("To enable this analysis, ensure coach background data is available.")

    # Summary
    print("\n" + "=" * 70)
    print("KEY FINDING: OFFENSIVE vs DEFENSIVE METRICS")
    print("=" * 70)

    off_avg = off_def_results['Offensive Metrics']['avg']
    def_avg = off_def_results['Defensive Metrics']['avg']

    if def_avg > off_avg:
        pct_diff = ((def_avg - off_avg) / off_avg) * 100
        print(f"\nDefensive metrics have {pct_diff:.1f}% higher average importance than offensive metrics.")
        print(f"  - Offensive: {off_avg:.4f} avg |SHAP| per feature")
        print(f"  - Defensive: {def_avg:.4f} avg |SHAP| per feature")
    else:
        pct_diff = ((off_avg - def_avg) / def_avg) * 100
        print(f"\nOffensive metrics have {pct_diff:.1f}% higher average importance than defensive metrics.")
        print(f"  - Offensive: {off_avg:.4f} avg |SHAP| per feature")
        print(f"  - Defensive: {def_avg:.4f} avg |SHAP| per feature")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
