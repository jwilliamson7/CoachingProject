"""
Analyze WAR prediction errors across coach subgroups.

Uses cross-validated predictions to determine which types of coaches
are easier or harder for the WAR model to predict.

Usage:
    python scripts/analyze_war_subgroups.py
    python scripts/analyze_war_subgroups.py --n-folds 10
    python scripts/analyze_war_subgroups.py --no-plots
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.war_regressor import WARRegressor
from model.config import MODEL_PATHS, WAR_CONFIG, MODEL_CONFIG
from model.evaluation import regression_metrics


# =============================================================================
# Data Loading
# =============================================================================

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load WAR prediction data and coach background classifications."""
    war_df = pd.read_csv(PROJECT_ROOT / MODEL_PATHS['war_data_file'])
    bg_df = pd.read_csv(PROJECT_ROOT / 'data' / 'coach_backgrounds.csv')
    return war_df, bg_df


def merge_backgrounds(war_df: pd.DataFrame, bg_df: pd.DataFrame) -> pd.DataFrame:
    """Merge coach background (Offensive/Defensive/Other) into WAR data."""
    bg_map = bg_df.set_index('Coach_Name')['Background'].to_dict()
    war_df = war_df.copy()
    war_df['background'] = war_df['Coach Name'].map(bg_map).fillna('Unknown')
    # Collapse Both into Other for cleaner groups
    war_df.loc[war_df['background'] == 'Both', 'background'] = 'Other'
    return war_df


# =============================================================================
# Cross-Validated Predictions
# =============================================================================

def get_cv_predictions(
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42
) -> np.ndarray:
    """
    Generate out-of-fold predictions via coach-level cross-validation.

    Each fold trains a separate model and predicts on held-out coaches,
    ensuring every data point gets exactly one prediction.
    """
    coaches = df['Coach Name'].values
    unique_coaches = np.array(df['Coach Name'].unique())

    rng = np.random.RandomState(random_state)
    rng.shuffle(unique_coaches)

    fold_size = len(unique_coaches) // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else len(unique_coaches)
        folds.append(set(unique_coaches[start:end]))

    predictions = np.full(len(y), np.nan)

    for fold_idx in range(n_folds):
        val_coaches = folds[fold_idx]
        train_coaches = set(unique_coaches) - val_coaches

        train_mask = np.array([c in train_coaches for c in coaches])
        val_mask = np.array([c in val_coaches for c in coaches])

        X_train, y_train = X[train_mask], y[train_mask]
        X_val = X[val_mask]

        model = WARRegressor()
        model.fit(X_train, y_train, verbose=False)
        predictions[val_mask] = model.predict(X_val)

        print(f"  Fold {fold_idx + 1}/{n_folds}: "
              f"train={train_mask.sum()}, val={val_mask.sum()} samples")

    assert not np.isnan(predictions).any(), "Some samples missing predictions"
    return predictions


# =============================================================================
# Subgroup Definitions
# =============================================================================

def define_subgroups(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Define subgroup labels for each data point.

    Returns dict mapping subgroup_name -> Series of category labels.
    """
    groups = {}

    # 1. Prior HC experience (Feature 2 = num_times_hc)
    groups['Prior HC Experience'] = df['Feature 2'].apply(
        lambda x: 'First-time HC' if x == 0 else 'Experienced HC'
    )

    # 2. Age at hire (Feature 1)
    groups['Age at Hire'] = pd.cut(
        df['Feature 1'],
        bins=[0, 44, 54, 100],
        labels=['Under 45', '45-54', '55+']
    )

    # 3. Era
    groups['Era'] = pd.cut(
        df['Year'],
        bins=[0, 1989, 2005, 2030],
        labels=['Pre-1990', '1990-2005', '2006+']
    )

    # 4. Tenure length (actual seasons coached)
    groups['Tenure'] = pd.cut(
        df['num_seasons'],
        bins=[0, 2, 4, 100],
        labels=['1-2 years', '3-4 years', '5+ years']
    )

    # 5. Coordinator background (from merged background column)
    groups['Coordinator Background'] = df['background']

    # 6. Total NFL experience (Features 6+7+8)
    nfl_exp = df['Feature 6'] + df['Feature 7'] + df['Feature 8']
    groups['NFL Experience'] = pd.cut(
        nfl_exp,
        bins=[-1, 9, 19, 100],
        labels=['<10 years', '10-19 years', '20+ years']
    )

    # 7. College HC experience (Feature 5)
    groups['College HC Background'] = df['Feature 5'].apply(
        lambda x: 'Former College HC' if x > 0 else 'No College HC'
    )

    return groups


# =============================================================================
# Subgroup Metrics
# =============================================================================

def compute_subgroup_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: Dict[str, pd.Series]
) -> Dict[str, pd.DataFrame]:
    """
    Compute MAE, RMSE, mean bias, and sample size for each subgroup.

    Returns dict mapping subgroup_name -> DataFrame of metrics per category.
    """
    residuals = y_true - y_pred
    abs_errors = np.abs(residuals)
    results = {}

    for group_name, labels in groups.items():
        rows = []
        for cat in labels.unique():
            if pd.isna(cat):
                continue
            mask = (labels == cat).values
            n = mask.sum()
            if n == 0:
                continue

            cat_residuals = residuals[mask]
            cat_abs_errors = abs_errors[mask]

            rows.append({
                'Category': cat,
                'N': n,
                'MAE': cat_abs_errors.mean(),
                'RMSE': np.sqrt((cat_residuals ** 2).mean()),
                'Mean Bias': cat_residuals.mean(),
                'Median AE': np.median(cat_abs_errors),
                'Actual Mean WAR': y_true[mask].mean(),
                'Predicted Mean WAR': y_pred[mask].mean(),
            })

        results[group_name] = pd.DataFrame(rows)

    return results


# =============================================================================
# Statistical Tests
# =============================================================================

def test_subgroup_differences(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: Dict[str, pd.Series]
) -> Dict[str, Dict]:
    """Run Kruskal-Wallis tests for significant differences in absolute error across groups."""
    abs_errors = np.abs(y_true - y_pred)
    test_results = {}

    for group_name, labels in groups.items():
        categories = [c for c in labels.unique() if not pd.isna(c)]
        if len(categories) < 2:
            continue

        group_errors = []
        for cat in categories:
            mask = (labels == cat).values
            group_errors.append(abs_errors[mask])

        if len(group_errors) >= 2 and all(len(g) >= 3 for g in group_errors):
            stat, p_value = stats.kruskal(*group_errors)
            test_results[group_name] = {
                'statistic': stat,
                'p_value': p_value,
                'n_groups': len(group_errors),
                'significant': p_value < 0.05
            }

    return test_results


# =============================================================================
# Printing
# =============================================================================

def print_results(
    overall_metrics: Dict,
    subgroup_metrics: Dict[str, pd.DataFrame],
    test_results: Dict[str, Dict]
):
    """Print formatted results tables."""
    print("\n" + "=" * 80)
    print("WAR PREDICTION ERROR ANALYSIS BY COACH SUBGROUP")
    print("=" * 80)

    print("\nOVERALL MODEL PERFORMANCE (Cross-Validated)")
    print("-" * 50)
    print(f"  MAE:         {overall_metrics['mae']:.4f}")
    print(f"  RMSE:        {overall_metrics['rmse']:.4f}")
    print(f"  R²:          {overall_metrics['r2']:.4f}")
    print(f"  Correlation: {overall_metrics['correlation']:.4f}")

    for group_name, df in subgroup_metrics.items():
        print(f"\n{'=' * 80}")
        print(f"  {group_name.upper()}")
        print(f"{'=' * 80}")

        # Sort by MAE descending (hardest first)
        df_sorted = df.sort_values('MAE', ascending=False)

        header = (f"  {'Category':<22} {'N':>5}  {'MAE':>8}  {'RMSE':>8}  "
                  f"{'Bias':>8}  {'Med AE':>8}  {'True WAR':>9}  {'Pred WAR':>9}")
        print(header)
        print("  " + "-" * 95)

        for _, row in df_sorted.iterrows():
            print(f"  {str(row['Category']):<22} {row['N']:>5}  "
                  f"{row['MAE']:>8.4f}  {row['RMSE']:>8.4f}  "
                  f"{row['Mean Bias']:>+8.4f}  {row['Median AE']:>8.4f}  "
                  f"{row['Actual Mean WAR']:>+9.4f}  {row['Predicted Mean WAR']:>+9.4f}")

        # Print statistical test if available
        if group_name in test_results:
            t = test_results[group_name]
            sig = "YES" if t['significant'] else "no"
            print(f"\n  Kruskal-Wallis test: H={t['statistic']:.3f}, "
                  f"p={t['p_value']:.4f} (significant: {sig})")


# =============================================================================
# Visualization
# =============================================================================

def create_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: Dict[str, pd.Series],
    output_dir: Path
):
    """Generate box plot visualizations of prediction errors by subgroup."""
    output_dir.mkdir(parents=True, exist_ok=True)
    residuals = y_true - y_pred
    abs_errors = np.abs(residuals)

    # Select subgroups to plot (skip those with too many categories)
    plot_groups = {k: v for k, v in groups.items()
                   if len([c for c in v.unique() if not pd.isna(c)]) <= 5}

    n_plots = len(plot_groups)
    fig, axes = plt.subplots(2, (n_plots + 1) // 2, figsize=(7 * ((n_plots + 1) // 2), 12))
    axes = axes.flatten()

    for idx, (group_name, labels) in enumerate(plot_groups.items()):
        ax = axes[idx]
        categories = sorted([c for c in labels.unique() if not pd.isna(c)], key=str)

        data_for_plot = []
        tick_labels = []
        for cat in categories:
            mask = (labels == cat).values
            data_for_plot.append(abs_errors[mask])
            n = mask.sum()
            tick_labels.append(f"{cat}\n(n={n})")

        bp = ax.boxplot(data_for_plot, tick_labels=tick_labels, patch_artist=True)
        colors = plt.cm.Set2(np.linspace(0, 1, len(data_for_plot)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(group_name, fontsize=13, fontweight='bold')
        ax.set_ylabel('Absolute Error (WAR)')
        ax.tick_params(axis='x', labelsize=9)
        ax.grid(axis='y', alpha=0.3)

    # Hide unused axes
    for idx in range(len(plot_groups), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('WAR Prediction Error by Coach Subgroup (Cross-Validated)',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    filepath = output_dir / 'war_subgroup_error_boxplots.png'
    fig.savefig(filepath, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\nBox plots saved to: {filepath}")
    plt.close(fig)

    # Second figure: bias plot (mean residual by subgroup)
    fig2, axes2 = plt.subplots(2, (n_plots + 1) // 2, figsize=(7 * ((n_plots + 1) // 2), 12))
    axes2 = axes2.flatten()

    for idx, (group_name, labels) in enumerate(plot_groups.items()):
        ax = axes2[idx]
        categories = sorted([c for c in labels.unique() if not pd.isna(c)], key=str)

        means = []
        ci_low = []
        ci_high = []
        tick_labels = []
        for cat in categories:
            mask = (labels == cat).values
            r = residuals[mask]
            n = len(r)
            mean = r.mean()
            se = r.std() / np.sqrt(n) if n > 1 else 0
            means.append(mean)
            ci_low.append(mean - 1.96 * se)
            ci_high.append(mean + 1.96 * se)
            tick_labels.append(f"{cat}\n(n={n})")

        x = np.arange(len(categories))
        colors = ['#e74c3c' if m > 0 else '#3498db' for m in means]
        ax.barh(x, means, color=colors, alpha=0.7, height=0.5)
        ax.errorbar(means, x, xerr=[np.array(means) - np.array(ci_low),
                                     np.array(ci_high) - np.array(means)],
                    fmt='none', color='black', capsize=4)
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.set_yticks(x)
        ax.set_yticklabels(tick_labels, fontsize=9)
        ax.set_title(group_name, fontsize=13, fontweight='bold')
        ax.set_xlabel('Mean Bias (Actual - Predicted WAR)')
        ax.grid(axis='x', alpha=0.3)

    for idx in range(len(plot_groups), len(axes2)):
        axes2[idx].set_visible(False)

    fig2.suptitle('WAR Prediction Bias by Coach Subgroup (Cross-Validated)',
                  fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    filepath2 = output_dir / 'war_subgroup_bias_barplots.png'
    fig2.savefig(filepath2, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Bias plots saved to: {filepath2}")
    plt.close(fig2)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyze WAR prediction errors by coach subgroup'
    )
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of CV folds (default: 5)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--output-dir', type=str,
                        default='figures/war',
                        help='Output directory for plots')
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    war_df, bg_df = load_data()
    war_df = merge_backgrounds(war_df, bg_df)
    print(f"  {len(war_df)} coaching hires loaded")

    # Prepare features and target
    feature_cols = [f'Feature {i}' for i in range(1, 141)]
    X = war_df[feature_cols].values
    y = war_df[WAR_CONFIG['target_column']].values

    # Cross-validated predictions
    print(f"\nRunning {args.n_folds}-fold coach-level cross-validation...")
    y_pred = get_cv_predictions(war_df, X, y, n_folds=args.n_folds)

    # Overall metrics
    overall = regression_metrics(y, y_pred)

    # Define subgroups
    groups = define_subgroups(war_df)

    # Compute subgroup metrics
    subgroup_results = compute_subgroup_metrics(y, y_pred, groups)

    # Statistical tests
    test_results = test_subgroup_differences(y, y_pred, groups)

    # Print results
    print_results(overall, subgroup_results, test_results)

    # Plots
    if not args.no_plots:
        print("\nGenerating plots...")
        create_plots(y, y_pred, groups, Path(args.output_dir))

    print("\nAnalysis complete.")


if __name__ == '__main__':
    main()
