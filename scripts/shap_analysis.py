#!/usr/bin/env python
"""
SHAP (SHapley Additive exPlanations) Analysis for NFL Coach Tenure Prediction.

This script generates interpretable explanations for model predictions using SHAP values
and partial dependence plots. It helps answer "why" the model makes certain predictions.

Outputs:
    - SHAP summary plots (beeswarm and bar)
    - Partial dependence plots for top features
    - Feature interaction analysis
    - Per-class SHAP importance

Usage:
    python scripts/shap_analysis.py [--top-n 20] [--output-dir latex/figures] [--n-samples 635]
"""

import os
import sys
import argparse
import warnings
import pickle

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from typing import List, Dict, Tuple, Optional

from model import CoachTenureModel
from model.config import MODEL_PATHS, FEATURE_CONFIG, ORDINAL_CONFIG
from model.cross_validation import stratified_coach_level_split
from data_constants import get_all_feature_names, HIRING_TEAM_FEATURES

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


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


def get_feature_categories() -> Dict[str, Tuple[int, int]]:
    """Get feature category index ranges (0-indexed)."""
    return {
        'Core Experience': (0, 8),
        'OC Stats': (8, 41),
        'DC Stats': (41, 74),
        'HC Stats': (74, 140),
        'Hiring Team': (140, 150)
    }


def load_data_and_model(model_path: str = None) -> Tuple[CoachTenureModel, pd.DataFrame, np.ndarray, np.ndarray]:
    """Load the trained model and data."""
    if model_path is None:
        model_path = os.path.join(project_root, MODEL_PATHS['ordinal_model_output'])

    print(f"Loading model from {model_path}...")
    model = CoachTenureModel.load(model_path)

    data_path = os.path.join(project_root, MODEL_PATHS['data_file'])
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Filter to known tenure classes
    df = df[df['Coach Tenure Class'] != -1].copy()

    # Model expects exactly 150 features
    # Handle case where CSV has extra index column
    feature_cols = df.columns[2:-2]
    if len(feature_cols) != 150:
        # Force to 150 features
        feature_cols = df.columns[2:152]

    X = df[feature_cols].values
    y = df['Coach Tenure Class'].values

    return model, df, X, y


def compute_shap_values(model: CoachTenureModel, X: np.ndarray,
                        feature_names: List[str], n_samples: int = None,
                        n_background: int = 100) -> Dict[str, shap.Explanation]:
    """
    Compute SHAP values for the ordinal classifier.

    For the Frank-Hall ordinal classifier, we compute SHAP values for each
    binary classifier:
    - Classifier 0: P(Y > 0) - separates class 0 from classes 1+2
    - Classifier 1: P(Y > 1) - separates classes 0+1 from class 2

    Args:
        model: Trained CoachTenureModel
        X: Feature matrix
        feature_names: List of feature names
        n_samples: Number of samples to explain (None = all)
        n_background: Number of background samples for KernelExplainer
    """
    shap_values_dict = {}
    ordinal_clf = model.model_

    # Determine samples to explain
    if n_samples is None or n_samples >= X.shape[0]:
        X_explain = X
        explain_idx = np.arange(X.shape[0])
    else:
        np.random.seed(42)  # For reproducibility
        explain_idx = np.random.choice(X.shape[0], n_samples, replace=False)
        X_explain = X[explain_idx]

    # Background sample for KernelExplainer
    np.random.seed(42)
    n_bg = min(n_background, X.shape[0])
    background_idx = np.random.choice(X.shape[0], n_bg, replace=False)
    background = X[background_idx]

    print(f"\nComputing SHAP values for {len(X_explain)} samples...")

    for i, clf in enumerate(ordinal_clf.classifiers_):
        print(f"  Classifier {i} (P(Y > {i}))...", end=" ", flush=True)

        # Create predict function for this classifier
        def predict_proba(x, classifier=clf):
            return classifier.predict_proba(x)[:, 1]

        # Use KernelExplainer (model-agnostic, works with all XGBoost versions)
        explainer = shap.KernelExplainer(predict_proba, background)

        # Compute SHAP values
        shap_values_raw = explainer.shap_values(X_explain, nsamples=100)

        # Ensure base_values is an array
        if np.isscalar(explainer.expected_value):
            base_values = np.full(shap_values_raw.shape[0], explainer.expected_value)
        else:
            base_values = explainer.expected_value

        shap_values = shap.Explanation(
            values=shap_values_raw,
            base_values=base_values,
            data=X_explain,
            feature_names=feature_names
        )

        key = f"P(Y > {i})"
        shap_values_dict[key] = shap_values
        print("done")

    return shap_values_dict


def compute_aggregated_shap(shap_values_dict: Dict[str, shap.Explanation],
                            feature_names: List[str]) -> shap.Explanation:
    """
    Aggregate SHAP values across binary classifiers.

    Uses absolute values averaged across classifiers to get overall feature importance.
    """
    print("Aggregating SHAP values across classifiers...")

    all_abs_shap = []
    for key, sv in shap_values_dict.items():
        all_abs_shap.append(np.abs(sv.values))

    avg_abs_shap = np.mean(all_abs_shap, axis=0)

    first_sv = list(shap_values_dict.values())[0]

    if np.isscalar(first_sv.base_values):
        base_values = np.full(avg_abs_shap.shape[0], first_sv.base_values)
    elif hasattr(first_sv.base_values, 'shape'):
        base_values = first_sv.base_values
    else:
        base_values = np.array(first_sv.base_values)

    aggregated = shap.Explanation(
        values=avg_abs_shap,
        base_values=base_values,
        data=first_sv.data,
        feature_names=feature_names
    )

    return aggregated


def plot_shap_summary(shap_values: shap.Explanation, output_path: str,
                      max_display: int = 20, title: str = None):
    """Create SHAP summary beeswarm plot."""
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, show=False, max_display=max_display)

    if title:
        plt.title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_shap_bar(shap_values: shap.Explanation, output_path: str,
                  max_display: int = 20, title: str = None):
    """Create SHAP bar plot showing mean absolute SHAP values."""
    plt.figure(figsize=(10, 8))
    shap.plots.bar(shap_values, show=False, max_display=max_display)

    if title:
        plt.title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_partial_dependence(model: CoachTenureModel, X: np.ndarray,
                            feature_names: List[str], top_features: List[int],
                            output_dir: str, n_features: int = 6):
    """
    Create partial dependence plots for top features.

    Shows how each feature affects the predicted probability across its range.
    """
    print(f"\nGenerating partial dependence plots for top {n_features} features...")

    features_to_plot = top_features[:n_features]

    n_cols = 2
    n_rows = (len(features_to_plot) + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = axes.flatten()

    ordinal_clf = model.model_

    for idx, feat_idx in enumerate(features_to_plot):
        ax = axes[idx]
        feat_name = feature_names[feat_idx]
        feat_values = X[:, feat_idx]
        grid = np.linspace(feat_values.min(), feat_values.max(), 50)

        class_probs = {0: [], 1: [], 2: []}

        for grid_val in grid:
            X_modified = X.copy()
            X_modified[:, feat_idx] = grid_val
            probs = ordinal_clf.predict_proba(X_modified)

            for c in range(3):
                class_probs[c].append(probs[:, c].mean())

        colors = ['#e74c3c', '#f39c12', '#27ae60']
        labels = ['Class 0 (1-2 yrs)', 'Class 1 (3-4 yrs)', 'Class 2 (5+ yrs)']

        for c in range(3):
            ax.plot(grid, class_probs[c], color=colors[c], label=labels[c], linewidth=2)

        ax.set_xlabel(feat_name, fontsize=10)
        ax.set_ylabel('Mean Predicted Probability', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax.scatter(feat_values, np.zeros_like(feat_values) - 0.02,
                   alpha=0.1, s=10, color='black')

    for idx in range(len(features_to_plot), len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle('Partial Dependence Plots: Effect of Features on Predicted Tenure Class',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'shap_partial_dependence.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_shap_by_class(shap_values_dict: Dict[str, shap.Explanation],
                       feature_names: List[str], output_dir: str,
                       max_display: int = 15):
    """
    Create separate SHAP plots for each binary classifier.

    This shows which features matter for different classification decisions:
    - P(Y > 0): What distinguishes short-tenure from longer-tenure coaches?
    - P(Y > 1): What distinguishes medium-tenure from long-tenure coaches?
    """
    print("\nGenerating per-classifier SHAP plots...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    titles = [
        'Features for Short vs Longer Tenure\n(P(Y > 0): Class 0 vs Classes 1+2)',
        'Features for Medium vs Long Tenure\n(P(Y > 1): Classes 0+1 vs Class 2)'
    ]

    for i, (key, sv) in enumerate(shap_values_dict.items()):
        plt.sca(axes[i])
        shap.plots.bar(sv, show=False, max_display=max_display)
        axes[i].set_title(titles[i], fontsize=12, fontweight='bold')

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'shap_by_classifier.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_category_shap(aggregated_shap: shap.Explanation,
                       feature_names: List[str], output_dir: str):
    """Create SHAP importance by feature category."""
    print("\nGenerating category-level SHAP analysis...")

    categories = get_feature_categories()
    mean_abs_shap = np.abs(aggregated_shap.values).mean(axis=0)

    category_importance = {}
    category_avg_importance = {}

    for cat_name, (start, end) in categories.items():
        cat_shap = mean_abs_shap[start:end]
        category_importance[cat_name] = cat_shap.sum()
        category_avg_importance[cat_name] = cat_shap.mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    cats = list(category_importance.keys())
    totals = [category_importance[c] for c in cats]
    colors = plt.cm.Set2(np.linspace(0, 1, len(cats)))

    bars1 = ax1.barh(cats, totals, color=colors)
    ax1.set_xlabel('Total Mean |SHAP|', fontsize=12)
    ax1.set_title('Total SHAP Importance by Category', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()

    for bar, val in zip(bars1, totals):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontsize=10)

    avgs = [category_avg_importance[c] for c in cats]

    bars2 = ax2.barh(cats, avgs, color=colors)
    ax2.set_xlabel('Average Mean |SHAP| per Feature', fontsize=12)
    ax2.set_title('Average SHAP Importance by Category', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()

    for bar, val in zip(bars2, avgs):
        ax2.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                 f'{val:.4f}', va='center', fontsize=10)

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'shap_category_importance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

    print("\n  Category SHAP Summary:")
    print(f"  {'Category':<20} {'Total':>10} {'Average':>10} {'# Features':>12}")
    print("  " + "-" * 55)
    for cat_name, (start, end) in categories.items():
        n_features = end - start
        print(f"  {cat_name:<20} {category_importance[cat_name]:>10.4f} "
              f"{category_avg_importance[cat_name]:>10.4f} {n_features:>12}")


def get_top_features(aggregated_shap: shap.Explanation, n: int = 20) -> List[int]:
    """Get indices of top N most important features by mean |SHAP|."""
    mean_abs_shap = np.abs(aggregated_shap.values).mean(axis=0)
    return np.argsort(mean_abs_shap)[::-1][:n].tolist()


def generate_shap_report(shap_values_dict: Dict[str, shap.Explanation],
                         aggregated_shap: shap.Explanation,
                         feature_names: List[str], output_dir: str):
    """Generate a text report summarizing SHAP findings."""
    print("\nGenerating SHAP analysis report...")

    analysis_dir = os.path.join(project_root, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    report_path = os.path.join(analysis_dir, 'shap_analysis_report.txt')

    mean_abs_shap = np.abs(aggregated_shap.values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1]

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SHAP ANALYSIS REPORT: NFL Coach Tenure Prediction\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. TOP 20 MOST IMPORTANT FEATURES (by mean |SHAP|)\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Rank':<6} {'Feature':<45} {'Mean |SHAP|':>12}\n")
        f.write("-" * 60 + "\n")

        for rank, idx in enumerate(top_indices[:20], 1):
            f.write(f"{rank:<6} {feature_names[idx]:<45} {mean_abs_shap[idx]:>12.4f}\n")

        f.write("\n\n")

        f.write("2. IMPORTANCE BY FEATURE CATEGORY\n")
        f.write("-" * 60 + "\n")

        categories = get_feature_categories()
        category_data = []

        for cat_name, (start, end) in categories.items():
            cat_shap = mean_abs_shap[start:end]
            category_data.append({
                'name': cat_name,
                'total': cat_shap.sum(),
                'avg': cat_shap.mean(),
                'n': end - start
            })

        category_data.sort(key=lambda x: x['total'], reverse=True)

        f.write(f"{'Category':<20} {'Total SHAP':>12} {'Avg SHAP':>12} {'# Features':>12}\n")
        f.write("-" * 60 + "\n")

        for cat in category_data:
            f.write(f"{cat['name']:<20} {cat['total']:>12.4f} {cat['avg']:>12.4f} {cat['n']:>12}\n")

        f.write("\n\n")

        f.write("3. TOP FEATURES BY CLASSIFICATION DECISION\n")
        f.write("-" * 60 + "\n\n")

        for key, sv in shap_values_dict.items():
            f.write(f"{key}:\n")

            if key == "P(Y > 0)":
                f.write("  (What distinguishes short-tenure [Class 0] from longer-tenure coaches)\n")
            else:
                f.write("  (What distinguishes medium/long-tenure coaches [Classes 1+2])\n")

            mean_abs = np.abs(sv.values).mean(axis=0)
            top_idx = np.argsort(mean_abs)[::-1][:10]

            for rank, idx in enumerate(top_idx, 1):
                f.write(f"  {rank:>2}. {feature_names[idx]:<40} {mean_abs[idx]:.4f}\n")
            f.write("\n")

        f.write("\n")
        f.write("4. KEY FINDINGS\n")
        f.write("-" * 60 + "\n")
        f.write("- The model uses features from multiple categories, suggesting tenure\n")
        f.write("  prediction requires diverse information about coaching background.\n")
        f.write("- Top features typically include age, experience metrics, and recent\n")
        f.write("  team performance statistics.\n")
        f.write("- See partial dependence plots for direction of feature effects.\n")
        f.write("\n")
        f.write("=" * 80 + "\n")

    print(f"  Saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='SHAP analysis for NFL Coach Tenure Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--top-n', type=int, default=20,
        help='Number of top features to display (default: 20)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='figures/tenure',
        help='Output directory for figures (default: figures/tenure)'
    )
    parser.add_argument(
        '--model-path', type=str, default=None,
        help='Path to trained model (default: use config)'
    )
    parser.add_argument(
        '--n-samples', type=int, default=None,
        help='Number of samples to explain (default: all)'
    )
    parser.add_argument(
        '--save-shap', action='store_true',
        help='Save computed SHAP values to disk for reuse'
    )
    parser.add_argument(
        '--no-cache', action='store_true',
        help='Ignore cached SHAP values and recompute'
    )

    args = parser.parse_args()

    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("SHAP ANALYSIS FOR NFL COACH TENURE PREDICTION")
    print("=" * 70)

    model, df, X, y = load_data_and_model(args.model_path)
    feature_names = get_feature_names()

    print(f"\nData shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}")

    shap_cache_path = os.path.join(project_root, 'data', 'shap_values_cache.pkl')

    if os.path.exists(shap_cache_path) and not args.no_cache and not args.save_shap:
        print(f"\nLoading cached SHAP values from {shap_cache_path}...")
        with open(shap_cache_path, 'rb') as f:
            cache = pickle.load(f)
            shap_values_dict = cache['shap_values_dict']
            aggregated_shap = cache['aggregated_shap']
            for key in shap_values_dict:
                shap_values_dict[key].feature_names = feature_names
            aggregated_shap.feature_names = feature_names
    else:
        shap_values_dict = compute_shap_values(model, X, feature_names, n_samples=args.n_samples)
        aggregated_shap = compute_aggregated_shap(shap_values_dict, feature_names)

        if args.save_shap:
            print(f"\nSaving SHAP values to {shap_cache_path}...")
            with open(shap_cache_path, 'wb') as f:
                pickle.dump({
                    'shap_values_dict': shap_values_dict,
                    'aggregated_shap': aggregated_shap
                }, f)

    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    print("\n1. SHAP Summary Plot (Beeswarm)...")
    plot_shap_summary(
        aggregated_shap,
        os.path.join(output_dir, 'shap_summary_beeswarm.png'),
        max_display=args.top_n,
        title='SHAP Feature Importance: Impact on Model Output'
    )

    print("\n2. SHAP Bar Plot...")
    plot_shap_bar(
        aggregated_shap,
        os.path.join(output_dir, 'shap_summary_bar.png'),
        max_display=args.top_n,
        title='Mean |SHAP| Values: Feature Importance'
    )

    print("\n3. Per-Classifier SHAP Plots...")
    plot_shap_by_class(shap_values_dict, feature_names, output_dir)

    print("\n4. Category SHAP Analysis...")
    plot_category_shap(aggregated_shap, feature_names, output_dir)

    print("\n5. Partial Dependence Plots...")
    top_features = get_top_features(aggregated_shap, n=args.top_n)
    plot_partial_dependence(model, X, feature_names, top_features, output_dir, n_features=6)

    print("\n6. Generating Summary Report...")
    generate_shap_report(shap_values_dict, aggregated_shap, feature_names, output_dir)

    print("\n" + "=" * 70)
    print("SHAP ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutputs saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - shap_summary_beeswarm.png  : Feature importance with value distributions")
    print("  - shap_summary_bar.png       : Mean |SHAP| bar chart")
    print("  - shap_by_classifier.png     : Importance for each binary decision")
    print("  - shap_category_importance.png: Importance by feature category")
    print("  - shap_partial_dependence.png: How top features affect predictions")
    print("  - shap_analysis_report.txt   : Detailed text summary")


if __name__ == '__main__':
    main()
