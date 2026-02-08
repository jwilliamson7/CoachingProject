#!/usr/bin/env python
"""
Parsimonious Model Analysis for NFL Coach Tenure Prediction.

Addresses JQAS reviewer feedback: "How would the model change if only those features
with high SHAP scores were used, rather than the whole category?"

This script:
1. Ranks features by mean |SHAP| value
2. Trains reduced ordinal models with top-K features (K = 5, 10, 20, ..., 150)
3. Evaluates across 50 independent train/test splits (matching main analysis)
4. Reports mean +/- std across seeds
5. Generates a performance-vs-parsimony curve figure
6. Outputs a LaTeX-ready results table

Usage:
    python scripts/parsimonious_model.py
    python scripts/parsimonious_model.py --n-seeds 50
    python scripts/parsimonious_model.py --feature-counts 10 20 50 150
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
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from model import (
    CoachTenureModel,
    stratified_coach_level_split,
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
    """Load precomputed SHAP values and rank features by mean |SHAP|."""
    cache_path = os.path.join(project_root, 'data', 'shap_values_cache.pkl')

    if os.path.exists(cache_path):
        print(f"Loading cached SHAP values from {cache_path}...")
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        aggregated = cache['aggregated_shap']
        mean_abs = np.abs(aggregated.values).mean(axis=0)
        ranking = np.argsort(mean_abs)[::-1]
        return ranking, mean_abs
    else:
        print("No cached SHAP values found. Computing from model...")
        return compute_shap_ranking()


def compute_shap_ranking():
    """Compute SHAP ranking from scratch."""
    from scripts.shap_analysis import (
        load_data_and_model, compute_shap_values,
        compute_aggregated_shap, get_feature_names,
    )

    model, df, X, y = load_data_and_model()
    feature_names = get_feature_names()
    shap_dict = compute_shap_values(model, X, feature_names, n_background=100)
    aggregated = compute_aggregated_shap(shap_dict, feature_names)

    mean_abs = np.abs(aggregated.values).mean(axis=0)
    ranking = np.argsort(mean_abs)[::-1]

    # Cache for future use
    cache_path = os.path.join(project_root, 'data', 'shap_values_cache.pkl')
    print(f"Saving SHAP cache to {cache_path}...")
    with open(cache_path, 'wb') as f:
        pickle.dump({
            'shap_values_dict': shap_dict,
            'aggregated_shap': aggregated,
        }, f)

    return ranking, mean_abs


def compute_metrics_from_arrays(y_true, y_pred, y_proba=None):
    """Compute ordinal metrics from arrays."""
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
    return result


def train_and_evaluate_single_seed(
    df, X, y, feature_indices, random_state=42
):
    """Train a model on a subset of features for a single seed and return metrics."""
    X_train, X_test, y_train, y_test, _ = stratified_coach_level_split(
        df, X, y,
        test_size=MODEL_CONFIG['test_size'],
        random_state=random_state,
    )

    X_train_sub = np.asarray(X_train)[:, feature_indices]
    X_test_sub = np.asarray(X_test)[:, feature_indices]
    y_train_arr = np.asarray(y_train)
    y_test_arr = np.asarray(y_test)

    model = CoachTenureModel(
        use_ordinal=True,
        n_classes=3,
        random_state=random_state,
    )
    model.fit(pd.DataFrame(X_train_sub), pd.Series(y_train_arr), verbose=0)

    y_pred = model.predict(pd.DataFrame(X_test_sub))
    y_proba = model.predict_proba(pd.DataFrame(X_test_sub))

    return compute_metrics_from_arrays(y_test_arr, y_pred, y_proba)


def run_parsimony_analysis(
    df, X, y, feature_ranking, feature_counts, n_seeds=50
):
    """Train models at each feature count across multiple seeds and collect results."""
    print(f"\n{'='*70}")
    print(f"PARSIMONY ANALYSIS ({n_seeds} seeds)")
    print(f"{'='*70}")

    all_results = {}

    for n_feat in feature_counts:
        print(f"\n  Top {n_feat} features across {n_seeds} seeds...")

        indices = feature_ranking[:n_feat]
        seed_metrics = []

        for seed in range(n_seeds):
            metrics = train_and_evaluate_single_seed(
                df, X, y,
                feature_indices=indices,
                random_state=seed,
            )
            seed_metrics.append(metrics)

        # Aggregate across seeds: mean +/- std
        metric_keys = seed_metrics[0].keys()
        aggregated = {}
        for k in metric_keys:
            vals = np.array([m[k] for m in seed_metrics if k in m])
            vals = vals[~np.isnan(vals)]
            aggregated[k] = {
                'mean': np.mean(vals),
                'std': np.std(vals),
                'ci_low': np.percentile(vals, 2.5),
                'ci_high': np.percentile(vals, 97.5),
                'all_seeds': vals.tolist(),
            }

        all_results[n_feat] = aggregated

        qwk = aggregated['qwk']
        mae = aggregated['mae']
        print(f"    QWK: {qwk['mean']:.4f} +/- {qwk['std']:.4f} [{qwk['ci_low']:.4f}, {qwk['ci_high']:.4f}]")
        print(f"    MAE: {mae['mean']:.4f} +/- {mae['std']:.4f} [{mae['ci_low']:.4f}, {mae['ci_high']:.4f}]")

    return all_results


def generate_parsimony_figure(all_results, output_dir):
    """Generate performance-vs-parsimony curve with mean +/- std bands."""
    print("\nGenerating parsimony curve figure...")

    os.makedirs(output_dir, exist_ok=True)

    feature_counts = sorted(all_results.keys())

    metrics_to_plot = [
        ('qwk', 'Quadratic Weighted Kappa', True),
        ('mae', 'Mean Absolute Error', False),
        ('macro_f1', 'Macro F1 Score', True),
        ('auroc', 'AUROC', True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax_idx, (metric, label, higher_better) in enumerate(metrics_to_plot):
        ax = axes[ax_idx]

        means = [all_results[n][metric]['mean'] for n in feature_counts]
        stds = [all_results[n][metric]['std'] for n in feature_counts]
        lo = [m - s for m, s in zip(means, stds)]
        hi = [m + s for m, s in zip(means, stds)]

        ax.plot(feature_counts, means, 'o-', color='#2c3e50', linewidth=2, markersize=6)
        ax.fill_between(
            feature_counts, lo, hi,
            alpha=0.2, color='#3498db', label=r'Mean $\pm$ 1 SD',
        )

        # Mark full model (150 features) with horizontal line
        if 150 in feature_counts:
            full_val = all_results[150][metric]['mean']
            ax.axhline(y=full_val, color='#e74c3c', linestyle='--', alpha=0.5, label=f'Full model ({full_val:.3f})')

        ax.set_xlabel('Number of Features', fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.suptitle(
        'Model Performance vs. Feature Count (SHAP-ranked features, 50 seeds)',
        fontsize=14, fontweight='bold', y=1.02,
    )
    plt.tight_layout()

    for subdir in [output_dir, os.path.join(project_root, 'latex', 'figures')]:
        os.makedirs(subdir, exist_ok=True)
        path = os.path.join(subdir, 'parsimony_curve.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")

    plt.close()


def generate_results_table(all_results, feature_ranking, shap_values):
    """Print LaTeX-ready results table with mean +/- std across seeds."""
    print(f"\n{'='*70}")
    print("PARSIMONY RESULTS TABLE")
    print(f"{'='*70}")

    feature_counts = sorted(all_results.keys())

    # Console table
    print(f"\n{'# Feat':>8} {'QWK':>18} {'MAE':>18} {'Adj Acc':>18} {'F1':>18} {'AUROC':>18}")
    print('-' * 100)

    for n in feature_counts:
        r = all_results[n]
        print(f"{n:>8} "
              f"{r['qwk']['mean']:>7.3f}+/-{r['qwk']['std']:.3f} "
              f"{r['mae']['mean']:>7.3f}+/-{r['mae']['std']:.3f} "
              f"{r['adjacent_accuracy']['mean']:>7.3f}+/-{r['adjacent_accuracy']['std']:.3f} "
              f"{r['macro_f1']['mean']:>7.3f}+/-{r['macro_f1']['std']:.3f} "
              f"{r['auroc']['mean']:>7.3f}+/-{r['auroc']['std']:.3f}")

    # Find best feature count by QWK
    best_n = max(feature_counts, key=lambda n: all_results[n]['qwk']['mean'])
    print(f"\n  Best QWK at {best_n} features: {all_results[best_n]['qwk']['mean']:.3f}")

    # LaTeX table
    print(f"\n--- LaTeX Table ---\n")
    print(r"\begin{table}[!ht]")
    print(r"\caption{Model performance with SHAP-ranked feature subsets. "
          r"Values are mean $\pm$ standard deviation across 50 independent train/test splits.}")
    print(r"\label{tab:parsimony}")
    print(r"\small")
    print(r"\begin{tabular}{rcccc}")
    print(r"\toprule")
    print(r"\# Features & QWK & MAE & Adj.\ Acc.\ & Macro F1 \\")
    print(r"\midrule")

    for n in feature_counts:
        r = all_results[n]
        qwk = r['qwk']
        mae = r['mae']
        adj = r['adjacent_accuracy']
        f1 = r['macro_f1']

        bold_start = r"\textbf{" if n == best_n else ""
        bold_end = "}" if n == best_n else ""

        line = (f"{n} & "
                f"{bold_start}{qwk['mean']:.3f} $\\pm$ {qwk['std']:.3f}{bold_end} & "
                f"{mae['mean']:.3f} $\\pm$ {mae['std']:.3f} & "
                f"{adj['mean']*100:.1f}\\% $\\pm$ {adj['std']*100:.1f}\\% & "
                f"{f1['mean']:.3f} $\\pm$ {f1['std']:.3f} \\\\")
        print(line)

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # Feature list for top-K models
    from scripts.shap_analysis import get_feature_names
    feature_names = get_feature_names()

    for k in [10, 20]:
        if k in feature_counts:
            print(f"\n--- Top {k} Features (by SHAP rank) ---")
            for rank, idx in enumerate(feature_ranking[:k], 1):
                print(f"  {rank:>2}. {feature_names[idx]:<45} |SHAP| = {shap_values[idx]:.4f}")


def save_results(all_results, feature_ranking, shap_values):
    """Save results for integration with other scripts."""
    analysis_dir = os.path.join(project_root, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    pickle_path = os.path.join(analysis_dir, 'parsimony_results.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump({
            'results': all_results,
            'feature_ranking': feature_ranking,
            'shap_values': shap_values,
        }, f)
    print(f"\nResults saved to: {pickle_path}")

    # CSV summary
    rows = []
    for n_feat in sorted(all_results.keys()):
        r = all_results[n_feat]
        row = {'n_features': n_feat}
        for metric in ['qwk', 'mae', 'adjacent_accuracy', 'exact_accuracy', 'macro_f1', 'auroc']:
            if metric in r:
                row[f'{metric}_mean'] = r[metric]['mean']
                row[f'{metric}_std'] = r[metric]['std']
                row[f'{metric}_ci_low'] = r[metric]['ci_low']
                row[f'{metric}_ci_high'] = r[metric]['ci_high']
        rows.append(row)

    csv_path = os.path.join(analysis_dir, 'parsimony_results.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"CSV saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Parsimonious model analysis')
    parser.add_argument(
        '--feature-counts', type=int, nargs='+',
        default=[5] + list(range(10, 151, 10)),
        help='Feature counts to test (default: 5 10 20 30 ... 150)',
    )
    parser.add_argument('--n-seeds', type=int, default=50,
                        help='Number of train/test splits (default: 50)')
    parser.add_argument('--output-dir', type=str, default='figures/tenure',
                        help='Output directory for figures')
    args = parser.parse_args()

    print("=" * 70)
    print("PARSIMONIOUS MODEL ANALYSIS")
    print(f"({args.n_seeds} seeds, {len(args.feature_counts)} feature counts)")
    print("=" * 70)

    # Load data
    df, X, y = load_data()
    print(f"Loaded {len(df)} instances with {X.shape[1]} features")

    # Get SHAP-based feature ranking
    feature_ranking, shap_values = get_shap_feature_ranking()

    print(f"\nTop 10 features by SHAP importance:")
    from scripts.shap_analysis import get_feature_names
    feature_names = get_feature_names()
    for i in range(10):
        idx = feature_ranking[i]
        print(f"  {i+1:>2}. {feature_names[idx]:<45} |SHAP| = {shap_values[idx]:.4f}")

    # Run parsimony analysis
    all_results = run_parsimony_analysis(
        df, X, y,
        feature_ranking=feature_ranking,
        feature_counts=sorted(args.feature_counts),
        n_seeds=args.n_seeds,
    )

    # Generate figure
    output_dir = os.path.join(project_root, args.output_dir)
    generate_parsimony_figure(all_results, output_dir)

    # Print results table
    generate_results_table(all_results, feature_ranking, shap_values)

    # Save results
    save_results(all_results, feature_ranking, shap_values)

    print(f"\n{'='*70}")
    print("PARSIMONY ANALYSIS COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
