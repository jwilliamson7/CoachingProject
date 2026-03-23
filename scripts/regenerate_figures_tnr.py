#!/usr/bin/env python
"""
Regenerate paper figures with Times New Roman font for IJCSS submission.

Outputs to ijcss/figures/ without modifying existing figures.
Regenerates all 4 paper figures:
  1. parsimony_curve.png - from parsimony_results.pkl
  2. confusion_matrix_50seed.png - recomputed across 50 seeds
  3. tenure_distribution.png - from data
  4. shap_partial_dependence.png - from trained model
"""

import os
import sys
import pickle
import warnings

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import confusion_matrix

from model import CoachTenureModel, stratified_coach_level_split
from model.config import MODEL_CONFIG, MODEL_PATHS, FEATURE_CONFIG, ORDINAL_CONFIG

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Set Times New Roman globally
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'mathtext.fontset': 'stix',  # Math font compatible with TNR
})

OUTPUT_DIR = os.path.join(project_root, 'ijcss', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    data_path = os.path.join(project_root, MODEL_PATHS['data_file'])
    df = pd.read_csv(data_path, index_col=0)
    df = df[df[FEATURE_CONFIG['target_column']] != -1].copy()
    X = df.iloc[:, FEATURE_CONFIG['feature_columns_start']:FEATURE_CONFIG['feature_columns_end']]
    y = df[FEATURE_CONFIG['target_column']]
    return df, X, y


def get_shap_feature_ranking():
    cache_path = os.path.join(project_root, 'data', 'shap_values_cache.pkl')
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    aggregated = cache['aggregated_shap']
    mean_abs = np.abs(aggregated.values).mean(axis=0)
    ranking = np.argsort(mean_abs)[::-1]
    return ranking


def subset_features(X, top_k):
    ranking = get_shap_feature_ranking()
    selected = sorted(ranking[:top_k])
    return X.iloc[:, selected]


# ---- Figure 1: Parsimony Curve ----
def generate_parsimony_curve():
    print("Generating parsimony curve...")
    pkl_path = os.path.join(project_root, 'analysis', 'parsimony_results.pkl')
    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f)
    all_results = raw['results']

    feature_counts = sorted(all_results.keys())

    metrics_info = [
        ('qwk', 'Quadratic Weighted Kappa'),
        ('mae', 'Mean Absolute Error'),
        ('adjacent_accuracy', 'Adjacent Accuracy'),
        ('macro_f1', 'Macro F1 Score'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax, (metric, label) in zip(axes, metrics_info):
        means = [all_results[n][metric]['mean'] for n in feature_counts]
        stds = [all_results[n][metric]['std'] for n in feature_counts]
        lo = [m - s for m, s in zip(means, stds)]
        hi = [m + s for m, s in zip(means, stds)]

        ax.plot(feature_counts, means, 'o-', color='#2c3e50', linewidth=2, markersize=6)
        ax.fill_between(feature_counts, lo, hi,
                        alpha=0.2, color='#3498db', label=r'Mean $\pm$ 1 SD')

        if 150 in feature_counts:
            full_val = all_results[150][metric]['mean']
            ax.axhline(y=full_val, color='#e74c3c', linestyle='--', alpha=0.5,
                        label=f'Full model ({full_val:.3f})')

        ax.set_xlabel('Number of Features', fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'parsimony_curve.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ---- Figure 2: Confusion Matrix (50 seeds) ----
def generate_confusion_matrix_fig():
    print("Generating confusion matrix (50 seeds)...")
    n_seeds = 50
    n_classes = 3
    top_k = 40

    df, X, y = load_data()
    X = subset_features(X, top_k)

    cm_accumulated = np.zeros((n_classes, n_classes), dtype=float)

    for seed in range(n_seeds):
        X_train, X_test, y_train, y_test, _ = stratified_coach_level_split(
            df, X, y,
            test_size=MODEL_CONFIG['test_size'],
            random_state=seed,
        )
        model = CoachTenureModel(use_ordinal=True, n_classes=n_classes, random_state=seed)
        model.fit(X_train, y_train, verbose=0)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(np.asarray(y_test), y_pred, labels=[0, 1, 2])
        cm_accumulated += cm
        if (seed + 1) % 10 == 0:
            print(f"    Seed {seed + 1}/{n_seeds}")

    cm_avg_counts = cm_accumulated / n_seeds
    row_sums = cm_accumulated.sum(axis=1, keepdims=True)
    cm_proportions = cm_accumulated / row_sums

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_proportions, cmap='Blues', vmin=0, vmax=1, aspect='equal')

    class_labels = [
        "Class 0\n(\u22642 yr)",
        "Class 1\n(3\u20134 yr)",
        "Class 2\n(5+ yr)",
    ]

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_labels, fontsize=11)
    ax.set_yticklabels(class_labels, fontsize=11)
    ax.set_xlabel("Predicted Class", fontsize=13, labelpad=8)
    ax.set_ylabel("True Class", fontsize=13, labelpad=8)
    ax.set_title("Averaged Confusion Matrix (50 Seeds)", fontsize=14, pad=12)

    for i in range(n_classes):
        for j in range(n_classes):
            prop = cm_proportions[i, j]
            avg_count = cm_avg_counts[i, j]
            text_color = 'white' if prop > 0.55 else 'black'
            ax.text(j, i, f"{prop:.3f}\n({avg_count:.1f})",
                    ha='center', va='center', fontsize=12, fontweight='bold',
                    color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proportion", fontsize=12)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'confusion_matrix_50seed.png')
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


# ---- Figure 3: Tenure Distribution ----
def generate_tenure_distribution():
    print("Generating tenure distribution...")
    df, X, y = load_data()

    class_counts = y.value_counts().sort_index()
    total = len(y)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#e41a1c', '#377eb8', '#4daf4a']
    labels = [
        f'Class 0: 1-2 years\n({class_counts[0]} coaches, {class_counts[0]/total*100:.1f}%)',
        f'Class 1: 3-4 years\n({class_counts[1]} coaches, {class_counts[1]/total*100:.1f}%)',
        f'Class 2: 5+ years\n({class_counts[2]} coaches, {class_counts[2]/total*100:.1f}%)',
    ]

    bars = ax.bar([0, 1, 2], [class_counts[0], class_counts[1], class_counts[2]],
                  color=colors, alpha=0.85, edgecolor='black', linewidth=0.8)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Number of Coaching Hires', fontsize=12)
    ax.set_title(f'Coach Tenure Classification Distribution (n={total})', fontsize=13)

    for bar, count in zip(bars, [class_counts[0], class_counts[1], class_counts[2]]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                str(count), ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylim(0, max(class_counts) * 1.12)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, 'tenure_distribution.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ---- Figure 4: SHAP Partial Dependence ----
def generate_partial_dependence():
    print("Generating partial dependence plots...")
    import shap
    from scripts.shap_analysis import get_feature_names

    df, X, y = load_data()

    # Train model on full training set
    X_train, X_test, y_train, y_test, _ = stratified_coach_level_split(
        df, X, y,
        test_size=MODEL_CONFIG['test_size'],
        random_state=MODEL_CONFIG['random_state'],
    )

    model = CoachTenureModel(use_ordinal=True, n_classes=3, random_state=42)
    model.fit(X_train, y_train, verbose=0)

    feature_names = get_feature_names()

    # Get SHAP ranking for top 6 features
    cache_path = os.path.join(project_root, 'data', 'shap_values_cache.pkl')
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    aggregated = cache['aggregated_shap']
    mean_abs = np.abs(aggregated.values).mean(axis=0)
    top_6_idx = np.argsort(mean_abs)[::-1][:6]

    # Get binary classifiers for SHAP
    classifiers = model.model_.classifiers_
    X_train_np = np.asarray(X_train)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    class_names = ['Class 0 (1-2 yr)', 'Class 1 (3-4 yr)', 'Class 2 (5+ yr)']
    class_colors = ['#e41a1c', '#377eb8', '#4daf4a']

    for plot_idx, feat_idx in enumerate(top_6_idx):
        ax = axes[plot_idx]
        feat_name = feature_names[feat_idx]
        feat_values = X_train_np[:, feat_idx]

        # Sort by feature value for smooth curves
        sort_idx = np.argsort(feat_values)
        feat_sorted = feat_values[sort_idx]

        # Get SHAP values from each binary classifier
        shap_per_class = np.zeros((len(X_train_np), 3))

        for clf_idx, clf in enumerate(classifiers):
            explainer = shap.TreeExplainer(clf)
            sv = explainer.shap_values(X_train_np)
            if clf_idx == 0:  # P(Y > 0)
                shap_per_class[:, 0] -= sv[:, feat_idx]  # P(Y=0) = 1 - P(Y>0)
                shap_per_class[:, 1] += sv[:, feat_idx]
            if clf_idx == 1:  # P(Y > 1)
                shap_per_class[:, 1] -= sv[:, feat_idx]
                shap_per_class[:, 2] += sv[:, feat_idx]

        # Smooth with rolling window
        window = max(len(feat_sorted) // 20, 5)
        for cls in range(3):
            vals_sorted = shap_per_class[sort_idx, cls]
            smoothed = pd.Series(vals_sorted).rolling(window, center=True, min_periods=1).mean()
            ax.plot(feat_sorted, smoothed, color=class_colors[cls],
                    label=class_names[cls], linewidth=1.5)

        ax.set_xlabel(feat_name, fontsize=10)
        ax.set_ylabel('SHAP value', fontsize=10)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.grid(True, alpha=0.2)
        if plot_idx == 0:
            ax.legend(fontsize=8, loc='best')

    plt.suptitle('Partial Dependence: Top 6 Features', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, 'shap_partial_dependence.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


if __name__ == '__main__':
    generate_parsimony_curve()
    generate_confusion_matrix_fig()
    generate_tenure_distribution()
    generate_partial_dependence()
    print(f"\nAll figures saved to: {OUTPUT_DIR}")
