#!/usr/bin/env python
"""
Generate an averaged confusion matrix heatmap across 50 seeds for the
NFL coaching tenure ordinal classification model.

For each seed (0-49):
  1. Split data into 80/20 train/test using coach-level stratified splitting
  2. Train ordinal model on train set (top 40 SHAP-ranked features)
  3. Predict on test set
  4. Accumulate confusion matrix counts

After all seeds, normalize each row to proportions and plot as a heatmap.
"""

import os
import sys
import warnings
import pickle

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import confusion_matrix

from model.pipeline import (
    load_modeling_data, top_k_indices, best_k, leakage_free_split, fit_model, ordinal_model,
)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def main():
    n_seeds = 50
    n_classes = 3

    print("Loading data...")
    df, X, y = load_modeling_data()
    feat_idx = top_k_indices()
    top_k = len(feat_idx)
    print(f"Loaded {len(df)} instances; using top {top_k} SHAP-ranked features")

    # Accumulate confusion matrix counts across seeds
    cm_accumulated = np.zeros((n_classes, n_classes), dtype=float)
    total_test_per_class = np.zeros(n_classes, dtype=float)

    for seed in range(n_seeds):
        # Shared leakage-free split (coach-level split, train-only imputation, top-K)
        split = leakage_free_split(df, X, y, seed, feature_indices=feat_idx)
        model = fit_model(split, ordinal_model, seed)
        y_pred = model.predict(pd.DataFrame(split.X_test))
        y_test = split.y_test

        # Build confusion matrix for this seed
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        cm_accumulated += cm

        # Track class counts
        for cls in range(n_classes):
            total_test_per_class[cls] += np.sum(y_test == cls)

        print(f"  Seed {seed:>2}/{n_seeds}: test_size={len(y_test)}, "
              f"accuracy={np.mean(y_pred == y_test):.3f}")

    # Average counts per seed
    cm_avg_counts = cm_accumulated / n_seeds

    # Normalize rows to proportions (each row sums to 1)
    row_sums = cm_accumulated.sum(axis=1, keepdims=True)
    cm_proportions = cm_accumulated / row_sums

    print(f"\nAccumulated confusion matrix (total across {n_seeds} seeds):")
    print(cm_accumulated)
    print(f"\nRow-normalized proportions:")
    print(cm_proportions)
    print(f"\nAverage counts per seed:")
    print(cm_avg_counts)

    # ---- Plot ----
    matplotlib.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
    })

    fig, ax = plt.subplots(figsize=(6, 5))

    # Use a subtle blue colormap
    im = ax.imshow(cm_proportions, cmap='Blues', vmin=0, vmax=1, aspect='equal')

    # Class labels
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

    # Annotate each cell with proportion and average count
    for i in range(n_classes):
        for j in range(n_classes):
            prop = cm_proportions[i, j]
            avg_count = cm_avg_counts[i, j]
            # Choose text color based on background intensity
            text_color = 'white' if prop > 0.55 else 'black'
            ax.text(j, i,
                    f"{prop:.3f}\n({avg_count:.1f})",
                    ha='center', va='center',
                    fontsize=12, fontweight='bold',
                    color=text_color)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proportion", fontsize=12)

    # Remove top/right spines for cleaner look
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)

    plt.tight_layout()

    # Save
    output_path = os.path.join(project_root, 'figures', 'confusion_matrix_50seed.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nFigure saved to: {output_path}")
    plt.close(fig)


if __name__ == '__main__':
    main()
