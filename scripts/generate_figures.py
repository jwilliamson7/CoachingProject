#!/usr/bin/env python
"""
Generate figures for the LaTeX report.

Usage:
    python scripts/generate_figures.py

Generates:
    - correlation_matrix.png: Feature correlation heatmap
    - tenure_predictions.png: Prediction vs ground truth visualization
    - tenure_feature_importance.png: Top feature importances
    - tenure_distribution.png: Class distribution histogram
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from model import (
    CoachTenureModel,
    stratified_coach_level_split,
)
from model.config import MODEL_PATHS, FEATURE_CONFIG, MODEL_CONFIG, ORDINAL_CONFIG


def setup_output_dir():
    """Create output directory for figures."""
    output_dir = os.path.join(project_root, 'latex', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def load_data():
    """Load the training data."""
    data_path = os.path.join(project_root, MODEL_PATHS['data_file'])
    df = pd.read_csv(data_path, index_col=0)
    df_known = df[df[FEATURE_CONFIG['target_column']] != -1]

    X = df_known.iloc[:, FEATURE_CONFIG['feature_columns_start']:FEATURE_CONFIG['feature_columns_end']]
    y = df_known[FEATURE_CONFIG['target_column']]

    return df_known, X, y


def generate_correlation_matrix(X, output_dir):
    """Generate feature correlation matrix heatmap."""
    print("Generating correlation matrix...")

    # Compute correlation matrix
    corr = X.corr()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create custom colormap (blue-white-red)
    cmap = LinearSegmentedColormap.from_list('custom', ['#2166ac', '#f7f7f7', '#b2182b'])

    # Plot heatmap
    im = ax.imshow(corr.values, cmap=cmap, aspect='auto', vmin=-1, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation Coefficient', fontsize=11)

    # Add feature group labels
    ax.set_xlabel('Feature Index', fontsize=11)
    ax.set_ylabel('Feature Index', fontsize=11)

    # Add tick marks at group boundaries
    group_boundaries = [0, 8, 41, 74, 107, 140, 150]
    group_labels = ['Core\n(1-8)', 'OC\n(9-41)', 'DC\n(42-74)',
                    'HC\n(75-107)', 'HC Opp\n(108-140)', 'Team\n(141-150)']

    # Set ticks at group centers
    group_centers = [(group_boundaries[i] + group_boundaries[i+1]) / 2
                     for i in range(len(group_boundaries)-1)]

    ax.set_xticks(group_centers)
    ax.set_xticklabels(group_labels, fontsize=9)
    ax.set_yticks(group_centers)
    ax.set_yticklabels(group_labels, fontsize=9)

    # Add grid lines at group boundaries
    for boundary in group_boundaries[1:-1]:
        ax.axhline(y=boundary-0.5, color='black', linewidth=0.5, alpha=0.5)
        ax.axvline(x=boundary-0.5, color='black', linewidth=0.5, alpha=0.5)

    ax.set_title('Feature Correlation Matrix\n(150 Features Grouped by Category)', fontsize=12)

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'correlation_matrix.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def generate_predictions_plot(df, X, y, model, output_dir):
    """Generate prediction vs ground truth visualization."""
    print("Generating predictions plot...")

    # Get test split
    X_train, X_test, y_train, y_test, _ = stratified_coach_level_split(
        df, X, y,
        test_size=MODEL_CONFIG['test_size'],
        random_state=MODEL_CONFIG['random_state']
    )

    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Sort by true class then by predicted class
    sort_idx = np.lexsort((y_pred, y_test.values))
    y_test_sorted = y_test.values[sort_idx]
    y_pred_sorted = y_pred[sort_idx]

    # Create figure with two rows
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 4), sharex=True,
                                    gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.1})

    # Define colors for classes
    colors = ['#e41a1c', '#377eb8', '#4daf4a']  # Red, Blue, Green
    class_labels = ['Class 0 (1-2 yrs)', 'Class 1 (3-4 yrs)', 'Class 2 (5+ yrs)']

    n_samples = len(y_test_sorted)

    # Plot ground truth (top row)
    for i, true_val in enumerate(y_test_sorted):
        ax1.bar(i, 1, color=colors[int(true_val)], width=1.0, edgecolor='none')

    # Plot predictions (bottom row)
    for i, (true_val, pred_val) in enumerate(zip(y_test_sorted, y_pred_sorted)):
        # Use hatching for misclassifications
        if true_val == pred_val:
            ax2.bar(i, 1, color=colors[int(pred_val)], width=1.0, edgecolor='none')
        else:
            ax2.bar(i, 1, color=colors[int(pred_val)], width=1.0, edgecolor='black',
                   linewidth=1.5, hatch='///')

    # Add row labels
    ax1.set_ylabel('Ground\nTruth', fontsize=10, rotation=0, ha='right', va='center')
    ax2.set_ylabel('Predicted', fontsize=10, rotation=0, ha='right', va='center')

    # Remove y-axis ticks
    ax1.set_yticks([])
    ax2.set_yticks([])

    # Set x limits
    ax1.set_xlim(-0.5, n_samples - 0.5)
    ax2.set_xlim(-0.5, n_samples - 0.5)

    # Add class boundary lines
    class_counts = [sum(y_test_sorted == c) for c in [0, 1, 2]]
    boundaries = [class_counts[0], class_counts[0] + class_counts[1]]

    for boundary in boundaries:
        ax1.axvline(x=boundary - 0.5, color='black', linestyle='-', linewidth=1.5)
        ax2.axvline(x=boundary - 0.5, color='black', linestyle='-', linewidth=1.5)

    # Add class labels at top
    class_centers = [class_counts[0]/2,
                     class_counts[0] + class_counts[1]/2,
                     class_counts[0] + class_counts[1] + class_counts[2]/2]
    for center, label, count in zip(class_centers, class_labels, class_counts):
        ax1.text(center, 1.15, f'{label}\n(n={count})', ha='center', va='bottom', fontsize=9)

    # Calculate metrics
    accuracy = (y_test_sorted == y_pred_sorted).mean()
    n_correct = sum(y_test_sorted == y_pred_sorted)
    n_wrong = n_samples - n_correct

    # Add legend
    patches = [mpatches.Patch(color=colors[i], label=f'Class {i}') for i in range(3)]
    patches.append(mpatches.Patch(facecolor='gray', edgecolor='black', hatch='///',
                                  label=f'Misclassified (n={n_wrong})'))

    ax2.set_xlabel('Test Set Instances (sorted by true class, then predicted class)', fontsize=10)

    # Place legend well below x-axis label
    fig.legend(handles=patches, loc='lower center', fontsize=8, ncol=4,
               bbox_to_anchor=(0.5, -0.08))

    # Title
    fig.suptitle(f'Ordinal Model: Ground Truth vs Predictions (Test Set, n={n_samples}, Accuracy={accuracy:.1%})',
                 fontsize=12, y=1.02)

    # Adjust subplot to make room for legend at bottom
    plt.subplots_adjust(bottom=0.15)

    output_path = os.path.join(output_dir, 'tenure_predictions.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def generate_feature_importance_plot(model, output_dir):
    """Generate feature importance bar chart."""
    print("Generating feature importance plot...")

    # Get feature importances
    importances = model.get_feature_importances()

    # Get top 20
    top_n = 20
    indices = np.argsort(importances)[::-1][:top_n]
    top_importances = importances[indices]

    # Feature descriptions (abbreviated)
    feature_descriptions = {
        103: "HC: 3rd down conv. %",
        6: "Years NFL position coach",
        12: "OC: team turnovers",
        145: "Hiring team: yds allowed",
        122: "HC opp: rushing TDs",
        35: "OC: points per drive",
        94: "HC: penalty 1st downs",
        83: "HC: passing TDs",
        70: "DC opp: 3rd down conv. %",
        136: "HC opp: 3rd down conv. %",
        135: "HC opp: 3rd down attempts",
        11: "OC: yards per play",
        73: "DC opp: red zone attempts",
        22: "OC: rushing yards",
        120: "HC opp: rushing attempts",
        48: "DC opp: passing attempts",
        138: "HC opp: 4th down conv. %",
        129: "HC opp: scoring %",
        101: "HC: points per drive",
        55: "DC opp: rushing yards",
    }

    # Create labels
    labels = [feature_descriptions.get(idx+1, f"Feature {idx+1}") for idx in indices]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by feature category
    def get_category_color(idx):
        feat_num = idx + 1
        if feat_num <= 8:
            return '#1f77b4'  # Core - blue
        elif feat_num <= 41:
            return '#ff7f0e'  # OC - orange
        elif feat_num <= 74:
            return '#2ca02c'  # DC - green
        elif feat_num <= 140:
            return '#d62728'  # HC - red (both team and opponent stats)
        else:
            return '#8c564b'  # Hiring team - brown

    colors = [get_category_color(idx) for idx in indices]

    y_pos = np.arange(top_n)
    ax.barh(y_pos, top_importances, color=colors, alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance', fontsize=11)
    ax.set_title('Top 20 Feature Importances (Ordinal Model)', fontsize=12)

    # Add legend for categories
    category_patches = [
        mpatches.Patch(color='#1f77b4', label='Core Experience'),
        mpatches.Patch(color='#ff7f0e', label='OC Stats'),
        mpatches.Patch(color='#2ca02c', label='DC Stats'),
        mpatches.Patch(color='#d62728', label='HC Stats'),
        mpatches.Patch(color='#8c564b', label='Hiring Team'),
    ]
    ax.legend(handles=category_patches, loc='lower right', fontsize=8)

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'tenure_feature_importance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def generate_distribution_plot(y, output_dir):
    """Generate tenure class distribution histogram."""
    print("Generating distribution plot...")

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # Count classes
    class_counts = y.value_counts().sort_index()

    # Define colors and labels
    colors = ['#e41a1c', '#377eb8', '#4daf4a']
    labels = ['Class 0\n(1-2 years)', 'Class 1\n(3-4 years)', 'Class 2\n(5+ years)']

    # Create bars
    x_pos = np.arange(3)
    bars = ax.bar(x_pos, class_counts.values, color=colors, alpha=0.8, edgecolor='black')

    # Add count labels on bars
    for bar, count in zip(bars, class_counts.values):
        height = bar.get_height()
        pct = count / len(y) * 100
        ax.annotate(f'{count}\n({pct:.1f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Number of Coaching Hires', fontsize=11)
    ax.set_title(f'Coach Tenure Classification Distribution (n={len(y)})', fontsize=12)

    # Set y-axis limit to accommodate labels
    ax.set_ylim(0, max(class_counts.values) * 1.2)

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'tenure_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def main():
    print("="*60)
    print("Generating Figures for LaTeX Report")
    print("="*60)

    # Setup
    output_dir = setup_output_dir()
    print(f"Output directory: {output_dir}\n")

    # Load data
    print("Loading data...")
    df, X, y = load_data()
    print(f"  Loaded {len(df)} instances with {X.shape[1]} features\n")

    # Load model
    print("Loading model...")
    model_path = os.path.join(project_root, MODEL_PATHS['ordinal_model_output'])
    model = CoachTenureModel.load(model_path)
    print(f"  Loaded ordinal model\n")

    # Generate figures
    generate_correlation_matrix(X, output_dir)
    generate_predictions_plot(df, X, y, model, output_dir)
    generate_feature_importance_plot(model, output_dir)
    generate_distribution_plot(y, output_dir)

    print("\n" + "="*60)
    print("All figures generated successfully!")
    print("="*60)

    # Print LaTeX include statements
    print("\nLaTeX include statements:")
    print("-" * 40)
    print("\\includegraphics[width=0.9\\textwidth]{figures/correlation_matrix.png}")
    print("\\includegraphics[width=0.9\\textwidth]{figures/tenure_predictions.png}")
    print("\\includegraphics[width=0.9\\textwidth]{figures/tenure_feature_importance.png}")
    print("\\includegraphics[width=0.8\\textwidth]{figures/tenure_distribution.png}")


if __name__ == '__main__':
    main()
