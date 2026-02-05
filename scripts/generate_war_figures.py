"""
Generate figures for WAR prediction analysis.

Creates:
1. WAR prediction correlation scatter plot
2. Recent hires WAR predictions bar chart
3. Predicted WAR x Predicted Tenure matrix
4. Feature importance by category
5. WAR distribution histogram

Usage:
    python scripts/generate_war_figures.py
    python scripts/generate_war_figures.py --output-dir figures/war
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.war_regressor import WARRegressor
from model.coach_tenure_model import CoachTenureModel
from model.config import MODEL_PATHS, WAR_CONFIG, ORDINAL_CONFIG, FEATURE_CONFIG, MODEL_CONFIG

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'tertiary': '#2ca02c',
    'class_0': '#d62728',  # Red for short tenure
    'class_1': '#ff7f0e',  # Orange for medium tenure
    'class_2': '#2ca02c',  # Green for long tenure
}


def load_models_and_data():
    """Load trained models and data."""
    # Load WAR model
    war_model = WARRegressor.load(MODEL_PATHS['war_model_output'])
    print(f"Loaded WAR model from {MODEL_PATHS['war_model_output']}")

    # Load tenure model
    tenure_model = CoachTenureModel.load(MODEL_PATHS['ordinal_model_output'])
    print(f"Loaded tenure model from {MODEL_PATHS['ordinal_model_output']}")

    # Load WAR prediction data
    war_df = pd.read_csv(MODEL_PATHS['war_data_file'])
    print(f"Loaded WAR data: {len(war_df)} rows")

    # Load master data - SVD imputed (used for all predictions for consistency)
    master_df = pd.read_csv(MODEL_PATHS['data_file'])
    print(f"Loaded master data: {len(master_df)} rows (SVD imputed)")

    return war_model, tenure_model, war_df, master_df


def get_tenure_predictions_with_leakage_fix(tenure_model, recent_df, master_df):
    """
    Get tenure predictions for recent hires, handling data leakage.

    Coaches with prior HC stints in training data are predicted using
    a model retrained without their prior data.

    All data uses SVD imputation for consistency with model training.

    Parameters
    ----------
    tenure_model : CoachTenureModel
        The trained tenure model
    recent_df : pd.DataFrame
        Recent hires data (SVD imputed)
    master_df : pd.DataFrame
        Full master data (SVD imputed) for identifying prior stints and retraining
    """
    # Load training data (known tenure instances)
    df_train = master_df[master_df['Coach Tenure Class'] != -1].copy()

    # Identify coaches with prior stints
    recent_coaches = set(recent_df['Coach Name'].unique())
    train_coaches = set(df_train['Coach Name'].unique())
    coaches_with_prior = recent_coaches & train_coaches

    print(f"  {len(coaches_with_prior)} coaches have prior stints - retraining model for them")

    # Initialize arrays
    n = len(recent_df)
    all_predictions = np.zeros(n, dtype=int)
    all_probabilities = np.zeros((n, 3))

    # Prepare feature columns
    feature_cols = [f'Feature {i}' for i in range(1, 151)]

    # Split coaches
    mask_prior = recent_df['Coach Name'].isin(coaches_with_prior)

    # Group 1: Coaches WITHOUT prior stints - use standard model
    if (~mask_prior).any():
        df_no_prior = recent_df[~mask_prior]
        X_no_prior = df_no_prior[feature_cols].values

        pred = tenure_model.predict(X_no_prior)
        prob = tenure_model.predict_proba(X_no_prior)

        indices = np.where(~mask_prior.values)[0]
        all_predictions[indices] = pred
        all_probabilities[indices] = prob

    # Group 2: Coaches WITH prior stints - retrain model excluding them
    if mask_prior.any():
        # Filter training data to exclude these coaches
        df_train_clean = df_train[~df_train['Coach Name'].isin(coaches_with_prior)]

        X_train = df_train_clean[feature_cols].values
        y_train = df_train_clean['Coach Tenure Class'].values

        # Retrain model
        clean_model = CoachTenureModel(
            use_ordinal=tenure_model.use_ordinal,
            n_classes=3,
            random_state=MODEL_CONFIG['random_state']
        )
        clean_model.fit(X_train, y_train, verbose=False)

        # Predict using raw data with 0-fill (matching predict.py)
        df_with_prior = recent_df[mask_prior]
        X_with_prior = df_with_prior[feature_cols].values

        pred = clean_model.predict(X_with_prior)
        prob = clean_model.predict_proba(X_with_prior)

        indices = np.where(mask_prior.values)[0]
        all_predictions[indices] = pred
        all_probabilities[indices] = prob

    return all_predictions, all_probabilities


def get_train_test_split(war_df, master_df, test_size=0.2, random_state=42):
    """
    Recreate the same train/test split used during training.
    This ensures we only evaluate on true held-out data.

    NOTE: Recent hires (tenure class -1) are excluded from the training data,
    so we must filter them out here as well.
    """
    # Get recent hire coaches to exclude (same as training)
    recent_hire_coaches = set(master_df[master_df['Coach Tenure Class'] == -1]['Coach Name'].unique())

    # Filter WAR data to exclude recent hires
    war_df_filtered = war_df[~war_df['Coach Name'].isin(recent_hire_coaches)]

    coaches = war_df_filtered['Coach Name'].values
    unique_coaches = war_df_filtered['Coach Name'].unique()

    np.random.seed(random_state)
    n_test = int(len(unique_coaches) * test_size)
    test_coaches = set(np.random.choice(unique_coaches, n_test, replace=False))
    train_coaches = set(unique_coaches) - test_coaches

    # Get indices in the filtered dataframe
    train_idx = np.array([i for i, c in enumerate(coaches) if c in train_coaches])
    test_idx = np.array([i for i, c in enumerate(coaches) if c in test_coaches])

    return war_df_filtered, train_idx, test_idx, train_coaches, test_coaches


def plot_war_correlation(war_model, war_df, master_df, output_dir):
    """
    Plot actual vs predicted WAR correlation using TEST SET ONLY.
    This avoids data leakage from showing training set performance.
    """
    # Get train/test split (same as training, excluding recent hires)
    war_df_filtered, train_idx, test_idx, train_coaches, test_coaches = get_train_test_split(war_df, master_df)

    # Prepare features - TEST SET ONLY
    feature_cols = [f'Feature {i}' for i in range(1, 141)]
    X_test = war_df_filtered.iloc[test_idx][feature_cols].values
    y_true = war_df_filtered.iloc[test_idx]['avg_war_per_season'].values

    # Get predictions on test set only
    y_pred = war_model.predict(X_test)

    # Calculate statistics
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    r2 = correlation ** 2
    mae = np.mean(np.abs(y_true - y_pred))

    print(f"  Using TEST SET ONLY: {len(test_idx)} samples, {len(test_coaches)} coaches (recent hires excluded)")

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5, c=COLORS['primary'], s=50, edgecolors='white', linewidth=0.5)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect prediction')

    # Regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, y_pred)
    x_line = np.linspace(min_val, max_val, 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color=COLORS['secondary'], linewidth=2, label=f'Fit (r={correlation:.3f})')

    # Labels and title
    ax.set_xlabel('Actual WAR (% wins above replacement)', fontsize=12)
    ax.set_ylabel('Predicted WAR (% wins above replacement)', fontsize=12)
    ax.set_title('WAR Prediction: Actual vs Predicted (Test Set)', fontsize=14, fontweight='bold')

    # Add statistics text box
    stats_text = f'r = {correlation:.3f}\nR² = {r2:.3f}\nMAE = {mae:.3f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.legend(loc='lower right')
    ax.set_aspect('equal')

    # Save
    output_path = output_dir / 'war_correlation.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_recent_hires_war(war_model, master_df, war_df, output_dir):
    """
    Plot predicted WAR for recent hires (coaches hired since 2022).

    NOTE: The WAR model was trained EXCLUDING recent hires to avoid data leakage,
    so these predictions are unbiased. Uses SVD imputed data for consistency
    with model training.
    """
    # Filter to recent hires - use SVD imputed data (consistent with training)
    recent_df = master_df[master_df['Coach Tenure Class'] == -1].copy()

    if len(recent_df) == 0:
        print("No recent hires found (Coach Tenure Class == -1)")
        return

    print(f"  Predicting WAR for {len(recent_df)} recent hires (model trained excluding these coaches)")

    # Prepare features (140 features for WAR model)
    feature_cols = [f'Feature {i}' for i in range(1, 141)]
    X = recent_df[feature_cols].values

    # Get WAR predictions
    war_preds = war_model.predict(X)
    recent_df['Predicted_WAR'] = war_preds

    # Sort by predicted WAR
    recent_df = recent_df.sort_values('Predicted_WAR', ascending=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Horizontal bar chart
    coaches = recent_df['Coach Name'].values
    wars = recent_df['Predicted_WAR'].values
    years = recent_df['Year'].values

    # Color based on WAR value
    colors = [COLORS['class_2'] if w > 0 else COLORS['class_0'] for w in wars]

    y_pos = np.arange(len(coaches))
    bars = ax.barh(y_pos, wars, color=colors, edgecolor='white', linewidth=0.5)

    # Add coach names with hire year
    labels = [f"{coach} ({int(year)})" for coach, year in zip(coaches, years)]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)

    # Add vertical line at 0
    ax.axvline(x=0, color='black', linewidth=1, linestyle='-')

    # Add value labels on bars
    for i, (bar, war) in enumerate(zip(bars, wars)):
        width = bar.get_width()
        label_x = width + 0.005 if width >= 0 else width - 0.005
        ha = 'left' if width >= 0 else 'right'
        ax.text(label_x, bar.get_y() + bar.get_height()/2,
                f'{war:.3f}', va='center', ha=ha, fontsize=9)

    # Labels and title
    ax.set_xlabel('Predicted WAR (% wins above replacement)', fontsize=12)
    ax.set_title('Predicted WAR for Recent Head Coach Hires', fontsize=14, fontweight='bold')

    # Add legend
    positive_patch = mpatches.Patch(color=COLORS['class_2'], label='Positive WAR')
    negative_patch = mpatches.Patch(color=COLORS['class_0'], label='Negative WAR')
    ax.legend(handles=[positive_patch, negative_patch], loc='lower right')

    # Save
    output_path = output_dir / 'recent_hires_war.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    return recent_df


def plot_war_tenure_matrix(war_model, tenure_model, master_df, war_df, output_dir):
    """
    Plot predicted WAR vs predicted tenure class for recent hires.

    Uses data leakage prevention for tenure predictions - coaches with
    prior HC stints are predicted using a model retrained without their data.

    All predictions use SVD imputed data for consistency with model training.

    Parameters
    ----------
    war_model : WARRegressor
        WAR prediction model
    tenure_model : CoachTenureModel
        Tenure classification model
    master_df : pd.DataFrame
        SVD imputed master data (used for all predictions)
    war_df : pd.DataFrame
        WAR prediction data
    output_dir : Path
        Output directory for figures
    """
    # Filter to recent hires using SVD imputed data (consistent with training)
    recent_df = master_df[master_df['Coach Tenure Class'] == -1].copy()

    if len(recent_df) == 0:
        print("No recent hires found")
        return

    # Get WAR predictions
    feature_cols_war = [f'Feature {i}' for i in range(1, 141)]
    X_war = recent_df[feature_cols_war].values
    war_preds = war_model.predict(X_war)

    # Get tenure predictions WITH leakage fix
    print("  Getting tenure predictions with leakage fix...")
    tenure_preds, tenure_probs = get_tenure_predictions_with_leakage_fix(
        tenure_model, recent_df, master_df
    )

    recent_df['Predicted_WAR'] = war_preds
    recent_df['Predicted_Tenure'] = tenure_preds
    recent_df['Tenure_Confidence'] = tenure_probs.max(axis=1)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Define colors by tenure class
    tenure_colors = {0: COLORS['class_0'], 1: COLORS['class_1'], 2: COLORS['class_2']}
    colors = [tenure_colors[t] for t in tenure_preds]

    # Jitter for y-axis (use fixed seed for reproducibility)
    np.random.seed(123)
    y_jitter = np.random.uniform(-0.15, 0.15, len(tenure_preds))

    # Scatter plot
    scatter = ax.scatter(
        war_preds, tenure_preds + y_jitter,
        c=colors, s=100, alpha=0.7, edgecolors='white', linewidth=1
    )

    # Add coach name labels
    for i, row in recent_df.iterrows():
        idx = recent_df.index.get_loc(i)
        ax.annotate(
            row['Coach Name'],
            (war_preds[idx], tenure_preds[idx] + y_jitter[idx]),
            fontsize=8, alpha=0.8,
            xytext=(5, 0), textcoords='offset points'
        )

    # Add quadrant lines
    ax.axvline(x=0, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    ax.axhline(y=0.5, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    ax.axhline(y=1.5, color='gray', linewidth=1, linestyle='--', alpha=0.5)

    # Labels
    ax.set_xlabel('Predicted WAR (% wins above replacement)', fontsize=12)
    ax.set_ylabel('Predicted Tenure Class', fontsize=12)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Class 0\n(1-2 years)', 'Class 1\n(3-4 years)', 'Class 2\n(5+ years)'])
    ax.set_title('Predicted WAR vs Predicted Tenure for Recent Hires', fontsize=14, fontweight='bold')

    # Add legend
    class_patches = [
        mpatches.Patch(color=COLORS['class_0'], label='Predicted: 1-2 years'),
        mpatches.Patch(color=COLORS['class_1'], label='Predicted: 3-4 years'),
        mpatches.Patch(color=COLORS['class_2'], label='Predicted: 5+ years')
    ]
    ax.legend(handles=class_patches, loc='upper left')

    # Add quadrant labels
    ax.text(0.02, 2.7, 'High WAR\nLong Tenure', fontsize=10, style='italic', alpha=0.7)
    ax.text(-0.15, 2.7, 'Low WAR\nLong Tenure', fontsize=10, style='italic', alpha=0.7, ha='right')
    ax.text(0.02, -0.3, 'High WAR\nShort Tenure', fontsize=10, style='italic', alpha=0.7)
    ax.text(-0.15, -0.3, 'Low WAR\nShort Tenure', fontsize=10, style='italic', alpha=0.7, ha='right')

    ax.set_ylim(-0.5, 3)

    # Save
    output_path = output_dir / 'war_tenure_matrix.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    return recent_df


def plot_feature_importance_by_category(war_model, output_dir):
    """
    Plot feature importance grouped by category.
    """
    importances = war_model.get_feature_importances()

    # Define feature categories (based on CLAUDE.md)
    categories = {
        'Core Experience': list(range(0, 8)),      # Features 1-8
        'OC Stats': list(range(8, 41)),            # Features 9-41
        'DC Stats': list(range(41, 74)),           # Features 42-74
        'HC Stats': list(range(74, 140))           # Features 75-140
    }

    # Calculate total and average importance per category
    category_totals = {}
    category_avgs = {}
    for cat_name, indices in categories.items():
        cat_importance = importances[indices]
        category_totals[cat_name] = cat_importance.sum()
        category_avgs[cat_name] = cat_importance.mean()

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Total importance
    cat_names = list(category_totals.keys())
    totals = list(category_totals.values())
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary'], '#9467bd']

    bars1 = ax1.bar(cat_names, totals, color=colors, edgecolor='white', linewidth=1)
    ax1.set_ylabel('Total Feature Importance', fontsize=12)
    ax1.set_title('Total Importance by Category', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=15)

    # Add value labels
    for bar, val in zip(bars1, totals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    # Plot 2: Average importance
    avgs = list(category_avgs.values())
    bars2 = ax2.bar(cat_names, avgs, color=colors, edgecolor='white', linewidth=1)
    ax2.set_ylabel('Average Feature Importance', fontsize=12)
    ax2.set_title('Average Importance by Category', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=15)

    # Add value labels
    for bar, val in zip(bars2, avgs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    # Add feature counts
    for ax, bars in [(ax1, bars1), (ax2, bars2)]:
        for bar, (cat_name, indices) in zip(bars, categories.items()):
            ax.text(bar.get_x() + bar.get_width()/2, 0.001,
                   f'n={len(indices)}', ha='center', va='bottom', fontsize=9, alpha=0.7)

    plt.suptitle('WAR Model Feature Importance by Category', fontsize=16, fontweight='bold', y=1.02)

    # Save
    output_path = output_dir / 'war_feature_importance_category.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_war_distribution(war_df, master_df, output_dir):
    """
    Plot distribution of actual WAR values (training data only, excluding recent hires).
    """
    # Exclude recent hires
    recent_hire_coaches = set(master_df[master_df['Coach Tenure Class'] == -1]['Coach Name'].unique())
    war_df_filtered = war_df[~war_df['Coach Name'].isin(recent_hire_coaches)]

    war_values = war_df_filtered['avg_war_per_season'].values

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram
    n, bins, patches = ax.hist(war_values, bins=30, color=COLORS['primary'],
                                edgecolor='white', linewidth=0.5, alpha=0.7)

    # Color bars based on positive/negative
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge < 0:
            patch.set_facecolor(COLORS['class_0'])
        else:
            patch.set_facecolor(COLORS['class_2'])

    # Add vertical line at 0 and mean
    ax.axvline(x=0, color='black', linewidth=2, linestyle='-', label='Replacement level')
    ax.axvline(x=war_values.mean(), color=COLORS['secondary'], linewidth=2,
               linestyle='--', label=f'Mean ({war_values.mean():.3f})')

    # Labels
    ax.set_xlabel('Average WAR per Season (% wins above replacement)', fontsize=12)
    ax.set_ylabel('Number of Coaching Stints', fontsize=12)
    ax.set_title('Distribution of Coach WAR Values', fontsize=14, fontweight='bold')

    # Statistics text
    stats_text = (f'n = {len(war_values)}\n'
                  f'Mean = {war_values.mean():.3f}\n'
                  f'Std = {war_values.std():.3f}\n'
                  f'Min = {war_values.min():.3f}\n'
                  f'Max = {war_values.max():.3f}')
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.legend(loc='upper left')

    # Save
    output_path = output_dir / 'war_distribution.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_top_features(war_model, output_dir, top_n=20):
    """
    Plot top N most important features.
    """
    importances = war_model.get_feature_importances()
    feature_names = [f'Feature {i}' for i in range(1, 141)]

    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]
    top_importances = importances[indices]
    top_names = [feature_names[i] for i in indices]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Horizontal bar chart
    y_pos = np.arange(len(top_names))
    bars = ax.barh(y_pos, top_importances, color=COLORS['primary'],
                   edgecolor='white', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names)
    ax.invert_yaxis()  # Top feature at top

    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Most Important Features for WAR Prediction',
                fontsize=14, fontweight='bold')

    # Add value labels
    for bar, imp in zip(bars, top_importances):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
               f'{imp:.4f}', va='center', fontsize=9)

    # Save
    output_path = output_dir / 'war_top_features.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate WAR prediction figures')
    parser.add_argument('--output-dir', type=str, default='figures/war',
                       help='Output directory for figures')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load models and data
    print("\n" + "=" * 60)
    print("Loading models and data...")
    print("=" * 60)
    war_model, tenure_model, war_df, master_df = load_models_and_data()

    # Generate figures
    print("\n" + "=" * 60)
    print("Generating figures...")
    print("=" * 60)

    # 1. WAR correlation plot
    print("\n1. WAR prediction correlation...")
    plot_war_correlation(war_model, war_df, master_df, output_dir)

    # 2. Recent hires WAR predictions
    print("\n2. Recent hires WAR predictions...")
    plot_recent_hires_war(war_model, master_df, war_df, output_dir)

    # 3. WAR x Tenure matrix (all using SVD imputed data for consistency)
    print("\n3. WAR vs Tenure matrix...")
    plot_war_tenure_matrix(war_model, tenure_model, master_df, war_df, output_dir)

    # 4. Feature importance by category
    print("\n4. Feature importance by category...")
    plot_feature_importance_by_category(war_model, output_dir)

    # 5. WAR distribution
    print("\n5. WAR distribution...")
    plot_war_distribution(war_df, master_df, output_dir)

    # 6. Top features
    print("\n6. Top features...")
    plot_top_features(war_model, output_dir)

    print("\n" + "=" * 60)
    print(f"All figures saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
