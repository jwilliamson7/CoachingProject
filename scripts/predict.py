#!/usr/bin/env python
"""
Prediction script for NFL Coach Tenure Prediction Model (Reviewer 2: practical
application -- which recent hires are predicted to last?).

The deployed model is a single saved artifact (model + imputer + top-K indices),
built once on the whole known-tenure population and reused. Run with --retrain to
rebuild it. Recent hires (tenure class == -1) are the prediction set.

Coaches who already appear in the training population (prior HC stints) are
predicted with a model retrained -- and an imputer re-fit -- without their own
rows, so the model never sees their past outcome.

Usage:
    python scripts/predict.py
    python scripts/predict.py --retrain
    python scripts/predict.py --output analysis/recent_hire_predictions.csv
    python scripts/predict.py --no-leakage-fix
"""

import argparse
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd

from model.config import FEATURE_CONFIG
from model.pipeline import (
    FSTART, FEND, load_modeling_data, load_recent_hires,
    fit_on_full, transform_select, save_production_model, load_production_model,
)

NAME = FEATURE_CONFIG['coach_name_column']
YEAR = FEATURE_CONFIG['year_column']


def format_predictions(df, predictions, probabilities, confidence_levels):
    """Format predictions into a dataframe."""
    class_labels = {0: "0 (1-2 years)", 1: "1 (3-4 years)", 2: "2 (5+ years)"}
    results = []
    for i, idx in enumerate(df.index):
        row = df.loc[idx]
        prob = probabilities[i]
        results.append({
            'Coach Name': row[NAME],
            'Year': int(row[YEAR]),
            'Predicted Class': int(predictions[i]),
            'Prediction Label': class_labels.get(int(predictions[i]), str(predictions[i])),
            'P(Class 0)': prob[0],
            'P(Class 1)': prob[1],
            'P(Class 2)': prob[2],
            'Max Probability': prob.max(),
            'Confidence': confidence_levels[i],
        })
    return pd.DataFrame(results)


def print_predictions_table(df_predictions, model_type="Ordinal"):
    """Print formatted predictions table."""
    print("\n" + "=" * 85)
    print("NFL HEAD COACH TENURE PREDICTIONS")
    print("=" * 85)
    print(f"Model: {model_type} XGBoost Classifier")
    print(f"Predictions for {len(df_predictions)} recent coaching hires")

    has_leakage_col = 'Leakage Fixed' in df_predictions.columns
    if has_leakage_col and df_predictions['Leakage Fixed'].any():
        print("(*) = Predicted with model retrained to exclude coach's prior HC data")
    print("=" * 85)

    print(f"{'Coach Name':<22} {'Year':>4}  {'Prediction':<14} {'Probability Distribution':<30} {'Confidence'}")
    print("-" * 85)
    for _, row in df_predictions.iterrows():
        prob_str = f"[{row['P(Class 0)']:.3f}, {row['P(Class 1)']:.3f}, {row['P(Class 2)']:.3f}]"
        marker = "*" if has_leakage_col and row.get('Leakage Fixed', False) else " "
        print(f"{marker} {row['Coach Name']:<20} {row['Year']:>4}  {row['Prediction Label']:<14} "
              f"{prob_str:<30} {row['Confidence']:<4} ({row['Max Probability']:.1%})")

    print("\n" + "-" * 80)
    print("SUMMARY STATISTICS")
    print("-" * 80)
    print(f"\nPrediction Confidence:")
    print(f"  Average confidence: {df_predictions['Max Probability'].mean():.1%}")
    print(f"  Confidence range: {df_predictions['Max Probability'].min():.1%} - "
          f"{df_predictions['Max Probability'].max():.1%}")

    class_counts = df_predictions['Predicted Class'].value_counts().sort_index()
    print(f"\nPredicted Tenure Distribution:")
    class_labels = ["Class 0 (1-2 years)", "Class 1 (3-4 years)", "Class 2 (5+ years)"]
    for cls in range(3):
        count = int(class_counts.get(cls, 0))
        print(f"  {class_labels[cls]}: {count} coaches ({count / len(df_predictions) * 100:.1f}%)")

    conf_counts = df_predictions['Confidence'].value_counts()
    print(f"\nPrediction Confidence Distribution:")
    for level in ['HIGH', 'MED', 'LOW']:
        print(f"  {level}: {int(conf_counts.get(level, 0))} coaches")

    print("\n" + "=" * 80)
    print("INTERPRETATION GUIDE:")
    print("Class 0 (1-2 years): Short tenure - likely to be replaced within 2 seasons")
    print("Class 1 (3-4 years): Medium tenure - moderate job security")
    print("Class 2 (5+ years):  Long tenure - likely to have extended tenure")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Predict NFL Coach Tenure for Recent Hires (leakage-free)",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output path for predictions CSV')
    parser.add_argument('--min-year', type=int, default=2022,
                        help='Minimum hire year to include (default: 2022)')
    parser.add_argument('--retrain', action='store_true',
                        help='Rebuild and save the deployed model bundle')
    parser.add_argument('--no-leakage-fix', action='store_true',
                        help='Disable the prior-stint retraining (one model for all)')
    args = parser.parse_args()

    # Load the deployed model bundle (build + save it once if missing / --retrain)
    if args.retrain:
        path, bundle = save_production_model()
        print(f"Built and saved deployed model bundle -> {path}")
    else:
        try:
            bundle = load_production_model()
        except FileNotFoundError:
            path, bundle = save_production_model()
            print(f"No saved model found; built and saved bundle -> {path}")
    model, imputer, top_idx = bundle['model'], bundle['imputer'], bundle['feature_indices']
    print(f"Deployed model: top-{bundle['best_k']} features, trained on {bundle['n_train']} known hires")

    df_recent = load_recent_hires()
    df_train, _, _ = load_modeling_data(known_only=True)
    print(f"Recent hires (class -1): {len(df_recent)}")

    if args.min_year:
        df_recent = df_recent[df_recent[YEAR] >= args.min_year]
        print(f"Filtered to {len(df_recent)} hires from {args.min_year} onwards")
    if len(df_recent) == 0:
        print("No recent coaching hires to predict.")
        return

    coaches_with_prior = set(df_recent[NAME]) & set(df_train[NAME])
    do_fix = bool(coaches_with_prior) and not args.no_leakage_fix
    if do_fix:
        print(f"\n*** DATA LEAKAGE PREVENTION ***")
        print(f"{len(coaches_with_prior)} recent hires also appear in training (prior stints):")
        for coach in sorted(coaches_with_prior):
            yrs = df_train[df_train[NAME] == coach][YEAR].tolist()
            print(f"  - {coach} (prior stints: {yrs})")
        print("These are predicted with a model retrained (imputer re-fit) without their data.\n")

    n = len(df_recent)
    all_pred = np.zeros(n, dtype=int)
    all_prob = np.zeros((n, 3))
    all_max = np.zeros(n)
    all_conf = np.empty(n, dtype=object)
    leakage_fixed = np.zeros(n, dtype=bool)

    mask_prior = df_recent[NAME].isin(coaches_with_prior).values

    # Group 1: no prior stints (or leakage fix disabled) -> the deployed model
    grp1 = ~mask_prior if do_fix else np.ones(n, dtype=bool)
    if grp1.any():
        X = transform_select(df_recent[grp1].iloc[:, FSTART:FEND], imputer, top_idx)
        pred, mx, conf = model.predict_with_confidence(X)
        idxs = np.where(grp1)[0]
        all_pred[idxs], all_prob[idxs] = pred, model.predict_proba(X)
        all_max[idxs], all_conf[idxs] = mx, conf

    # Group 2: prior stints -> model + imputer refit excluding those coaches
    if do_fix and mask_prior.any():
        df_clean = df_train[~df_train[NAME].isin(coaches_with_prior)]
        print(f"Retraining on {len(df_clean)} instances (excluding prior-stint coaches)...")
        clean_model, clean_imputer = fit_on_full(
            df_clean.iloc[:, FSTART:FEND], df_clean[FEATURE_CONFIG['target_column']],
            feature_indices=top_idx)
        X = transform_select(df_recent[mask_prior].iloc[:, FSTART:FEND], clean_imputer, top_idx)
        pred, mx, conf = clean_model.predict_with_confidence(X)
        idxs = np.where(mask_prior)[0]
        all_pred[idxs], all_prob[idxs] = pred, clean_model.predict_proba(X)
        all_max[idxs], all_conf[idxs] = mx, conf
        leakage_fixed[idxs] = True

    df_predictions = format_predictions(df_recent, all_pred, all_prob, all_conf)
    df_predictions['Leakage Fixed'] = leakage_fixed
    df_predictions = df_predictions.sort_values('Year').reset_index(drop=True)

    print_predictions_table(df_predictions, "Ordinal")

    out = args.output or os.path.join(project_root, 'analysis', 'recent_hire_predictions.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df_predictions.to_csv(out, index=False)
    print(f"\nPredictions saved to {out}")
    print("\nPrediction complete!")


if __name__ == '__main__':
    main()
