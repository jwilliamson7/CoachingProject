#!/usr/bin/env python
"""
Prediction script for NFL Coach Tenure Prediction Model.

Makes predictions for recent coaching hires (tenure class == -1).

Usage:
    python scripts/predict.py --model model.pkl

Examples:
    # Predict with ordinal model
    python scripts/predict.py --model coach_tenure_ordinal_model.pkl

    # Predict and save to CSV
    python scripts/predict.py --model coach_tenure_model.pkl --output predictions.csv
"""

import argparse
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd

from model import CoachTenureModel
from model.config import MODEL_PATHS, FEATURE_CONFIG, CONFIDENCE_THRESHOLDS, ORDINAL_CONFIG


def load_model(model_path: str) -> CoachTenureModel:
    """Load a trained model."""
    print(f"Loading model from {model_path}...")
    return CoachTenureModel.load(model_path)


def load_recent_hires(data_path: str = None) -> pd.DataFrame:
    """Load recent coaching hires (tenure class == -1)."""
    if data_path is None:
        data_path = os.path.join(project_root, MODEL_PATHS['raw_data_file'])

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, index_col=0)

    # Handle missing values
    df.fillna(0, inplace=True)

    # Filter to recent hires only (tenure class == -1)
    df_recent = df[df[FEATURE_CONFIG['target_column']] == -1]

    print(f"Found {len(df_recent)} recent coaching hires to predict")
    return df_recent


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features from dataframe."""
    X = df.iloc[:, FEATURE_CONFIG['feature_columns_start']:FEATURE_CONFIG['feature_columns_end']]
    return X


def format_predictions(
    df: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    confidence_levels: np.ndarray
) -> pd.DataFrame:
    """Format predictions into a nice dataframe."""
    results = []

    for i, idx in enumerate(df.index):
        row = df.loc[idx]
        pred_class = predictions[i]
        prob = probabilities[i]
        max_prob = prob.max()
        confidence = confidence_levels[i]

        # Class label
        class_labels = {
            0: "0 (1-2 years)",
            1: "1 (3-4 years)",
            2: "2 (5+ years)"
        }

        results.append({
            'Coach Name': row[FEATURE_CONFIG['coach_name_column']],
            'Year': int(row[FEATURE_CONFIG['year_column']]),
            'Predicted Class': pred_class,
            'Prediction Label': class_labels.get(pred_class, str(pred_class)),
            'P(Class 0)': prob[0],
            'P(Class 1)': prob[1],
            'P(Class 2)': prob[2],
            'Max Probability': max_prob,
            'Confidence': confidence
        })

    return pd.DataFrame(results)


def print_predictions_table(df_predictions: pd.DataFrame, model_type: str = "Ordinal"):
    """Print formatted predictions table."""
    print("\n" + "="*80)
    print("NFL HEAD COACH TENURE PREDICTIONS")
    print("="*80)
    print(f"Model: {model_type} XGBoost Classifier")
    print(f"Predictions for {len(df_predictions)} recent coaching hires")
    print("="*80)

    print(f"{'Coach Name':<22} {'Year':>4}  {'Prediction':<14} {'Probability Distribution':<30} {'Confidence'}")
    print("-"*80)

    for _, row in df_predictions.iterrows():
        coach_name = row['Coach Name']
        year = row['Year']
        pred_label = row['Prediction Label']
        prob_str = f"[{row['P(Class 0)']:.3f}, {row['P(Class 1)']:.3f}, {row['P(Class 2)']:.3f}]"
        max_prob = row['Max Probability']
        confidence = row['Confidence']

        print(f"  {coach_name:<20} {year:>4}  {pred_label:<14} {prob_str:<30} {confidence:<4} ({max_prob:.1%})")

    # Summary statistics
    print("\n" + "-"*80)
    print("SUMMARY STATISTICS")
    print("-"*80)

    print(f"\nPrediction Confidence:")
    print(f"  Average confidence: {df_predictions['Max Probability'].mean():.1%}")
    print(f"  Confidence range: {df_predictions['Max Probability'].min():.1%} - {df_predictions['Max Probability'].max():.1%}")

    # Count predictions by class
    class_counts = df_predictions['Predicted Class'].value_counts().sort_index()
    print(f"\nPredicted Tenure Distribution:")
    class_labels = ["Class 0 (1-2 years)", "Class 1 (3-4 years)", "Class 2 (5+ years)"]
    for cls in range(3):
        count = class_counts.get(cls, 0)
        pct = count / len(df_predictions) * 100
        print(f"  {class_labels[cls]}: {count} coaches ({pct:.1f}%)")

    # Count by confidence level
    conf_counts = df_predictions['Confidence'].value_counts()
    print(f"\nPrediction Confidence Distribution:")
    for level in ['HIGH', 'MED', 'LOW']:
        count = conf_counts.get(level, 0)
        print(f"  {level}: {count} coaches")

    print("\n" + "="*80)
    print("INTERPRETATION GUIDE:")
    print("Class 0 (1-2 years): Short tenure - likely to be fired within 2 seasons")
    print("Class 1 (3-4 years): Medium tenure - moderate job security")
    print("Class 2 (5+ years):  Long tenure - likely to have extended tenure")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Predict NFL Coach Tenure for Recent Hires",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--model', '-m', type=str, default=None,
        help='Path to trained model file'
    )
    parser.add_argument(
        '--data', '-d', type=str, default=None,
        help='Path to data CSV file containing recent hires'
    )
    parser.add_argument(
        '--output', '-o', type=str, default=None,
        help='Output path for predictions CSV'
    )
    parser.add_argument(
        '--min-year', type=int, default=2022,
        help='Minimum hire year to include in predictions (default: 2022)'
    )

    args = parser.parse_args()

    # Determine model path
    if args.model is None:
        model_path = os.path.join(project_root, MODEL_PATHS['ordinal_model_output'])
        if not os.path.exists(model_path):
            model_path = os.path.join(project_root, MODEL_PATHS['multiclass_model_output'])
            if not os.path.exists(model_path):
                print("No trained model found. Run training first:")
                print("  python scripts/train.py")
                return
    else:
        model_path = args.model

    # Load model
    model = load_model(model_path)
    model_type = "Ordinal" if model.use_ordinal else "Multiclass"

    # Load recent hires
    df_recent = load_recent_hires(args.data)

    if len(df_recent) == 0:
        print("No recent coaching hires found (tenure class == -1)")
        return

    # Filter by minimum year if specified
    if args.min_year:
        df_recent = df_recent[df_recent[FEATURE_CONFIG['year_column']] >= args.min_year]
        print(f"Filtered to {len(df_recent)} hires from {args.min_year} onwards")

    if len(df_recent) == 0:
        print(f"No coaching hires found from {args.min_year} onwards")
        return

    # Prepare features
    X = prepare_features(df_recent)

    # Make predictions
    print("\nGenerating predictions...")
    predictions, max_probs, confidence_levels = model.predict_with_confidence(X)
    probabilities = model.predict_proba(X)

    # Format results
    df_predictions = format_predictions(
        df_recent, predictions, probabilities, confidence_levels
    )

    # Print formatted table
    print_predictions_table(df_predictions, model_type)

    # Save to CSV if requested
    if args.output:
        df_predictions.to_csv(args.output, index=False)
        print(f"\nPredictions saved to {args.output}")

    print("\nPrediction complete!")


if __name__ == '__main__':
    main()
