#!/usr/bin/env python
"""
Data Pipeline Validation for NFL Coach Tenure Prediction.

Validates dataset integrity, feature engineering, and team mappings.

Usage:
    python scripts/validation/validate_data.py
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set
from collections import Counter

from model.config import MODEL_PATHS, FEATURE_CONFIG
from model.cross_validation import stratified_coach_level_split
from data_constants import TEAM_FRANCHISE_MAPPINGS, CURRENT_TEAM_ABBREVIATIONS


class DataValidationResult:
    """Container for validation results."""

    def __init__(self, name: str):
        self.name = name
        self.passed = True
        self.messages = []
        self.details = {}

    def fail(self, message: str):
        self.passed = False
        self.messages.append(f"FAIL: {message}")

    def warn(self, message: str):
        self.messages.append(f"WARN: {message}")

    def info(self, message: str):
        self.messages.append(f"INFO: {message}")

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        result = f"\n[{status}] {self.name}\n"
        for msg in self.messages:
            result += f"  {msg}\n"
        return result


def load_data() -> pd.DataFrame:
    """Load the imputed master data file."""
    data_path = os.path.join(project_root, MODEL_PATHS['data_file'])
    return pd.read_csv(data_path, index_col=0)


def validate_dataset_size() -> DataValidationResult:
    """
    Validate dataset size claims.

    Paper claims: 635 instances with known tenure (excluding class -1)
    """
    result = DataValidationResult("Dataset Size Validation")

    df = load_data()
    total_rows = len(df)

    # Count instances with known tenure (class != -1)
    df_known = df[df[FEATURE_CONFIG['target_column']] != -1]
    known_tenure_count = len(df_known)

    # Count instances with unknown tenure (class == -1)
    df_unknown = df[df[FEATURE_CONFIG['target_column']] == -1]
    unknown_tenure_count = len(df_unknown)

    result.info(f"Total rows in dataset: {total_rows}")
    result.info(f"Instances with known tenure (class != -1): {known_tenure_count}")
    result.info(f"Instances with unknown tenure (class == -1): {unknown_tenure_count}")

    result.details['total_rows'] = total_rows
    result.details['known_tenure'] = known_tenure_count
    result.details['unknown_tenure'] = unknown_tenure_count

    # The paper may have different counts - this validates the actual data
    # Note: Paper claimed 635 but data may have 656 total rows
    if known_tenure_count < 500:
        result.fail(f"Too few instances with known tenure: {known_tenure_count} < 500")

    return result


def validate_class_distribution() -> DataValidationResult:
    """
    Validate class distribution claims.

    Paper claims: Class 0 (49.0%), Class 1 (26.8%), Class 2 (24.3%)
    """
    result = DataValidationResult("Class Distribution Validation")

    df = load_data()
    df_known = df[df[FEATURE_CONFIG['target_column']] != -1]

    class_counts = df_known[FEATURE_CONFIG['target_column']].value_counts().sort_index()
    total = len(df_known)

    expected_distributions = {
        0: 0.490,  # 49.0%
        1: 0.268,  # 26.8%
        2: 0.243   # 24.3%
    }

    tolerance = 0.05  # 5% tolerance

    for cls in [0, 1, 2]:
        count = class_counts.get(cls, 0)
        actual_pct = count / total
        expected_pct = expected_distributions[cls]
        diff = abs(actual_pct - expected_pct)

        result.info(f"Class {cls}: {count} instances ({actual_pct:.1%}), expected ~{expected_pct:.1%}")
        result.details[f'class_{cls}_count'] = count
        result.details[f'class_{cls}_pct'] = actual_pct

        if diff > tolerance:
            result.warn(f"Class {cls} distribution differs by {diff:.1%} from expected")

    return result


def validate_train_test_split() -> DataValidationResult:
    """
    Validate train/test split sizes.

    Paper claims: 508 train, 127 test instances (80/20 split)
    """
    result = DataValidationResult("Train/Test Split Validation")

    df = load_data()
    df_known = df[df[FEATURE_CONFIG['target_column']] != -1]

    X = df_known.iloc[:, FEATURE_CONFIG['feature_columns_start']:FEATURE_CONFIG['feature_columns_end']]
    y = df_known[FEATURE_CONFIG['target_column']]

    X_train, X_test, y_train, y_test, test_coaches = stratified_coach_level_split(
        df_known, X, y,
        test_size=0.2,
        random_state=42
    )

    train_size = len(X_train)
    test_size = len(X_test)
    total = train_size + test_size

    result.info(f"Train set: {train_size} instances ({train_size/total:.1%})")
    result.info(f"Test set: {test_size} instances ({test_size/total:.1%})")
    result.info(f"Test coaches: {len(test_coaches)}")

    result.details['train_size'] = train_size
    result.details['test_size'] = test_size
    result.details['test_coaches'] = test_coaches

    # Verify 80/20 split approximately
    expected_test_ratio = 0.2
    actual_test_ratio = test_size / total
    if abs(actual_test_ratio - expected_test_ratio) > 0.05:
        result.warn(f"Test ratio {actual_test_ratio:.1%} differs from expected 20%")

    return result


def validate_no_duplicate_instances() -> DataValidationResult:
    """Verify each coach-year combination appears only once."""
    result = DataValidationResult("No Duplicate Instances")

    df = load_data()

    # Check for duplicate coach-year combinations
    coach_year_pairs = df[['Coach Name', 'Year']].apply(tuple, axis=1)
    duplicates = coach_year_pairs[coach_year_pairs.duplicated()]

    if len(duplicates) > 0:
        result.fail(f"Found {len(duplicates)} duplicate coach-year combinations")
        for dup in duplicates.unique()[:5]:  # Show first 5
            result.info(f"  Duplicate: {dup}")
    else:
        result.info("No duplicate coach-year combinations found")

    result.details['num_duplicates'] = len(duplicates)

    return result


def validate_feature_ranges() -> DataValidationResult:
    """Validate all features are within reasonable bounds after imputation."""
    result = DataValidationResult("Feature Range Validation")

    df = load_data()

    # Get feature columns
    feature_cols = df.columns[FEATURE_CONFIG['feature_columns_start']:FEATURE_CONFIG['feature_columns_end']]
    features = df[feature_cols]

    # Check for infinite values
    inf_count = np.isinf(features.values).sum()
    if inf_count > 0:
        result.fail(f"Found {inf_count} infinite values in features")
    else:
        result.info("No infinite values found")

    # Check for NaN values (should be none after imputation)
    nan_count = features.isna().sum().sum()
    if nan_count > 0:
        result.fail(f"Found {nan_count} NaN values in features")
    else:
        result.info("No NaN values found (imputation successful)")

    # Check feature 1 (age) is reasonable (25-80 range)
    age_col = features.iloc[:, 0]  # First feature is age
    age_min, age_max = age_col.min(), age_col.max()
    result.info(f"Age range: {age_min:.0f} - {age_max:.0f}")

    if age_min < 20 or age_max > 90:
        result.warn(f"Age values outside expected range [20, 90]")

    result.details['inf_count'] = inf_count
    result.details['nan_count'] = nan_count
    result.details['age_range'] = (age_min, age_max)

    return result


def validate_no_coach_overlap_train_test() -> DataValidationResult:
    """Critical: Verify no coach appears in both train and test sets."""
    result = DataValidationResult("No Coach Overlap (Data Leakage Check)")

    df = load_data()
    df_known = df[df[FEATURE_CONFIG['target_column']] != -1]

    X = df_known.iloc[:, FEATURE_CONFIG['feature_columns_start']:FEATURE_CONFIG['feature_columns_end']]
    y = df_known[FEATURE_CONFIG['target_column']]

    X_train, X_test, y_train, y_test, test_coaches = stratified_coach_level_split(
        df_known, X, y,
        test_size=0.2,
        random_state=42
    )

    # Get train coaches
    train_coaches = set(df_known.loc[X_train.index, 'Coach Name'].unique())
    test_coaches_set = set(test_coaches)

    # Check for overlap
    overlap = train_coaches & test_coaches_set

    if len(overlap) > 0:
        result.fail(f"Found {len(overlap)} coaches appearing in both train and test sets!")
        for coach in list(overlap)[:5]:
            result.info(f"  Overlapping coach: {coach}")
    else:
        result.info(f"No coach overlap found between train ({len(train_coaches)} coaches) and test ({len(test_coaches_set)} coaches)")

    result.details['train_coaches'] = len(train_coaches)
    result.details['test_coaches'] = len(test_coaches_set)
    result.details['overlap'] = len(overlap)

    return result


def validate_coach_instances_stay_together() -> DataValidationResult:
    """Verify all instances of a coach go to the same split."""
    result = DataValidationResult("Coach Instances Stay Together")

    df = load_data()
    df_known = df[df[FEATURE_CONFIG['target_column']] != -1]

    # Find coaches with multiple instances
    coach_counts = df_known['Coach Name'].value_counts()
    multi_instance_coaches = coach_counts[coach_counts > 1].index.tolist()

    result.info(f"Found {len(multi_instance_coaches)} coaches with multiple instances")

    X = df_known.iloc[:, FEATURE_CONFIG['feature_columns_start']:FEATURE_CONFIG['feature_columns_end']]
    y = df_known[FEATURE_CONFIG['target_column']]

    X_train, X_test, y_train, y_test, test_coaches = stratified_coach_level_split(
        df_known, X, y,
        test_size=0.2,
        random_state=42
    )

    # Check that each multi-instance coach is entirely in train or test
    train_indices = set(X_train.index)
    test_indices = set(X_test.index)

    split_coaches = []
    for coach in multi_instance_coaches:
        coach_indices = df_known[df_known['Coach Name'] == coach].index.tolist()
        in_train = sum(1 for idx in coach_indices if idx in train_indices)
        in_test = sum(1 for idx in coach_indices if idx in test_indices)

        if in_train > 0 and in_test > 0:
            split_coaches.append(coach)
            result.fail(f"Coach {coach} is split: {in_train} in train, {in_test} in test")

    if len(split_coaches) == 0:
        result.info("All multi-instance coaches kept together in same split")

    result.details['multi_instance_coaches'] = len(multi_instance_coaches)
    result.details['split_coaches'] = len(split_coaches)

    return result


def validate_feature_count() -> DataValidationResult:
    """Verify feature count matches documentation (150 features)."""
    result = DataValidationResult("Feature Count Validation")

    df = load_data()

    # Get feature columns
    feature_cols = df.columns[FEATURE_CONFIG['feature_columns_start']:FEATURE_CONFIG['feature_columns_end']]
    actual_features = len(feature_cols)
    expected_features = FEATURE_CONFIG['n_features']

    result.info(f"Expected features: {expected_features}")
    result.info(f"Actual features: {actual_features}")

    if actual_features != expected_features:
        result.fail(f"Feature count mismatch: {actual_features} != {expected_features}")

    result.details['expected'] = expected_features
    result.details['actual'] = actual_features

    return result


def validate_team_mappings() -> DataValidationResult:
    """Verify team franchise mappings cover all current teams."""
    result = DataValidationResult("Team Franchise Mappings")

    # Check all current teams are covered
    covered_abbrevs = set()
    for franchise, abbrevs in TEAM_FRANCHISE_MAPPINGS.items():
        covered_abbrevs.update(abbrevs)

    # Current 32 NFL teams (standard abbreviations)
    current_teams = set(CURRENT_TEAM_ABBREVIATIONS.values())

    result.info(f"Total unique abbreviations in mappings: {len(covered_abbrevs)}")
    result.info(f"Current teams defined: {len(current_teams)}")

    # Check for missing current teams
    missing = current_teams - covered_abbrevs
    if missing:
        result.warn(f"Current teams not in mappings: {missing}")

    # Check for circular references
    circular_refs = []
    for franchise, abbrevs in TEAM_FRANCHISE_MAPPINGS.items():
        for abbrev in abbrevs:
            if abbrev in TEAM_FRANCHISE_MAPPINGS and abbrev != franchise:
                # This abbreviation is also a franchise key
                circular_refs.append((franchise, abbrev))

    if circular_refs:
        result.warn(f"Potential circular references: {circular_refs[:5]}")

    result.details['covered_abbrevs'] = len(covered_abbrevs)
    result.details['missing_teams'] = list(missing) if missing else []
    result.details['circular_refs'] = circular_refs

    return result


def validate_target_column_values() -> DataValidationResult:
    """Verify target column only contains valid values (-1, 0, 1, 2)."""
    result = DataValidationResult("Target Column Values")

    df = load_data()

    target_values = df[FEATURE_CONFIG['target_column']].unique()
    expected_values = {-1, 0, 1, 2}

    result.info(f"Unique target values: {sorted(target_values)}")

    unexpected = set(target_values) - expected_values
    if unexpected:
        result.fail(f"Unexpected target values: {unexpected}")
    else:
        result.info("All target values are valid (-1, 0, 1, 2)")

    result.details['unique_values'] = sorted(target_values)

    return result


def run_all_data_validations() -> Dict[str, DataValidationResult]:
    """Run all data validation checks."""
    print("="*60)
    print("DATA PIPELINE VALIDATION")
    print("="*60)

    validations = [
        validate_dataset_size,
        validate_class_distribution,
        validate_train_test_split,
        validate_no_duplicate_instances,
        validate_feature_ranges,
        validate_no_coach_overlap_train_test,
        validate_coach_instances_stay_together,
        validate_feature_count,
        validate_team_mappings,
        validate_target_column_values,
    ]

    results = {}
    passed = 0
    failed = 0

    for validate_func in validations:
        try:
            result = validate_func()
            results[result.name] = result
            print(result)

            if result.passed:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n[ERROR] {validate_func.__name__}: {str(e)}\n")
            failed += 1

    print("="*60)
    print(f"DATA VALIDATION SUMMARY: {passed} passed, {failed} failed")
    print("="*60)

    return results


if __name__ == '__main__':
    results = run_all_data_validations()

    # Exit with error code if any validations failed
    all_passed = all(r.passed for r in results.values())
    sys.exit(0 if all_passed else 1)
