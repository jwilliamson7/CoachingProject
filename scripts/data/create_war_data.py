"""
Create WAR prediction dataset by merging coach features with WAR targets.

This script:
1. Loads the master coaching data (with 150 features per hire)
2. Loads the WAR trajectories data (season-by-season WAR)
3. Identifies coaching stints for coaches with multiple HC positions
4. Calculates average WAR per season for each hiring instance
5. Creates merged dataset with Features 1-140 and avg_war_per_season target

Usage:
    python scripts/data/create_war_data.py
    python scripts/data/create_war_data.py --output data/war_prediction_data.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_war_data(filepath: str) -> pd.DataFrame:
    """
    Load WAR trajectories CSV.

    Parameters
    ----------
    filepath : str
        Path to coach_war_trajectories.csv

    Returns
    -------
    pd.DataFrame
        WAR data with columns: Coach, Year, Season_Number, Annual_WAR, etc.
    """
    df = pd.read_csv(filepath)
    print(f"Loaded WAR data: {len(df)} rows, {df['Coach'].nunique()} coaches")
    return df


def load_master_data(filepath: str) -> pd.DataFrame:
    """
    Load master coaching data CSV.

    Parameters
    ----------
    filepath : str
        Path to svd_imputed_master_data.csv

    Returns
    -------
    pd.DataFrame
        Master data with Coach Name, Year, Features 1-150, targets
    """
    df = pd.read_csv(filepath)
    print(f"Loaded master data: {len(df)} rows, {df['Coach Name'].nunique()} coaches")
    return df


def identify_coach_stints(war_df: pd.DataFrame) -> Dict[str, List[Tuple[int, int, str]]]:
    """
    For each coach, identify (start_year, end_year, team) for each head coaching stint.

    Stints are detected by:
    1. Year gaps > 1 year
    2. Team changes (even without year gap)

    For example:
    - Belichick: 1991-1995 (CLE), 2000-2023 (NWE) = 2 stints
    - Ditka: 1982-1992 (CHI), 1997-1999 (NOR) = 2 stints
    - Caldwell: 2009-2011 (CLT), 2014-2017 (DET) = 2 stints

    Parameters
    ----------
    war_df : pd.DataFrame
        WAR trajectories data with Team column

    Returns
    -------
    Dict[str, List[Tuple[int, int, str]]]
        Mapping from coach name to list of (start_year, end_year, team) tuples
    """
    stints_by_coach = {}

    # Check if Team column exists
    has_team = 'Team' in war_df.columns

    for coach in war_df['Coach'].unique():
        coach_data = war_df[war_df['Coach'] == coach].sort_values('Year')
        years = coach_data['Year'].values
        teams = coach_data['Team'].values if has_team else [None] * len(years)

        stints = []
        current_stint_start = None
        current_team = None
        prev_year = None

        for year, team in zip(years, teams):
            # New stint if: first year, year gap > 1, or team change
            is_new_stint = (
                prev_year is None or
                year - prev_year > 1 or
                (has_team and team != current_team)
            )

            if is_new_stint:
                # Save previous stint
                if current_stint_start is not None:
                    stints.append((current_stint_start, prev_year, current_team))
                current_stint_start = year
                current_team = team

            prev_year = year

        # Add the last stint
        if current_stint_start is not None:
            stints.append((current_stint_start, prev_year, current_team))

        stints_by_coach[coach] = stints

    # Print summary
    multi_stint_coaches = [c for c, s in stints_by_coach.items() if len(s) > 1]
    print(f"Identified stints for {len(stints_by_coach)} coaches")
    print(f"  - {len(multi_stint_coaches)} coaches with multiple stints")

    # Show examples
    if multi_stint_coaches and has_team:
        print("\n  Example multi-stint coaches:")
        for coach in multi_stint_coaches[:3]:
            stints = stints_by_coach[coach]
            stint_strs = [f"{s[2]} {s[0]}-{s[1]}" for s in stints]
            print(f"    {coach}: {', '.join(stint_strs)}")

    return stints_by_coach


def match_hire_to_stint(
    coach: str,
    hire_year: int,
    stints: Dict[str, List[Tuple[int, int, str]]]
) -> Optional[Tuple[int, int, str]]:
    """
    Find which stint contains the hire year.

    Parameters
    ----------
    coach : str
        Coach name
    hire_year : int
        Year the coach was hired
    stints : Dict[str, List[Tuple[int, int, str]]]
        Mapping from coach name to list of (start_year, end_year, team) tuples

    Returns
    -------
    Optional[Tuple[int, int, str]]
        (start_year, end_year, team) of matching stint, or None if not found
    """
    if coach not in stints:
        return None

    for start_year, end_year, team in stints[coach]:
        if start_year <= hire_year <= end_year:
            return (start_year, end_year, team)

    # No matching stint found (hire year not in any stint range)
    return None


def calculate_average_war(
    war_df: pd.DataFrame,
    coach: str,
    start_year: int,
    end_year: int,
    team: str = None
) -> float:
    """
    Calculate mean Annual_WAR for a coach's specified tenure period.

    Parameters
    ----------
    war_df : pd.DataFrame
        WAR trajectories data
    coach : str
        Coach name
    start_year : int
        First year of the stint
    end_year : int
        Last year of the stint
    team : str, optional
        Team abbreviation to filter by (if available in data)

    Returns
    -------
    float
        Mean Annual_WAR over the stint
    """
    mask = (
        (war_df['Coach'] == coach) &
        (war_df['Year'] >= start_year) &
        (war_df['Year'] <= end_year)
    )

    # Also filter by team if provided and column exists
    if team is not None and 'Team' in war_df.columns:
        mask = mask & (war_df['Team'] == team)

    stint_data = war_df[mask]

    if len(stint_data) == 0:
        return np.nan

    return stint_data['Annual_WAR'].mean()


def create_war_dataset(
    master_df: pd.DataFrame,
    war_df: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create merged dataset with Features 1-140 and avg_war_per_season target.

    This function:
    1. Identifies stints for all coaches in WAR data
    2. For each hire in master data, matches to corresponding WAR stint
    3. Calculates average WAR per season for each hire
    4. Excludes hiring team features (141-150)

    Parameters
    ----------
    master_df : pd.DataFrame
        Master coaching data with 150 features
    war_df : pd.DataFrame
        WAR trajectories data
    verbose : bool
        Whether to print progress information

    Returns
    -------
    pd.DataFrame
        Merged dataset with Coach Name, Year, Features 1-140, avg_war_per_season
    """
    # Identify stints
    stints = identify_coach_stints(war_df)

    # Prepare output data
    rows = []
    matched_count = 0
    unmatched_count = 0

    for idx, row in master_df.iterrows():
        coach_name = row['Coach Name']
        hire_year = row['Year']

        # Try to match hire to WAR stint
        stint = match_hire_to_stint(coach_name, hire_year, stints)

        if stint is None:
            unmatched_count += 1
            continue

        start_year, end_year, team = stint
        avg_war = calculate_average_war(
            war_df, coach_name, start_year, end_year, team
        )

        if np.isnan(avg_war):
            unmatched_count += 1
            continue

        # Extract Features 1-140 (columns Feature 1 through Feature 140)
        # In master data, features are in columns 3 to 152 (0-indexed: 3:143)
        # Feature 1 = column index 3, Feature 140 = column index 142
        feature_cols = [f'Feature {i}' for i in range(1, 141)]
        features = row[feature_cols].values

        # Build row for output
        output_row = {
            'Coach Name': coach_name,
            'Year': hire_year,
            'Team': team,
            'avg_war_per_season': avg_war,
            'stint_start': start_year,
            'stint_end': end_year,
            'num_seasons': end_year - start_year + 1
        }

        # Add features
        for i, feat_val in enumerate(features, start=1):
            output_row[f'Feature {i}'] = feat_val

        rows.append(output_row)
        matched_count += 1

    # Create DataFrame
    result_df = pd.DataFrame(rows)

    # Reorder columns: Coach Name, Year, Team, Features, then metadata and target
    feature_cols = [f'Feature {i}' for i in range(1, 141)]
    col_order = (
        ['Coach Name', 'Year', 'Team'] +
        feature_cols +
        ['avg_war_per_season', 'stint_start', 'stint_end', 'num_seasons']
    )
    result_df = result_df[col_order]

    if verbose:
        print(f"\nDataset creation complete:")
        print(f"  - Matched hires: {matched_count}")
        print(f"  - Unmatched hires: {unmatched_count}")
        print(f"  - Output shape: {result_df.shape}")
        print(f"\nTarget variable (avg_war_per_season) stats:")
        print(f"  - Mean: {result_df['avg_war_per_season'].mean():.4f}")
        print(f"  - Std:  {result_df['avg_war_per_season'].std():.4f}")
        print(f"  - Min:  {result_df['avg_war_per_season'].min():.4f}")
        print(f"  - Max:  {result_df['avg_war_per_season'].max():.4f}")

    return result_df


def validate_multi_stint_coaches(result_df: pd.DataFrame, war_df: pd.DataFrame):
    """
    Validate that multi-stint coaches have correct WAR values for each stint.

    Prints validation info for coaches like Belichick who have multiple hires.
    """
    # Find coaches with multiple hires in result
    coach_counts = result_df['Coach Name'].value_counts()
    multi_hire_coaches = coach_counts[coach_counts > 1].index.tolist()

    if not multi_hire_coaches:
        print("\nNo multi-hire coaches found in result dataset")
        return

    # Check for Team column
    has_team = 'Team' in result_df.columns

    print(f"\nValidation: {len(multi_hire_coaches)} coaches with multiple hires")
    for coach in multi_hire_coaches[:5]:  # Show first 5
        coach_rows = result_df[result_df['Coach Name'] == coach]
        print(f"\n  {coach}:")
        for _, row in coach_rows.iterrows():
            team_str = f" ({row['Team']})" if has_team else ""
            print(f"    Hire {int(row['Year'])}{team_str}: "
                  f"stint {int(row['stint_start'])}-{int(row['stint_end'])}, "
                  f"avg WAR = {row['avg_war_per_season']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Create WAR prediction dataset'
    )
    parser.add_argument(
        '--master-data',
        type=str,
        default='data/svd_imputed_master_data.csv',
        help='Path to master coaching data'
    )
    parser.add_argument(
        '--war-data',
        type=str,
        default='data/coach_war_trajectories_with_team.csv',
        help='Path to WAR trajectories data (with Team column)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/war_prediction_data.csv',
        help='Output path for merged dataset'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate multi-stint coaches'
    )

    args = parser.parse_args()

    # Load data
    print("=" * 60)
    print("Creating WAR Prediction Dataset")
    print("=" * 60)

    master_df = load_master_data(args.master_data)
    war_df = load_war_data(args.war_data)

    # Create merged dataset
    result_df = create_war_dataset(master_df, war_df)

    # Validate if requested
    if args.validate:
        validate_multi_stint_coaches(result_df, war_df)

    # Save
    result_df.to_csv(args.output, index=False)
    print(f"\nSaved to: {args.output}")

    return result_df


if __name__ == '__main__':
    main()
