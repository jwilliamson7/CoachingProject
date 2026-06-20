"""
Constants and configuration data for NFL coaching analysis.

This module contains all the static dictionaries, lists, and mappings
used throughout the coaching data analysis pipeline.
"""

from typing import Dict, List, Union, Optional
import pandas as pd


# Team franchise abbreviation mappings for historical name changes and relocations
TEAM_FRANCHISE_MAPPINGS = {
    # Stable franchises (never relocated, consistent abbreviations)
    "chi": ["chi"],      # Chicago Bears
    "sfo": ["sfo"],      # San Francisco 49ers
    "gnb": ["gnb"],      # Green Bay Packers
    "dal": ["dal"],      # Dallas Cowboys
    "nwe": ["nwe"],      # New England Patriots
    "jax": ["jax"],      # Jacksonville Jaguars
    "nor": ["nor"],      # New Orleans Saints
    # Franchises with historical relocations or name changes
    "ind": ["ind", "clt"],
    "ari": ["ari", "crd"],
    "hou": ["hou", "htx", "oti"],
    "ten": ["ten", "oti"],
    "oak": ["oak", "rai"], 
    "stl": ["stl", "ram", "sla", "gun"],
    "lar": ["lar", "ram"],
    "ram": ["lar", "ram"],
    "bal": ["bal", "rav", "clt"],
    "lac": ["lac", "sdg"],
    "chr": ["chr", "cra"],
    "frn": ["frn", "fyj"],
    "nyt": ["nyt", "nyj"],
    "can": ["can", "cbd"],
    "bos": ["bos", "byk", "was", "ptb"],
    "pot": ["pot", "ptb"],
    "lad": ["lad", "lda"],
    "evn": ["evn", "ecg"],
    "pho": ["pho", "crd"],
    "prt": ["prt", "det"],
    "lvr": ["lvr", "rai"],
    "chh": ["chh", "cra"],
    "htx": ["htx", "oti"],
    "buf": ["buf", "bff", "bba"],
    "cle": ["cle", "cti", "cib", "cli"],
    "min": ["min", "mnn"],
    "kan": ["kan", "kcb"],
    "det": ["det", "dwl", "dpn", "dhr", "dti"],
    "nyy": ["nyy", "naa", "nya"],
    "cin": ["cin", "red", "ccl"],
    "mia": ["mia", "msa"],
    "was": ["was", "sen"],
    "dtx": ["kan", "dtx"],
    "nyg": ["nyg", "ng1"],
    "cli": ["cli", "cib"],
    "nyb": ["nyb", "nyy"]
}

# Current team name to abbreviation mapping for new hires
CURRENT_TEAM_ABBREVIATIONS = {
    "Chicago Bears": "chi",
    "Jacksonville Jaguars": "jax",
    "New Orleans Saints": "nor",
    "New York Jets": "nyj",
    "Dallas Cowboys": "dal",
    "New England Patriots": "nwe",
    "Las Vegas Raiders": "rai",
    "San Francisco 49ers": "sfo",
    "New York Giants": "nyg",
    "Los Angeles Chargers": "lac", 
    "Green Bay Packers": "gnb",
    "Los Angeles Rams": "ram"
}

# Core coaching experience features
CORE_COACHING_FEATURES = [
    "age",
    "num_times_hc",
    "num_yr_col_pos",
    "num_yr_col_coor",
    "num_yr_col_hc",
    "num_yr_nfl_pos",
    "num_yr_nfl_coor",
    "num_yr_nfl_hc"
]

# Base team statistics that get role-specific suffixes
BASE_TEAM_STATISTICS = [
    "PF (Points For)",
    "Yds",
    "Y/P",
    "TO",
    "1stD",
    "Cmp Passing",
    "Att Passing",
    "Yds Passing",
    "TD Passing",
    "Int Passing",
    "NY/A Passing",
    "1stD Passing",
    "Att Rushing",
    "Yds Rushing",
    "TD Rushing",
    "Y/A Rushing",
    "1stD Rushing",
    "Pen",
    "Yds Penalties",
    "1stPy",
    "#Dr",
    "Sc%",
    "TO%",
    "Time Average Drive",
    "Plays Average Drive",
    "Yds Average Drive",
    "Pts Average Drive",
    "3DAtt",
    "3D%",
    "4DAtt",
    "4D%",
    "RZAtt",
    "RZPct"
]

# Role suffixes for team statistics.
# Canonical layout (Jun 2026): ALL FOUR parallel "your-unit" performance blocks
# are pooled into ONE oriented, season-weighted "__unit" block in create_data
# (positive = good unit, see unit_stat_sign):
#   OC offense (team)        -> +sign
#   HC offense (team)        -> +sign
#   DC defense (opp allowed) -> -sign   (low allowed = good defense)
#   HC defense (opp allowed) -> -sign   (the team's defense under the HC; PFR's
#                                        "Opponent" table is a team's defensive ledger)
# An HC season contributes BOTH sides (he owns offense and defense); an OC only
# offense, a DC only defense. The hiring-team CONTEXT block (HIRING_TEAM_FEATURES)
# stays separate -- it is the inherited team's state, not the coach's performance.
# Legacy per-role suffixes retained only for the archived 4-block dataset.
ROLE_SUFFIXES = {
    "unit": "__unit",
}
LEGACY_ROLE_SUFFIXES = {
    "offensive_coordinator": "__oc",
    "defensive_coordinator": "__dc",
    "head_coach": "__hc",
    "head_coach_opponent": "__opp__hc",
}

# Stats where a HIGHER value is WORSE for the unit (giveaways and penalties).
# Used to orient the pooled unit block so that positive always means "good unit".
UNIT_NEGATIVE_STAT_TOKENS = ("TO", "Int", "Pen")


def unit_stat_sign(stat: str) -> float:
    """+1.0 if a higher value of ``stat`` is good for the unit, -1.0 if bad.

    Turnovers (TO, TO%), interceptions thrown (Int Passing) and penalties (Pen,
    Yds Penalties) are negatively oriented; everything else is positive.
    """
    return -1.0 if any(tok in stat for tok in UNIT_NEGATIVE_STAT_TOKENS) else 1.0

# Hiring team context features
HIRING_TEAM_FEATURES = [
    "hiring_team_win_pct",
    "hiring_team_points_scored", 
    "hiring_team_points_allowed",
    "hiring_team_yards_offense",
    "hiring_team_yards_allowed",
    "hiring_team_yards_per_play",
    "hiring_team_yards_per_play_allowed",
    "hiring_team_turnovers_forced",
    "hiring_team_turnovers_committed",
    "hiring_team_num_playoff_appearances"
]

# Words/phrases that exclude a coaching role from consideration
EXCLUDED_ROLE_KEYWORDS = [
    "Consultant",
    "Scout", 
    "Analyst",
    "Athletic Director",
    "Advisor",
    "Intern",
    "Sports Science",
    "Quality Control",
    "Emeritus",
    "Freshman ",
    "/Freshman",
    "Passing Game Coordinator",
    "Pass Gm. Coord.",
    "Recruiting",
    "Reserve",
    "earnings",
    "Strength and Conditioning",
    "Strength & Conditioning", 
    "Video",
    "Senior Assistant",
    "Associate Head Coach"
]

# Coaching tenure classification thresholds
TENURE_CLASSIFICATIONS = {
    "short": (0, 2),      # 0-2 years
    "medium": (3, 4),     # 3-4 years  
    "long": (5, float('inf'))  # 5+ years
}

# Data file configurations
DATA_FILES = {
    "coach_tables": ["all_coaching_results", "all_coaching_ranks", "all_coaching_history"],
    "league_tables": ["league_team_data_normalized", "league_opponent_data_normalized"],
    "team_record": "team_record.csv",
    "team_playoff": "team_playoff_record.csv"
}

# Current analysis parameters
# expected_feature_count: 2 (Coach Name, Year) + 41 (8 core + 33 unit) + 10
# hiring-team + 1 (Avg 2Y Win Pct) + 1 (Coach Tenure Class) = 55.
# (Was 154 under the legacy 4-block, 150-stat layout.)
ANALYSIS_CONFIG = {
    "cutoff_year": 2022,
    "current_year": 2025,
    "expected_feature_count": 55,
    "hiring_context_years": [1, 2]  # Look back 1-2 years for team context
}

# Coaches who were fired/resigned and should get actual tenure classification
# (not marked as -1 for insufficient data)
FIRED_COACHES = [
    "Doug Pederson",
    "Frank Reich",
    "Antonio Pierce",
    "Jerod Mayo"
]

# Coach-year hiring instances to exclude from final dataset
# Format: (coach_name, hire_year) tuples
EXCLUDED_HIRING_INSTANCES = [
    ("Sean Payton", 2013)  # Interim/temporary hire that should not be included in analysis
]


def get_all_feature_names() -> List[str]:
    """Complete feature-name list in canonical order (collapsed unit-block layout).

    Core (8) + pooled unit block (33, ``__unit``) = 41 names. The single unit
    block replaces all four legacy parallel blocks (OC ``__oc``, DC ``__dc``,
    HC-team ``__hc``, HC-opp ``__opp__hc``); see ``unit_stat_sign`` and
    create_data._load_league_data for the per-side orientation. The 10
    hiring-team CONTEXT features are appended separately downstream
    (get_output_column_names) since they describe the inherited team, not the
    coach's own unit performance.
    """
    # Core coaching experience features (Features 1-8)
    core_features = [
        "age",
        "num_times_hc",
        "num_yr_col_pos",
        "num_yr_col_coor",
        "num_yr_col_hc",
        "num_yr_nfl_pos",
        "num_yr_nfl_coor",
        "num_yr_nfl_hc"
    ]

    # Pooled, orientation-corrected unit performance block (Features 9-41)
    unit_features = [f"{stat}__unit" for stat in BASE_TEAM_STATISTICS]

    return core_features + unit_features


def get_feature_dict() -> Dict[str, Union[int, List]]:
    """Create feature dictionary with appropriate default values.

    Core features accumulate as integer counters; the unit and HC-opponent
    statistics accumulate as per-season lists (later reduced by ``_safe_mean``,
    which pools all of a coach's role-seasons and drops missing-league seasons).
    """
    feature_dict = {}

    # Core features start as integers
    for feature in CORE_COACHING_FEATURES:
        feature_dict[feature] = 0

    # Pooled unit block + HC-opponent block start as lists
    for stat in BASE_TEAM_STATISTICS:
        for suffix in ROLE_SUFFIXES.values():
            feature_dict[f"{stat}{suffix}"] = []

    return feature_dict


def get_hiring_team_stat_dict() -> Dict[str, List]:
    """Create hiring team statistics dictionary"""
    return {feature: [] for feature in HIRING_TEAM_FEATURES}


def get_output_column_names() -> List[str]:
    """Generate output CSV column names"""
    columns = ['Coach Name', 'Year']

    # Add numbered feature columns
    total_features = len(get_all_feature_names()) + len(HIRING_TEAM_FEATURES)
    for i in range(1, total_features + 1):
        columns.append(f'Feature {i}')

    columns.extend(['Avg 2Y Win Pct', 'Coach Tenure Class'])
    return columns


def standardize_team_abbreviation(team: str, year: Optional[int] = None) -> str:
    """
    Standardize a team abbreviation to a canonical PFR franchise key, resolving
    historical relocations and abbreviations that meant different franchises in
    different eras (BAL, HOU, STL).

    Mirrors Coach_WAR/crawlers/utils/data_constants.py so the two projects
    identify franchises identically. The returned key uniquely identifies the
    franchise (e.g. Baltimore Colts and Indianapolis Colts both -> 'clt';
    Baltimore Ravens -> 'rav'; Houston Oilers and Tennessee Titans -> 'oti').

    Examples:
        standardize_team_abbreviation('BAL', 1975) -> 'clt'  # Baltimore Colts
        standardize_team_abbreviation('BAL', 2000) -> 'rav'  # Baltimore Ravens
        standardize_team_abbreviation('HOU', 1990) -> 'oti'  # Houston Oilers
        standardize_team_abbreviation('HOU', 2010) -> 'htx'  # Houston Texans
    """
    if not team or pd.isna(team):
        return team

    team = str(team).upper().strip()

    # Year-based franchise changes take priority (shared abbreviation, two teams)
    if year is not None:
        if team == 'BAL':
            return 'clt' if year <= 1983 else 'rav'  # Colts (->Indy) vs Ravens
        elif team == 'HOU':
            return 'oti' if year <= 1996 else 'htx'  # Oilers (->Tenn) vs Texans
        elif team == 'STL':
            return 'crd' if year <= 1987 else 'ram'  # Cardinals (->AZ) vs Rams

    # Relocations/renames without era-based ambiguity
    standard_mappings = {
        'ARI': 'crd', 'IND': 'clt', 'LAC': 'sdg', 'LAR': 'ram', 'LVR': 'rai',
        'LV': 'rai', 'OAK': 'rai', 'PHO': 'crd', 'TEN': 'oti', 'BOS': 'nwe',
        'GB': 'gnb', 'KC': 'kan', 'NE': 'nwe', 'NO': 'nor', 'SF': 'sfo',
        'TB': 'tam',
    }
    if team in standard_mappings:
        return standard_mappings[team]

    # Unmapped abbreviations (incl. PFR canon like 'rai','ram','sdg') -> lowercase
    return team.lower()


# Model file paths (relative to project root)
MODEL_PATHS = {
    'data_dir': 'data',
    'data_file': 'data/svd_imputed_master_data.csv',
    'raw_data_file': 'data/master_data.csv',
    'models_dir': 'data/models',
    'default_model_output': 'data/models/coach_tenure_model.pkl',
    'ordinal_model_output': 'data/models/coach_tenure_ordinal_model.pkl',
    'multiclass_model_output': 'data/models/coach_tenure_multiclass_model.pkl'
}

# Ordinal classification configuration
ORDINAL_CONFIG = {
    'n_classes': 3,
    'class_names': [
        'Class 0 (1-2 yrs)',
        'Class 1 (3-4 yrs)',
        'Class 2 (5+ yrs)'
    ],
    'class_thresholds': {
        0: (1, 2),   # 1-2 years tenure
        1: (3, 4),   # 3-4 years tenure
        2: (5, None) # 5+ years tenure
    }
}