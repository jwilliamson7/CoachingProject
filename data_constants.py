"""
Constants and configuration data for NFL coaching analysis.

This module contains all the static dictionaries, lists, and mappings
used throughout the coaching data analysis pipeline.
"""

from typing import Dict, List, Union


# Team franchise abbreviation mappings for historical name changes and relocations
TEAM_FRANCHISE_MAPPINGS = {
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

# Role suffixes for team statistics
ROLE_SUFFIXES = {
    "offensive_coordinator": "__oc",
    "defensive_coordinator": "__dc", 
    "head_coach": "__hc",
    "head_coach_opponent": "__opp__hc"
}

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
ANALYSIS_CONFIG = {
    "cutoff_year": 2022,
    "current_year": 2025,
    "expected_feature_count": 154,
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
    """Generate complete list of feature names in correct order matching Excel file"""
    
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
    
    # NFL OC team statistics (Features 9-41)
    oc_features = []
    for stat in BASE_TEAM_STATISTICS:
        oc_features.append(f"{stat}__oc")
    
    # NFL DC opponent statistics (Features 42-74) 
    dc_features = []
    for stat in BASE_TEAM_STATISTICS:
        dc_features.append(f"{stat}__dc")
    
    # NFL HC team statistics (Features 75-107)
    hc_features = []
    for stat in BASE_TEAM_STATISTICS:
        hc_features.append(f"{stat}__hc")
    
    # NFL HC opponent statistics (Features 108-140)
    hc_opp_features = []
    for stat in BASE_TEAM_STATISTICS:
        hc_opp_features.append(f"{stat}__opp__hc")
    
    # Combine all in exact Excel order
    feature_names = core_features + oc_features + dc_features + hc_features + hc_opp_features
    
    return feature_names


def get_feature_dict() -> Dict[str, Union[int, List]]:
    """Create feature dictionary with appropriate default values"""
    feature_dict = {}
    
    # Core features start as integers
    for feature in CORE_COACHING_FEATURES:
        feature_dict[feature] = 0
    
    # Team statistics features start as lists
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