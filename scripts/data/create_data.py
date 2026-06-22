import os
import sys
import pandas as pd
import math
from pathlib import Path
from numpy import mean, nan, nanmean
from typing import List, Dict, Tuple, Union, Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data_constants import (
    TEAM_FRANCHISE_MAPPINGS,
    standardize_team_abbreviation,
    CURRENT_TEAM_ABBREVIATIONS,
    EXCLUDED_ROLE_KEYWORDS,
    TENURE_CLASSIFICATIONS,
    DATA_FILES,
    ANALYSIS_CONFIG,
    FIRED_COACHES,
    EXCLUDED_HIRING_INSTANCES,
    get_all_feature_names,
    get_feature_dict,
    get_hiring_team_stat_dict,
    get_output_column_names,
    unit_stat_sign,
)


class CoachingDataProcessor:
    """Processes NFL coaching career data into feature-engineered datasets"""

    def __init__(self, coaches_dir: str = None, teams_dir: str = None,
                 league_dir: str = None):
        """Initialize with data directory paths (relative to project root)"""
        self.coaches_dir = Path(coaches_dir) if coaches_dir else project_root / "Coaches"
        self.teams_dir = Path(teams_dir) if teams_dir else project_root / "Teams"
        self.league_dir = Path(league_dir) if league_dir else project_root / "League Data"
        self._primary_idx = None  # lazily-loaded Coach_WAR Primary_Coach index
        
    def _classify_coach_tenure(self, years: int) -> int:
        """Classify coaching tenure into categories"""
        if years <= 0:
            return -1
        elif years <= TENURE_CLASSIFICATIONS["short"][1]:
            return 0
        elif years <= TENURE_CLASSIFICATIONS["medium"][1]:
            return 1
        else:
            return 2
    
    def _classify_coaching_role(self, role: str) -> str:
        """Classify coaching role, excluding interim and non-coaching positions"""
        if not role or not isinstance(role, str):
            return "None"
        
        # Check for excluded keywords
        for keyword in EXCLUDED_ROLE_KEYWORDS:
            if keyword in role:
                return "None"
        
        # Exclude generic assistant roles
        if ("Assistant" in role or "Asst" in role) and "/" not in role and "\\" not in role:
            return "None"
        
        # Classify specific roles
        if "Head Coach" in role and "Ass" not in role and "Interim" not in role:
            return "Head Coach"
        
        if "Coordinator" in role:
            if "Offensive Coordinator" in role and "Interim O" not in role:
                return "Offensive Coordinator"
            elif "Defensive Coordinator" in role and "Interim D" not in role:
                return "Defensive Coordinator"
            elif "Special" in role and "Interim S" not in role:
                return "Special Teams Coordinator"
            else:
                return "Position"
        
        return "Position"
    
    def _classify_coaching_level(self, level: str) -> str:
        """Classify coaching level (College, NFL, or None)"""
        if not level:
            return "None"
        if "College" in level:
            return "College"
        if "NFL" in level and level != "NFL Europe":
            return "NFL"
        return "None"
    
    def _safe_mean(self, values: Union[List, float, int]) -> float:
        """Calculate mean of list, handling NaN values and non-list inputs"""
        if not isinstance(values, list):
            return values
        
        clean_values = [x for x in values if not (isinstance(x, float) and math.isnan(x))]
        if len(clean_values) == 0:
            return nan
        return nanmean(clean_values)
    
    def _get_team_abbreviation(self, team_name: str) -> str:
        """Get team abbreviation for current team names"""
        return CURRENT_TEAM_ABBREVIATIONS.get(team_name, team_name.lower()[:3])
    
    def _franchise_file_candidates(self, team_abbrev: str, year: int = None) -> List[str]:
        """Candidate data-file abbreviations for a team-season, tried in order
        against that season's files until one matches.

        The YEAR-AWARE canonical key (standardize_team_abbreviation) is tried FIRST,
        because a flat lookup alone is NOT a safe file locator for an abbreviation
        that historically meant two different franchises: the league files store
        each team under its PFR canonical abbreviation, and the wrong era's abbr is
        a real-but-different team that would otherwise match. Concretely 'stl' is
        the Cardinals ('crd') before 1988 but the Rams ('ram') from 1995, and 'bos'
        is the 1970 Patriots ('nwe') not the Braves/Redskins->Washington lineage;
        without the year, 'stl' would fall through to 'ram' and 'bos' to 'was'. The
        flat TEAM_FRANCHISE_MAPPINGS is kept only as a fallback set of legacy file
        spellings.
        """
        cands = []
        if year is not None:
            canon = standardize_team_abbreviation(team_abbrev, year)
            if canon:
                cands.append(canon)
        franchise_list = TEAM_FRANCHISE_MAPPINGS.get(team_abbrev, team_abbrev)
        flat = franchise_list if isinstance(franchise_list, list) else [franchise_list]
        for a in flat:
            if a not in cands:
                cands.append(a)
        return cands
    
    def _load_league_data(self, year: int, team_list: List[str], role: str, 
                         feature_dict: Dict) -> int:
        """Load and process league data for a specific year and role"""
        
        year_dir = self.league_dir / str(year)
        if not year_dir.exists():
            return 0
        
        try:
            # Load team and opponent data
            team_file = year_dir / f"{DATA_FILES['league_tables'][0]}.csv"
            opponent_file = year_dir / f"{DATA_FILES['league_tables'][1]}.csv"
            
            team_df = pd.read_csv(team_file)
            opponent_df = pd.read_csv(opponent_file)
            
        except FileNotFoundError:
            return 0
        
        # Find team in data
        team_found = False
        team_index = 0
        
        for i, team_abbrev in enumerate(team_list):
            team_row = team_df[team_df['Team Abbreviation'] == team_abbrev]
            if not team_row.empty:
                team_found = True
                team_index = i
                break
        
        if not team_found:
            print(f"Error: {team_list[0]} not found in year {year}")
            return -1
        
        # Get corresponding opponent row
        team_abbrev = team_list[team_index]
        team_row = team_df[team_df['Team Abbreviation'] == team_abbrev]
        opponent_row = opponent_df[opponent_df['Team Abbreviation'] == team_abbrev]
        
        # Process data based on role -- pool every "your-unit" role-season-SIDE into
        # the single oriented __unit block (positive = good unit). All four legacy
        # parallel blocks are merged here:
        #   OC season -> +sign * team offense
        #   HC season -> +sign * team offense  AND  -sign * opponent allowed (defense)
        #   DC season -> -sign * opponent offense allowed  (low allowed = good defense)
        # An HC season therefore contributes BOTH sides; OC offense only; DC defense
        # only. _safe_mean later averages across all pooled role-season-sides,
        # dropping any whose league value is missing (the exact season-weighted pool).
        def _add_offense(row):
            for col in row.columns:
                if col != 'Team Abbreviation':
                    key = f"{col}__unit"
                    if key in feature_dict:
                        feature_dict[key].append(unit_stat_sign(col) * row[col].values[0])

        def _add_defense(row):
            for col in row.columns:
                if col != 'Team Abbreviation':
                    key = f"{col}__unit"
                    if key in feature_dict:
                        feature_dict[key].append(-unit_stat_sign(col) * row[col].values[0])

        if role == "HC":
            _add_offense(team_row)       # team's offense under the HC
            _add_defense(opponent_row)   # team's defense under the HC (points allowed)
        elif role == "OC":
            _add_offense(team_row)
        elif role == "DC":
            _add_defense(opponent_row)

        return team_index
    
    def _get_hiring_team_context(self, franchise: str, hire_year: int) -> List[float]:
        """Get hiring team's recent performance context"""
        
        team_dir = self.teams_dir / franchise
        if not team_dir.exists():
            return [nan] * len(get_hiring_team_stat_dict())
        
        # Load team record
        record_file = team_dir / DATA_FILES['team_record']
        if not record_file.exists():
            return [nan] * len(get_hiring_team_stat_dict())
        
        team_record = pd.read_csv(record_file)
        feature_dict = get_hiring_team_stat_dict()
        
        # Look at previous years
        for prev_year in [hire_year - 1, hire_year - 2]:
            year_records = team_record[team_record['Year'] == prev_year]
            if year_records.empty:
                continue
            
            record = year_records.iloc[0]
            wins = record['W']
            losses = record['L'] 
            ties = record.get('T', 0)
            
            # Calculate win percentage
            total_games = wins + losses + ties
            win_pct = (wins + 0.5 * ties) / total_games if total_games > 0 else 0
            feature_dict["hiring_team_win_pct"].append(win_pct)
            
            # Get league performance data
            year_dir = self.league_dir / str(prev_year)
            if year_dir.exists():
                try:
                    team_file = year_dir / f"{DATA_FILES['league_tables'][0]}.csv"
                    opponent_file = year_dir / f"{DATA_FILES['league_tables'][1]}.csv"
                    
                    team_df = pd.read_csv(team_file)
                    opponent_df = pd.read_csv(opponent_file)
                    
                    team_row = team_df[team_df['Team Abbreviation'] == franchise]
                    opponent_row = opponent_df[opponent_df['Team Abbreviation'] == franchise]
                    
                    if not team_row.empty and not opponent_row.empty:
                        feature_dict["hiring_team_points_scored"].append(team_row['PF (Points For)'].values[0])
                        feature_dict["hiring_team_points_allowed"].append(opponent_row['PF (Points For)'].values[0])
                        feature_dict["hiring_team_yards_offense"].append(team_row['Yds'].values[0])
                        feature_dict["hiring_team_yards_allowed"].append(opponent_row['Yds'].values[0])
                        feature_dict["hiring_team_yards_per_play"].append(team_row['Y/P'].values[0])
                        feature_dict["hiring_team_yards_per_play_allowed"].append(opponent_row['Y/P'].values[0])
                        feature_dict["hiring_team_turnovers_forced"].append(opponent_row['TO'].values[0])
                        feature_dict["hiring_team_turnovers_committed"].append(team_row['TO'].values[0])
                        
                except FileNotFoundError:
                    continue
        
        # Check playoff appearances
        playoff_file = team_dir / DATA_FILES['team_playoff']
        if playoff_file.exists():
            playoff_df = pd.read_csv(playoff_file)
            for prev_year in [hire_year - 1, hire_year - 2]:
                playoff_games = playoff_df[playoff_df['Year'] == prev_year]
                if prev_year == hire_year - 1:
                    feature_dict["hiring_team_num_playoff_appearances"].append(1 if not playoff_games.empty else 0)
                elif not feature_dict["hiring_team_num_playoff_appearances"]:
                    feature_dict["hiring_team_num_playoff_appearances"].append(1 if not playoff_games.empty else 0)
                elif not playoff_games.empty:
                    feature_dict["hiring_team_num_playoff_appearances"][0] += 1
        
        return [self._safe_mean(values) for values in feature_dict.values()]
    
    def _dedupe_career_rows(self, history_df: pd.DataFrame) -> pd.DataFrame:
        """Collapse PFR sub-role duplicate rows so each season is counted once.

        PFR coaching histories sometimes list the SAME season as two rows with
        different role text (e.g. 'Defensive Coordinator' and 'Defensive
        Coordinator/Linebackers', or a literally duplicated 'Offensive
        Coordinator'). Because the experience/performance loop runs PER ROW, those
        rows would double-count the season's experience counter and double-weight
        its pooled performance. Keep one row per (Year, classified level, classified
        role) -- a genuine same-year role change (e.g. coordinator promoted to
        in-season head coach) classifies to a DIFFERENT role and is preserved. Rows
        that classify to 'None' never increment a counter, so they are all kept to
        leave the demotion/sequence logic untouched.
        """
        if history_df.empty:
            return history_df
        seen = set()
        keep_idx = []
        for idx, row in history_df.iterrows():
            role = self._classify_coaching_role(row.get('Role', ''))
            if role == "None":
                keep_idx.append(idx)
                continue
            key = (row.get('Year'), self._classify_coaching_level(row.get('Level', '')), role)
            if key in seen:
                continue
            seen.add(key)
            keep_idx.append(idx)
        return history_df.loc[keep_idx]

    def _classify_hire(self, coach_name: str, hire_year: int):
        """Resolve interim caretakers / interim->permanent promotions for a detected
        hire year, returning ('clean'|'interim_only'|'reanchor', anchored_year).

        Reuses the Coach_WAR Primary_Coach resolution (engineer_career_features) so
        the single source of "what is a real hire" lives in one place. A coach absent
        from that table comes back 'clean' (kept as detected).
        """
        from scripts.data.engineer_career_features import (
            classify_hire_instance, _primary_coach_index, MIN_HIRE_YEAR)
        # Pre-1970 hires are removed by the downstream era filter and are NOT covered
        # by the Coach_WAR Primary_Coach table. Classifying them treats the
        # pre-coverage season as "not primary" and would spuriously re-anchor a clean
        # pre-1970 hire forward INTO the modeled window (e.g. Chuck Noll 1969 -> 1970).
        # Mirror engineer's filter-then-classify order by keeping them as detected.
        if hire_year < MIN_HIRE_YEAR:
            return "clean", hire_year
        if self._primary_idx is None:
            try:
                self._primary_idx = _primary_coach_index()
            except Exception:
                self._primary_idx = {}
        return classify_hire_instance(self._primary_idx, coach_name, hire_year)

    def _finalize_prev_tenure(self, career_instances, year, last_hc_year):
        """When a new hire begins, append the tenure class to the previous still-open
        instance (measured to this season, gap-aware). This is a placeholder that
        engineer recomputes relocation/partial-season-aware for known hires; it only
        ensures the previous instance reaches full length before the next is added."""
        if career_instances and len(career_instances[-1]) == ANALYSIS_CONFIG['expected_feature_count'] - 1:
            prev_instance = career_instances[-1]
            prev_hire_year = prev_instance[1]
            if last_hc_year is not None and year - last_hc_year > 1:
                tenure_years = (last_hc_year + 1) - prev_hire_year
            else:
                tenure_years = year - prev_hire_year
            prev_instance.append(self._classify_coach_tenure(tenure_years))

    def _emit_hire_instance(self, coach_name, hire_year, feature_dict,
                            franchise_list, results_df, career_instances):
        """Snapshot the current cumulative experience/performance as a hiring instance
        anchored at hire_year (prior experience only -- the hire season's own
        performance is NOT yet pooled). feature_dict['age'] must already be set to the
        age at hire_year. For a re-anchored interim->permanent hire the caller folds
        the interim partial HC season into feature_dict BEFORE calling, so it lands in
        prior experience; hire_year is then the season-opening year."""
        # 2-year winning percentage from the (anchored) hire year forward.
        win_results = []
        for result_year in [hire_year, hire_year + 1]:
            if not results_df.empty:
                result_rows = results_df[results_df['Year'] == result_year]
                if not result_rows.empty:
                    result = result_rows.iloc[0]
                    wins = result.get('W', 0)
                    losses = result.get('L', 0)
                    ties = result.get('T', 0)
                    total = wins + losses + ties
                    if total > 0:
                        win_results.append((wins + 0.5 * ties) / total)

        # Snapshot features in canonical order.
        new_instance = [coach_name, hire_year]
        for feature_name in get_all_feature_names():
            if feature_name in feature_dict:
                new_instance.append(self._safe_mean(feature_dict[feature_name]))
            else:
                new_instance.append(0)

        # Hiring-team context as-of the (anchored) hire year.
        team_index = 0
        for i, team_abbrev in enumerate(franchise_list):
            year_dir = self.league_dir / str(hire_year)
            if year_dir.exists():
                try:
                    team_df = pd.read_csv(year_dir / f"{DATA_FILES['league_tables'][0]}.csv")
                    if not team_df[team_df['Team Abbreviation'] == team_abbrev].empty:
                        team_index = i
                        break
                except FileNotFoundError:
                    continue
        new_instance.extend(self._get_hiring_team_context(franchise_list[team_index], hire_year))

        # 2-year win pct (recent-hire sentinel when fewer than two seasons exist).
        if hire_year >= 2024 and len(win_results) < 2:
            new_instance.append(-1.0 if len(win_results) == 0 else win_results[0])
        else:
            new_instance.append(self._safe_mean(win_results))

        if (coach_name, hire_year) not in EXCLUDED_HIRING_INSTANCES:
            career_instances.append(new_instance)
            feature_dict["num_times_hc"] += 1
        else:
            print(f"\tExcluded hiring instance: {coach_name} ({hire_year})")

    def _process_coach_career(self, coach_name: str) -> List[List]:
        """Process a single coach's career into training instances"""
        
        coach_dir = self.coaches_dir / coach_name
        if not coach_dir.exists():
            return []
        
        # Load coach data files
        dfs = []
        for table_name in DATA_FILES['coach_tables']:
            file_path = coach_dir / f"{table_name}.csv"
            if file_path.exists():
                dfs.append(pd.read_csv(file_path))
            else:
                dfs.append(pd.DataFrame())
        
        if len(dfs) < 3 or dfs[2].empty:  # Need coaching history
            return []

        results_df, ranks_df, history_df = dfs
        # Collapse PFR sub-role duplicate rows so each season is counted once.
        history_df = self._dedupe_career_rows(history_df)

        # Track coaching progression
        career_instances = []
        feature_dict = get_feature_dict()
        is_head_coach = False
        prev_year_check = None
        last_hc_year = None  # Track last year as head coach for gap detection
        last_hc_key = None   # Canonical (year-aware) franchise of the last HC season
        last_hc_employer = None  # Employer name of the last HC season (ranks-Tm backup)
        
        # Process each year of coaching history
        for _, row in history_df.iterrows():
            year = row['Year']
            level = self._classify_coaching_level(row.get('Level', ''))
            role = self._classify_coaching_role(row.get('Role', ''))
            age = row.get('Age', 0)
            
            # Handle demotion from head coach
            if is_head_coach and (level != 'NFL' or role != "Head Coach"):
                if career_instances:
                    # Calculate tenure for previous instance
                    prev_instance = career_instances[-1]
                    prev_hire_year = prev_instance[1]
                    tenure_years = (prev_year_check + 1) - prev_hire_year if prev_year_check else year - prev_hire_year
                    prev_instance.append(self._classify_coach_tenure(tenure_years))
                is_head_coach = False
            
            # Update experience counters
            if level == "College":
                if role == "Position":
                    feature_dict["num_yr_col_pos"] += 1
                elif role in ["Offensive Coordinator", "Defensive Coordinator", "Special Teams Coordinator"]:
                    feature_dict["num_yr_col_coor"] += 1
                elif role == "Head Coach":
                    feature_dict["num_yr_col_hc"] += 1
                    
            elif level == "NFL":
                if role == "Position":
                    feature_dict["num_yr_nfl_pos"] += 1
                elif role != "None":
                    # Count NFL coordinator experience from HISTORY, regardless of
                    # whether PFR ranked the unit that season. Special-teams
                    # coordinators are never in the ranks table (PFR ranks only
                    # offense/defense), and some early OC/DC seasons predate it, so
                    # gating the counter on ranks availability silently dropped real
                    # experience (e.g. John Harbaugh's 9 STC years -> 0). This mirrors
                    # num_yr_nfl_pos above; performance is still loaded only when the
                    # ranked unit can actually be located (inside the block below).
                    if role in ["Offensive Coordinator", "Defensive Coordinator", "Special Teams Coordinator"]:
                        feature_dict["num_yr_nfl_coor"] += 1

                    # Get team abbreviation from ranks data or derive for current year hires
                    franchise = None
                    franchise_list = None
                    
                    if not ranks_df.empty:
                        team_rows = ranks_df[ranks_df['Year'] == year]
                        if not team_rows.empty:
                            franchise = team_rows.iloc[0]['Tm'].lower()
                            franchise_list = self._franchise_file_candidates(franchise, year)

                    # For current year head coach hires without ranking data, derive team from role
                    # This applies to NEW hires and team changes in current year
                    if (franchise is None and role == "Head Coach" and
                        year >= ANALYSIS_CONFIG['current_year']):
                        # Parse team from history data for current year hires/moves
                        employer = row.get('Employer', '')
                        if employer:
                            current_franchise = self._get_team_abbreviation(employer)

                            # Only set the franchise for a genuine new hire / team change,
                            # judged by the YEAR-AWARE canonical key (not the flat map): a
                            # current-year HC row that continues last season at the same
                            # franchise is not a new instance.
                            cur_key = standardize_team_abbreviation(current_franchise, year)
                            if not (last_hc_year is not None and year - last_hc_year <= 1
                                    and cur_key == last_hc_key):
                                franchise = current_franchise
                                franchise_list = self._franchise_file_candidates(current_franchise, year)
                    
                    if franchise_list:

                        # Load performance data for ranked roles (the coordinator
                        # counter was already incremented above, independent of ranks).
                        if role in ["Offensive Coordinator", "Defensive Coordinator", "Special Teams Coordinator"]:
                            # Load performance data
                            if role == "Offensive Coordinator":
                                self._load_league_data(year, franchise_list, "OC", feature_dict)
                            elif role == "Defensive Coordinator":
                                self._load_league_data(year, franchise_list, "DC", feature_dict)
                                
                        elif role == "Head Coach":
                            # A new hire is a HC season that does NOT continue an
                            # immediately-prior HC season at the same franchise. Continuity
                            # is judged two ways, either of which suffices: (a) the YEAR-AWARE
                            # canonical franchise key, which absorbs relocations and renames
                            # whose employer NAME changes (Houston Oilers -> Tennessee Titans,
                            # Baltimore -> Indianapolis Colts); and (b) the head-coaching
                            # row's employer NAME, which absorbs a mid-season firing whose
                            # ranks 'Tm' points at the coach's next club rather than the team
                            # he led (Arnsparger's 1976 Tm reads Miami though he was the
                            # Giants' HC). Together they also collapse the spurious same-year
                            # demotion-then-rehire fragment a mid-season exit creates.
                            cur_key = standardize_team_abbreviation(franchise, year)
                            hc_employer = str(row.get('Employer', '')).strip()
                            is_continuation = (
                                last_hc_year is not None and
                                year - last_hc_year <= 1 and
                                (cur_key == last_hc_key or
                                 (hc_employer != '' and hc_employer == last_hc_employer)))
                            is_new_hire = not is_continuation

                            folded_interim = False
                            if is_new_hire:
                                # Resolve interim caretakers and interim->permanent
                                # promotions up front (Coach_WAR Primary_Coach), so the
                                # modeled hire is the season-opening year and the interim
                                # audition season is folded into prior experience rather
                                # than dropped or stored as a separate instance.
                                kind, anchor = self._classify_hire(coach_name, year)

                                if kind == "interim_only":
                                    # Mid-season caretaker who never opened a season as the
                                    # primary HC -- not a real hire, so emit no instance. The
                                    # season's experience/performance still accrues below.
                                    pass
                                elif kind == "reanchor" and anchor == year + 1:
                                    # Interim takeover kept on as the permanent coach the next
                                    # season. Fold THIS partial HC season into prior experience
                                    # (its unit performance + the HC-year count) and anchor the
                                    # modeled hire to the season-opening year, with age and
                                    # hiring-team context taken as-of that anchored year.
                                    self._load_league_data(year, franchise_list, "HC", feature_dict)
                                    feature_dict["num_yr_nfl_hc"] += 1
                                    folded_interim = True
                                    is_head_coach = True
                                    feature_dict["age"] = age + 1
                                    self._finalize_prev_tenure(career_instances, year, last_hc_year)
                                    self._emit_hire_instance(coach_name, anchor, feature_dict,
                                                             franchise_list, results_df, career_instances)
                                else:
                                    # Clean new hire: instance as-of this season, prior
                                    # (pre-hire) experience only.
                                    is_head_coach = True
                                    feature_dict["age"] = age
                                    self._finalize_prev_tenure(career_instances, year, last_hc_year)
                                    self._emit_hire_instance(coach_name, year, feature_dict,
                                                             franchise_list, results_df, career_instances)

                            # Accrue this season's HC performance for FUTURE hiring instances,
                            # unless it was already folded into a re-anchored hire above
                            # (avoid double-counting the interim audition season).
                            if not folded_interim:
                                self._load_league_data(year, franchise_list, "HC", feature_dict)
                                feature_dict["num_yr_nfl_hc"] += 1

                            # Update continuity trackers (all HC seasons, not just new hires).
                            last_hc_year = year
                            last_hc_key = cur_key
                            last_hc_employer = hc_employer
            
            prev_year_check = year
        
        # Handle final instance
        if career_instances and len(career_instances[-1]) == ANALYSIS_CONFIG['expected_feature_count'] - 1:
            final_instance = career_instances[-1]
            hire_year = final_instance[1]
            
            # Determine tenure classification
            if hire_year > ANALYSIS_CONFIG['current_year']:
                # Future hire - exclude completely
                career_instances.pop()
                print(f'\tExcluded future hire: {coach_name}, hire year: {hire_year}')
            elif hire_year >= ANALYSIS_CONFIG['cutoff_year']:
                # Recent hire - check if still active or was fired
                # Special handling for known fired coaches vs still active
                # For coaches hired 2022+, check if we have evidence they were fired
                if coach_name in FIRED_COACHES or prev_year_check < ANALYSIS_CONFIG['current_year'] - 1:
                    # Was fired/resigned - we know the actual tenure
                    tenure_years = (prev_year_check + 1) - hire_year
                    final_instance.append(self._classify_coach_tenure(tenure_years))
                    print(f'\tCalculated tenure for fired coach: {coach_name}, tenure: {tenure_years} years')
                else:
                    # Still active - insufficient data for final classification
                    final_instance.append(-1)
                    print(f'\tExcluded tenure classification: {coach_name}')
            else:
                # Calculate final tenure
                tenure_years = (prev_year_check + 1) - hire_year
                final_instance.append(self._classify_coach_tenure(tenure_years))
        
        # Validate instance lengths
        for i, instance in enumerate(career_instances):
            if len(instance) != ANALYSIS_CONFIG['expected_feature_count']:
                print(f'Error: {coach_name}, instance {i}, length {len(instance)} != {ANALYSIS_CONFIG["expected_feature_count"]}')
        
        return career_instances
    
    def process_all_coaches(self) -> pd.DataFrame:
        """Process all coaches and return combined dataset"""
        
        coach_dirs = [d for d in self.coaches_dir.iterdir() if d.is_dir()]
        all_instances = []
        
        print(f"Processing {len(coach_dirs)} coaches...")
        
        for i, coach_dir in enumerate(coach_dirs, 1):
            coach_name = coach_dir.name
            print(f"Parsing coach {i}: {coach_name}")
            
            coach_instances = self._process_coach_career(coach_name)
            all_instances.extend(coach_instances)
            
            
            
        
        # Create DataFrame
        columns = get_output_column_names()
        df = pd.DataFrame(data=all_instances, columns=columns)
        
        
        print(f'Processed {len(all_instances)} coaching instances')
        return df


def main():
    """Build the single canonical modeling dataset.

    Constructs the all-era base hiring instances in memory, then engineers them into
    the final modern-era (1970+) modeling table written to data/master_data.csv.
    There is no intermediate/_extended file: the feature engineering is a function
    call (engineer_career_features.build_modeling_dataset), so there is ONE builder
    and ONE output file.
    """
    print("Starting Coaching Data Processing...")

    processor = CoachingDataProcessor()
    base_df = processor.process_all_coaches()          # all-era base instances (in-memory)

    # Lazy import avoids an import cycle (engineer's survival-script consumers import
    # this module's classifiers indirectly); build the final modeling table.
    from scripts.data.engineer_career_features import build_modeling_dataset
    final_df = build_modeling_dataset(base_df)

    output_file = project_root / "data" / "master_data.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_file)

    print(f"\nProcessing completed!")
    print(f"Dataset saved to: {output_file}")
    print(f"Total modeling instances: {len(final_df)}")


if __name__ == "__main__":
    main()