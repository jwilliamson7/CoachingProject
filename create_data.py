import os
import pandas as pd
import math
import operator
from numpy import mean, nan
import re


def get_point_features():
    features = ['Coach Name', 'Year']
    for i in range(1, 25 + 1):
        features.append('Feature {}'.format(i))
    features.append('Avg 2Y Win Pct')
    features.append('Coach Tenure Class')
    return features


def get_team_dict():
    return {"ind":"clt", "ari":"crd", "hou":["htx", "oti"], "ten":"oti", "oak":"rai", 
            "stl":["ram", "sla", "gun"], "lar":"ram", "bal":["rav","clt"], "lac":"sdg", "chr":"cra",
            "frn":"fyj", "nyt":"nyj", "can":"cbd", "bos":["byk", "was", "ptb"], "pot":"ptb",
            "lad":"lda", "evn":"ecg", "pho":"crd", "prt":"det", "lvr":"rai",
            "chh":"cra", "htx":"oti", "buf":["buf", "bff", "bba"], "cle":["cle", "cti", "cib", "cli"],
            "min":["min", "mnn"], "kan":["kan", "kcb"], "det":["det", "dwl", "dpn", "dhr", "dti"], "nyy": ["nyy", "naa", "nya"],
            "cin":["cin", "red", "ccl"], "mia":["mia", "msa"], "was":["was", "sen"], "dtx":["kan", "dtx"],
            "nyg":["nyg", "ng1"], "cli":["cli", "cib"]
            }


def classify_coach_tenure(years):
    if years <= 0:
        return -1
    if years <= 2:
        return 0
    elif years > 2 and years <= 4:
        return 1
    elif years > 4 and years <= 7:
        return 2
    else:
        return 3


def classify_role(role):
    if role == "" or type(role) != type('string') or "Consultant" in role or "Scout" in role or "Analyst" in role or "Athletic Director" in role or "Advisor" in role or "Intern" in role or "Sports Science" in role:
        return "None"
    if ("Assistant" in role or "Asst" in role) and "/" not in role and "\\" not in role:
        return "None"
    if "Head Coach" in role and "Ass" not in role:
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


def get_norm_val(rank, out_of):
    return 1 - (rank - 1) / (out_of - 1)


def classify_level(level):
    if "College" in level:
        return "College"
    if "NFL" in level:
        return "NFL"
    return "None"


def quick_mean(lis):
    if len(lis) == 0:
        return nan
    else:
        return mean(lis)

def get_num_team_in_league(franchise, year, team_path):
    if not isinstance(franchise, list):
        franchise = [franchise]
    for single in franchise + []:
        file_path = team_path + "\\" + str(single) + "\\team_record.csv"
        df = pd.read_csv(file_path)
        result = df.loc[df['Year'] == year]['out of'].values
        if len(result) == 0:
            continue
        else:
            return int(result[0]), single
    return None, franchise

def get_hiring_team_stats(franchise, year, team_path):
    file_path = team_path + "\\" + str(franchise) + "\\"
    record_path = file_path + "team_record.csv"
    playoff_path = file_path + "team_playoff_record.csv"
    feature_dict = {
        "hiring_team_win_pct":          [],
        "hiring_team_norm_turnover":    [],
        "hiring_team_norm_point":       [],
        "hiring_team_norm_yard":        [],
        "hiring_team_norm_div":         [],
        "hiring_team_playoff_app":      [],
        "hiring_team_playoff_win":      []
    }
    
    regex = re.compile(r'[^\d.]+')
    df_record = pd.read_csv(record_path)
    for prev_year in [year - 1, year - 2]:
        record_row = df_record.loc[df_record['Year'] == prev_year]
        if record_row.shape[0] == 0:
            continue
        num_wins = record_row['W'].values[0]
        num_losses = record_row['L'].values[0]
        num_ties = record_row['T'].values[0]
        num_teams = record_row['out of'].values[0]
        raw_div_text = record_row['Div. Finish'].values[0]
        raw_div_text = raw_div_text.partition(' of ')
        div_place = int(regex.sub('', raw_div_text[0]))
        num_teams_in_division = int(raw_div_text[2])
        feature_dict["hiring_team_win_pct"].append((num_wins + .5 * num_ties) / (num_losses + num_ties + num_wins))
        feature_dict["hiring_team_norm_turnover"].append(get_norm_val(record_row['T/G'].values[0], num_teams))
        feature_dict["hiring_team_norm_point"].append(get_norm_val(record_row['Pts±'].values[0], num_teams))
        feature_dict["hiring_team_norm_yard"].append(get_norm_val(record_row['Yds±'].values[0], num_teams))
        feature_dict["hiring_team_norm_div"].append(get_norm_val(div_place, num_teams_in_division))
    
    
    if os.path.exists(playoff_path):
        df_playoff = pd.read_csv(playoff_path)
        for prev_year in [year - 1, year - 2]:
            record_rows = df_playoff.loc[df_playoff['Year'] == prev_year]
            num_games = record_rows.shape[0]
            if num_games == 0:
                feature_dict["hiring_team_playoff_app"].append(0)
                feature_dict["hiring_team_playoff_win"].append(0)
            else:
                feature_dict["hiring_team_playoff_app"].append(1)
                feature_dict["hiring_team_playoff_win"].append(sum(1 for game_result in record_rows.iloc[:,6].values if game_result == 'W'))

    return [quick_mean(value) for value in feature_dict.values()]


def parse_coach_career(coach_name, coach_path, team_path):
    team_dict = get_team_dict()
    rows = []
    table_names = ["all_coaching_results", "all_coaching_ranks", "all_coaching_history"]
    file_path = coach_path + "\\" + str(coach_name) + "\\"
    feature_dict = {
        "age":                  0,
        "num_times_hc":         0,
        "num_yr_col_pos":       0,
        "num_yr_col_coor":      0,
        "num_yr_col_hc":        0,
        "num_yr_nfl_pos":       0,
        "num_yr_nfl_coor":      0,
        "num_yr_nfl_hc":        0,
        "demotion_presence":    0,
        "nfl_oc_norm_yard":     [],
        "nfl_oc_norm_point":    [],
        "nfl_oc_norm_giveaway": [],
        "nfl_dc_norm_yard":     [],
        "nfl_dc_norm_point":    [],
        "nfl_dc_norm_turnover": [],
        "nfl_hc_norm_yard":     [],
        "nfl_hc_norm_point":    [],
        "nfl_hc_norm_turnover": []
    }
    feature_transform_dict = {
        "nfl_oc_norm_yard":     quick_mean,
        "nfl_oc_norm_point":    quick_mean,
        "nfl_oc_norm_giveaway": quick_mean,
        "nfl_dc_norm_yard":     quick_mean,
        "nfl_dc_norm_point":    quick_mean,
        "nfl_dc_norm_turnover": quick_mean,
        "nfl_hc_norm_yard":     quick_mean,
        "nfl_hc_norm_point":    quick_mean,
        "nfl_hc_norm_turnover": quick_mean
    }
    dfs = []
    for table_name in table_names:
        dfs.append(pd.read_csv(file_path + table_name + ".csv"))
    # For each year in their career
    prev_franchise_abrev = None
    is_head_coach = False
    previous_year_check = None
    year = 0
    for row in dfs[2].itertuples(index=False):
        level = classify_level(row[3])
        role = classify_role(row[5])
        year = row[1]
        
        if is_head_coach and (level == 'College' or role != "Head Coach"):
            is_head_coach = False
            feature_dict["demotion_presence"] = 1
            prev_year = rows[-1][1]
            math_year = previous_year_check + 1 if year != previous_year_check + 1 else year
            rows[-1].append(classify_coach_tenure(math_year - prev_year))

        if level == "College":
            if role == "Position":
                feature_dict["num_yr_col_pos"] += 1
            elif role == "Offensive Coordinator" or role == "Defensive Coordinator" or role == "Special Teams Coordinator":
                feature_dict["num_yr_col_coor"] += 1
            elif role == "Head Coach":
                feature_dict["num_yr_col_hc"] += 1
            # Else None
        elif level == "NFL":
            if role == "Position":
                feature_dict["num_yr_nfl_pos"] += 1
            elif role != "None":
                result = dfs[1].loc[dfs[1]['Year'] == year]['Tm'].values
                if len(result) == 0:
                    continue
                else:
                    franchise = result[0].lower()
                franchise_abrev = team_dict.get(franchise, franchise)
                num_teams_in_league, franchise_abrev = get_num_team_in_league(franchise_abrev, year, team_path)
                if num_teams_in_league == None:
                    print('Error: {}, {} not found with year {}'.format(coach_name, franchise_abrev, year))
                    continue
                if role == "Offensive Coordinator" or role == "Defensive Coordinator" or role == "Special Teams Coordinator":
                    feature_dict["num_yr_nfl_coor"] += 1
                    if role == "Offensive Coordinator":
                        feature_dict["nfl_oc_norm_yard"].append(get_norm_val(dfs[1].loc[dfs[1]['Year'] == year]['Yds'].values[0], num_teams_in_league))
                        feature_dict["nfl_oc_norm_point"].append(get_norm_val(dfs[1].loc[dfs[1]['Year'] == year]['Pts'].values[0], num_teams_in_league))
                        feature_dict["nfl_oc_norm_giveaway"].append(get_norm_val(dfs[1].loc[dfs[1]['Year'] == year]['GvA'].values[0], num_teams_in_league))
                    elif role == "Defensive Coordinator":
                        feature_dict["nfl_dc_norm_yard"].append(get_norm_val(dfs[1].loc[dfs[1]['Year'] == year]['Yds Passing Off'].values[0], num_teams_in_league))
                        feature_dict["nfl_dc_norm_point"].append(get_norm_val(dfs[1].loc[dfs[1]['Year'] == year]['Pts Passing Off'].values[0], num_teams_in_league))
                        feature_dict["nfl_dc_norm_turnover"].append(get_norm_val(dfs[1].loc[dfs[1]['Year'] == year]['TkA'].values[0], num_teams_in_league))
                    # Nothing necessary if special teams
                elif role == "Head Coach":
                    if not is_head_coach or franchise_abrev != prev_franchise_abrev:
                        is_head_coach = True
                        feature_dict["age"] = row[2]
                        prev_franchise_abrev = franchise_abrev
                        winning_result = []
                        for new_year in [year, year + 1]:
                            record_row = dfs[0].loc[dfs[0]['Year'] == new_year]
                            if record_row.shape[0] == 0:
                                continue
                            num_wins = record_row['W'].values[0]
                            num_losses = record_row['L'].values[0]
                            num_ties = record_row['T'].values[0]
                            winning_result.append((num_wins + .5 * num_ties) / (num_wins + num_ties + num_losses))
                        # Not first time hire, previous career to calculate
                        if len(rows) != 0 and len(rows[-1]) == 28:
                            prev_year = rows[-1][1]
                            math_year = previous_year_check + 1 if year != previous_year_check + 1 else year
                            rows[-1].append(classify_coach_tenure(math_year - prev_year))

                        # Adding the data
                        new_row = [coach_name, year] + [value if key not in feature_transform_dict else feature_transform_dict[key](value) for key, value in feature_dict.items()]
                        new_row += get_hiring_team_stats(franchise_abrev, year, team_path)
                        new_row.append(quick_mean(winning_result))
                        
                        # add a row
                        rows.append(new_row)
                    feature_dict["nfl_hc_norm_yard"].append(get_norm_val(dfs[1].loc[dfs[1]['Year'] == year]['Yds±'].values[0], num_teams_in_league))
                    feature_dict["nfl_hc_norm_point"].append(get_norm_val(dfs[1].loc[dfs[1]['Year'] == year]['Pts±'].values[0], num_teams_in_league))
                    feature_dict["nfl_hc_norm_turnover"].append(get_norm_val(dfs[1].loc[dfs[1]['Year'] == year]['T/G'].values[0], num_teams_in_league))
                    feature_dict["num_times_hc"] += 1
                    feature_dict["num_yr_nfl_hc"] += 1
                    # Add
        previous_year_check = year
    
    #Handle final hire length if not demotion
    if len(rows) != 0 and len(rows[-1]) == 28:
        prev_year = rows[-1][1]
        # Does not consider hires made in 2020 as they do not have 
        # at least one full season of coached games
        cutoff_year = 2014
        current_year = 2020
        if prev_year == current_year:
            rows.pop()
            print('\tExcluded last hire for both classification: {}, hire year: {}'.format(coach_name, prev_year))
        else:
            # Necessary as the year count is not iterated like all previous checks
            year += 1
            # Does not handle current coach tenure classification for those
            # hired since 2014 and still employed since not enough time has passed 
            # for fair classification
            if year > current_year and prev_year >= cutoff_year:
                print('\tExcluded last hire tenure classification: {}'.format(coach_name))
            rows[-1].append(classify_coach_tenure(year - prev_year) if prev_year < cutoff_year else -1)


    #Checks length
    for i in range(0, len(rows)):
        if len(rows[i]) != 29:
            print('Error: {} {}'.format(coach_name, rows[i]))
    #print(feature_dict)
    #print(rows)
    return rows


def main():
    master_data = []
    root_path = os.getcwd()
    coach_path = root_path + "\\Coaches"
    team_path = root_path + "\\Teams"

    count = 1
    list_subfolders_with_paths = [f.path for f in os.scandir(coach_path) if f.is_dir()]
    for sub in list_subfolders_with_paths:
        coach_name = sub.split('\\')[-1]
        print("Parsing coach {}, {}".format(count, coach_name))
        for new_row in parse_coach_career(coach_name, coach_path, team_path):
            master_data.append(new_row)
        count += 1
        """if count == 6:
            break"""
    df = pd.DataFrame(data=master_data, columns=get_point_features())
    df.to_csv("master_data.csv")
    df.to_feather("master_data.feather")
    print('Parsed {} Hiring Instances'.format(len(master_data)))

if __name__ == "__main__":
    main()