import os
import pandas as pd
import math
import operator
from numpy import mean, nan


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
        return None
    if years <= 2:
        return 0
    elif years > 2 and years <= 4:
        return 1
    elif years > 4 and years <= 8:
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

def get_num_team_in_league(franchise, year, team_path, second_iteration=False):
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
    
    df_record = pd.read_csv(record_path)
    df_playoff = None if not os.path.exists(playoff_path) else pd.read_csv(playoff_path)

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
    for row in dfs[2].itertuples(index=False):
        level = classify_level(row[3])
        role = classify_role(row[5])
        year = row[1]
        
        if is_head_coach and (level == 'College' or role != "Head Coach"):
            is_head_coach = False
            feature_dict["demotion_presence"] = 1

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
                        feature_dict["nfl_oc_norm_yard"].append(dfs[1].loc[dfs[1]['Year'] == year]['Yds'].values[0] / num_teams_in_league)
                        feature_dict["nfl_oc_norm_point"].append(dfs[1].loc[dfs[1]['Year'] == year]['Pts'].values[0] / num_teams_in_league)
                        feature_dict["nfl_oc_norm_giveaway"].append(dfs[1].loc[dfs[1]['Year'] == year]['GvA'].values[0] / num_teams_in_league)
                    elif role == "Defensive Coordinator":
                        feature_dict["nfl_dc_norm_yard"].append(dfs[1].loc[dfs[1]['Year'] == year]['Yds Passing Off'].values[0] / num_teams_in_league)
                        feature_dict["nfl_dc_norm_point"].append(dfs[1].loc[dfs[1]['Year'] == year]['Pts Passing Off'].values[0] / num_teams_in_league)
                        feature_dict["nfl_dc_norm_turnover"].append(dfs[1].loc[dfs[1]['Year'] == year]['TkA'].values[0] / num_teams_in_league)
                    # Nothing necessary if special teams
                elif role == "Head Coach":
                    if not is_head_coach or franchise_abrev != prev_franchise_abrev:
                        is_head_coach = True
                        feature_dict["age"] = row[2]
                        # TODO fix check
                        prev_team = dfs[1].loc[dfs[1]['Year'] == year]['Tm'].values[0].lower()
                        prev_franchise_abrev = team_dict.get(prev_team, prev_team)

                        # Adding the data
                        new_row = [coach_name, year] + [value if key not in feature_transform_dict else feature_transform_dict[key](value) for key, value in feature_dict.items()]
                        #new_row += get_hiring_team_stats(franchise_abrev, year, team_path)
                        
                        # add a row
                        rows.append(new_row)
                    feature_dict["nfl_hc_norm_yard"].append(dfs[1].loc[dfs[1]['Year'] == year]['Yds±'].values[0] / num_teams_in_league)
                    feature_dict["nfl_hc_norm_point"].append(dfs[1].loc[dfs[1]['Year'] == year]['Pts±'].values[0] / num_teams_in_league)
                    feature_dict["nfl_hc_norm_turnover"].append(dfs[1].loc[dfs[1]['Year'] == year]['T/G'].values[0] / num_teams_in_league)
                    feature_dict["num_times_hc"] += 1
                    feature_dict["num_yr_nfl_hc"] += 1
                    # Add
        
    #print(feature_dict)
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
        """if count == 2:
            break"""
    #df = pd.DataFrame(data=master_data, columns=get_point_features())
    #df.to_csv("master_data.csv")
    #df.to_feather("master_data.feather")
    print(len(master_data))

if __name__ == "__main__":
    main()