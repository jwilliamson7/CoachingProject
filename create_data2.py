from doctest import master
import os
import pandas as pd
import math
import operator
from numpy import mean, nan
import re

from sympy import comp


def get_point_features():
    features = ['Coach Name', 'Year']
    for i in range(1, len(get_feature_name_dict()) + len(get_hiring_team_stat_dict().keys()) + 1):
        features.append('Feature {}'.format(i))
    features.append('Avg 2Y Win Pct')
    features.append('Coach Tenure Class')
    return features


def get_team_dict():
    return {"ind":["ind", "clt"], 
            "ari":["ari", "crd"],
            "hou":["hou", "htx", "oti"],
            "ten":["ten", "oti"],
            "oak":["oak", "rai"], 
            "stl":["stl", "ram", "sla", "gun"],
            "lar":["lar", "ram"],
            "bal":["bal", "rav","clt"],
            "lac":["lac", "sdg"],
            "chr":["chr", "cra"],
            "frn":["frn", "fyj"],
            "nyt":["nyt", "nyj"],
            "can":["can", "cbd"],
            "bos":["bos", "byk", "was", "ptb"],
            "pot":["pot", "ptb"],
            "lad":["lad", "lda"],
            "evn":["evn", "ecg"],
            "pho":["pho", "crd"],
            "prt":["prt", "det"],
            "lvr":["lvr", "rai"],
            "chh":["chh", "cra"],
            "htx":["htx", "oti"],
            "buf":["buf", "bff", "bba"],
            "cle":["cle", "cti", "cib", "cli"],
            "min":["min", "mnn"],
            "kan":["kan", "kcb"],
            "det":["det", "dwl", "dpn", "dhr", "dti"],
            "nyy":["nyy", "naa", "nya"],
            "cin":["cin", "red", "ccl"],
            "mia":["mia", "msa"],
            "was":["was", "sen"],
            "dtx":["kan", "dtx"],
            "nyg":["nyg", "ng1"],
            "cli":["cli", "cib"]
            }

#TODO Fix
def get_position_name_list():
    return [
        "Running Backs"
    ]

#Defines feature names
def get_feature_name_dict():
    #TODO add initial features
    return_list = [
        "age",
        "num_times_hc",
        "num_yr_col_pos",
        "num_yr_col_coor",
        "num_yr_col_hc",
        "num_yr_nfl_pos",
        "num_yr_nfl_coor",
        "num_yr_nfl_hc"
    ]
    base_list =  [
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
    complete_list = [[], [], [], []]
    for name in base_list:
        complete_list[0].append(name + "__oc")
        complete_list[1].append(name + "__dc")
        complete_list[2].append(name + "__hc")
        complete_list[3].append(name + "__opp__hc")
    for temp_list in complete_list:
        return_list.extend(temp_list)
    return return_list

#Returns the dictionary of features
def get_feature_dict():
    my_dict = {}
    for feature_name in get_feature_name_dict():
        if "__oc" in feature_name or "__dc" in feature_name or "__hc" in feature_name:
            my_dict[feature_name] = []
        else:
            my_dict[feature_name] = 0
    return my_dict


# Classifies coach tenure into three classes, respresenting differing levels of coaching success
def classify_coach_tenure(years):
    if years <= 0:
        return -1
    if years <= 2:
        return 0
    elif years > 2 and years <= 4:
        return 1
    else:
        return 2


#Classifies the coaching role, excludes interim head coach experience
def classify_role(role):
    if role == "" or type(role) != type('string'):
        return "None"
    exclude_words = [
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
    for word in exclude_words:
        if word in role:
            return "None"
    if ("Assistant" in role or "Asst" in role) and "/" not in role and "\\" not in role:
        return "None"
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


def classify_level(level):
    if "College" in level:
        return "College"
    if "NFL" in level:
        return "NFL"
    return "None"


def quick_mean(lis):
    if not isinstance(lis, list):
        return lis
    if len(lis) == 0:
        return nan
    else:
        return mean(lis)


def get_hiring_team_stat_dict():
    return {
        "hiring_team_win_pct":          [],
        "hiring_team_points_scored":    [],
        "hiring_team_points_allowed" : [],
        "hiring_team_yards_offense" : [],
        "hiring_team_yards_allowed" : [],
        "hiring_team_yards_per_play" : [],
        "hiring_team_yards_per_play_allowed" : [],
        "hiring_team_turnovers_forced" : [],
        "hiring_team_turnovers_committed" : [],
        "hiring_team_num_playoff_appearances" : []
    }


def get_features_from_year(year, team_list, role, feature_dict, league_path):
    league_table_names = ["league_team_data_normalized", "league_opponent_data_normalized"]
    my_team = 0
    opponent_team = 1
    file_path = league_path + "\\" + str(year) + "\\"
    dfs = []
    for table_name in league_table_names:
        dfs.append(pd.read_csv(file_path + table_name + ".csv"))
    rows = []
    list_iterator = 0
    for df in dfs:
        temp_df = df.loc[df['Team Abbreviation']==team_list[list_iterator]]
        while temp_df.empty and list_iterator < len(team_list):
            list_iterator += 1
            temp_df = df.loc[df['Team Abbreviation']==team_list[list_iterator]]
        if list_iterator == len(team_list):
            print("Error: {} not found in year {}".format(team_list[0], year))
            return 0
        else:
            rows.append(temp_df)
    if role == "HC":
        for key, value in rows[my_team].to_dict("list").items():
            temp_key = key + "__hc"
            if temp_key in feature_dict:
                feature_dict[temp_key].append(value[0])
        for key, value in rows[opponent_team].to_dict("list").items():
            temp_key = key + "__opp__hc"
            if temp_key in feature_dict:
                feature_dict[temp_key].append(value[0])
    elif role == "OC":
        for key, value in rows[my_team].to_dict("list").items():
            temp_key = key + "__oc"
            if temp_key in feature_dict:
                feature_dict[temp_key].append(value[0])
    elif role == "DC":
        for key, value in rows[opponent_team].to_dict("list").items():
            temp_key = key + "__dc"
            if temp_key in feature_dict:
                feature_dict[temp_key].append(value[0])
    else:
        print("Error: {} is not a valid role".format(role))
        return 0
    return 1



#TODO fix
def get_hiring_team_stats(franchise, year, team_path):
    file_path = team_path + "\\" + str(franchise) + "\\"
    record_path = file_path + "team_record.csv"
    playoff_path = file_path + "team_playoff_record.csv"
    feature_dict = get_hiring_team_stat_dict()
    
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


def parse_coach_career(coach_name, coach_path, team_path, league_path):
    team_dict = get_team_dict()
    rows = []
    coach_table_names = ["all_coaching_results", "all_coaching_ranks", "all_coaching_history"]
    file_path = coach_path + "\\" + str(coach_name) + "\\"
    feature_dict = get_feature_dict()

    dfs = []
    for table_name in coach_table_names:
        dfs.append(pd.read_csv(file_path + table_name + ".csv"))
    
    prev_franchise_abrev = None
    is_head_coach = False
    previous_year_check = None
    year = 0


    # For each year in their career
    for row in dfs[2].itertuples(index=False):
        level = classify_level(row[3])
        role = classify_role(row[5])
        year = row[1]
        
        #Checks to see if the coach was demoted
        if is_head_coach and (level == 'College' or level == 'None' or role != "Head Coach"):
            is_head_coach = False
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
                team_abbreviation_list = dfs[1].loc[dfs[1]['Year'] == year]['Tm'].values
                if len(team_abbreviation_list) == 0:
                    continue
                else:
                    franchise = team_abbreviation_list[0].lower()
                franchise_abbrev_list = team_dict.get(franchise, franchise)
                if not isinstance(franchise_abbrev_list, list):
                    franchise_abbrev_list = [franchise_abbrev_list]

                if role == "Offensive Coordinator" or role == "Defensive Coordinator" or role == "Special Teams Coordinator":
                    feature_dict["num_yr_nfl_coor"] += 1
                    if role == "Offensive Coordinator":
                        get_features_from_year(year, franchise_abbrev_list, "OC", feature_dict, league_path)
                    elif role == "Defensive Coordinator":
                        get_features_from_year(year, franchise_abbrev_list, "DC", feature_dict, league_path)
                    # Nothing necessary if special teams
                elif role == "Head Coach":
                    #TODO fix second check
                    if not is_head_coach or franchise_abbrev_list != prev_franchise_abrev:
                        is_head_coach = True
                        feature_dict["age"] = row[2]
                        prev_franchise_abrev = franchise_abbrev_list
                        winning_result = []
                        for new_year in [year, year + 1]:
                            record_row = dfs[0].loc[dfs[0]['Year'] == new_year]
                            if record_row.shape[0] == 0:
                                continue
                            num_wins = record_row['W'].values[0]
                            num_losses = record_row['L'].values[0]
                            num_ties = record_row['T'].values[0]
                            winning_result.append((num_wins + .5 * num_ties) / (num_wins + num_ties + num_losses))
                        #TODO Fix
                        # Not first time hire, previous career to calculate
                        if len(rows) != 0 and len(rows[-1]) == 143:
                            prev_year = rows[-1][1]
                            math_year = previous_year_check + 1 if year != previous_year_check + 1 else year
                            rows[-1].append(classify_coach_tenure(math_year - prev_year))
                        # Adding the data
                        new_row = [coach_name, year] + [quick_mean(value) for value in feature_dict.values()]
                        #TODO Unhide
                        #new_row += get_hiring_team_stats(franchise_abrev, year, team_path)
                        new_row.append(quick_mean(winning_result))
                        
                        # add a row
                        rows.append(new_row)
                        feature_dict["num_times_hc"] += 1
                    
                    get_features_from_year(year, franchise_abbrev_list, "HC", feature_dict, league_path)
                    feature_dict["num_yr_nfl_hc"] += 1
                    # Add
        previous_year_check = year
        
    
    #Handle final hire length if not demotion
    if len(rows) != 0 and len(rows[-1]) == 143:
        prev_year = rows[-1][1]
        # Does not consider hires made in 2020 as they do not have 
        # at least one full season of coached games
        cutoff_year = 2017
        current_year = 2022
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
        if len(rows[i]) != 144:
            print('Error: {} {}'.format(coach_name, rows[i]))
    
    #print(feature_dict)
    return rows



def main():
    master_data = []
    root_path = os.getcwd()
    coach_path = root_path + "\\Coaches"
    team_path = root_path + "\\Teams"
    league_path = root_path + "\\League Data"

    
    count = 1
    list_subfolders_with_paths = [f.path for f in os.scandir(coach_path) if f.is_dir()]
    for sub in list_subfolders_with_paths:
        #TODO Unhide
        coach_name = "Don Shula"
        #coach_name = sub.split('\\')[-1]
        print("Parsing coach {}, {}".format(count, coach_name))
        for new_row in parse_coach_career(coach_name, coach_path, team_path, league_path):
            master_data.append(new_row)
        count += 1
        #TODO delete
        break
    print(master_data)
    """
    df = pd.DataFrame(data=master_data, columns=get_point_features())
    df.to_csv("master_data2.csv")
    """
    
    print('Parsed {} Hiring Instances'.format(len(master_data)))

if __name__ == "__main__":
    main()