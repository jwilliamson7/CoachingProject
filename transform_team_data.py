import os
import pandas as pd
import math
import operator
from numpy import mean, nan
import re

from sympy import root

def make_directory(name):
    if not os.path.exists(name):
        os.mkdir(name)

def main():
    list_of_team_tables = {}
    list_of_opponent_tables = {}
    root_path = os.getcwd()
    team_path = root_path + "\\Teams"
    data_path = root_path + "\\League Data"
    table_names = ["yearly_team_stats", "yearly_opponent_stats"]

    count = 1
    list_subfolders_with_paths = [f.path for f in os.scandir(team_path) if f.is_dir()]
    for sub in list_subfolders_with_paths:
        team_name = sub.split('\\')[-1]
        print("Parsing team {}, {}".format(count, team_name))
        file_path = team_path + "\\" + str(team_name) + "\\"
        dfs = []
        for table_name in table_names:
            dfs.append(pd.read_csv(file_path + table_name + ".csv"))
        
        #team stats
        for i in range(dfs[0].shape[0]):
            row = dfs[0].iloc[i].values.flatten().tolist()
            year = int(row[1])
            row = row[2:]
            row.insert(0, team_name)
            if year not in list_of_team_tables:
                list_of_team_tables[year] = [row]
            else:
                list_of_team_tables[year].append(row)
        #opponent stats
        for i in range(dfs[1].shape[0]):
            row = dfs[1].iloc[i].values.flatten().tolist()
            year = int(row[1])
            row = row[2:]
            row.insert(0, team_name)
            if year not in list_of_opponent_tables:
                list_of_opponent_tables[year] = [row]
            else:
                list_of_opponent_tables[year].append(row)
        count += 1
    
    #Writing the new tables
    os.chdir(data_path)
    
    #Team tables
    for year, table in list_of_team_tables.items():
        df = pd.DataFrame(data=table, columns=dfs[0].columns[1:])
        df.rename({'Year': 'Team Abreviation'}, axis='columns', inplace=True)
        make_directory(str(year))
        os.chdir(str(year))
        df.to_csv("league_team_data.csv")
        os.chdir('../')

    #Opponent tables
    for year, table in list_of_opponent_tables.items():
        df = pd.DataFrame(data=table, columns=dfs[1].columns[1:])
        df.rename({'Year': 'Team Abreviation'}, axis='columns', inplace=True)
        make_directory(str(year))
        os.chdir(str(year))
        df.to_csv("league_opponent_data.csv")
        os.chdir('../')
    

if __name__ == "__main__":
    main()