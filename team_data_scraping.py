from multiprocessing.sharedctypes import Value
import requests
import urllib.request
import time
import os
import pandas as pd
from random import randint
from bs4 import BeautifulSoup, Comment
import math

def make_directory(name):
    if not os.path.exists(name):
        os.mkdir(name)

def scrape_team_data(url):
    base_url = "https://www.pro-football-reference.com/teams/"
    start = time.time_ns()
    response = requests.get(base_url + url)
    duplicate_names = ['Def Rank']
    table_name = "team_record"
    
    soup = BeautifulSoup(response.text, 'lxml')
    make_directory(url)
    os.chdir(url)
    table = soup.find('div', {'id':'div_team_index'}).find('table')
    headers = []
    rows = []
    header_count = 0
    for th in table.find_all("th"):
        class_list = th.get('class')
        if class_list != None and "poptip" in class_list:
            text = th.text.strip()
            while text in headers:
                text = text + " " + duplicate_names[header_count]
                if text in headers:
                    header_count += 1 
            headers.append(text)
    
    for tr in table.find('tbody').find_all('tr'):
        class_list = tr.get('class')
        if class_list != None and "thead" in class_list:
            continue
        row = [tr.find('th').text.strip()]
        for td in tr.find_all('td'):
            row.append(td.text.strip())
        rows.append(row)
    
    # Creates a vector of the years with team stats and then calls the scrape yearly stats function on that vector
    years = []
    for row in rows:
        years.append(row[0])
    end_time = scrape_team_yearly_stats(base_url + url, years, start)

    #Saves the output of the team record data (which primarily includes ranks)
    df = pd.DataFrame(data=rows, columns=headers)
    df.to_csv(table_name + ".csv")
    df.to_feather(table_name + ".feather")

    check_time_and_pause(end_time, time.time_ns())
    scrape_team_playoff_data(url)
    os.chdir('../')

# Function that attempts to create a single table with all topline team stats (not rankings of stats)
def scrape_team_yearly_stats(url, years, start_time):
    base_url = url
    headers = ['Year']
    team_rows = []
    opponent_rows = []
    count = 1
    headers = [
        'Year',
        'PF (Points For)',
        'Yds',
        'Offensive Plays',
        'Y/P',
        'TO',
        'FL+',
        '1stD',
        'Cmp Passing',
        'Att Passing',
        'Yds Passing',
        'TD Passing',
        'Int Passing',
        'NY/A Passing',
        '1stD Passing',
        'Att Rushing',
        'Yds Rushing',
        'TD Rushing',
        'Y/A Rushing',
        '1stD Rushing',
        'Pen',
        'Yds Penalties',
        '1stPy',
        '#Dr',
        'Sc%',
        'TO%',
        'Start Average Drive',
        'Time Average Drive',
        'Plays Average Drive',
        'Yds Average Drive',
        'Pts Average Drive',
        '3DAtt',
        '3DConv',
        '3D%',
        '4DAtt',
        '4DConv',
        '4D%',
        'RZAtt',
        'RZTD',
        'RZPct',
        ]

    for year in years:
        print("Scraping year %s" % (year))
        #Checks the time to prevent too frequent of requests
        check_time_and_pause(start_time, time.time_ns())
        response = requests.get(base_url + '/' + year + '.htm')
        start_time = time.time_ns()

        soup = BeautifulSoup(response.text, 'lxml')

        #Pulls the "Team stats and Rankings" table
        table = soup.find('div', {'id':'div_team_stats'}).find('table')
        #Pulls the "Team Conversions" table
        table2 = None
        if int(year) > 1998:
            table2 = soup.find('div', {'id':'div_team_conversions'}).find('table')
        #pulls the table headers for the first website call
        """
        if count == 1:
            duplicate_names = ['', ' Passing', ' Rushing', ' Penalties', ' Average Drive']
            header_count = 0
            #Table headers from table 1
            for th in table.find_all("th"):
                class_list = th.get('class')
                if class_list != None and "poptip" in class_list:
                    text = th.text.strip() + duplicate_names[header_count]
                    if text in headers:
                        header_count += 1 
                        text = th.text.strip() + duplicate_names[header_count]
                    if text != "Player":
                        headers.append(text)
            #table headers from table 2
            for th in table2.find_all("th"):
                class_list = th.get('class')
                if class_list != None and "poptip" in class_list:
                    text = th.text.strip()
                    if text != "Player":
                        headers.append(text)
        """
        row_number = 0
        for tr in table.find('tbody').find_all('tr')[0:2]:
            row = [year]            
            for td in tr.find_all('td'):
                row.append(td.text.strip())    
            if table2 != None:
                tr2 = table2.find('tbody').find_all('tr')[row_number]
                for td2 in tr2.find_all('td'):
                    row.append(td2.text.strip())
            else:
                row.extend([''] * (len(headers) - len(row)))
            if row_number == 0:
                team_rows.append(row)
            else:
                opponent_rows.append(row)
            row_number += 1
        count += 1
    df = pd.DataFrame(data=team_rows, columns=headers)
    df.to_csv("yearly_team_stats.csv")
    df = pd.DataFrame(data=opponent_rows, columns=headers)
    df.to_csv("yearly_opponent_stats.csv")
    return time.time_ns()


def scrape_team_playoff_data(url):
    base_url = "https://www.pro-football-reference.com/teams/"
    tail = '/playoffs.htm'
    table_name = 'team_playoff_record'
    duplicate_names = ['Offense', 'Defense']
    response = requests.get(base_url + url + tail)
    soup = BeautifulSoup(response.text, 'lxml')
    table = soup.find('div', {'id':'div_playoff_game_log'})
    if table == None:
        return
    table = table.find('table')
    if table == None:
        return
    headers = ['Year']
    rows = []
    header_count = 0
    for th in table.find_all("th"):
        class_list = th.get('class')
        if class_list != None and "poptip" in class_list:
            text = th.text.strip()
            while text in headers:
                text = text + " " + duplicate_names[header_count]
                if text in headers:
                    header_count += 1 
            headers.append(text)
    row_year = ''
    for tr in table.find('tbody').find_all('tr'):
        class_list = tr.get('class')
        if class_list != None and "thead" in class_list:
            row_year = str(tr.text.strip().partition(' ')[0])
            continue
        row = [row_year, tr.find('th').text.strip()]
        for td in tr.find_all('td'):
            row.append(td.text.strip())
        rows.append(row)
    df = pd.DataFrame(data=rows, columns=headers)
    df.to_csv(table_name + ".csv")
    df.to_feather(table_name + ".feather")


def check_time_and_pause(start, stop, threshold=5e9):
    if stop - start < threshold:
        pause_duration = randint(0,1) + math.ceil((threshold - (stop - start)) * 1e-9)
        print('***Waiting %d seconds' % pause_duration)
        time.sleep(pause_duration)

def parse_active_teams():
    os.chdir('Teams')
    url = 'https://www.pro-football-reference.com/teams/'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    out = open('visited_teams.txt','w')

    start = time.time_ns()
    count = 1
    for link in soup.find('table', {"id": "teams_active"}).find_all('a'): 
        link_text = link.get('href')
        if 'teams' not in link_text:
            continue
        team_brev = link_text.split('/')[-2]
        print("Scraping team %d, %s" % (count, team_brev))
        check_time_and_pause(start, time.time_ns())
        start = time.time_ns()
        scrape_team_data(team_brev)
        out.write(team_brev + "\n")
        count += 1
    out.close()


def parse_inactive_teams():
    os.chdir('Teams')
    url = 'https://www.pro-football-reference.com/teams/'
    response = requests.get(url)

    soup = BeautifulSoup(response.text, "html.parser")

    out = open('visited_teams.txt','w')

    comment = soup.find('div', {"id":"all_teams_inactive"}).find(text=lambda text: isinstance(text, Comment))
    if comment.find("<table ") > 0:
        comment_soup = BeautifulSoup(str(comment), 'html.parser')
        t = comment_soup.find("table")

    start = time.time_ns()
    count = 1
    for link in t.find_all('a'):
        link_text = link.get('href')
        if 'teams' not in link_text:
            continue
        team_brev = link_text.split('/')[-2]
        print("Scraping team %d, %s" % (count, team_brev))
        check_time_and_pause(start, time.time_ns())
        start = time.time_ns()
        scrape_team_data(team_brev)
        out.write(team_brev + "\n")
        count += 1
    out.close()


if __name__ == "__main__":
    parse_active_teams()
    #parse_inactive_teams()