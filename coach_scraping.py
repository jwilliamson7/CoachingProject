import requests
import urllib.request
import time
import os
import pandas as pd
from random import randint
from bs4 import BeautifulSoup, Comment

def make_directory(name):
    if not os.path.exists(name):
        os.mkdir(name)

def scrape_coach_data(url):
    table_names = ["all_coaching_results", "all_coaching_ranks", "all_coaching_history"]
    duplicate_names = [['Playoffs'],['Rushing Off', 'Passing Off', 'Defense', 'Rushing Def', 'Passing Def'],[]]
    response = requests.get('https://www.pro-football-reference.com' + url)
    soup = BeautifulSoup(response.text, 'lxml')
    coach_name = soup.find('h1').text.strip()
    make_directory(coach_name)
    os.chdir(coach_name)
    tables = [soup.find("div", {"id": "all_coaching_results"}), soup.find("div", {"id": "all_coaching_ranks"}), soup.find("div", {"id": "all_coaching_history"})]  
    count = 0
    for table in tables:
        table_soup = BeautifulSoup(str(table), "lxml")
        t = table_soup.find('table')
        if t == None:
            comment = table_soup.find(text=lambda text: isinstance(text, Comment))
            if comment == None:
                break
            if comment.find("<table ") > 0:
                comment_soup = BeautifulSoup(str(comment), 'html.parser')
                t = comment_soup.find("table")
        headers = []
        rows = []
        header_count = 0
        for th in t.find_all("th"):
            class_list = th.get('class')
            if class_list != None and "poptip" in class_list:
                text = th.text.strip()
                while text in headers:
                    text = text + " " + duplicate_names[count][header_count]
                    if text in headers:
                        header_count += 1 
                headers.append(text)
        for tr in t.find('tbody').find_all('tr'):
            row = [tr.find('th').text.strip()]
            for td in tr.find_all('td'):
                row.append(td.text.strip())
            rows.append(row) 
        df = pd.DataFrame(data=rows, columns=headers)
        df.to_csv(table_names[count] + ".csv")
        df.to_feather(table_names[count] + ".feather")
        count = count + 1
    os.chdir('../')
    return 0

os.chdir('Coaches')
url = 'https://www.pro-football-reference.com/coaches/'
response = requests.get(url)

soup = BeautifulSoup(response.text, "html.parser")

out = open('visited.txt','w')

start = time.time_ns()
stop = 0
temp_soup = None
count = 1
temp_count = 1
for link in soup.find_all('a'):
    if temp_count < 518:
        temp_count = temp_count + 1
        continue
    link_text = link.get('href')
    if link_text.find('/coaches/') == -1:
        continue
    print("Scraping coach %d, %s" % (count, link_text))
    stop = time.time_ns()
    if stop - start < 1e9:
        time.sleep(randint(1,5))
    scrape_coach_data(link.get('href'))
    out.write(link_text + "\n")
    start = time.time_ns()
    count += 1

out.close()