from bs4 import BeautifulSoup
from requests_html import HTMLSession # use this to get through to the embedded javascript 
import numpy as np
def parser(url):

    # use HTMLSession to read the javascript embedded tables on the website
    session = HTMLSession()
    resp = session.get(url)

    # from here start parsing through the html using beautifulsoup
    soup = BeautifulSoup(resp.html.html, 'lxml') # (file, parsing method)
    rows = soup.find_all('tr')
    return rows


def get_team_data(rows,stat,stat_op,call_name):
    """
    This function takes in a table and loops through the rows and 
    then loops through the elements in each row to get a team and stat.


    Parameters
    ----------
    rows : bs4.element module 
        the meat. what is being scraped through.
    stat_list : list
        list of stats that eventually get turned into a column in an array
    team_list : list
        list of teams that evenually gets turned into a column in an array
        

    Returns
    -------
    out : numpy array
        an array [team, stat]

    """

    week_numbers = [0]
    trigger1 = True
    opponent = False
    for row in rows: 
        items = row.find_all('td') # The elements in the row
        weeks = row.find_all('th')
        for week in weeks:
            # print(week.contents)
            if week.contents == []:
                continue
            elif week['data-stat'] == 'week_num' and len(week.contents[0]) < 3:
                week_numbers.append(int(week.contents[0]))
                # print(week_numbers[-1])
                # print(len(week_numbers))
        
        # print(row.prettify())       
        if week_numbers[-1] <= 17 and not opponent:
            for i, v in enumerate(items):
                # print(v.contents) # this HTML is different than the other parts of the website. Use .contents to get the contents inside the "td" level. 
                if len(week_numbers)-1 != week_numbers[-1] and trigger1: # test for the bye week in the html
                    stat.append(float('NaN')) # add NaN for the bye week
                    trigger1 = False
                elif v['data-stat'] == call_name and len(v.contents) > 0: 
                    # print(v.contents)
                    stuff = float(v.contents[0])
                    stat.append(stuff) # add the stat to the list
                else:
                    continue
        elif week_numbers[-1] <= 17 and opponent:
            for i, v in enumerate(items):
                # print(v.contents) # this HTML is different than the other parts of the website. Use .contents to get the contents inside the "td" level. 
                # print(stat_op)
                if len(week_numbers)-1 != week_numbers[-1] and trigger1: # test for the bye week in the html
                    stat_op.append(float('NaN')) # add NaN for the bye week
                    trigger1 = False
                elif v['data-stat'] == call_name and len(v.contents) > 0: 
                    # print(v.contents)
                    stuff = float(v.contents[0])
                    stat_op.append(stuff) # add the stat to the list
                else:
                    continue
        if week_numbers[-1] >= 17:
            week_numbers = [0]
            opponent = True # move to the data for the opponents
            trigger1 = True 
    stat = np.array([stat],dtype = object).T
    stat_op = np.array([stat_op],dtype = object).T
    return stat,stat_op

def get_team_data_loc(rows,stat,stat_op,call_name):
    """
    This function takes in a table and loops through the rows and 
    then loops through the elements in each row to get a team and stat.


    Parameters
    ----------
    rows : bs4.element module 
        the meat. what is being scraped through.
    stat_list : list
        list of stats that eventually get turned into a column in an array
    team_list : list
        list of teams that evenually gets turned into a column in an array
        

    Returns
    -------
    out : numpy array
        an array [team, stat]

    """

    week_numbers = [0]
    trigger1 = True
    opponent = False
    for row in rows: 
        items = row.find_all('td') # The elements in the row
        weeks = row.find_all('th')
        for week in weeks:
            # print(week.contents)
            if week.contents == []:
                continue
            elif week['data-stat'] == 'week_num' and len(week.contents[0]) < 3:
                week_numbers.append(int(week.contents[0]))
                # print(week_numbers[-1])
                # print(len(week_numbers))
        
        # print(row.prettify())       
        if week_numbers[-1] <= 17 and not opponent:
            for i, v in enumerate(items):
                # print(v.contents) # this HTML is different than the other parts of the website. Use .contents to get the contents inside the "td" level. 
                # print(v['data-stat'])
                if len(week_numbers)-1 != week_numbers[-1] and trigger1: # test for the bye week in the html
                    stat.append(float('NaN')) # add NaN for the bye week
                    trigger1 = False
                elif v['data-stat'] == call_name and len(v.contents) > 0: 
                    # print(v.contents)
                    stuff = 0
                    stat.append(stuff) # add the stat to the list
                elif v['data-stat'] == call_name and len(v.contents) == 0: 
                    # print(v.contents)
                    stuff = 1
                    stat.append(stuff) # add the stat to the list
                else:
                    continue
        elif week_numbers[-1] <= 17 and opponent:
            for i, v in enumerate(items):
                # print(v.contents) # this HTML is different than the other parts of the website. Use .contents to get the contents inside the "td" level. 
                # print(stat_op)
                if len(week_numbers)-1 != week_numbers[-1] and trigger1: # test for the bye week in the html
                    stat_op.append(float('NaN')) # add NaN for the bye week
                    trigger1 = False
                elif v['data-stat'] == call_name and len(v.contents) > 0: 
                    # print(v.contents)
                    stuff = 0
                    stat_op.append(stuff) # add the stat to the list
                elif v['data-stat'] == call_name and len(v.contents) == 0: 
                    # print(v.contents)
                    stuff = 1
                    stat_op.append(stuff) # add the stat to the list
                else:
                    continue
        if week_numbers[-1] >= 17:
            week_numbers = [0]
            opponent = True # move to the data for the opponents
            trigger1 = True 
    stat = np.array([stat],dtype = object).T
    stat_op = np.array([stat_op],dtype = object).T
    return stat,stat_op