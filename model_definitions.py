import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

# =============================================================================
# v11
# =============================================================================
# stat averager
def stat_average_v11(df):
    # get the teams to loop through
    teams = df.team.unique()
    # find the average stats for each team in the wins frame
    df_avgs = pd.DataFrame(columns=df.columns)
    for team in teams:
        team_avgs = np.mean(df[df.team == team])
        df_avgs = df_avgs.append(team_avgs,ignore_index = True)
    df_avgs.drop(columns=['year','week','opponent','team'],inplace=True)
    df_avgs.index = teams
    return df_avgs

# make a rank system
def stat_rank_v11(df_avgs,stats,desc_stats):
    low_stats = []
    medium_stats = ['ppg','pypg','rypg','trdcpg','spg']
    high_stats = []
    xhigh_stats = []
    low = 1
    medium = 1
    high = 1
    xhigh = 1
    teams = df_avgs.index.values
    ranks = pd.DataFrame()
    spots = np.arange(1,33).reshape(-1,1)
    for stat in stats:
        if any([stat == v for v in desc_stats]): # add all the stats that sort descending
            rank = np.argsort(df_avgs[stat],axis=0)[::-1]
        else:
            rank = np.argsort(df_avgs[stat],axis=0)
        temp = teams[rank].reshape(-1,1)
        temp_cat = np.concatenate((temp,spots),axis=1)
        temp_sort = np.argsort(temp_cat[:,0])
        sort_data = temp_cat[temp_sort,:]   
        ranks[stat] = sort_data[:,1]
    # sum_ranks = ranks.drop(columns = 'team')
    ranks[low_stats] = ranks[low_stats]*low
    ranks[medium_stats] = ranks[medium_stats]*medium
    ranks[high_stats] = ranks[high_stats]*high
    ranks[xhigh_stats] = ranks[xhigh_stats]*xhigh
    ranks['sum'] = np.sum(ranks,axis=1)
    ranks.index = df_avgs.index
    return ranks
# =============================================================================
# Main Calculator for v10
# =============================================================================
def calculator(data1,data2,stat,calc):
    stat_for = []
    stat_allowed = []
    teams = data1.team.unique() 
    for team in teams:
        if calc == 'sum':
            stat_for.append(np.sum(data1[data1.team == team].loc[:,stat]))
            stat_allowed.append(np.sum(data2[data2.team == team].loc[:,stat]))
        elif calc == 'avg':
            stat_for.append(np.mean(data1[data1.team == team].loc[:,stat]))
            stat_allowed.append(np.mean(data2[data2.team == team].loc[:,stat]))
    stat_for = pd.Series(stat_for,index=teams) # convert to a series
    stat_allowed = pd.Series(stat_allowed,index=teams) # convert to a series
    return stat_for,stat_allowed
# =============================================================================
# average the data into bins
# =============================================================================
def data_averager(X,Y,step):
    # Average the points per game for every number of rushing attempts
    X = np.array(X).reshape(-1,1)
    Y = np.array(Y).reshape(-1,1)
    sort = np.argsort(X,axis=0)
    X = X[sort].reshape(-1,1)
    X = [v for i,v in enumerate(X) if v == v] # test for and get rid of nan 
    X = np.array(X).reshape(-1,1)
    Y = Y[sort].reshape(-1,1)
    Y = [v for i,v in enumerate(Y) if v == v] # test for and get rid of nan 
    Y = np.array(Y).reshape(-1,1)
    
    # X_unique = np.unique(X)
    X_unique = np.arange(np.floor(X[0]),X[-1]+1,step)
    li = 0
    Y_avg = []
    for i,v in enumerate(X_unique):
        Y_temp = []
        try:
            while X[li] >= v and X[li] < X_unique[i+1]:
                Y_temp.append(Y[li])
                li += 1
            Y_avg.append(np.mean(Y_temp)) 
        except:
            Y_avg.append(np.mean(Y_temp)) 
            break
    while len(X_unique) > len(Y_avg):
        X_unique = np.delete(X_unique,-1)
    X_unique = np.array(X_unique).reshape(-1,1)
    Y_avg = np.array(Y_avg).reshape(-1,1)
    mask = Y_avg != Y_avg
    Y_avg = Y_avg[~mask]
    X_unique = X_unique[~mask]
    X_unique = np.array(X_unique).reshape(-1,1)
    Y_avg = np.array(Y_avg).reshape(-1,1)
    return X_unique, Y_avg
    
# =============================================================================
# Average the stats for each team
# =============================================================================
def stat_averager(data):
    data_avgs = []
    total_data = []
    for index, row in data.iterrows():
        team_data = row.values
        mask = team_data != team_data
        team_data = team_data[~mask]
        data_avgs.append(np.mean(team_data))
        [total_data.append(i) for i in team_data]
    data_avgs.append(np.mean(np.array(total_data).reshape(-1,1)))
    return data_avgs
# =============================================================================
# Create an adjuster for each stat and team
# =============================================================================
def stat_adjust_calc(data):
    data_avgs = []
    total_data = []
    for index, row in data.iterrows():
        team_data = row.values
        mask = team_data != team_data
        team_data = team_data[~mask]
        data_avgs.append(np.mean(team_data))
        [total_data.append(i) for i in team_data]
    league_avg = np.mean(np.array(total_data).reshape(-1,1))
    adjuster = data_avgs / league_avg
    return adjuster

def stat_adj_calc2(opps,ratios,stat,var_name):
    stat_ratios = []
    for team in np.array(opps.values).reshape(-1,1):
        if team == team:  
            stat_ratio = ratios.loc[var_name,team].values[0]
            stat_ratios.append(stat_ratio)
        else:
            stat_ratios.append(float('nan'))
    stat_ratios = np.array(stat_ratios).reshape(len(stat),len(stat.T))
    return stat_ratios

def opp_avgs(data,opponents):
    opp_avgs = []
    for teams in opponents.values:
        mask = teams == teams
        teams = teams[mask]
        # print(teams)
        team_opponents = data.loc[teams]
        team_avgs = []
        for index, row in team_opponents.iterrows():
            team_data = row.values
            mask = team_data != team_data
            team_data = team_data[~mask]
            team_avgs.append(np.mean(team_data))
        opp_avgs.append(np.mean(team_avgs))
    return opp_avgs

def model_creation(X,Y,degree):
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1)
    model = LinearRegression()
    poly_reg = PolynomialFeatures(degree = degree)
    x_poly = poly_reg.fit_transform(X_train)
    model.fit(x_poly,Y_train)
    return model,poly_reg,X_test,Y_test