# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 13:54:00 2020

@author: jerem
"""
# import os
# print(os.getcwd())
# path = 'C://Users//jerem//OneDrive//Documents//python_projects//nfl_project//Version 3'
# os.chdir(path)
# print(os.getcwd())
#%% Import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nfl_package.model_definitions as md
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

#%% Load Data
# df1: data for a team | df2: data against a team
import xlrd
# xls = pd.ExcelFile('C://Users//jerem//OneDrive//Documents//python_projects//nfl_project//Version 3//Line Data Analysis (Import into Pandas).xlsx')
# df1 = pd.read_excel(xls,sheet_name="Data_for",index_col=0,header=0)
# df2 = pd.read_excel(xls,sheet_name="Data_against",index_col=0,header=0)
# df3 = pd.read_excel(xls,sheet_name="Lines",index_col=0,header=0)
# xls.close()
# #%% save the data to a .pkl file
# df1.to_pickle('df1')
# df2.to_pickle('df2')
# df3.to_pickle('df3')
#%% load the data from the .pkl file
df1 = pd.read_pickle('df1')
df2 = pd.read_pickle('df2')
df3 = pd.read_pickle('df3')
#%% do stuff
# get the teams in a list
teams = df1.team.unique()

# By iterating through the df, get the difference between f2 and opponent f2


# get the wins and losses and put them in the dataset
net = df1.ppg.to_numpy() - df2.ppg.to_numpy()
mask = net > 0 # the winners 
net[mask] = 1
net[~mask] = 0
df1['win'] = net
df2['win'] = net
# df3['win'] = net


# put the current pythagorean ratio together with the line for the game to find a predictor for the winner
from sklearn.linear_model import LogisticRegression
X = df3.loc[:,['line', 'pyth ratio']]
y = df3.loc[:,'win/loss']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


fig = plt.figure()
plt.scatter(X,y,color='blue',marker='.')

# completion percentage
pcppg_1 = df1.pcpg/df1.papg
df1.insert(5,'pcppg',pcppg_1)
pcppg_2 = df2.pcpg/df2.papg
df2.insert(5,'pcppg',pcppg_1)

# 2019
df1_19 = df1[df1.year == 2019]
df2_19 = df2[df2.year == 2019]
# 2020 
df1_20 = df1[df1.year == 2020]
df2_20 = df2[df2.year == 2020]
# wins in 2020
df1_20_w = df1_20[df1_20.win == 1]
df2_20_w = df2_20[df2_20.win == 1]
# loses in 2020
df1_20_l = df1_20[df1_20.win == 0]
df2_20_l = df2_20[df2_20.win == 0]

# split the data into wins and losses and average
df1_20_w_avgs = md.stat_average_v11(df1_20_w)
df2_20_w_avgs = md.stat_average_v11(df2_20_w)
df1_20_l_avgs = md.stat_average_v11(df1_20_l)
df2_20_l_avgs = md.stat_average_v11(df2_20_l)

# the difference in stat averages for wins and losses
    # this is used as an inspection device to determine what stats lead to a loss
df1_20_avgs_dif = df1_20_w_avgs.iloc[:,1:] - df1_20_l_avgs.iloc[:,1:]
df1_20_avgs_perc = df1_20_w_avgs.iloc[:,1:] / df1_20_l_avgs.iloc[:,1:]
avg_percent_dif = np.mean(df1_20_avgs_perc)

# team averages for 2020
df1_20_tr_avgs = md.stat_average_v11(df1_20)
df2_20_tr_avgs = md.stat_average_v11(df2_20)
    
avg_errors = []
win_percentages = []
key_stats = ['ppg','pypg','rypg','trdcpg','spg']
descending_stats_1 = ['ppg','pypg','rypg','trdcpg']
descending_stats_2 = ['spg'] # 'papg','ipg' too

fig = plt.figure()
for i in range(5,17):
    # the training weeks for 2020
    qual1 = df1_20.week <= i  
    qual2 = df1_20.week >= i-5
    qual = qual1 & qual2
    df1_20tr = df1_20[qual1]
    df2_20tr = df2_20[qual1]
    # the testing weeks for the model
    df1_20tst = df1_20[df1_20.week == i+1]
    df2_20tst = df2_20[df2_20.week == i+1]
    
    # the opponents faced for the training weeks
    opponents_tr = df1_20tr.opponent
    teams_tr = df1_20tr.team
    
    # get the weeks opponents with respect to the alphabetical order of the teams
    opponents_cur = df2_20tst.opponent
    teams_cur = df2_20tst.team
    
    # team averages for 2020 training
    df1_20_tr_avgs = md.stat_average_v11(df1_20tr)
    df2_20_tr_avgs = md.stat_average_v11(df2_20tr)

    # averages of key stats - for
    key_stat_avgs_1 = df1_20_tr_avgs.loc[teams,key_stats].T
    avgs_1 = np.mean(key_stat_avgs_1,axis=1)
    key_stat_avgs_1['avg'] = avgs_1
    # averages of key stats - allowed
    key_stat_avgs_2 = df2_20_tr_avgs.loc[teams,key_stats].T
    avgs_2 = np.mean(key_stat_avgs_2,axis=1)
    key_stat_avgs_2['avg'] = avgs_2
    
    # ratios of team avg to league avg - for
    key_stat_ratios_1 = (np.array(key_stat_avgs_1.iloc[:,:-1].values) /
    np.array(key_stat_avgs_1.iloc[:,-1].values).reshape(-1,1))
    key_stat_ratios_1 = pd.DataFrame(key_stat_ratios_1,index=key_stats,columns = teams)
    key_stat_ratios_1.T[descending_stats_2] = 1/key_stat_ratios_1.T[descending_stats_2]
    key_stat_ratios_1.loc['sum'] = np.sum(key_stat_ratios_1)
    # ratios of team avg to league avg - allowed
    key_stat_ratios_2 = (np.array(key_stat_avgs_2.iloc[:,:-1].values) /
    np.array(key_stat_avgs_2.iloc[:,-1].values).reshape(-1,1))
    key_stat_ratios_2 = pd.DataFrame(key_stat_ratios_2,index=key_stats,columns = teams)
    key_stat_ratios_2.T[descending_stats_1] = 1/key_stat_ratios_2.T[descending_stats_1]
    key_stat_ratios_2.loc['sum'] = np.sum(key_stat_ratios_2)
    
    # team quality number
    qual_num = key_stat_ratios_1.loc['sum'] + key_stat_ratios_2.loc['sum']
    qual_num_lg_avg = np.mean(qual_num)
    
    # get the quality numbers for all the opponents
    qual_opp = qual_num[opponents_tr.values]
    qual_opp.index = teams_tr.values
    # calculate ratios for toughness of schedule
    qual_opp_avgs = []
    for team in teams:
        qual_opp_avgs.append(np.mean(qual_opp[team]))
    qual_opp_avgs = np.array(qual_opp_avgs)
    qual_opp_ratios = qual_opp_avgs/qual_num_lg_avg
    qual_opp_avgs = pd.Series(qual_opp_avgs,index=teams)
    qual_opp_ratios = pd.Series(qual_opp_ratios,index=teams)
    
    # adjust the quality number by the toughness of the teams schedule
    qual_num_adj = qual_num*qual_opp_ratios
    
    # net adjusted quality number for the weeks matchups
    net_qual_num_adj = qual_num_adj[teams_cur].values - qual_num_adj[opponents_cur].values
    
    # net points scored this week
    ppg_net = df1_20tst.ppg - df2_20tst.ppg
    
    # plot the net quality vs the net score
    plt.scatter(net_qual_num_adj,ppg_net)
    plt.xlabel('NET Quality')
    plt.ylabel('NET ppg')
    plt.title('Rank vs ppg | NET')
    
    # stats
    test1 = net_qual_num_adj > 1 
    ppg_net_stat_ttl = ppg_net.values[test1]
    test2 = ppg_net_stat_ttl > 0
    ppg_net_stat_good = ppg_net_stat_ttl[test2]
    win_perc = len(ppg_net_stat_good)/len(ppg_net_stat_ttl)
    win_percentages.append(win_perc)
    
    # rank the offense and defense of each team using key stats
    ranks_1 = md.stat_rank_v11(df1_20_tr_avgs,key_stats,descending_stats_1)
    ranks_1 = ranks_1.loc[teams_cur,:]
    ranks_2 = md.stat_rank_v11(df2_20_tr_avgs,key_stats,descending_stats_2)
    ranks_2 = ranks_2.loc[teams_cur,:]
    
    # defensive ranking in ratio form
    ranks_2_ratio = ranks_2.T
    ranks_2_ratio['avg'] = np.mean(ranks_2_ratio,axis=1)
    
    
    # find the net rank for the team and the defense they are playing
    net_ranks = ranks_1.values - ranks_2.loc[opponents_cur,:].values
    net_ranks = pd.DataFrame(net_ranks,columns=ranks_2.columns,index=teams_cur)
    net_ranks = net_ranks.drop(columns=['sum'])
    net_ranks['sum'] = np.sum(net_ranks,axis=1)
    # add the offense and defensive ranking together for an overall team rank
    rank_ovr = ranks_1['sum'] + ranks_2['sum']
    rank_ovr.index = teams_cur
    
    # find the net rank for the game (negative numbers are the favorite)
    a = rank_ovr[opponents_cur]
    rank_net = rank_ovr.values - rank_ovr[opponents_cur].values
    rank_net = pd.Series(rank_net,index=teams_cur)
    
    
    
    

# =============================================================================
#     # average points scored and allowed for the training data set
#     pts_avg_1_tr,pts_avg_2_tr = md.calculator(df1_20tr,df2_20tr,'ppg','avg') 
#     # total points scored and allowed for the 2019 data set
#     pts_ttl_1_19, pts_ttl_2_19 = md.calculator(df1_19,df2_19,'ppg','sum')
#     # total wins for the 2019 data set
#     wins_ttl_1_19,trash = md.calculator(df1_19,df2_19,'win','sum')
#     # total wins for the training data set
#     wins_ttl_1_tr,trash = md.calculator(df1_20tr,df2_20tr,'win','sum')
#     # average passer rating for the training data set
#     qbr_avg_1_tr, qbr_avg_2_tr = md.calculator(df1_20tr,df2_20tr,'ppg','avg')
# =============================================================================
 
# =============================================================================
#     # =============================================================================
#     # calculate the needed stats
#     # =============================================================================
#     opp_cur = df1_20tst.opponent
#     team_cur = df1_20tst.team
#     # net total points in the current year
#     pts_avg_1_tr, pts_avg_2_tr = md.calculator(df1_20tr,df2_20tr,'ppg','avg')
#     a = (pts_avg_1_tr[df1_20tst.team].values / pts_avg_2_tr[df1_20tst.opponent].values)*5
#     a = np.array(a).reshape(-1,1)
# 
#     # net total passing yards in the current year
#     pyds_avg_1_tr, pyds_avg_2_tr = md.calculator(df1_20tr,df2_20tr,'pypg','avg')
#     a_test = pyds_avg_1_tr[df1_20tst.team].values
#     b_test = pyds_avg_2_tr[df1_20tst.opponent].values
#     b = (pyds_avg_1_tr[df1_20tst.team].values / pyds_avg_2_tr[df1_20tst.opponent].values)*5
#     b = np.array(b).reshape(-1,1)
#     
#     # passing yards per attempt
#     patm_avg_1_tr, patm_avg_2_tr = md.calculator(df1_20tr,df2_20tr,'papg','avg')
#     pypa_avg_1_tr = pyds_avg_1_tr / patm_avg_1_tr
#     pypa_avg_2_tr = pyds_avg_2_tr / patm_avg_2_tr
#     c = (pypa_avg_1_tr[df1_20tst.team].values / pypa_avg_2_tr[df1_20tst.opponent].values)*5
#     c = np.array(c).reshape(-1,1)
#     
#     
#     
#     # net points scored in the game (with respect to the team, not the opponent)
#     net_pts = df1_20tst.ppg.values - df2_20tst.ppg.values
#     net_pts = np.array(net_pts).reshape(-1,1)
#     # net_pts = pd.Series(net_pts,index=df1_20tst.team)
#     
#     stat = []
#     # combine the potential predictors and the dependent var into a dataframe
#     data = np.concatenate((stat,net_pts),axis=1)
#     data = pd.DataFrame(data,columns=['x','y'])
#     
#     # split the data into a test and train set
#     x = data.iloc[:,:-1]
#     y = data.iloc[:,-1]
#     
#     # train model
#     model = LinearRegression()
#     model.fit(x,y)
#     
#     # predict the dependent var
#     pred = model.predict(x)
#     
#     # calculate the average error
#     error = np.abs(y - pred)
#     avg_error = np.mean(error)
#     avg_errors.append(avg_error)
#     print(avg_error)
#     
#     # test how good it is at picking the winners
#     test1a = pred >= 0 
#     test1b =  y.values >= 0
#     maska = test1a & test1b
#     test2a = pred <= 0 
#     test2b = y.values <= 0
#     maskb = test2a & test2b
#     mask = maska | maskb
#     mdl_score = mask
#     win_perc = sum(mdl_score)/len(mdl_score)
#     win_percentages.append(win_perc)
#     print(win_perc)
#     # get metrics
#     coefs = model.coef_
#     a_score = model.score(x,y)
#     print('\n')
# =============================================================================
    
# ttl_avg_error = np.mean(avg_errors)
# print(ttl_avg_error)
ttl_win_perc = np.mean(win_percentages)
print(ttl_win_perc)

print('\n')