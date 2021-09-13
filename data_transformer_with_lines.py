import pandas as pd
import numpy as np
count = 0
for year in range(2011,2021):
    yr = str(year)
    filepath = 'C:\\Users\\jerem\\OneDrive\\Documents\\python_projects\\nfl_project\\Version 3\\data_files\\'
    ppg = pd.read_csv(
    	'%sppg_%s.csv'
    	%(filepath,yr), index_col = 0) # points per game data
    ppga = pd.read_csv(
    	'%sppga_%s.csv' 
    	%(filepath,yr), index_col = 0) # points per game allowed data
    opponents = pd.read_csv(
    	'%sopponents_%s.csv' 
    	%(filepath,yr), index_col = 0) # the opponents faced for each team
    loc = pd.read_csv(
    	'%sloc_%s.csv' 
    	%(filepath,yr), index_col = 0) # the opponents faced for each team
    line = pd.read_csv(
    	'%sout_of_format\\lines_%s.csv' 
    	%(filepath,yr), index_col = 0) # the opponents faced for each team

    # Stats for scoring team
    pcpg = pd.read_csv(
            '%spcpg_%s.csv' 
            %(filepath,yr), index_col = 0) # passing completions per game
    papg = pd.read_csv(
            '%spapg_%s.csv' 
            %(filepath,yr), index_col = 0) # passing attempets per game

    pypg = pd.read_csv(
            '%spypg_%s.csv' 
            %(filepath,yr), index_col = 0) # passing yards per game

    rapg = pd.read_csv(
            '%srapg_%s.csv' 
            %(filepath,yr), index_col = 0) # rushing attempts per game
    rypg = pd.read_csv(
            '%srypg_%s.csv' 
            %(filepath,yr), index_col = 0) # rushing yards per game

    rtdpg = pd.read_csv(
            '%srtdpg_%s.csv' 
            %(filepath,yr), index_col = 0) # rushing TD's per game

    trdapg = pd.read_csv(
            '%strdapg_%s.csv' 
            %(filepath,yr), index_col = 0) # third down attempts per game
    trdcpg = pd.read_csv(
            '%strdcpg_%s.csv' 
            %(filepath,yr), index_col = 0) # third down completions per game

    qbrpg = pd.read_csv(
    	'%sqbrpg_%s.csv' 
    	%(filepath,yr), index_col = 0) # Quarterback rating per game

    ptdpg = pd.read_csv(
    	'%sptdpg_%s.csv' 
    	%(filepath,yr), index_col = 0) # passing TD's per game

    toppg = pd.read_csv(
    	'%stoppg_%s.csv' 
    	%(filepath,yr), index_col = 0) # time of possesion

    ipg = pd.read_csv(
    	'%sipg_%s.csv' 
    	%(filepath,yr), index_col = 0) # interceptions thrown per game

    spg = pd.read_csv(
    	'%sspg_%s.csv' 
    	%(filepath,yr), index_col = 0) # sacks allowed per game

    # Stats for defending team
    pcpga = pd.read_csv(
    	'%spcpga_%s.csv' 
    	%(filepath,yr), index_col = 0) # passing completions per game
    papga = pd.read_csv(
    	'%spapga_%s.csv' 
    	%(filepath,yr), index_col = 0) # passing attempets per game

    pypga = pd.read_csv(
    	'%spypga_%s.csv' 
    	%(filepath,yr), index_col = 0) # passing yards per game

    rapga = pd.read_csv(
    	'%srapga_%s.csv' 
    	%(filepath,yr), index_col = 0) # rushing attempts per game
    rypga = pd.read_csv(
    	'%srypga_%s.csv' 
    	%(filepath,yr), index_col = 0) # rushing yards per game

    rtdpga = pd.read_csv(
            '%srtdpga_%s.csv' 
            %(filepath,yr), index_col = 0) # rushing TD's per game

    trdapga = pd.read_csv(
    	'%strdapga_%s.csv' 
    	%(filepath,yr), index_col = 0) # third down attempts per game
    trdcpga = pd.read_csv(
    	'%strdcpga_%s.csv' 
    	%(filepath,yr), index_col = 0) # third down completions per game

    qbrpga = pd.read_csv(
    	'%sqbrpga_%s.csv' 
    	%(filepath,yr), index_col = 0) # Quarterback rating per game

    ptdpga = pd.read_csv(
    	'%sptdpga_%s.csv' 
    	%(filepath,yr), index_col = 0) # passing TD's per game

    toppga = pd.read_csv(
    	'%stoppga_%s.csv' 
    	%(filepath,yr), index_col = 0) # time of possesion

    ipga = pd.read_csv(
    	'%sipga_%s.csv' 
    	%(filepath,yr), index_col = 0) # interceptions per game 

    spga = pd.read_csv(
    	'%sspga_%s.csv' 
    	%(filepath,yr), index_col = 0) # sacks achieved per game

    # convert lines data into correct format
    line = line.T
   
    # find a way to go through the data sets and linearize them, with the weeks as a new column
    # get the columns into an array
    weeks = np.array(ppg.columns)
    # print(weeks)
    # get the weeks for the line data (that is missing the bye)
    weeks_line = weeks[0:16]
    # get the indexes into an array
    teams = np.array(ppg.index)
    # print(teams)
    # linearize the data for the stat
    
    # loop through the teams and make individual lists than can than be added together
    stats = [opponents,loc,ppg,qbrpg,pcpg,papg,pypg,ptdpg,ipg,spg,rapg,rypg,rtdpg,toppg,trdapg,trdcpg]
    stats_a = [opponents,loc,ppga,qbrpga,pcpga,papga,pypga,ptdpga,ipga,spga,rapga,rypga,rtdpga,toppga,trdapga,trdcpga]
    stats_line_calc = [opponents,loc,ppg,ppga]
    def dataset_maker(year,weeks,teams,stats):
        # NOTE: something in this process converts all the data to type str
        initial = True
        for stat in stats:
                year_col = []
                week_col = []
                team_col = []   
                stat_data = []
                if initial:
                        initial = False
                        for team in teams:
                                [stat_data.append(v) for v in stat.loc[team,:].values]
                                [week_col.append(i+1) for i in range(len(weeks))]
                                [team_col.append(team) for i in range(len(weeks))]
                                [year_col.append(year) for i in range(len(weeks))]
                        dataset = np.vstack((year_col,week_col))
                        dataset = np.vstack((dataset,team_col))
                        dataset = np.vstack((dataset,stat_data))
                else:
                        for team in teams:
                                [stat_data.append(float(v)) for v in stat.loc[team,:].values]
                        dataset = np.vstack((dataset,stat_data))

        dataset = dataset.T
        df = pd.DataFrame(dataset)
        mask = df == 'nan'
        df[mask] = float('nan')
        df.dropna(inplace=True)
        return df
        
    
    cols = ['year','week','team','opponent','location','ppg','qbrpg','pcpg','papg','pypg','ptdpg','ipg','spg',
             'rapg','rypg','rtdpg','toppg','trdapg','trdcpg']
    cols_line_calc = ['year','week','team','opponent','location','ppg','ppga']
    df1 = dataset_maker(year,weeks,teams,stats)
    df1.columns = cols
    df1.reset_index(inplace=True)
    df1.drop(columns='index',inplace=True)
    df2 = dataset_maker(year,weeks,teams,stats_a)
    df2.columns = cols
    df2.reset_index(inplace=True)
    df2.drop(columns='index',inplace=True)
    df3 = dataset_maker(year,weeks,teams,stats_line_calc)
    df3.columns = cols_line_calc
    df3.reset_index(inplace=True)
    df3.drop(columns='index',inplace=True)
    df_lines = dataset_maker(year,weeks_line,teams,[line])
    df_lines.columns = ['year','week','team','line']
    df_lines.drop(columns='week',inplace=True)
    df_lines.reset_index(inplace=True)
    df_lines.drop(columns='index',inplace=True)
    
    #add the lines to the main dataframes
    df3 = df3.astype({'ppg': 'float'})
    df3 = df3.astype({'ppga': 'float'})
    df3['net'] = df3['ppg'] - df3['ppga']
    df3['line'] = df_lines['line']
    df3 = df3.astype({'line': 'float'})
# =============================================================================
#     # get the wins and losses 
#     net = print(type(df1.ppg.to_numpy()[0])) #- df2.ppg.to_numpy()
#     mask = net > 1 # the winners 
#     net[mask] = 1
#     net[~mask] = 0
#     df1['win'] = net
#     df2['win'] = net
# =============================================================================
    
    if count < 1:
        df1_final = df1
        df2_final = df2
        df3_final = df3
    else:
        df1_final = pd.concat([df1_final,df1])
        df2_final = pd.concat([df2_final,df2])
        df3_final = pd.concat([df3_final,df3])
        
    count += 1

df1_final.reset_index(inplace=True,drop=True)
df2_final.reset_index(inplace=True,drop=True)
df3_final.reset_index(inplace=True,drop=True)
with pd.ExcelWriter('Data_raw.xlsx') as writer:  
        df1_final.to_excel(writer, sheet_name='Data_for', float_format='%.2f')
        df2_final.to_excel(writer, sheet_name='Data_against', float_format='%.2f')
        df3_final.to_excel(writer, sheet_name='Lines', float_format='%.2f')
                        

        

