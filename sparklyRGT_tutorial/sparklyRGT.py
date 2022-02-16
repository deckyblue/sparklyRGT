"""
This module includes functions to load in MEDPC rGT data and outputs a dataframe. 

Authors: Brett Hathaway & Dexter Kim 
"""

print("I am being executed!")

#main imports 
import os
import pandas as pd
import numpy as np

# plotting imports 
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# stats imports 
import scipy.stats as stats
import math
import seaborn as sns
import pingouin as pg

#the following line prevents pandas from giving unecessary errors 
pd.options.mode.chained_assignment = None

#---------------------------------------------------------------#

def load_data(fnames, reset_sessions = False): #when reset_sessions = False --> load_data runs like normal 
#load data from computer
    for i,file in enumerate(fnames):
        if i == 0:
            df = pd.read_excel(fnames[i])
            if reset_sessions:
                for i,session in enumerate(df.Session.unique()):
                    for j in range(len(df)):
                        if df.at[j,'Session'] == session:
                            df.at[j,'Session'] = i + 1
        else:
            df2 = pd.read_excel(fnames[i])
            if reset_sessions:
                for i,session in enumerate(df2.Session.unique()):
                    for j in range(len(df2)):
                        if df2.at[j,'Session'] == session:
                            df2.at[j,'Session'] = i + 1
            df = df.append(df2, ignore_index = True)
    return df

def load_multiple_data(fnames, reset_sessions=False): 
#load multiple datasets from computer and redo subject numbers (for multiple cohorts) 
    for i,file in enumerate(fnames):
        if i == 0:
            df = pd.read_excel(fnames[i])
            df['Subject'] += 100 #rat 1 becomes rat 101
            if reset_sessions:
                for i,session in enumerate(df.Session.unique()):
                    for j in range(len(df)):
                        if df.at[j,'Session'] == session:
                            df.at[j,'Session'] = i + 1
        else:
            df2 = pd.read_excel(fnames[i])
            df2['Subject'] += 100 * (1+i) #rat 1 becomes rat 201, 301, etc. 
            if reset_sessions:
                for i,session in enumerate(df2.Session.unique()):
                    for j in range(len(df2)):
                        if df2.at[j,'Session'] == session:
                            df2.at[j,'Session'] = i + 1
            df = df.append(df2, ignore_index = True)
    return df

def check_sessions(df): ##checks that the 'Session' column has correct, and non-missing session numbers
    pd.set_option('display.max_rows', None) ##otherwise it will ... the middle rows (only give the head and tail)
    print(df.groupby(['Subject','StartDate','Session'])['Trial'].max())
    pd.set_option('display.max_rows',df.Subject.max()) ##this sets the number of displayed rows to the number of subjects
    
def check_groups(df):
    pd.set_option('display.max_rows', None)
    print(df.groupby(['Subject','Group'])['Trial'].max())
    pd.set_option('display.max_rows',df.Subject.max())
    
def drop_sessions(df, session_nums):
    'Takes in a list of session numbers, and removes the data from specified session numbers'
    for s in session_nums:
        drop_sess = list(df.loc[df['Session'] == s].index)
        df1 = df.drop(drop_sess, inplace = True)
        df1 = df.reset_index(drop = True, inplace = True)
    return df ##could replace with check_sessions(df)

def drop_groups(df, group_nums):
    'Takes in a list of group numbers, and removes the data from specified group numbers'
    for s in group_nums:
        drop_group = list(df.loc[df['Group'] == s].index)
        df.drop(drop_group, inplace = True)
        df.reset_index(drop = True, inplace = True)
    return df ##could replace with check_groups(df) or df

def drop_subjects(df, subs):
    for sub in subs:
        df.drop(list(df.loc[df['Subject']==sub].index), inplace=True)
    df.reset_index(drop = True, inplace=True)
    return df

def edit_sessions(df, orig_sess = None, new_sess = None,subs = 'all'):
    if subs == 'all':
        for i,sess in enumerate(orig_sess):
            for j in range(len(df)):
                if df.at[j,'Session']== sess:
                    df.at[j,'Session'] = new_sess[i]
    else:
        for sub in subs:
            index = list(df.loc[df['Subject']==sub].index)
            for i,sess in enumerate(orig_sess):
                for idx in index:
                    if df.at[idx,'Session'] == sess:
                        df.at[idx,'Session'] = new_sess[i]
    df.reset_index(drop = True, inplace = True)
    return df


def edit_groups(df, orig_group, new_group, subs = 'all'):
    if subs == 'all':
        for i,group in enumerate(orig_group):
            for j in range(len(df)):
                if df.at[j,'Group'] == group:
                    df.at[j,'Group'] = new_group[i]
    else:
        for sub in subs:
            index = list(df.loc[df['Subject']==sub].index)
            for i,group in enumerate(orig_group):
                for idx in index:
                    if df.at[idx,'Group'] == group:
                        df.at[idx,'Group'] = new_group[i]
    df.reset_index(drop = True, inplace = True)
    return df
#------------------------------FILTER FUNCTIONS---------------------------------#    

def filt_vars(df_long_sum, variables = None, task = None): 
    """Takes in long df, variables (list object), and task... and outputs/updates variables"""
    if variables == None: #then run all variables
        if task == 'choiceRGT':
            variables = df_long_sum.columns[2:25] #from 'P1_C' to 'pref'
        else:
            variables = df_long_sum.columns[2:12] #from 'P1' to 'prem'
    return variables

def filt_sess(df_long_sum, sessions = None):   
    """Takes in long df and sessions (list object)... and outputs/updates sessions"""
    if sessions != None:
        df_long_sum = df_long_sum[df_long_sum.Session.isin(sessions)] 
    else:
        sessions = list(df_long_sum.Session.unique()) #all sessions
    return sessions

#------------------------------ANALYZE BY SESSION/GROUP---------------------------------#

def get_choices(df):
    configA = np.array([1, 4, 0, 2, 3]) #this is the order for version A - i.e., hole 1 corresponds to P1
    configB = np.array([4, 1, 0, 3, 2]) #this is the order for version B - i.e., hole 1 corresponds to P4
    
    #this uses the configurations above to assign the correct option, 
    #based on whether the MSN name contains 'A' or 'B'
    df['option'] = df['MSN'].str.contains("B").values*configB[df['Chosen'].astype('int').ravel()-1].astype('int') + \
        df['MSN'].str.contains("A").values*configA[df['Chosen'].astype('int').ravel()-1].astype('int')

    #the above code changes any zero in the chosen column to a three in the option column - fixed with this code
    for i in range(len(df)): 
        if df['Chosen'][i] == 0: 
            df['option'][i] = 0 
    return df    

def get_sum_choice(num, df, mode = 'Session', task = None):
    #get choice information for given group number or session number
    if task == 'choiceRGT':
        df_cued = df.loc[(df[mode] == num) & (df['Cued_Chosen']==1)]
        df_uncued = df.loc[(df[mode] == num) & (df['Uncued_Chosen']==1)]
        subs = df.Subject.unique()
        subs.sort()
        cued_percentage = pd.DataFrame(columns=[str(num) + 'P1_C',str(num) + 'P2_C',str(num) + 'P3_C',str(num) + 'P4_C'])
        uncued_percentage = pd.DataFrame(columns=[str(num) + 'P1_U',str(num) + 'P2_U',str(num) + 'P3_U',str(num) + 'P4_U'])
        for sub in subs:
            for i,column in enumerate(cued_percentage.columns):
                if len(df_cued.loc[(df_cued['option'] != 0) & (df_cued.Subject == sub)]) != 0:
                    cued_percentage.at[sub,column] = (len(df_cued.loc[(df_cued.option == i + 1) & 
                                                    (df_cued.Subject == sub)]))/(len(df_cued.loc[(df_cued['option'] != 0) & 
                                                                                        (df_cued.Subject == sub)])) *100
                elif len(df_cued.loc[(df_cued['option'] != 0) & (df_cued.Subject == sub)]) == 0:
                    cued_percentage.at[sub,column] = 0
            for i,column in enumerate(uncued_percentage.columns):
                if len(df_uncued.loc[(df_uncued['option'] != 0) & (df_uncued.Subject == sub)]) != 0:
                    uncued_percentage.at[sub,column] = (len(df_uncued.loc[(df_uncued.option == i + 1) & 
                                                    (df_uncued.Subject == sub)]))/(len(df_uncued.loc[(df_uncued['option'] != 0) & 
                                                                                        (df_uncued.Subject == sub)])) *100
                elif len(df_uncued.loc[(df_uncued['option'] != 0) & (df_uncued.Subject == sub)]) == 0:
                    uncued_percentage.at[sub,column] = 0
        percentage = pd.concat([cued_percentage,uncued_percentage], axis = 1)
          
    else:
        df1 = df.loc[df[mode] == num]
        subs = df1.Subject.unique()
        subs.sort()
        percentage = pd.DataFrame(columns=[str(num) + 'P1',str(num) + 'P2',str(num) + 'P3',str(num) + 'P4'])
        for sub in subs:
            for i,column in enumerate(percentage.columns):
                if len(df1.loc[(df1['option'] != 0) & (df1.Subject == sub)]) == 0:
                    percentage.at[sub,column] = 0
                else:
                    percentage.at[sub,column] = (len(df1.loc[(df1.option == i + 1) & 
                                                    (df1.Subject == sub)]))/(len(df1.loc[(df1['option'] != 0) & 
                                                                                        (df1.Subject == sub)])) *100
    return percentage

def get_sum_choice_all(df, mode = 'Session', task = None):
    df_sess = []
    for num in np.sort(df[mode].unique()):
        df_sess.append(get_sum_choice(num,df,mode, task = task))
    df1 = pd.concat(df_sess, axis=1)
    if task == 'choiceRGT':
        for num in np.sort(df[mode].unique()):
            df1['risk_cued'+ str(num)] = df1[str(num)+'P1_C'] + df1[str(num)+'P2_C']- df1[str(num)+'P3_C'] - df1[str(num)+'P4_C']
            df1['risk_uncued' + str(num)] = df1[str(num)+'P1_U'] + df1[str(num)+'P2_U']- df1[str(num)+'P3_U'] - df1[str(num)+'P4_U']
    else:
        for num in np.sort(df[mode].unique()):
            df1['risk'+ str(num)] = df1[str(num)+'P1'] + df1[str(num)+'P2']- df1[str(num)+'P3'] - df1[str(num)+'P4']
    return df1

def get_premature(df_raw,df_sum,mode = 'Session', task = None):
    #extract premature response information on either group or session
    
    if task == 'choiceRGT':
        df_uncued = df_raw.loc[df_raw['Uncued_Chosen'] == 1]
        df_cued = df_raw.loc[df_raw['Cued_Chosen'] == 1]
        
        prem_resp_cued = df_cued.groupby(['Subject', mode],as_index=False)['Premature_Resp'].sum()
        prem_resp_cued['Trials'] = df_cued.groupby(['Subject',mode],as_index=False)['Trial'].count()['Trial']
        prem_resp_cued['prem_percent'] = prem_resp_cued['Premature_Resp']/prem_resp_cued['Trials'] * 100
        
        prem_resp_uncued = df_uncued.groupby(['Subject', mode],as_index=False)['Premature_Resp'].sum()
        prem_resp_uncued['Trials'] = df_uncued.groupby(['Subject',mode],as_index=False)['Trial'].count()['Trial']
        prem_resp_uncued['prem_percent'] = prem_resp_uncued['Premature_Resp']/prem_resp_uncued['Trials'] * 100

        for num in np.sort(df_raw[mode].unique()):
            df_sum['prem_cued' + str(num)] = prem_resp_cued.loc[prem_resp_cued[mode]==num].set_index('Subject')['prem_percent']
            df_sum['prem_uncued' + str(num)] = prem_resp_uncued.loc[prem_resp_uncued[mode]==num].set_index('Subject')['prem_percent']
        return df_sum 
    
    prem_resp = df_raw.groupby(['Subject', mode],as_index=False)['Premature_Resp'].sum()
    
    prem_resp['Trials'] = df_raw.groupby(['Subject',mode],as_index=False)['Trial'].count()['Trial']

    prem_resp['prem_percent'] = prem_resp['Premature_Resp']/prem_resp['Trials'] * 100

    for num in np.sort(df_raw[mode].unique()):
        df_sum['prem' + str(num)] = prem_resp.loc[prem_resp[mode]==num].set_index('Subject')['prem_percent']
        

    return df_sum

def get_latencies(df_raw,df_sum,mode = 'Session', task = None):
    #extract collect and choice lat information
    
    if task == 'choiceRGT':
        #collect lat
        df_cued = df_raw.loc[(df_raw['Rewarded'] == 1) & (df_raw['Cued_Chosen'] == 1)]
        df_uncued = df_raw.loc[(df_raw['Rewarded'] == 1) & (df_raw['Uncued_Chosen'] == 1)]
        collect_lat_uncued = df_uncued.groupby(['Subject',mode],as_index=False)['Collect_Lat'].mean()
        collect_lat_cued = df_cued.groupby(['Subject',mode],as_index=False)['Collect_Lat'].mean()
    
        for num in np.sort(df_raw[mode].unique()):
            df_sum['co_lat_cued' + str(num)] = collect_lat_cued.loc[collect_lat_cued[mode]==num].set_index('Subject')['Collect_Lat']
            df_sum['co_lat_uncued' + str(num)] = collect_lat_uncued.loc[collect_lat_uncued[mode]==num].set_index('Subject')['Collect_Lat']
            
        #choice lat
        df_cued = df_raw.loc[(df_raw['Chosen'] != 0) & (df_raw['Cued_Chosen'] == 1)]
        df_uncued = df_raw.loc[(df_raw['Chosen'] != 0) & (df_raw['Uncued_Chosen'] == 1)]
        choice_lat_uncued = df_uncued.groupby(['Subject',mode],as_index=False)['Choice_Lat'].mean()
        choice_lat_cued = df_cued.groupby(['Subject',mode],as_index=False)['Choice_Lat'].mean()
        for num in np.sort(df_raw[mode].unique()):
            df_sum['ch_lat_cued' + str(num)] = choice_lat_cued.loc[choice_lat_cued[mode]==num].set_index('Subject')['Choice_Lat']
            df_sum['ch_lat_uncued' + str(num)] = choice_lat_uncued.loc[choice_lat_uncued[mode]==num].set_index('Subject')['Choice_Lat']
        
        #lever latency
        df_uncued = df_raw.loc[df_raw['Uncued_Chosen'] == 1]
        df_cued = df_raw.loc[df_raw['Cued_Chosen'] == 1]
        
        uncued_lever_lat = df_uncued.groupby(['Subject',mode], as_index = False)['Lever_Latency'].mean()
        cued_lever_lat = df_cued.groupby(['Subject',mode], as_index = False)['Lever_Latency'].mean()
        
        for num in np.sort(df_raw[mode].unique()):
            df_sum['cued_lev_lat' + str(num)] = cued_lever_lat.loc[cued_lever_lat[mode]==num].set_index('Subject')['Lever_Latency']
#         for num in np.sort(df_raw[mode].unique()):
            df_sum['uncued_lev_lat' + str(num)] = uncued_lever_lat.loc[uncued_lever_lat[mode]==num].set_index('Subject')['Lever_Latency']
        return df_sum
    
    df_raw = df_raw.loc[df_raw['Chosen'] != 0]
    df_raw2 = df_raw.loc[df_raw['Rewarded'] == 1]    
    
    collect_lat = df_raw2.groupby(['Subject',mode],as_index=False)['Collect_Lat'].mean()
    choice_lat = df_raw.groupby(['Subject',mode],as_index=False)['Choice_Lat'].mean()
   
    
    for num in np.sort(df_raw[mode].unique()):
        df_sum['collect_lat' + str(num)] = collect_lat.loc[collect_lat[mode]==num].set_index('Subject')['Collect_Lat']
    for num in np.sort(df_raw[mode].unique()):
        df_sum['choice_lat' + str(num)] = choice_lat.loc[choice_lat[mode]==num].set_index('Subject')['Choice_Lat']
        

    return df_sum

def get_omit(df_raw,df_sum,mode = 'Session', task = None):
    
    if task == 'choiceRGT': 
        df_cued = df_raw.loc[df_raw['Cued_Chosen'] == 1]
        df_uncued = df_raw.loc[df_raw['Uncued_Chosen'] == 1]
        
        cued_omit = df_cued.groupby(['Subject',mode], as_index = False)['Omit'].sum()
        cued_omit['Trials'] = df_cued.groupby(['Subject',mode],as_index=False)['Trial'].count()['Trial']
        cued_omit['omit_percent'] = cued_omit['Omit']/cued_omit['Trials'] * 100
        
        uncued_omit = df_uncued.groupby(['Subject',mode], as_index = False)['Omit'].sum()
        uncued_omit['Trials'] = df_uncued.groupby(['Subject',mode],as_index=False)['Trial'].count()['Trial']
        uncued_omit['omit_percent'] = uncued_omit['Omit']/uncued_omit['Trials'] * 100
        
        lev_omit = df_raw.groupby(['Subject',mode],as_index=False)['Choice_Omit'].sum()
        lev_omit['Trials'] = df_raw.groupby(['Subject',mode],as_index=False)['Trial'].count()['Trial']
        lev_omit['lev_omit_percent'] = lev_omit['Choice_Omit']/lev_omit['Trials'] * 100
        
        for num in np.sort(df_raw[mode].unique()):
            df_sum['cued_omit' + str(num)] = cued_omit.loc[cued_omit[mode]==num].set_index('Subject')['omit_percent']
            df_sum['uncued_omit' + str(num)] = uncued_omit.loc[uncued_omit[mode]==num].set_index('Subject')['omit_percent']
            df_sum['lev_omit' + str(num)] = lev_omit.loc[lev_omit[mode]==num].set_index('Subject')['lev_omit_percent']
        
        return df_sum 
    
    
#         lev_omit = df_raw.groupby(['Subject',mode],as_index=False)['Choice_Omit'].sum()
#         for num in np.sort(df_raw[mode].unique()):
    
    omit = df_raw.groupby(['Subject',mode],as_index=False)['Omit'].sum()
    for num in np.sort(df_raw[mode].unique()):
        df_sum['omit' + str(num)] = omit.loc[omit[mode]==num].set_index('Subject')['Omit']
    return df_sum

def get_trials(df_raw,df_sum,mode = 'Session', task = None):
    if task == 'choiceRGT':
        trials = df_raw.groupby(['Subject',mode],as_index=False)['Trial'].count()
        for num in np.sort(df_raw[mode].unique()):
            df_sum['trial_init' + str(num)] = trials.loc[trials[mode]==num].set_index('Subject')['Trial']
        return df_sum
#         df_cued = df_raw.loc[df_raw['Cued_Chosen'] == 1]
            
#         trials_cued = df_cued.groupby(['Subject', mode],as_index=False)['Trial'].max()
#         for num in np.sort(df_cued[mode].unique()):
#             df_sum['trial_cued_' + str(num)] = trials_cued.loc[trials_cued[mode]==num].set_index('Subject')['Trial']
            
#         df_uncued = df_raw.loc[df_raw['Uncued_Chosen'] == 1]
            
#         trials_uncued = df_uncued.groupby(['Subject', mode],as_index=False)['Trial'].max()
#         for num in np.sort(df_uncued[mode].unique()):
#             df_sum['trial_uncued_' + str(num)] = trials_uncued.loc[trials_uncued[mode]==num].set_index('Subject')['Trial']
#         return df_sum

# ---

#         df_cued = df_raw.loc[df_raw['Cued_Chosen'] == 1]
#         df_uncued = df_raw.loc[df_raw['Uncued_Chosen'] == 1]
        
#         trials_uncued = df_uncued.groupby(['Subject', mode],as_index=False)['Trial'].max()
#         trials_cued = df_cued.groupby(['Subject', mode],as_index=False)['Trial'].max()

#         for num in np.sort(df_raw[mode].unique()):
#             df_sum['trial_cued_' + str(num)] = trials_cued.loc[trials_cued[mode]==num].set_index('Subject')['Trial']
#             df_sum['trial_uncued_' + str(num)] = trials_uncued.loc[trials_uncued[mode]==num].set_index('Subject')['Trial']
            
#         return df_sum
        
        
    trials = df_raw.groupby(['Subject', mode],as_index=False)['Trial'].max()
    for num in np.sort(df_raw[mode].unique()):
        df_sum['trial' + str(num)] = trials.loc[trials[mode]==num].set_index('Subject')['Trial']
    return df_sum

def get_trials_init(df_raw,df_sum,mode = 'Session'):
    ##This function is not included in get_summary_data
    trials = df_raw.groupby(['Subject',mode],as_index=False)['Trial'].count()
    for num in np.sort(df_raw[mode].unique()):
        df_sum['trial_init' + str(num)] = trials.loc[trials[mode]==num].set_index('Subject')['Trial']
    return df_sum

def get_preference_score(df_raw,df_sum,mode = 'Session'):
    uncued_picks = df_raw.groupby(['Subject', mode],as_index=False)['Uncued_Chosen'].sum()
    uncued_picks['cued_picks'] = df_raw.groupby(['Subject', mode],as_index=False)['Cued_Chosen'].sum()['Cued_Chosen']
    uncued_picks['pref_score'] = uncued_picks['cued_picks']/(uncued_picks['cued_picks'] + uncued_picks['Uncued_Chosen']) * 100
    
    for num in np.sort(df_raw[mode].unique()):
        df_sum['pref' + str(num)] = uncued_picks.loc[uncued_picks[mode]==num].set_index('Subject')['pref_score'] 
    return df_sum

#-------------------------------GET SUMMARY DATA--------------------------------#

def get_summary_data(df_raw, mode = 'Session', task = None):
    df_raw = get_choices(df_raw)
    df_sum = get_sum_choice_all(df_raw,mode,task)
    df_sum = get_latencies(df_raw,df_sum,mode,task)
    df_sum = get_omit(df_raw,df_sum,mode,task)
    df_sum = get_trials(df_raw,df_sum,mode,task)
    df_sum = get_premature(df_raw,df_sum,mode,task)
    if task == 'choiceRGT':
        df_sum = get_preference_score(df_raw,df_sum,mode)
    return df_sum

def get_long_summary_data(df_edited, df_sum, task = None): 
    "takes in the edited df (after dropping subjects and sessions) and wide-summary df, and outputs a long-summary df"
    "Cannot take subjects with different session numbers"
    subs = df_edited.Subject.unique() #list of subjects
    subs.sort() 

    sess = list(df_edited.Session.unique())
    sess.sort()
    all_sess = []
    for i in range(len(subs)): #for all subjects 
        all_sess.append(sess) #all_sess is now a list of lists (sess)

    df_temp = df_edited.groupby(['Subject','Session'],as_index=False)['Trial'].max()
    df_temp.drop('Trial', inplace=True, axis=1)

    if task == 'choiceRGT':
        choice_names = ['P1_C','P2_C','P3_C','P4_C','P1_U','P2_U','P3_U','P4_U']
    else:
        choice_names = ['P1','P2','P3','P4'] #for P1-P4 

    for col in choice_names:
        df_temp[col] = [0] * len(df_temp) #create a column of zeros for the length
        for session in sess: #for each session
            for rat in subs: #for each subject
                idx = df_temp.loc[np.logical_and(df_temp['Subject'] == rat, df_temp['Session'] == session)].index.values
                df_temp.at[idx, col] = df_sum.at[rat, str(session) + col] 

    if task == 'choiceRGT':
        column_names = ['risk_cued','risk_uncued','co_lat_cued','co_lat_uncued','ch_lat_cued','ch_lat_uncued',
                        'cued_lev_lat','uncued_lev_lat','cued_omit','uncued_omit','lev_omit','trial_init','prem_cued','prem_uncued','pref']
    else:
        column_names = ['risk','collect_lat','choice_lat','omit','trial','prem'] #for variables ending in session

    for col in column_names:
        df_temp[col] = [0] * len(df_temp)
        for session in sess:
            for rat in subs:
                idx = df_temp.loc[np.logical_and(df_temp['Subject'] == rat, df_temp['Session'] == session)].index.values
                df_temp.at[idx, col] = df_sum.at[rat, col + str(session)]

    return df_temp

def impute_missing_data(df1, session = None, subject = None, choice = None, vars = None):
    if choice == 'all':
        for i in range(1,5):
            df1.at[subject,str(session)+'P'+str(i)] = np.mean([df1.at[subject,str(session-1)+'P'+str(i)],
                                                               df1.at[subject,str(session+1)+'P'+str(i)]])
    elif choice != None:
        for i in choice:
            df1.at[subject,str(session)+'P'+str(i)] = np.mean([df1.at[subject,str(session-1)+'P'+str(i)],
                                                               df1.at[subject,str(session+1)+'P'+str(i)]])
    if vars == 'all':
        vars = ['risk','collect_lat','choice_lat','omit','trial','prem']
        for var in vars:
            df1.at[subject,var+str(session)] = np.mean([df1.at[subject,var+str(session-1)],
                                                        df1.at[subject,var + str(session+1)]])
    elif vars != None:
        for var in vars:
            df1.at[subject,var+str(session)] = np.mean([df1.at[subject,var+str(session-1)],
                                                        df1.at[subject,var + str(session+1)]])
    return df1

#--------------------------------GET LONG SUMMARY DATA VARIABLES-------------------------------#

def get_risk_status_long(df_long_sum, sessions = None): 
    """takes in long df summary data and list of sessions, and gets the mean risks and risk status and appends it to df_long
    if sessions is not passed, all sessions in df_long are used"""
   
    #objects
    subs = df_long_sum.Subject.unique()
    mean_risk_list = []
    sessions = filt_sess(df_long_sum, sessions)
         
    for s in subs: 
        df_sub = df_long_sum.loc[(df_long_sum['Session'].isin(sessions)) & (df_long_sum['Subject'] == s)] #df where Subject == s and where Session == startsess to endsess 
        mean_risk = df_sub['risk'].mean() #mean_risk
        for s in sessions:
            mean_risk_list.append(mean_risk)
    df_long_sum["mean_risk"] = mean_risk_list 
    
    for row in df_long_sum.index: #for each row
        if df_long_sum.at[row,'mean_risk'] > 0:
            df_long_sum.at[row,'risk_status'] = 1 
        elif df_long_sum.at[row,'mean_risk'] < 0: 
            df_long_sum.at[row,'risk_status'] = 2 
    return df_long_sum

def get_group_long(df_long, group_list):
    """takes in df_long and group_list, and creates a column called group. Group == 1 represents the first group passed to group_list, and so on."""
    for row in df_long.index: 
        for group in group_list: 
            if np.isin(df_long.at[row,'Subject'], group):
                if group == group_list[0]:
                    df_long.at[row,'group'] = 1
                elif group == group_list[1]:
                    df_long.at[row,'group'] = 2
                elif group == group_list[2]:
                    df_long.at[row,'group'] = 3
                elif group == group_list[3]:
                    df_long.at[row,'group'] = 4
    return df_long

#--------------------------------ANOVA-------------------------------#

def rm_anova(df_long_sum, variables = None, sessions = None, task = None): 
    """Takes in long-summary data (df), outcome variables (list), sessions (list) and task... then runs a RM ANOVA on those variables across those sessions.
    If nothing is passed to variables or sessions, the RM ANOVA will be run on all variables and sessions in df_long_sum
    Must pass at least 2 session numbers to sessions
    The function will print the unstable df, and stable df"""
    
    #filter rows by sessions, and select columns by variables
    variables = filt_vars(df_long_sum, variables, task)
    sessions = filt_sess(df_long_sum, sessions)
    
    #run anova
    unstable_dict = {} #dict was required to make the list run down the dataframe (as opposed to across)
    unstable_pvals = []
    unstable_vars = []
    stable_dict = {}
    stable_pvals = []
    stable_vars = []
    
    for var in variables: #for each variable, run a RM anova
        res = pg.rm_anova(dv=var, within='Session', subject='Subject', data=df_long_sum, detailed=True)
        pval = res['p-unc'][0]
        if pval < 0.05: 
            unstable_pvals.append(pval)
            unstable_vars.append(var)
        else: 
            stable_pvals.append(pval)
            stable_vars.append(var)
            
    unstable_dict['variable'] = unstable_vars
    unstable_dict['p-value'] = unstable_pvals
    unstable_df = pd.DataFrame(data=unstable_dict)
    stable_dict['variable'] = stable_vars
    stable_dict['p-value'] = stable_pvals
    stable_df = pd.DataFrame(data=stable_dict)
            
    #print unstable and stable dataframes
    print(f'Unstable df: {unstable_df}\nStable df: {stable_df}')

def mixed_anova(df_long_sum, bsf, variables = None, sessions = None, show_df = None, task = None):
    """Takes in long-summary data with between-subjects factors of interest as separate columns (dataframe), 1 between-subjects factor (string), outcome variables (list), sessions (list),
    specific variables (list) and task...and runs a mixed ANOVA on those variables across those sessions.
    If nothing is passed to variables or sessions, the RM ANOVA will be run on all variables and sessions in df_long_sum
    Must pass at least 2 session numbers to sessions
    The function will output a list of unstable variable(s) and stable variables, and dfs for the variables specified in show_df"""
    
    #filter rows by sessions, and select columns by variables
    variables = filt_vars(df_long_sum, variables, task)
    sessions = filt_sess(df_long_sum, sessions)
        
    #run anova
    unstable_list = []
    stable_list = []
    if show_df == None: 
        show_df = []
    for var in variables: 
        pvals = []
        res = pg.mixed_anova(df_long_sum, dv=var, within='Session', subject='Subject', between = bsf)
        if (var in show_df):
            print(f'{res}{var}')
        for pval in range(3):
            pvals.append(res['p-unc'][pval])
        if any(x < 0.05 for x in pvals):
            unstable_list.append(var)
        else: 
            stable_list.append(var)
            
    #return unstable and stable list
    print(f'Unstable list: {unstable_list}\nStable list: {stable_list}')

#--------------------------------GET RISK STATUS-------------------------------#

def get_risk_status(df_sum, startsess, endsess, task = None):
    #get risk status from specified sessions
    #create lists for indexing based on risk status
    risky = []
    optimal = []
    startsess = 'risk' + str(startsess)
    endsess = 'risk' + str(endsess)
    #calculate the mean risk score from the specified sessions
    df_sum['mean_risk'] = df_sum.loc[:,startsess:endsess].mean(axis=1) ###did this create a 'mean_risk' column?
    for sub in df_sum.index: #for each subject
        if df_sum.at[sub,'mean_risk'] > 0: #if the mean risk for that subject is above zero
            df_sum.at[sub,'risk_status'] = 1 #assign them a risk status of 1
            optimal.append(sub) #and add them to the 'optimal' list
        elif df_sum.at[sub,'mean_risk'] < 0: #if the mean risk for that subject is below zero
            df_sum.at[sub,'risk_status'] = 2 #assign them a risk status of 2
            risky.append(sub) #and append them to the 'risky' list
    return df_sum, risky, optimal

def get_risk_status_vehicle(df1):
    #get risk status from 'risk1' only - ie in the case of LS, the saline dose
    #create lists for indexing based on risk score
    risky = []
    optimal = []
    for sub in df1.index:
        if df1.at[sub,'risk1'] > 0:
            df1.at[sub,'risk_status'] = 1
            optimal.append(sub)
        elif df1.at[sub,'risk1'] < 0:
            df1.at[sub,'risk_status'] = 2
            risky.append(sub)
    return df1, risky, optimal

#-------------------------------EXPORT TO EXCEL--------------------------------#

def export_to_excel(df, groups = None, column_name = 'group', new_file_name = 'summary_data', asin = False):
    if groups == None:
        df_export = df.copy()
        if asin:
            col_list = [col for col in df_export.columns if 'P' in col] + [col for col in df_export.columns if 'prem' in col]
            for col in col_list:
                for sub in df_export.index:
                    df_export.at[sub,col] = np.arcsin(math.sqrt(df_export.at[sub,col]/100))
    else:
        dfs = []
        for group in groups: #this splits the dataframe by group
            dfs.append(df.loc[group])
        for i,df in enumerate(dfs): #this assigns a number to the group column - in this case, 0 for control, 1 for experimental
            df[column_name] = i ##i should be 0 and 1
        df_export = pd.concat(dfs) #this recombines the dataframes
        if asin:
            col_list = [col for col in df_export.columns if 'P' in col] + [col for col in df_export.columns if 'prem' in col]
            for col in col_list:
                for sub in df_export.index:
                    df_export.at[sub,col] = np.arcsin(math.sqrt(df_export.at[sub,col]/100))
        df_export.sort_index(inplace = True) #this sorts the subjects so they're in the right order after combining
    df_export.to_excel(new_file_name, index_label = 'Subject')
    

#------------------------------GET EXPERIMENTAL/CONTROL GROUP MEANS---------------------------------#
   
def get_means_sem(df_sum, groups = None, group_names = None): 
    if groups == None:
        mean_scores = pd.DataFrame(columns=list(df_sum.columns))
        SEM = pd.DataFrame(columns=mean_scores.columns)
        for column in mean_scores.columns:
            mean_scores.at[0,column] = df_sum[column].mean()
            SEM.at[0,column] = stats.sem(df_sum[column])
        mean_scores.rename(index={0: 'All rats'},inplace = True)
        SEM.rename(index={0: 'All rats'}, inplace = True)
    else:
        dfs = []
        #first split the dataframe based on experimental vs control
        for group in groups:
            dfs.append(df_sum.loc[group])
        #create two dataframes - one for the means, one for the SEM
        mean_scores = pd.DataFrame(columns=list(df_sum.columns))
        SEM = pd.DataFrame(columns=mean_scores.columns)
        #calculate the mean and standard errors, and store them in the above dataframes
        for column in mean_scores.columns:
            for i in range(len(groups)):
                mean_scores.at[i,column] = dfs[i][column].mean()
                SEM.at[i,column] = stats.sem(dfs[i][column])
        #rename the rows to be the group_names (i.e., transgene positive and transgene negative)   
        mean_scores.rename(index=group_names,inplace = True)
        SEM.rename(index=group_names, inplace = True)
    return mean_scores, SEM


#------------------------------PLOTTING BY SESSION---------------------------------#
    
def rgt_plot(variable,startsess,endsess,title,scores,sem, group_names = None, highlight = None, y_label = None, x_label = 'Session'):
    if y_label == None:
        y_label = variable
    plt.rcParams.update({'font.size': 18})
    fig,ax = plt.subplots(figsize = (15,8))
    ax.set_ylabel(y_label, fontweight='bold', fontsize = 20)
    ax.set_xlabel(x_label, fontweight = 'bold', fontsize = 20)
    ax.set_title(title + ': ' + y_label + '\n' + x_label + ' ' + str(startsess) + '-' + str(endsess),
                fontweight = 'bold', fontsize = 22, pad = 20)
    ax.spines['right'].set_linewidth(0)
    ax.spines['top'].set_linewidth(0)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xticks(np.arange(startsess,endsess+1))
    
    #define x axis (range of session numbers we are graphing)
    x=np.arange(startsess,endsess+1)
    #extract column names to include in figure
    columns = []
    for col in scores.columns:
        for session_number in [str(i) for i in x]: #turns session numbers in x into strings
            #check if variable name is in column name and column name ends with last letter of variable name + session number
            if variable in col and col.endswith(variable[-1] + session_number):
                #if so, append column name to columns list
                columns.append(col)

    if group_names == None:
        y = scores.loc['All rats',columns]
        plt.errorbar(x, y,
                     yerr = sem.loc['All rats',columns], 
                     linewidth=5, capsize = 8)
    else:
        for i,group in enumerate(group_names.values()):
            y = scores.loc[group,columns]
            plt.errorbar(x, y,
                         yerr = sem.loc[group,columns], 
                         label=group,linewidth=5, capsize = 8)
            ax.legend()
       
    if highlight != None:
        plt.axvline(highlight, 0, 1, color = 'gray', lw = 1)
        ax.fill_between([highlight,endsess], ax.get_ylim()[0], ax.get_ylim()[1], facecolor='gray', alpha=0.2)
        
def rgt_bar_plot(variable,startsess,endsess,title,scores,sem, group_names = None, y_label = None):
    if y_label == None:
        y_label = variable
    plt.rcParams.update({'font.size': 18}) 
    plt.rcParams["figure.figsize"] = (15,8)
    
    bars = [0]*len(scores.index)
    err = [0]*len(scores.index)
    sess = [str(x) for x in np.arange(startsess,endsess+1)] #list of session numbers as strings
    #extract column names to include in figure
    columns = []
    for col in scores.columns:
        for session_number in sess: 
            #check if variable name is in column name and column name ends with last letter of variable name + session number
            if variable in col and col.endswith(variable[-1] + session_number):
                #if so, append column name to columns list
                columns.append(col)
    
    for i,group in enumerate(scores.index):
        bars[i] = scores.loc[group, columns].mean()
        err[i] = sem.loc[group,columns].mean()
        plt.bar(i, bars[i], yerr = err[i], capsize = 8, width = .7, color = ['C'+str(i)], label = scores.index[i])
   
    ax = plt.gca()
    plt.xticks([])
    plt.xlabel('Group', labelpad = 20, fontweight = 'bold', fontsize = 20)
    ax.set_ylabel(y_label, fontweight = 'bold', fontsize = 20)
    #ax.set_ylim(0,50)
    ax.set_title(title + ': ' + y_label + '\n' + 'Session ' + str(startsess) + '-' + str(endsess),
                 fontweight = 'bold', fontsize = 22, pad = 20)
    ax.spines['right'].set_linewidth(0)
    ax.spines['top'].set_linewidth(0)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.legend()
        
        
def choice_bar_plot(startsess, endsess, scores, sem, task = None):
    sess = list(range(startsess,endsess + 1))
    if task == 'choiceRGT':
        labels = ['P1_C','P2_C','P3_C','P4_C','P1_U','P2_U','P3_U','P4_U']
    else:
        labels = ['P1','P2','P3','P4']
    df = pd.DataFrame()
    df1 = pd.DataFrame()
    

    for choice in labels:
        df[choice] = scores.loc[:, [col for col in scores.columns if choice in col 
                                    and int(col[:col.index('P')]) in sess]].mean(axis = 1)
        df1[choice] = sem.loc[:, [col for col in scores.columns if choice in col 
                                    and int(col[:col.index('P')]) in sess]].mean(axis = 1)
    ax = df.transpose().plot.bar(rot = 0, yerr = df1.transpose(), capsize = 8, figsize = (15,8))
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.rcParams.update({'font.size': 18})
    ax.set_ylabel('% Choice', fontweight = 'bold', fontsize = 20)
    ax.set_title( 'P1-P4 Choice'  '\n' + 'Session ' + str(startsess) + '-' + str(endsess), 
                 fontweight = 'bold', fontsize = 22, pad = 20)
    ax.set_ylim(bottom = 0)
    ax.xaxis.label.set_size(20)
    ax.spines['right'].set_linewidth(0)
    ax.spines['top'].set_linewidth(0)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.legend()
    
#------------------------------PLOTTING for Latin Squares---------------------------------#
    
def ls_bar_plot(figure_group, group_means, sem):
    labels = ['P1','P2','P3','P4']
    veh_means =  list(group_means.loc[figure_group,[col for col in group_means.columns if col.startswith('1')]])
    dose1_means = list(group_means.loc[figure_group,[col for col in group_means.columns if col.startswith('2')]])
    dose2_means = list(group_means.loc[figure_group,[col for col in group_means.columns if col.startswith('3')]])
    dose3_means = list(group_means.loc[figure_group,[col for col in group_means.columns if col.startswith('4')]])

    x = np.arange(len(labels))*3  # the label locations
    width = 0.5  # the width of the bars

    fig, ax = plt.subplots(figsize = (15,8))
    rects1 = ax.bar(x - width/2-width, veh_means, width, label='Vehicle',
                    yerr = list(sem.loc[figure_group,[col for col in sem.columns if col.startswith('1')]]), capsize = 8, ecolor='C0')
    rects2 = ax.bar(x - width/2, dose1_means, width, label='Dose 1', 
                    yerr =list(sem.loc[figure_group,[col for col in sem.columns if col.startswith('2')]]), capsize = 8, ecolor='C1')
    rects3 = ax.bar(x + width/2, dose2_means, width, label='Dose 2',
                    yerr = list(sem.loc[figure_group,[col for col in sem.columns if col.startswith('3')]]),capsize = 8, ecolor='C3',color = 'C3')
    rects4 = ax.bar(x + width/2+width, dose3_means, width, label='Dose 3',capsize = 8, 
                    yerr = list(sem.loc[figure_group,[col for col in sem.columns if col.startswith('4')]]), ecolor='C2', color = 'C2')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('% Choice', fontweight = 'bold', fontsize = 20)
    ax.set_title(figure_group + ': P1-P4', fontweight = 'bold', fontsize = 24, pad = 20)
    ax.set_ylim(bottom = 0,top = 100)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.spines['right'].set_linewidth(0)
    ax.spines['top'].set_linewidth(0)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.legend()