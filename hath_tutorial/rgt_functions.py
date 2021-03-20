"""
This module includes functions to load in MEDPC data and outputs a dataframe. 

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
        df.drop(drop_sess, inplace = True)
        df.reset_index(inplace = True)
    return None ##could replace with check_sessions(df)

def drop_groups(df, group_nums):
    'Takes in a list of session numbers, and removes the data from specified session numbers'
    for s in group_nums:
        drop_group = list(df.loc[df['Group'] == s].index)
        df.drop(drop_group, inplace = True)
        df.reset_index(inplace = True)
    return None ##could replace with check_groups(df) or df

def drop_subjects(df, subs):
    for sub in subs:
        df.drop(list(df.loc[df['Subject']==sub].index), inplace=True)
    df.reset_index(inplace=True)
    return df

def edit_sess(df, orig_sess, new_sess,subs = 'all'):
    if subs == 'all':
        for i,sess in enumerate(orig_sess):
            for i in range(len(df)):
                if df.at[i,'Session']== sess:
                    df.at[i,'Session'] = new_sess[i]
    else:
        for sub in subs:
            index = list(df.loc[df['Subject']==sub].index)
            for i,sess in enumerate(orig_sess):
                for idx in index:
                    if df.at[idx,'Session'] == sess:
                        df.at[idx,'Session'] = new_sess[i]
    return df
    
#------------------------------ANALYZE BY SESSION/GROUP---------------------------------#

def get_choices(df):
    configA = np.array([1, 4, 0, 2, 3]) #this is the order for version A - i.e., hole 1 corresponds to P1
    configB = np.array([4, 1, 0, 3, 2]) #this is the order for version B - i.e., hole 1 corresponds to P4

    #I took the following code from someone else, so honestly I'm not entirely sure how it works haha
    #the important thing is that it uses the configurations above to assign the correct option, 
    #based on whether the MSN name contains 'A' or 'B'
    df['option'] = df['MSN'].str.contains("B").values*configB[df['Chosen'].astype('int').ravel()-1].astype('int') + \
        df['MSN'].str.contains("A").values*configA[df['Chosen'].astype('int').ravel()-1].astype('int')
    
    ###I can just take your word for it right? haha
    ####Yes

    #the above code changes any zero in the chosen column to a three in the option column - don't need to know why
    #so we need to fix that (zeros represent either a premature response or an omission)
    for i in range(len(df)): ##range gives me a list from 0 to the len(df), which should be all the indices
        if df['Chosen'][i] == 0: ###can we say this in English? 
            ##the same as df.at
            ##if the index of the 'Chosen' column gets 0 --> option equals 0 
            df['option'][i] = 0 
    return df    

def get_sum_choice(num, df, mode = 'Session'):
    #get choice information for given group number or session number
    df1 = df.loc[df[mode] == num]
    subs = df1.Subject.unique()
    subs.sort()
    percentage = pd.DataFrame(columns=[str(num) + 'P1',str(num) + 'P2',str(num) + 'P3',str(num) + 'P4'])
    for sub in subs:
        for i,column in enumerate(percentage.columns):
            percentage.at[sub,column] = (len(df1.loc[(df1.option == i + 1) & 
                                            (df1.Subject == sub)]))/(len(df1.loc[(df1['option'] != 0) & 
                                                                                (df.Subject == sub)])) *100
    return percentage

def get_sum_choice_all(df, mode = 'Session'):
    df_sess = []
    for num in np.sort(df[mode].unique()):
        df_sess.append(get_sum_choice(num,df,mode))
    df1 = pd.concat(df_sess, axis=1)
    for num in np.sort(df[mode].unique()):
        df1['risk'+ str(num)] = df1[str(num)+'P1'] + df1[str(num)+'P2']- df1[str(num)+'P3'] - df1[str(num)+'P4']
    return df1

def get_premature(df_raw,df_sum,mode = 'Session'):
    #extract premature response information on either group or session
    prem_resp = df_raw.groupby(['Subject', mode],as_index=False)['Premature_Resp'].sum()

    prem_resp['Trials'] = df_raw.groupby(['Subject',mode],as_index=False)['Trial'].count()['Trial']

    prem_resp['prem_percent'] = prem_resp['Premature_Resp']/prem_resp['Trials'] * 100

    for num in np.sort(df_raw[mode].unique()):
        df_sum['prem' + str(num)] = prem_resp.loc[prem_resp[mode]==num].set_index('Subject')['prem_percent']
    return df_sum

def get_latencies(df_raw,df_sum,mode = 'Session'):
    #extract collect and choice lat information
    df_raw = df_raw.loc[df_raw['Chosen'] != 0]
    collect_lat = df_raw.groupby(['Subject',mode],as_index=False)['Collect_Lat'].mean()
    choice_lat = df_raw.groupby(['Subject',mode],as_index=False)['Choice_Lat'].mean()
    for num in np.sort(df_raw[mode].unique()):
        df_sum['collect_lat' + str(num)] = collect_lat.loc[collect_lat[mode]==num].set_index('Subject')['Collect_Lat']
    for num in np.sort(df_raw[mode].unique()):
        df_sum['choice_lat' + str(num)] = choice_lat.loc[choice_lat[mode]==num].set_index('Subject')['Choice_Lat']
    return df_sum

def get_omit(df_raw,df_sum,mode = 'Session'):
    omit = df_raw.groupby(['Subject',mode],as_index=False)['Omit'].sum()
    for num in np.sort(df_raw[mode].unique()):
        df_sum['omit' + str(num)] = omit.loc[omit[mode]==num].set_index('Subject')['Omit']
    return df_sum

def get_trials(df_raw,df_sum,mode = 'Session'):
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

#-------------------------------GET SUMMARY DATA--------------------------------#

def get_summary_data(df_raw, mode = 'Session'):
    df_raw = get_choices(df_raw)
    df_sum = get_sum_choice_all(df_raw,mode)
    df_sum = get_latencies(df_raw,df_sum,mode)
    df_sum = get_omit(df_raw,df_sum,mode)
    df_sum = get_trials(df_raw,df_sum,mode)
    df_sum = get_premature(df_raw,df_sum,mode)
    return df_sum

#--------------------------------GET RISK STATUS-------------------------------#

def get_risk_status(df_sum, startsess, endsess):
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

def export_to_excel(df,groups,column_name = 'group',file_name = 'summary_data'):
    dfs = []
    for group in groups: #this splits the dataframe by group
        dfs.append(df.loc[group])
    for i,df in enumerate(dfs): #this assigns a number to the tg_status column - in this case, 0 for control, 1 for experimental
        df[column_name] = i ##i should be 0 and 1
    df_export = pd.concat(dfs) #this recombines the dataframes
    df_export.sort_index(inplace = True) #this sorts the subjects so they're in the right order after combining
    df_export.to_excel(file_name, index_label = 'Subject')
    
#------------------------------GET EXPERIMENTAL/CONTROL GROUP MEANS---------------------------------#
   
def get_group_means_sem(df_sum,groups, group_names): ##exact same in ls and data_prep --> but objects are named differently
    dfs = []
    #first split the dataframe based on experimental vs control
    for group in groups:
        dfs.append(df_sum.loc[group])
    #create two dataframes - one for the means, one for the SEM
    mean_scores = pd.DataFrame(columns=list(df_sum.columns))
    stderror = pd.DataFrame(columns=mean_scores.columns)
    #calculate the mean and standard errors, and store them in the above dataframes
    for column in mean_scores.columns:
        for i in range(len(groups)):
            mean_scores.at[i,column] = dfs[i][column].mean()
            stderror.at[i,column] = stats.sem(dfs[i][column])
    #rename the rows to be the group_names (i.e., transgene positive and transgene negative)   
    mean_scores.rename(index=group_names,inplace = True)
    stderror.rename(index=group_names, inplace = True)
    return mean_scores, stderror

#------------------------------PLOTTING BY SESSION---------------------------------#

def rgt_plot(variable,startsess,endsess,group_names,title,scores,sem, highlight = None, var_title = None):
    ##startsess and endsess allow us to clip the session data 
    if var_title == None:
        var_title = variable
    plt.rcParams.update({'font.size': 22})
    fig,ax = plt.subplots(figsize = (20,10))
    ax.set_ylabel(var_title)
    ax.set_xlabel('Session')
    ax.set_xlim(startsess,endsess)
    ax.set_title(title + ': ' + var_title + '\n' + 'Session ' + str(startsess) + '-' + str(endsess))
    ax.spines['right'].set_linewidth(0)
    ax.spines['top'].set_linewidth(0)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.set_xlim(startsess-.1,endsess+.1)
    x=np.arange(startsess,endsess+1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
   
    for i,group in enumerate(group_names):
        y = scores.loc[group,variable+str(startsess):variable+str(endsess)]
        plt.errorbar(x, y,
                     yerr = sem.loc[group,variable+str(startsess):variable+str(endsess)], 
                     label=group,linewidth=4, capsize = 8)
    if highlight != None:
        plt.axvline(highlight, 0, 1, color = 'gray', lw = 1)
        ax.fill_between([highlight,endsess], ax.get_ylim()[0], ax.get_ylim()[1], facecolor='gray', alpha=0.2)
    ax.legend()

def choice_bar_plot(startsess, endsess, scores, sem,cmap = 'default'):
    sess = list(range(startsess,endsess + 1))
    labels = ['P1','P2','P3','P4']
    df = pd.DataFrame()
    df1 = pd.DataFrame()
    if cmap == 'Paired':
        colors = [plt.cm.Paired(5),plt.cm.Paired(1),plt.cm.Paired(4),plt.cm.Paired(0)]
    if cmap == 'default':
        colors = [plt.cm.Set1(1),plt.cm.Set1(0)]
    for choice in labels:
        df[choice] = scores.loc[:, [col for col in scores.columns if choice in col 
                                    and int(col[:col.index('P')]) in sess]].mean(axis = 1)
        df1[choice] = sem.loc[:, [col for col in scores.columns if choice in col 
                                    and int(col[:col.index('P')]) in sess]].mean(axis = 1)
    ax = df.transpose().plot.bar(rot = 0, yerr = df1.transpose(), capsize = 8, figsize = (20,8))
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.rcParams.update({'font.size': 18})
    ax.set_ylabel('% Choice', fontweight = 'bold', fontsize = 18)
    ax.set_title('P1-P4 Choice', fontweight = 'bold', fontsize = 22, pad = 20)
    ax.set_ylim(bottom = 0)
    ax.spines['right'].set_linewidth(0)
    ax.spines['top'].set_linewidth(0)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.legend()
    
#------------------------------PLOTTING BY GROUPS---------------------------------#
    
def ls_bar_plot(figure_group, group_means, sem):
    labels = ['P1','P2','P3','P4']
    veh_means =  list(group_means.loc[figure_group,[col for col in group_means.columns if col.startswith('1')]])
    dose1_means = list(group_means.loc[figure_group,[col for col in group_means.columns if col.startswith('2')]])
    dose2_means = list(group_means.loc[figure_group,[col for col in group_means.columns if col.startswith('3')]])
    dose3_means = list(group_means.loc[figure_group,[col for col in group_means.columns if col.startswith('4')]])

    x = np.arange(len(labels))*3  # the label locations
    width = 0.5  # the width of the bars

    fig, ax = plt.subplots(figsize = (10,5))
    rects1 = ax.bar(x - width/2-width, veh_means, width, label='Vehicle',
                    yerr = list(sem.loc[figure_group,[col for col in sem.columns if col.startswith('1')]]), capsize = 8, ecolor='C0')
    rects2 = ax.bar(x - width/2, dose1_means, width, label='Dose 1', 
                    yerr =list(sem.loc[figure_group,[col for col in sem.columns if col.startswith('2')]]), capsize = 8, ecolor='C1')
    rects3 = ax.bar(x + width/2, dose2_means, width, label='Dose 2',
                    yerr = list(sem.loc[figure_group,[col for col in sem.columns if col.startswith('3')]]),capsize = 8, ecolor='C3',color = 'C3')
    rects4 = ax.bar(x + width/2+width, dose3_means, width, label='Dose 3',capsize = 8, 
                    yerr = list(sem.loc[figure_group,[col for col in sem.columns if col.startswith('4')]]), ecolor='C2', color = 'C2')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('% Choice', fontweight = 'bold', fontsize = 22)
    ax.set_title(figure_group + ': P1-P4', fontweight = 'bold', fontsize = 24, pad = 20)
    ax.set_ylim(bottom = 0,top = 85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.spines['right'].set_linewidth(0)
    ax.spines['top'].set_linewidth(0)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.legend()
    
def rgt_bar_plot(variable,startsess,endsess,group_names,title,scores,sem, var_title = None):
    if var_title == None:
        var_title = variable
    plt.rcParams.update({'font.size': 20}) 
    plt.rcParams["figure.figsize"] = (15,8)
    
    bars = [0,0,0,0,0]
    err = [0,0,0,0,0]
    sess = [variable + str(s) for s in list(range(startsess,endsess+1))]
    for i,group in enumerate(scores.index):
        bars[i] = scores.loc[group, [col for col in scores.columns if col in sess]].mean()
        err[i] = sem.loc[group, [col for col in sem.columns if col in sess]].mean()
        plt.bar(i, bars[i], yerr = err[i], capsize = 8, width = .7, color = ['C'+str(i)], label = scores.index[i])
   
    ax = plt.gca()
    plt.xticks([])
    plt.xlabel('Task', labelpad = 20)
    ax.set_ylabel('% Premature')
    ax.set_ylim(0,50)
    ax.set_title(var_title,fontweight = 'bold', fontsize = 22, pad = 20)
    ax.spines['right'].set_linewidth(0)
    ax.spines['top'].set_linewidth(0)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.legend()