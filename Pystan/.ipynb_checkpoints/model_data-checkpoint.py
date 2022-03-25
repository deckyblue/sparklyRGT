import pandas as pd
import numpy as np
import os
import arviz as az
import math
from scipy import stats
from scipy.stats import norm
import xarray as xr
import copy
import matplotlib.pyplot as plt

def get_model_data(df,numsessions, subjects):
    df = get_choices(df) 
    df = extract_data(df, numsessions, subjects)
    startSubject = start_subject(df)
    startSession = start_session(df)
    model_data = get_data_dict(df, startSubject, startSession)
    return model_data


def load_data(fnames, reset_sessions = False):
#load data from computer
    
    for i,file in enumerate(fnames):
        
        if i == 0:
            df = pd.read_csv(fnames[i])
            if reset_sessions:
                for i,session in enumerate(df.Session.unique()):
                    for j in range(len(df)):
                        if df.at[j,'Session'] == session:
                            df.at[j,'Session'] = i + 1
        else:
            df2 = pd.read_csv(fnames[i])
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
            df = pd.read_excel(fnames[i], engine='openpyxl')
            df['Subject'] += 100 #rat 1 becomes rat 101
            if reset_sessions:
                for i,session in enumerate(df.Session.unique()):
                    for j in range(len(df)):
                        if df.at[j,'Session'] == session:
                            df.at[j,'Session'] = i + 1
        else:
            df2 = pd.read_excel(fnames[i], engine='openpyxl')
            df2['Subject'] += 100 * (1+i) #rat 1 becomes rat 201, 301, etc. 
            if reset_sessions:
                for i,session in enumerate(df2.Session.unique()):
                    for j in range(len(df2)):
                        if df2.at[j,'Session'] == session:
                            df2.at[j,'Session'] = i + 1 #makes sessions start from 1 
            df = df.append(df2, ignore_index = True)
    return df


def extract_data(df, numsessions, subjects):
    "take the data for subjects in subjects list from the same condition (task, drug, etc. - specified prior to this function) for the specified number of sessions"
#take only columns needed
    df = df.loc[:, ['Subject','Session','Trial','Pellets','Chosen','Pun_Dur','Option']]
#extract specified data
#extract the number of sessions specified above (numsessions)
    df_small = df.loc[df['Session'] < min(df.Session) + numsessions]

#get rid of chosen = 0
    df_small = df.loc[df['Chosen'] != 0]
    
#extract the conditions based on given list of subject numbers
   #df = df_small.loc[np.logical_and(df_small['Cue'] == cue, df_small['Condition'] == condition)]
    df = df_small[df_small['Subject'].isin(subjects)]
    
    #sort dataset by subject and then session
    df = df.sort_values(by=['Subject', 'Session'], ignore_index = True)
    
    return df



def get_choices(df):
    configA = np.array([1, 4, 0, 2, 3]) #this is the order for version A - i.e., hole 1 corresponds to P1
    configB = np.array([4, 1, 0, 3, 2]) #this is the order for version B - i.e., hole 1 corresponds to P4

    #I took the following code from someone else, so honestly I'm not entirely sure how it works haha
    #the important thing is that it uses the configurations above to assign the correct option, 
    #based on whether the MSN name contains 'A' or 'B'
    df['Option'] = df['MSN'].str.contains("B").values*configB[df['Chosen'].astype('int').ravel()-1].astype('int') + \
        df['MSN'].str.contains("A").values*configA[df['Chosen'].astype('int').ravel()-1].astype('int')
    
    ###I can just take your word for it right? haha
    ####Yes

    #the above code changes any zero in the chosen column to a three in the option column - don't need to know why
    #so we need to fix that (zeros represent either a premature response or an omission)
    for i in range(len(df)): ##range gives me a list from 0 to the len(df), which should be all the indices
        if df['Chosen'][i] == 0: ###can we say this in English? 
            ##the same as df.at
            ##if the index of the 'Chosen' column gets 0 --> option equals 0 
            df['Option'][i] = 0 
    return df    



def start_subject(df):
#startSubject is a list the length of the df, where it equals the subject number at the start of that subject's data 
#and zero otherwise

#redo subject numbers in a new column going from 1 - N number of subjects

    subs = df.Subject.unique()
    sub_dict = {}
    for i, sub in enumerate(subs):
    	sub_dict[sub] = i + 1
    df['Sub'] = 0
    for i in range(len(df)):
    	df['Sub'][i] = sub_dict[df['Subject'][i]]
    
    #find the index for the row before where the subject number changes
    s = np.diff(df.Sub)
    s_idx = np.where(s != 0)

    startSubject = [0] * len(df)

    #set the startSubject list value to the subject number when the subject number changes (row after index)
    for index in s_idx[0]:
        startSubject[index + 1] = df['Sub'][index + 1]

    #set the first value to the first subject number
    startSubject[0] = min(df.Sub)
    return startSubject


def start_session(df):
#startSession is a list the length of the df, where it equals the session number at the start of that session 
#and zero otherwise

#find the index for the row before where the session number changes
    s = np.diff(df.Session)
    s_idx = np.where(s != 0)

    startSession = [0] * len(df)

#set the startSession list value to the session number when the session number changes (row after index)
    for index in s_idx[0]:
        startSession[index + 1] = df['Session'][index + 1]
    
#set the first value to the first session number
    startSession[0] = min(df.Session)
    return startSession
    



def get_data_dict(df,startSubject, startSession):
#extract variables from df
    N = int(len(df.Subject.unique()))
    ntr = int(len(df))

    C = list([int(i) for i in df.Chosen])
    R = list([int(i) for i in df.Pellets])
    P = list([int(i) for i in df.Pun_Dur])
    O = list([int(i) for i in df.Option])
    startSubject = [int(i) for i in startSubject]
    startSession = [int(i) for i in startSession]

    #save them in dictionary
    model_data = {'N': N, 
                  'ntr': ntr,
                  'startSubject': startSubject,
                  'startSession': startSession,
                  'C': C, 
                  'R': R, 
                  'P': P, 
                  'O': O}
    
    return model_data

##--------------------rgt_model_posterior_figures----------------------###
def round_up(n):
    if n - math.floor(n) < 0.5:
        return math.floor(n)
    return math.ceil(n)

def sample_diff(num_params, fit1, fit2):
    """Takes in num_params (ex. basic = 4), fit1 and fit2 (4 chains) and outputs the sample difference distribution (fit1 - fit2) for the group parameters
    Also outputs Cohen's d, probability of superiority (Prob Sup)... and the distribution's mean, 94% HDI, and probability density to the left and right of 0
    If the 94% HDI does not contain 0, then the parameters are credibly different"""
    
    #variables
    dist = stats.norm()
    _, ax = plt.subplots(round_up(num_params/2), 2, figsize=(20, 10), constrained_layout=True)
    comparisons = [(j) for j in range(num_params)]
    pos = [(k, l) for k in range(round_up(num_params/2)) for l in range(2)]
    #comparisons
    for (j), (k, l) in zip(comparisons, pos):
        print(k,l)
        means_diff = fit1.posterior.mu_pr[:,:,j] - fit2.posterior.mu_pr[:,:,j] #means_diff holds the distribution of the sample difference 
        d_cohen = (means_diff / np.sqrt((fit1.posterior.sigma[:,:,j]**2 + fit2.posterior.sigma[:,:,j]**2) / 2)).mean() 
        ps = dist.cdf(d_cohen/(2**0.5))
        #plots posteriors in 2D array of locations: for example, mu_0 - mu_0 will be in the 0,0 box (top left)
        az.plot_posterior(means_diff, ref_val=0, ax=ax[k, l]) 
        ax[k, l].set_title(f'$\mu_{j}-\mu_{j}$')
        ax[k, l].plot(0, label=f"Cohen's d = {d_cohen.values:.2f}\nProb sup = {ps:.2f}", alpha=0)
        ax[k, l].legend();

def transform1(dataset):
    variable = norm.pdf(dataset.mu_pr.values)
    dataset.mu_pr.values = variable
    return dataset

def transform_beta(dataset):
    variable = norm.pdf(dataset.mu_pr.values)*100
    dataset.mu_pr.values = variable
    return dataset
        
def get_mean_params(model_fit, model):
    # access data in Pystan/nc_file folder
    # delete model_data.nc (if error, comment out this line, no file exists yet)
    os.remove('model_param.nc')
    
    param_est = []

    # save original data to file
    model_fit.to_netcdf("model_param.nc")
    file_name = "model_param.nc"

    temp = az.from_netcdf(file_name)
    param_est.append(az.summary(transform_beta(temp.posterior))['mean'][0])
    
    if model == "pindep":
        temp = az.from_netcdf(file_name)
        # get etapos, etaneg
        eta = az.summary(transform1(temp.posterior))['mean']
        for i in range(1,3):
            param_est.append(eta[i]) 
        temp = az.from_netcdf(file_name)
        p_val = az.summary(temp)['mean']
        for i in range(3,7):
            param_est.append(p_val[i])
    elif model == 'pindepstar':
        temp = az.from_netcdf(file_name)
        # get etapos, etaneg
        param_est.append(az.summary(transform1(temp.posterior))['mean'][1])
        temp = az.from_netcdf(file_name)
        p_val = az.summary(temp)['mean']
        for i in range(2,6):
            param_est.append(p_val[i])
    elif "star" in model:
        temp = az.from_netcdf(file_name)
        # get eta
        param_est.append(az.summary(transform1(temp.posterior))['mean'][1])
        temp = az.from_netcdf(file_name)
        p_val = az.summary(temp)['mean']
        for i in range(2,4):
            param_est.append(p_val[i])
    else:
        temp = az.from_netcdf(file_name)
        eta = az.summary(transform1(temp.posterior))['mean']
        for i in range(1,3):
            param_est.append(eta[i])
        temp = az.from_netcdf(file_name)
        p_val = az.summary(temp)['mean']
        for i in range(3,5):
            param_est.append(p_val[i]) 
                   
    return param_est