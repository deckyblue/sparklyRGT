###PACKAGES###
import numpy as np
import os 
import arviz as az
import scipy as sp
import scipy.io as sio
import scipy.stats as stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import importlib
import math
import xarray as xr
os.chdir("..")
import model_data as md

###---SUB_DF HELPER FUNCTIONS---###

def md_subs(df, numsessions, subs): 
    """gets md_df and unique subs (without subject 0)"""
    md_dict = md.get_model_data(df, numsessions, subs) #model data as a dict
    md_df = pd.DataFrame.from_dict(md_dict) #model data as a df
    
    subs_unique = md_df.startSubject.unique()
    subs_unique = np.delete(subs_unique, 1) #remove subject 0 (not a real subject)
    
    return md_df, subs_unique

def ntr_sub(md_df, subs_unique): 
    """ntr_sub takes in md_df, and gets the range of trials for a given subject within subs"""
    
    ntr_sub_ranges = []
    ntr_df = pd.DataFrame({'subject#': subs_unique})
    
    #get range for sub: 
    for s in subs_unique: 
        first_t = md_df.index[md_df['startSubject'] == s].tolist() #first trial
        last_t = md_df.index[md_df['startSubject'] == (s+1)].tolist() #last trial
        
        if last_t != []: #if s+1 does not exist (when s = final subject#), last_t = []
            ntr_sub = list(range(first_t[0], last_t[0]))
            ntr_sub_ranges.append(ntr_sub)
        elif last_t == []:
            last_t = [len(md_df)] #last_t = length of md_df
            ntr_sub = list(range(first_t[0], last_t[0]))
            ntr_sub_ranges.append(ntr_sub)
        
    ntr_df['range_ntr'] = ntr_sub_ranges
    return ntr_df #outputs df with subject, and range_ntr as the next column

def chosen_outcome_data(md_df, ntr_df, subs_unique): 
    """takes in md_df, subs_unique and gets chosen and outcome data"""
    
    for row in md_df.index: #add outcome column (where loss == 0, and win == 1)
        if md_df.at[row,'R'] == 0: #loss
            md_df.at[row,'outcome'] = 0
        if md_df.at[row,'P'] == 0: #win
            md_df.at[row,'outcome'] = 1
    
    chosen = []
    outcome = []
    ntr_df = ntr_sub(md_df, subs_unique)
    
    for row in ntr_df.index:
        range_ts = ntr_df.at[row, 'range_ntr'] #range of trials
        filtered_md_df = md_df[range_ts[0]:(range_ts[-1]+1)]
        chosen.append(filtered_md_df['O'].values)
        outcome.append(filtered_md_df['outcome'].values)
                               
    return chosen, outcome

###---get_params FUNCITONS---###

def get_params_basic(fit, subs_unique):
    """takes in fit, subs_unique (list from [1, final sub#]), and outputs params_s using ALL CHAINS
    function is called basic due to the parameters chosen"""
    
    fit_beta_df = fit.posterior.beta.to_dataframe()
    fit_etaP_df = fit.posterior.etaPositive.to_dataframe()
    fit_etaN_df = fit.posterior.etaNegative.to_dataframe()
    fit_m_df = fit.posterior.m.to_dataframe()
    
    params_s = []
    
    for s in subs_unique:
        params = np.concatenate([fit_beta_df[fit_beta_df.index.get_level_values('beta_dim_0').isin([s-1])],
                                  fit_etaP_df[fit_etaP_df.index.get_level_values('etaPositive_dim_0').isin([s-1])],
                                  fit_etaN_df[fit_etaN_df.index.get_level_values('etaNegative_dim_0').isin([s-1])],
                                  fit_m_df[fit_m_df.index.get_level_values('m_dim_0').isin([s-1])]],
                                  axis = 1)
        params_s.append(params)
    return params_s

def get_params_basicstar(fit, subs_unique):
    """takes in fit, subs_unique (list from [1, final sub#]), and outputs params_s using ALL CHAINS
    function is called basicstar due to the parameters chosen"""
    
    fit_beta_df = fit.posterior.beta.to_dataframe()
    fit_eta_df = fit.posterior.eta.to_dataframe()
    fit_m_df = fit.posterior.m.to_dataframe()
    
    params_s = []
    
    for s in subs_unique:
        params = np.concatenate([fit_beta_df[fit_beta_df.index.get_level_values('beta_dim_0').isin([s-1])],
                                  fit_eta_df[fit_eta_df.index.get_level_values('eta_dim_0').isin([s-1])],
                                  fit_m_df[fit_m_df.index.get_level_values('m_dim_0').isin([s-1])]],
                                  axis = 1)
        params_s.append(params)
    return params_s

def get_params_pscale(fit, subs_unique):
    """takes in fit, subs_unique (list from [1, final sub#]), and outputs params_s using ALL CHAINS
    function is called basic due to the parameters chosen"""
    
    fit_beta_df = fit.posterior.beta.to_dataframe()
    fit_etaP_df = fit.posterior.etaPositive.to_dataframe()
    fit_etaN_df = fit.posterior.etaNegative.to_dataframe()
    fit_m_df = fit.posterior.m.to_dataframe()
    fit_b_df = fit.posterior.b.to_dataframe()
    
    params_s = []
    
    for s in subs_unique:
        params = np.concatenate([fit_beta_df[fit_beta_df.index.get_level_values('beta_dim_0').isin([s-1])],
                                  fit_etaP_df[fit_etaP_df.index.get_level_values('etaPositive_dim_0').isin([s-1])],
                                  fit_etaN_df[fit_etaN_df.index.get_level_values('etaNegative_dim_0').isin([s-1])],
                                  fit_m_df[fit_m_df.index.get_level_values('m_dim_0').isin([s-1])],
                                  fit_b_df[fit_b_df.index.get_level_values('b_dim_0').isin([s-1])]],
                                  axis = 1)
        params_s.append(params)
    return params_s

def get_params_pscalestar(fit, subs_unique):
    """takes in fit, subs_unique (list from [1, final sub#]), and outputs params_s using ALL CHAINS
    function is called basic due to the parameters chosen"""
    
    fit_beta_df = fit.posterior.beta.to_dataframe()
    fit_eta_df = fit.posterior.eta.to_dataframe()
    fit_m_df = fit.posterior.m.to_dataframe()
    fit_b_df = fit.posterior.b.to_dataframe()
    
    params_s = []
    
    for s in subs_unique:
        params = np.concatenate([fit_beta_df[fit_beta_df.index.get_level_values('beta_dim_0').isin([s-1])],
                                  fit_eta_df[fit_eta_df.index.get_level_values('eta_dim_0').isin([s-1])],
                                  fit_m_df[fit_m_df.index.get_level_values('m_dim_0').isin([s-1])],
                                  fit_b_df[fit_b_df.index.get_level_values('b_dim_0').isin([s-1])]],
                                  axis = 1)
        params_s.append(params)
    return params_s

###---sub_df FUNCTIONS---###

def sub_df_basic(df, numsessions, subs, fit):
    """takes the output from md_subs, ntr_sub and chosen_outcome_data functions and outputs the complete df"""
    md_df, subs_unique = md_subs(df, numsessions, subs)
    ntr_df = ntr_sub(md_df, subs_unique)
    chosen_data, outcome_data = chosen_outcome_data(md_df, ntr_df, subs_unique)
    params_s = get_params_basic(fit, subs_unique)
    
    ntr_df['chosen_data'] = chosen_data
    ntr_df['outcome_data'] = outcome_data
    ntr_df['params_s'] = params_s
    
    return ntr_df, md_df

def sub_df_basicstar(df, numsessions, subs, fit):
    """takes the output from md_subs, ntr_sub and chosen_outcome_data functions and outputs the complete df"""
    md_df, subs_unique = md_subs(df, numsessions, subs)
    ntr_df = ntr_sub(md_df, subs_unique)
    chosen_data, outcome_data = chosen_outcome_data(md_df, ntr_df, subs_unique)
    params_s = get_params_basicstar(fit, subs_unique)
    
    ntr_df['chosen_data'] = chosen_data
    ntr_df['outcome_data'] = outcome_data
    ntr_df['params_s'] = params_s
    
    return ntr_df, md_df

def sub_df_pscale(df, numsessions, subs, fit):
    """takes the output from md_subs, ntr_sub and chosen_outcome_data functions and outputs the complete df"""
    md_df, subs_unique = md_subs(df, numsessions, subs)
    ntr_df = ntr_sub(md_df, subs_unique)
    chosen_data, outcome_data = chosen_outcome_data(md_df, ntr_df, subs_unique)
    params_s = get_params_pscale(fit, subs_unique)
    
    ntr_df['chosen_data'] = chosen_data
    ntr_df['outcome_data'] = outcome_data
    ntr_df['params_s'] = params_s
    
    return ntr_df, md_df

def sub_df_pscalestar(df, numsessions, subs, fit):
    """takes the output from md_subs, ntr_sub and chosen_outcome_data functions and outputs the complete df"""
    md_df, subs_unique = md_subs(df, numsessions, subs)
    ntr_df = ntr_sub(md_df, subs_unique)
    chosen_data, outcome_data = chosen_outcome_data(md_df, ntr_df, subs_unique)
    params_s = get_params_pscalestar(fit, subs_unique)
    
    ntr_df['chosen_data'] = chosen_data
    ntr_df['outcome_data'] = outcome_data
    ntr_df['params_s'] = params_s
    
    return ntr_df, md_df

###---log_lik_values FUNCTIONS---###

def basic_log_lik_values(sub_df):
    """takes in sub_df, and gets all the log_likelihood values in a list"""
    
    log_lik_values = [] #stores log_lik_values for all subjects
    
    #model
    p_win = [0.9,0.8,0.5,0.4]
    win_amount = [1,2,3,4]
    pun_dur = [5,10,30,40]
    
    for s in sub_df.index: #note: s is 0-indexed
        
        #subject data
        ntr = sub_df.at[s,'range_ntr']
        chosen_data = sub_df.at[s,'chosen_data']
        outcome_data = sub_df.at[s,'outcome_data']
        params = sub_df.at[s, 'params_s'] #4000 sets of parameters 
        
        #updating model values
        V = np.zeros(4) # [0,0,0,0]
#         Q = np.zeros([4,ntr]) #1 array of 4 rows and ntr columns
        
        paramsXtrial = [(p, t) for p in range(4000) for t in range(ntr[-1]+1 - ntr[0])] #4000 sets of parameters, for ntr trials
        for (p, t) in paramsXtrial: 

            if t == 0: #if using a new set of parameters (if first trial, reset)...
                V = np.zeros(4) #reset the V values

            p_action = np.exp(params[p][0]*V)/np.sum(np.exp(params[p][0]*V)) #p_action holds the probabilities of each action (P1-P4), starting at 0.25 for each

            if outcome_data[t] == 1: 
                V[chosen_data[t]-1] += params[p][1]*(win_amount[chosen_data[t]-1] - V[chosen_data[t]-1]) #subtract 1 because chosen_data stores P1-P4 as 1-4 instead 0-3
            else: 
                V[chosen_data[t]-1] += params[p][2]*(-params[p][3]*pun_dur[chosen_data[t]-1] - V[chosen_data[t]-1])
            log_lik_values.append(math.log(p_action[chosen_data[t]-1])) 
    
    return log_lik_values

def basicstar_log_lik_values(sub_df):
    """takes in sub_df, and gets all the log_likelihood values in a list"""
    
    log_lik_values = [] #stores log_lik_values for all subjects
    
    #model
    p_win = [0.9,0.8,0.5,0.4]
    win_amount = [1,2,3,4]
    pun_dur = [5,10,30,40]
    
    for s in sub_df.index: #note: s is 0-indexed
        
        #subject data
        ntr = sub_df.at[s,'range_ntr']
        chosen_data = sub_df.at[s,'chosen_data']
        outcome_data = sub_df.at[s,'outcome_data']
        params = sub_df.at[s, 'params_s'] #4000 sets of parameters 
        
        #updating model values
        V = np.zeros(4) # [0,0,0,0]
#         Q = np.zeros([4,ntr]) #1 array of 4 rows and ntr columns
        
        paramsXtrial = [(p, t) for p in range(4000) for t in range(ntr[-1]+1 - ntr[0])] #4000 sets of parameters, for ntr trials
        for (p, t) in paramsXtrial: 

            if t == 0: #if using a new set of parameters (if first trial, reset)...
                V = np.zeros(4) #reset the V values

            p_action = np.exp(params[p][0]*V)/np.sum(np.exp(params[p][0]*V)) #p_action holds the probabilities of each action (P1-P4), starting at 0.25 for each

            if outcome_data[t] == 1: 
                V[chosen_data[t]-1] += params[p][1]*(win_amount[chosen_data[t]-1] - V[chosen_data[t]-1]) #subtract 1 because chosen_data stores P1-P4 as 1-4 instead 0-3
            else: 
                V[chosen_data[t]-1] += params[p][1]*(-params[p][2]*pun_dur[chosen_data[t]-1] - V[chosen_data[t]-1])
            log_lik_values.append(math.log(p_action[chosen_data[t]-1])) 
    
    return log_lik_values

def pscale_log_lik_values(sub_df):
    """takes in sub_df, and gets all the log_likelihood values in a list"""
    
    log_lik_values = [] #stores log_lik_values for all subjects
    
    #model
    p_win = [0.9,0.8,0.5,0.4]
    win_amount = [1,2,3,4]
    pun_dur = [5,10,30,40]
    
    for s in sub_df.index: #note: s is 0-indexed
        
        #subject data
        ntr = sub_df.at[s,'range_ntr']
        chosen_data = sub_df.at[s,'chosen_data']
        outcome_data = sub_df.at[s,'outcome_data']
        params = sub_df.at[s, 'params_s'] #4000 sets of parameters 
        
        #updating model values
        V = np.zeros(4) # [0,0,0,0]
#         Q = np.zeros([4,ntr]) #1 array of 4 rows and ntr columns
        
        paramsXtrial = [(p, t) for p in range(4000) for t in range(ntr[-1]+1 - ntr[0])] #4000 sets of parameters, for ntr trials
        for (p, t) in paramsXtrial: 

            if t == 0: #if using a new set of parameters (if first trial, reset)...
                V = np.zeros(4) #reset the V values

            p_action = np.exp(params[p][0]*V)/np.sum(np.exp(params[p][0]*V)) #p_action holds the probabilities of each action (P1-P4), starting at 0.25 for each

            if outcome_data[t] == 1: 
                V[chosen_data[t]-1] += params[p][1]*(win_amount[chosen_data[t]-1] - V[chosen_data[t]-1]) #subtract 1 because chosen_data stores P1-P4 as 1-4 instead 0-3
            else: 
                V[chosen_data[t]-1] += params[p][2]*(params[p][4] - params[p][3]*pun_dur[chosen_data[t]-1] - V[chosen_data[t]-1])
            log_lik_values.append(math.log(p_action[chosen_data[t]-1])) 
    
    return log_lik_values

def pscalestar_log_lik_values(sub_df):
    """takes in sub_df, and gets all the log_likelihood values in a list"""
    
    log_lik_values = [] #stores log_lik_values for all subjects
    
    #model
    p_win = [0.9,0.8,0.5,0.4]
    win_amount = [1,2,3,4]
    pun_dur = [5,10,30,40]
    
    for s in sub_df.index: #note: s is 0-indexed
        
        #subject data
        ntr = sub_df.at[s,'range_ntr']
        chosen_data = sub_df.at[s,'chosen_data']
        outcome_data = sub_df.at[s,'outcome_data']
        params = sub_df.at[s, 'params_s'] #4000 sets of parameters 
        
        #updating model values
        V = np.zeros(4) # [0,0,0,0]
#         Q = np.zeros([4,ntr]) #1 array of 4 rows and ntr columns
        
        paramsXtrial = [(p, t) for p in range(4000) for t in range(ntr[-1]+1 - ntr[0])] #4000 sets of parameters, for ntr trials
        for (p, t) in paramsXtrial: 

            if t == 0: #if using a new set of parameters (if first trial, reset)...
                V = np.zeros(4) #reset the V values

            p_action = np.exp(params[p][0]*V)/np.sum(np.exp(params[p][0]*V)) #p_action holds the probabilities of each action (P1-P4), starting at 0.25 for each

            if outcome_data[t] == 1: 
                V[chosen_data[t]-1] += params[p][1]*(win_amount[chosen_data[t]-1] - V[chosen_data[t]-1]) #subtract 1 because chosen_data stores P1-P4 as 1-4 instead 0-3
            else: 
                V[chosen_data[t]-1] += params[p][1]*(params[p][3] - params[p][2]*pun_dur[chosen_data[t]-1] - V[chosen_data[t]-1])
            log_lik_values.append(math.log(p_action[chosen_data[t]-1])) 
    
    return log_lik_values

###---get_inference_data FUNCTION---###

def get_inference_data_log_lik(md_df, log_lik_values, fit):
    """takes in log_lik_values from get_log_lik_values, and creates an inference data object"""
    
    total_ntr = md_df.at[0, 'ntr'] #total trials in md_df, which represents the total number of trials for a cue variant 
    print(total_ntr)
    print(4000)
    print(len(log_lik_values))
    
    #build index
    iterables = [[0,1,2,3], list(range(0,1000)), list(range(0, total_ntr))] #4 chains, 1000 draws, total_ntr trials
    index = pd.MultiIndex.from_product(iterables, names=["chain", "draw", "trial"])
    
    #build InferenceData (id) object 
    s_log_likelihood = pd.Series(log_lik_values, index=index)
    df_log_likelihood = s_log_likelihood.to_frame(name = "log_likelihood")
    x_log_likelihood = df_log_likelihood.to_xarray()
    id_log_likelihood = fit.assign(x_log_likelihood)
    
    return id_log_likelihood

###---waic FUNCTIONS---###

def waic_basic_fit(df, numsessions, subs, fit):
    """summary function: takes in all parameters and outputs WAIC table using az.waic() // parameters:
    df - product of rgt.load_multiple_data,
    numsessions = 5,
    subs = cue variant (list), 
    fit = .nc object storing the model and fit"""
    
    sub_df, md_df = sub_df_basic(df, numsessions, subs, fit)
    log_lik_values = basic_log_lik_values(sub_df)
    id_log_likelihood = get_inference_data_log_lik(md_df, log_lik_values, fit)
    
    waic = az.waic(id_log_likelihood)
    return waic

def waic_basicstar_fit(df, numsessions, subs, fit):
    """summary function: takes in all parameters and outputs WAIC table using az.waic() // parameters:
    df - product of rgt.load_multiple_data,
    numsessions = 5,
    subs = cue variant (list), 
    fit = .nc object storing the model and fit"""
    
    sub_df, md_df = sub_df_basicstar(df, numsessions, subs, fit)
    log_lik_values = basicstar_log_lik_values(sub_df)
    id_log_likelihood = get_inference_data_log_lik(md_df, log_lik_values, fit)
    
    waic = az.waic(id_log_likelihood)
    return waic

def waic_pscale_fit(df, numsessions, subs, fit):
    """summary function: takes in all parameters and outputs WAIC table using az.waic() // parameters:
    df - product of rgt.load_multiple_data,
    numsessions = 5,
    subs = cue variant (list), 
    fit = .nc object storing the model and fit"""
    
    sub_df, md_df = sub_df_pscale(df, numsessions, subs, fit)
    log_lik_values = pscale_log_lik_values(sub_df)
    id_log_likelihood = get_inference_data_log_lik(md_df, log_lik_values, fit)
    
    waic = az.waic(id_log_likelihood)
    return waic

def waic_pscalestar_fit(df, numsessions, subs, fit):
    """summary function: takes in all parameters and outputs WAIC table using az.waic() // parameters:
    df - product of rgt.load_multiple_data,
    numsessions = 5,
    subs = cue variant (list), 
    fit = .nc object storing the model and fit"""
    
    sub_df, md_df = sub_df_pscalestar(df, numsessions, subs, fit)
    log_lik_values = pscalestar_log_lik_values(sub_df)
    id_log_likelihood = get_inference_data_log_lik(md_df, log_lik_values, fit)
    
    waic = az.waic(id_log_likelihood)
    return waic


