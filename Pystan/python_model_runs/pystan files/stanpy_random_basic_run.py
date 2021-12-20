import os
from cmdstanpy import cmdstan_path, set_cmdstan_path, CmdStanModel
import model_data as md 
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None
import arviz as az
import xarray as xr

#---------------------------

#number of sessions to do modelling on
numsessions = 5


fnames = ['BH09_raw-free_S1-5_corrected.xlsx','CH02_corrected.xlsx','NA01_raw_free-choice_S8-18.xlsx',"CH01_corrected.xlsx"]
#change these to the names of the excel files that you want to run modeling on 


#--------------------------

#load in excel files into one df, resetting the session numbers to start at 1
df = md.load_multiple_data(fnames, reset_sessions = True)


#creates lists of subjects run on each task (classic A, classic B, etc.)

#rename MSNs so that the rats on the outcome task don't have "loss" in the MSN
for i in range(len(df)):
    if df.at[i, 'MSN'] == 'LossrGT_A-losscue_v1':
        df.at[i,'MSN'] = 'outcomeRGT_A'
    if df.at[i, 'MSN'] == 'LossrGT_B-losscue_v1':
        df.at[i,'MSN'] = 'outcomeRGT_B'
        
#rename MSNs so that the rats on the random task don't have "loss" in the MSN
for i in range(len(df)):
    if df.at[i,'MSN'] == 'AnarchyrGT_B-losscue_v6':
        df.at[i,'MSN'] = 'RandomRGT_B'
    if df.at[i,'MSN'] == 'AnarchyrGT_A-losscue_v6':
        df.at[i,'MSN'] = 'RandomRGT_A'
        
        
task_list = df.groupby(['MSN'])['Subject'].unique()

#concatenate together the lists of subjects that run the same task (i.e., puts version A 
#and version B together) - based on unique string for each task name

##change between model runs
subs = np.concatenate(task_list[[task for task in df.MSN.unique() if 'Random' in task]])

#---------------------------------------------

#set path to cmdstanpy location
set_cmdstan_path('/scratch/st-caw42-1/cmdstan-2.28.2')


#extract model data from dataframe
model_data = md.get_model_data(df, numsessions, subs)

#stan code loaded in from file, uploaded to cmdstan folder in scratch
stan_file = os.path.join(cmdstan_path(), 'rGT_RLmodel_basic.stan')

#compile stan code
model = CmdStanModel(stan_file=stan_file, cpp_options={'STAN_THREADS':'true'})

#do the sampling
fit = model.sample(data = model_data, chains = 4, iter_sampling = 800, iter_warmup = 800)

#save all the output and summary df 
fit.save_csvfiles(dir='/scratch/st-caw42-1/stan_output')
fit_az = az.from_cmstanpy(fit)
fit_summary = fit.summary().to_csv('/scratch/st-caw42-1/stan_output/random_basic_fit_summary_.csv')
fit_az.to_netcdf('/scratch/st-caw42-1/stan_output/random_basic_fit.nc')