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

#task to do modelling on - should be 'Classic' for uncued task, 'cue' for cued task 
task_name = 'cue'

#insert the names of the files to do the modeling on 
fnames = ['sockeye_data.csv']

#choose the model (.stan file should be uploaded to cmdstan directory in scratch)
model_file_name = 'rGT_RLmodel_basic.stan'
#shortened name of model for file naming later on: (basic, basicstar, pscale, pscalestar, pindep, pindepstar)
model_name = 'basic'

#--------------------------

#load in excel files into one df, resetting the session numbers to start at 1
df = md.load_multiple_data(fnames, reset_sessions = True)

#if just one file, you can use:
#df = md.load_data(fnames, reset_sessions = True)


#if running on rgt variant dataset, uncomment the following lines: 

# df = df.replace(to_replace = {'MSN':
#                          {'LossrGT_A-losscue_v1': 'outcomeRGT_A',
#                         'LossrGT_B-losscue_v1': 'outcomeRGT_B',
#                          'AnarchyrGT_A-losscue_v6':'RandomRGT_A',
#                         'AnarchyrGT_B-losscue_v6': 'RandomRGT_B',
#                          'MisrGT_A-cue':'RevRGT_A',
#                          'MisrGT_B-cue':'RevRGT_B',
#                          'RevRGT_A-cue':'RevRGT_A',
#                          'RevRGT_B-cue':'RevRGT_B',
#                          'LossRGT_A':'lossRGT_A'
#                         }})


#create lists of subjects run on each task (classic A, classic B, etc.)
task_list = df.groupby(['MSN'])['Subject'].unique()

#concatenate together the lists of subjects that run the same task (i.e., puts version A 
#and version B together) - based on unique string for task name (stored in task_name)

##change between model runs
subs = np.concatenate(task_list[[task for task in df.MSN.unique() if task_name in task]])

#---------------------------------------------

#set path to cmdstanpy location
set_cmdstan_path('/scratch/st-caw42-1/cmdstan-2.28.2')

#extract model data from dataframe
model_data = md.get_model_data(df, numsessions, subs)

#load in stan model code from file
#should be uploaded to cmdstan-2.28.2 directory in scratch
stan_file = os.path.join(cmdstan_path(), model_file_name)

#compile stan code
model = CmdStanModel(stan_file=stan_file, cpp_options={'STAN_THREADS':'true'})

#do the sampling
fit = model.sample(data = model_data, chains = 4, iter_sampling = 800, iter_warmup = 800)

#---------------------------------------------

#save all the output and summary df 
fit.save_csvfiles(dir='/scratch/st-caw42-1/stan_output')
fit_summary = fit.summary().to_csv('/scratch/st-caw42-1/stan_output/' + task_name + '_' + model_name + '_fit_summary.csv')
fit_az = az.from_cmdstanpy(fit)
fit_az.to_netcdf('/scratch/st-caw42-1/stan_output/' + task_name + '_' + model_name +  '_fit.nc')