###PACKAGES###
import numpy as np
import os 
import arviz as az
import pandas as pd
import math
import xarray as xr
import model_data as md

###ASSIGN FIT OBJECTS###

loss_basic_fit = az.from_netcdf('loss_basic_fit.nc')
outcome_basic_fit = az.from_netcdf('outcome_basic_fit.nc')
random_basic_fit = az.from_netcdf('random_basic_fit.nc')
reverse_basic_fit = az.from_netcdf('reverse_basic_fit.nc')
standard_basic_fit = az.from_netcdf('standard_basic_fit.nc')
uncued_basic_fit = az.from_netcdf('uncued_basic_fit.nc')

loss_basicstar_fit = az.from_netcdf('loss_basic-star_fit.nc')
outcome_basicstar_fit = az.from_netcdf('outcome_basic-star_fit.nc')
random_basicstar_fit = az.from_netcdf('random_basic-star_fit.nc')
reverse_basicstar_fit = az.from_netcdf('reverse_basic-star_fit.nc')
standard_basicstar_fit = az.from_netcdf('standard_basic-star_fit.nc')
uncued_basicstar_fit = az.from_netcdf('uncued_basic-star_fit.nc')

loss_pscale_fit = az.from_netcdf('loss_pscale_fit.nc')
outcome_pscale_fit = az.from_netcdf('outcome_pscale_fit.nc')
random_pscale_fit = az.from_netcdf('random_pscale_fit.nc')
reverse_pscale_fit = az.from_netcdf('reverse_pscale_fit.nc')
standard_pscale_fit = az.from_netcdf('standard_pscale_fit.nc')
uncued_pscale_fit = az.from_netcdf('uncued_pscale_fit.nc')

loss_pscalestar_fit = az.from_netcdf('loss_pscale-star_fit.nc')
outcome_pscalestar_fit = az.from_netcdf('outcome_pscale-star_fit.nc')
random_pscalestar_fit = az.from_netcdf('random_pscale-star_fit.nc')
reverse_pscalestar_fit = az.from_netcdf('reverse_pscale-star_fit.nc')
standard_pscalestar_fit = az.from_netcdf('standard_pscale-star_fit.nc')
uncued_pscalestar_fit = az.from_netcdf('uncued_pscale-star_fit.nc')

loss_pindep_fit = az.from_netcdf('loss_pindep_fit.nc')
outcome_pindep_fit = az.from_netcdf('outcome_pindep_fit.nc')
random_pindep_fit = az.from_netcdf('random_pindep_fit.nc')
reverse_pindep_fit = az.from_netcdf('reverse_pindep_fit.nc')
standard_pindep_fit = az.from_netcdf('standard_pindep_fit.nc')
uncued_pindep_fit = az.from_netcdf('uncued_pindep_fit.nc')

loss_pindepstar_fit = az.from_netcdf('loss_pindep-star_fit.nc')
outcome_pindepstar_fit = az.from_netcdf('outcome_pindep-star_fit.nc')
random_pindepstar_fit = az.from_netcdf('random_pindep-star_fit.nc')
reverse_pindepstar_fit = az.from_netcdf('reverse_pindep-star_fit.nc')
standard_pindepstar_fit = az.from_netcdf('standard_pindep-star_fit.nc')
uncued_pindepstar_fit = az.from_netcdf('uncued_pindep-star_fit.nc')

###LOAD IN DATA###

fnames = ['BH09_raw-free_S1-5_corrected.xlsx','CH02_corrected.xlsx','NA01_raw_free-choice_S8-18.xlsx',"CH01_corrected.xlsx"]
df = md.load_multiple_data(fnames, reset_sessions = True)

###RENAME DATA###

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

###DEFINE SUBS###

uncued_subs = np.concatenate(task_list[[task for task in df.MSN.unique() if 'Classic' in task]])
standard_subs = np.concatenate((task_list['rGT_A-cue'], task_list['rGT_B-cue']))
reverse_subs = np.concatenate((np.concatenate(task_list[[task for task in df.MSN.unique() if 'Mis' in task]]),
                              np.concatenate(task_list[[task for task in df.MSN.unique() if 'Rev' in task]])))
outcome_subs = np.concatenate(task_list[[task for task in df.MSN.unique() if 'outcome' in task]])
random_subs = np.concatenate(task_list[[task for task in df.MSN.unique() if 'Random' in task]])
loss_subs = np.concatenate(task_list[[task for task in df.MSN.unique() if 'oss' in task]])

###WAIC LISTS###

loss_basic_waic = wc.waic_fit(df, 5, loss_subs, loss_basic_fit, "basic")
outcome_basic_waic = wc.waic_fit(df, 5, outcome_subs, outcome_basic_fit, "basic")
random_basic_waic = wc.waic_fit(df, 5, random_subs, random_basic_fit, "basic")
reverse_basic_waic = wc.waic_fit(df, 5, reverse_subs, reverse_basic_fit, "basic")
standard_basic_waic = wc.waic_fit(df, 5, standard_subs, standard_basic_fit, "basic")
uncued_basic_waic = wc.waic_fit(df, 5, uncued_subs, uncued_basic_fit, "basic")

basic_list = [loss_basic_waic.waic, outcome_basic_waic.waic, random_basic_waic.waic, reverse_basic_waic.waic, standard_basic_waic.waic, uncued_basic_waic.waic]
basic_list_SE = [loss_basic_waic.waic_se, outcome_basic_waic.waic_se, random_basic_waic.waic_se, reverse_basic_waic.waic_se, standard_basic_waic.waic_se, uncued_basic_waic.waic_se]

###

loss_basicstar_waic = wc.waic_fit(df, 5, loss_subs, loss_basicstar_fit, "basicstar")
outcome_basicstar_waic = wc.waic_fit(df, 5, outcome_subs, outcome_basicstar_fit, "basicstar")
random_basicstar_waic = wc.waic_fit(df, 5, random_subs, random_basicstar_fit, "basicstar")
reverse_basicstar_waic = wc.waic_fit(df, 5, reverse_subs, reverse_basicstar_fit, "basicstar")
standard_basicstar_waic = wc.waic_fit(df, 5, standard_subs, standard_basicstar_fit, "basicstar")
uncued_basicstar_waic = wc.waic_fit(df, 5, uncued_subs, uncued_basicstar_fit, "basicstar")

basicstar_list = [loss_basicstar_waic.waic, outcome_basicstar_waic.waic, random_basicstar_waic.waic, reverse_basicstar_waic.waic, standard_basicstar_waic.waic, uncued_basicstar_waic.waic]
basicstar_list_SE = [loss_basicstar_waic.waic_se, outcome_basicstar_waic.waic_se, random_basicstar_waic.waic_se, reverse_basicstar_waic.waic_se, standard_basicstar_waic.waic_se, uncued_basicstar_waic.waic_se]

###

loss_pscale_waic = wc.waic_fit(df, 5, loss_subs, loss_pscale_fit, "pscale")
outcome_pscale_waic = wc.waic_fit(df, 5, outcome_subs, outcome_pscale_fit, "pscale")
random_pscale_waic = wc.waic_fit(df, 5, random_subs, random_pscale_fit, "pscale")
reverse_pscale_waic = wc.waic_fit(df, 5, reverse_subs, reverse_pscale_fit, "pscale")
standard_pscale_waic = wc.waic_fit(df, 5, standard_subs, standard_pscale_fit, "pscale")
uncued_pscale_waic = wc.waic_fit(df, 5, uncued_subs, uncued_pscale_fit, "pscale")

pscale_list = [loss_pscale_waic.waic, outcome_pscale_waic.waic, random_pscale_waic.waic, reverse_pscale_waic.waic, standard_pscale_waic.waic, uncued_pscale_waic.waic]
pscale_list_SE = [loss_pscale_waic.waic_se, outcome_pscale_waic.waic_se, random_pscale_waic.waic_se, reverse_pscale_waic.waic_se, standard_pscale_waic.waic_se, uncued_pscale_waic.waic_se]

###

loss_pscalestar_waic = wc.waic_fit(df, 5, loss_subs, loss_pscalestar_fit, "pscalestar")
outcome_pscalestar_waic = wc.waic_fit(df, 5, outcome_subs, outcome_pscalestar_fit, "pscalestar")
random_pscalestar_waic = wc.waic_fit(df, 5, random_subs, random_pscalestar_fit, "pscalestar")
reverse_pscalestar_waic = wc.waic_fit(df, 5, reverse_subs, reverse_pscalestar_fit, "pscalestar")
standard_pscalestar_waic = wc.waic_fit(df, 5, standard_subs, standard_pscalestar_fit, "pscalestar")
uncued_pscalestar_waic = wc.waic_fit(df, 5, uncued_subs, uncued_pscalestar_fit, "pscalestar")

pscalestar_list = [loss_pscalestar_waic.waic, outcome_pscalestar_waic.waic, random_pscalestar_waic.waic, reverse_pscalestar_waic.waic, standard_pscalestar_waic.waic, uncued_pscalestar_waic.waic]
pscalestar_list_SE = [loss_pscalestar_waic.waic_se, outcome_pscalestar_waic.waic_se, random_pscalestar_waic.waic_se, reverse_pscalestar_waic.waic_se, standard_pscalestar_waic.waic_se, uncued_pscalestar_waic.waic_se]

###

loss_pindep_waic = wc.waic_fit(df, 5, loss_subs, loss_pindep_fit, "pindep")
outcome_pindep_waic = wc.waic_fit(df, 5, outcome_subs, outcome_pindep_fit, "pindep")
random_pindep_waic = wc.waic_fit(df, 5, random_subs, random_pindep_fit, "pindep")
reverse_pindep_waic = wc.waic_fit(df, 5, reverse_subs, reverse_pindep_fit, "pindep")
standard_pindep_waic = wc.waic_fit(df, 5, standard_subs, standard_pindep_fit, "pindep")
uncued_pindep_waic = wc.waic_fit(df, 5, uncued_subs, uncued_pindep_fit, "pindep")

pindep_list = [loss_pindep_waic.waic, outcome_pindep_waic.waic, random_pindep_waic.waic, reverse_pindep_waic.waic, standard_pindep_waic.waic, uncued_pindep_waic.waic]
pindep_list_SE = [loss_pindep_waic.waic_se, outcome_pindep_waic.waic_se, random_pindep_waic.waic_se, reverse_pindep_waic.waic_se, standard_pindep_waic.waic_se, uncued_pindep_waic.waic_se]

###

loss_pindepstar_waic = wc.waic_fit(df, 5, loss_subs, loss_pindepstar_fit, "pindepstar")
outcome_pindepstar_waic = wc.waic_fit(df, 5, outcome_subs, outcome_pindepstar_fit, "pindepstar")
random_pindepstar_waic = wc.waic_fit(df, 5, random_subs, random_pindepstar_fit, "pindepstar")
reverse_pindepstar_waic = wc.waic_fit(df, 5, reverse_subs, reverse_pindepstar_fit, "pindepstar")
standard_pindepstar_waic = wc.waic_fit(df, 5, standard_subs, standard_pindepstar_fit, "pindepstar")
uncued_pindepstar_waic = wc.waic_fit(df, 5, uncued_subs, uncued_pindepstar_fit, "pindepstar")

pindepstar_list = [loss_pindepstar_waic.waic, outcome_pindepstar_waic.waic, random_pindepstar_waic.waic, reverse_pindepstar_waic.waic, standard_pindepstar_waic.waic, uncued_pindepstar_waic.waic]
pindepstar_list_SE = [loss_pindepstar_waic.waic_se, outcome_pindepstar_waic.waic_se, random_pindepstar_waic.waic_se, reverse_pindepstar_waic.waic_se, standard_pindepstar_waic.waic_se, uncued_pindepstar_waic.waic_se]

###CREATE AND EXPORT DATAFRAME###

waic_values = pd.DataFrame(
    {'basic_waic': basic_list,
     'basic_waic_SE': basic_list_SE, 
     'basicstar_waic': basicstar_list,
     'basicstar_waic_SE': basicstar_list_SE,
     'pscale_waic': pscale_list,
     'pscale_waic_SE': pscale_list_SE,
     'pscalestar_waic': pscalestar_list,
     'pscalestar_SE': pscalestar_list_SE,
     'pindep_waic': pindep_list,
     'pindep_waic_SE': pindep_list_SE,
     'pindepstar_waic': pindepstar_list,
     'pindepstar_waic_SE': pindepstar_list_SE},
     index = ["loss", "outcome", "random", "reverse", "standard", "uncued"])

waic_values.to_excel("waic_values.xlsx")