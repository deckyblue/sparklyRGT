###PACKAGES###
import numpy as np
import os 
import arviz as az
import pandas as pd
import math
import xarray as xr
import model_data as md
import waic as wc

##ASSIGN FIT OBJECTS
cued_basic_fit = az.from_netcdf('/scratch/st-caw42-1/stan_outputs/cue_basic_fit.nc')
classic_basic_fit = az.from_netcdf('/scratch/st-caw42-1/stan_outputs/Classic_basic_classic_fit.nc')

cued_pscale_fit = az.from_netcdf('/scratch/st-caw42-1/stan_outputs/cue_pscale_fit.nc')
classic_pscale_fit = az.from_netcdf('/scratch/st-caw42-1/stan_outputs/Classic_pscale_fit.nc')

cued_pindep_fit = az.from_netcdf('/scratch/st-caw42-1/stan_outputs/cue_pindep_fit.nc')
classic_pindep_fit = az.from_netcdf('../stan_outputs/Classic_pindep_fit.nc')

cued_basicstar_fit = az.from_netcdf('/scratch/st-caw42-1/stan_outputs/cue_basicstar_fit.nc')
classic_basicstar_fit = az.from_netcdf('/scratch/st-caw42-1/stan_outputs/Classic_basicstar_fit.nc')

cued_pscalestar_fit = az.from_netcdf('/scratch/st-caw42-1/stan_outputs/cue_pscale_fit.nc')
classic_pscalestar_fit = az.from_netcdf('/scratch/st-caw42-1/stan_outputs/Classic_pscalestar_fit.nc')

cued_pindepstar_fit = az.from_netcdf('/scratch/st-caw42-1/stan_outputs/cue_pindepstar_fit.nc')
classic_pindepstar_fit = az.from_netcdf('/scratch/st-caw42-1/stan_outputs/Classic_pindepstar_fit.nc')

##LOAD IN DATA
fnames = ['langdon_data.csv']
df = md.load_data(fnames)

##DEFINE SUBS
task_list = df.groupby(['MSN'])['Subject'].unique()
classic_subs = np.concatenate(task_list[[task for task in df.MSN.unique() if 'Classic' in task]])
cued_subs = np.concatenate((task_list['rGT_A-cue'], task_list['rGT_B-cue']))

##WAIC LISTS
cued_basic_waic = wc.waic_fit(df, 5, cued_subs, cued_basic_fit, "basic")
classic_basic_waic = wc.waic_fit(df, 5, classic_subs, classic_basic_fit, "basic")

basic_list = [cued_basic_waic.waic, classic_basic_waic.waic]
basic_list_SE = [cued_basic_waic.waic_se, classic_basic_waic.waic_se]

##
cued_basicstar_waic = wc.waic_fit(df, 5, cued_subs, cued_basicstar_fit, "basicstar")
classic_basic_waic = wc.waic_fit(df, 5, classic_subs, classic_basicstar_fit, "basicstar")

basicstar_list = [cued_basicstar_waic.waic, classic_basicstar_waic.waic]
basicstar_list_SE = [cued_basicstar_waic.waic_se, classic_basicstar_waic.waic_se]

##
cued_pscale_waic = wc.waic_fit(df, 5, cued_subs, cued_pscale_fit, "pscale")
classic_pscale_waic = wc.waic_fit(df, 5, classic_subs, classic_pscale_fit, "pscale")

pscale_list = [cued_pscale_waic.waic, classic_pscale_waic.waic]
pscale_list_SE = [cued_pscale_waic.waic_se, classic_pscale_waic.waic_se]

##
cued_pscalestar_waic = wc.waic_fit(df, 5, cued_subs, cued_pscalestar_fit, "pscalestar")
classic_pscalestar_waic = wc.waic_fit(df, 5, classic_subs, classic_pscalestar_fit, "pscalestar")

pscalestar_list = [cued_pscalestar_waic.waic, classic_pscalestar_waic.waic]
pscalestar_list_SE = [cued_pscalestar_waic.waic_se, classic_pscalestar_waic.waic_se]

##
cued_pindep_waic = wc.waic_fit(df, 5, cued_subs, cued_pindep_fit, "pindep")
classic_pindep_waic = wc.waic_fit(df, 5, classic_subs, classic_pindep_fit, "pindep")

pindep_list = [cued_pindep_waic.waic, classic_pindep_waic.waic]
pindep_list_SE = [cued_pindep_waic.waic_se, classic_pindep_waic.waic_se]

##
cued_pindepstar_waic = wc.waic_fit(df, 5, cued_subs, cued_pindepstar_fit, "pindepstar")
classic_pindepstar_waic = wc.waic_fit(df, 5, classic_subs, classic_pindepstar_fit, "pindepstar")

pindep_list = [cued_pindepstar_waic.waic, classic_pindepstar_waic.waic]
pindep_list_SE = [cued_pindepstar_waic.waic_se, classic_pindepstar_waic.waic_se]

##CREATE AND EXPORT DATAFRAME
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
     index = ["cued", "classic"])

waic_values.to_excel("langdon_waic_values.xlsx")