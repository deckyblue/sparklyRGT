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


##LOAD IN DATA
fnames = ['langdon_data.csv']
df = md.load_data(fnames)

##DEFINE SUBS
classic_subs = [201]
cued_subs = [173]

##WAIC LISTS
cued_basic_waic = wc.waic_fit(df, 5, cued_subs, cued_basic_fit, "basic")
classic_basic_waic = wc.waic_fit(df, 5, classic_subs, classic_basic_fit, "basic")

basic_list = [cued_basic_waic.waic, classic_basic_waic.waic]
basic_list_SE = [cued_basic_waic.waic_se, classic_basic_waic.waic_se]


##CREATE AND EXPORT DATAFRAME
waic_values = pd.DataFrame(
    {'basic_waic': basic_list,
     'basic_waic_SE': basic_list_SE,},
     index = ["cued", "classic"])

waic_values.to_excel("test_langdon_waic_values.xlsx")