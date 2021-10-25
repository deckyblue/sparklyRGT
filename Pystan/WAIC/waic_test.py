import numpy as np
import os 
import arviz as az
import pandas as pd
import math
import xarray as xr
import model_data as md

data = {'col_1': [3, 2, 1, 0], 'col_2': ['a', 'b', 'c', 'd']}
data_df = pd.DataFrame.from_dict(data)
data_df.to_excel("data_df.xlsx")

