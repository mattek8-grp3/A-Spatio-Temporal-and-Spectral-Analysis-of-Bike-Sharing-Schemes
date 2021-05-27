"""
@author: Mattek Group 3
"""

from scipy.stats import ttest_ind
import numpy as np
import pandas as pd
import bikeshare as bs

#%%
city = "london"
year = 2019
month = 9
Data = bs.Data(city, year, month)

precip_threshold = 1
request, rain = bs.get_weather(city, year, month)
df = Data.df
df['hour'] = pd.to_datetime(df['start_t']).dt.hour
df['day'] = pd.to_datetime(df['start_t']).dt.day
n_tot = Data.stat.n_tot
id_index = Data.stat.id_index

trips_pr_hour = np.zeros((Data.num_days, 24))
day = df['start_dt'].dt.day
hour = df['start_dt'].dt.hour

for d, h in zip(day, hour):
    trips_pr_hour[d-1, h] += 1

trips_pr_hour = trips_pr_hour.reshape(720)
rain['trips_pr_hour'] = trips_pr_hour

# # Wet
indexes_w = rain.query('precipMM > {0}'.format(precip_threshold)).index
trips_pr_hour_wet = rain['trips_pr_hour'][indexes_w].values

# Dry
indexes_d = rain.query('precipMM <= {0}'.format(precip_threshold)).index
trips_pr_hour_dry = rain['trips_pr_hour'][indexes_d].values

# t-test
t_statistic, p = ttest_ind(trips_pr_hour_dry, trips_pr_hour_wet)
