"""
@author: Mattek Group 3
"""

import numpy as np
import bikeshare as bs

#%%
city = "washDC"
Data = bs.Data(city, year = 2019, month = 9)
df = Data.df
locations = Data.stat.locations
n = len(locations)

distances = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        distances[i,j] = bs.distance(locations[i][1], locations[i][0],
                                  locations[j][1], locations[j][0])

mdcds = 0
for i in range(n):
    tmp = np.argsort(distances[:,i])
    mdcds += distances[tmp[1], i]

mdcds = mdcds/n

#%% Filtered
cutoff_dict_month = {"chic": 14.8, "london": 20.5, "madrid": 11.1,
                     "mexico": 15.7, "nyc": 19.1, "sfran": 8.6,
                     "taipei": 9.4, "washDC": 14.3}

cutoff = cutoff_dict_month[city]
September = range(1,31)
adj = Data.adjacency(September, threshold=0)

T, filterarray = bs.TotalVariation(adj, cutoff)
id_index = Data.stat.id_index
df_filtered = bs.subframe(filterarray, df, id_index, low = False)
total_station_id_filtered = set(df_filtered['start_stat_id']).union(set(df_filtered['end_stat_id']))
n_tot_filtered = len(total_station_id_filtered)
id_index_filtered = dict(zip(sorted(total_station_id_filtered), np.arange(n_tot_filtered)))
locations_filtered = bs.station_locations(df_filtered, id_index_filtered)
m = len(locations_filtered)

distances_filtered = np.zeros((m, m))

for i in range(m):
    for j in range(m):
        distances_filtered[i,j] = bs.distance(locations_filtered[i][1],
                                           locations_filtered[i][0],
                                           locations_filtered[j][1],
                                           locations_filtered[j][0])

mdcds_filtered = 0
for i in range(m):
    tmp = np.argsort(distances_filtered[:,i])
    mdcds_filtered += distances_filtered[tmp[1], i]

mdcds_filtered = mdcds_filtered/m
