"""
In this script we calculate the Louvain clustering partition for a city of 
interrest and calculates the modularity and coverage of the final partition. 
"""


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import community.community_louvain
import contextily as ctx
import pandas as pd
import bikeshare as bs
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


#%% Louvian clustering
city = "mexico"
Data = bs.Data(city, year = 2019, month = 9)
df = Data.df

#weekdays = [2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 23, 24, 25, 26, 27, 30]
#weekends = [1, 7, 8, 14, 15, 21, 22, 28, 29]
September = range(1, 31)
adj = Data.adjacency(September, threshold = 0, remove_self_loops=True)
g = nx.from_numpy_matrix(adj)

# n_tot = Data.stat.n_tot
# id_index = Data.stat.id_index
# df['hour'] = pd.to_datetime(df['start_t']).dt.hour
# df['day'] = pd.to_datetime(df['start_t']).dt.day
# wdays = df.loc[df['day'].isin(weekdays)]
# wday_hours = dict()
# for hour in range(24):
#     wday_hours[hour] = wdays.loc[wdays['hour'] == hour]

# rush hours
# week_rush = pd.DataFrame()
# week_rush = pd.concat([ wday_hours[7], wday_hours[8], wday_hours[9],
#                         wday_hours[16], wday_hours[17], wday_hours[18],
#                         wday_hours[19]])
# adj_rush = bs.adjacency(week_rush, n_tot, id_index)
# g = nx.from_numpy_matrix(adj_rush)

#non rush
# week_nrush = pd.DataFrame()
# week_nrush = pd.concat([wday_hours[10], wday_hours[11], wday_hours[12],
#                         wday_hours[13], wday_hours[14], wday_hours[15]])
# adj_nrush = bs.adjacency(week_nrush, n_tot, id_index)
# g = nx.from_numpy_matrix(adj_nrush)

id_index = Data.stat.id_index
locations = dict()
for e in id_index.keys():
    if df[df['start_stat_id'] == e]['start_stat_lat'].shape[0]:
        locations[id_index[e]] = (df[df['start_stat_id'] == e]['start_stat_long'].iloc[0],
                                  df[df['start_stat_id'] == e]['start_stat_lat'].iloc[0])
    else:
        locations[id_index[e]] = (df[df['end_stat_id'] == e]['end_stat_long'].iloc[0],
                                  df[df['end_stat_id'] == e]['end_stat_lat'].iloc[0])

out_nodes = []
isolated_nodes = []
for e in nx.connected_components(g):
    if len(e) < 2:
        isolated_nodes.append(e.pop())

isolated_nodes.extend(out_nodes)
isolated_nodes = list(set(isolated_nodes))
g.remove_nodes_from(isolated_nodes)

for e in nx.connected_components(g):
    if len(e) < 2:
        print(e)

max_mod = 0
part = None
for i in range(50):
    partition = community.community_louvain.best_partition(g, weight='weight')
    c_g = [set() for e in range(len(partition))]
    for k, v in partition.items():
        c_g[v].add(k)
    mod = community.community_louvain.modularity(partition, g, weight='weight')
    if max_mod < mod:
        max_mod = mod
        part = partition

coverage = bs.coverage(g, part)

#%% Plotting
lat = [Data.stat.loc_merc[i][0] for i in range(Data.stat.n_tot)]
long = [Data.stat.loc_merc[i][1] for i in range(Data.stat.n_tot)]

extent = np.array([np.min(lat)-1000, np.max(lat)+1000, np.min(long)-1000, np.max(long)+1000])

fig, ax = plt.subplots(figsize=(5,8))
ax.axis(extent)
ax.set_aspect(1)
plt.tight_layout()
print("Adding basemap...")
ctx.add_basemap(ax, source=ctx.providers.Stamen.Terrain(apikey='7b7ec633799740d18aa9c877e5515b78'), attribution='(C) Stamen Design, (C) OpenStreetMap contributors')

color_cluster = ['yellow', 'blue', 'red', 'green', 'cyan', 'purple',
                 'orange','gray', 'pink', 'magenta', 'lime', 'white']

vec_col = [color_cluster[e] for e in part.values()]

print('Drawing network...')
nx.draw_networkx_nodes(g, Data.stat.loc_merc, node_size=20, node_color=vec_col, ax=ax)
nx.draw_networkx_edges(g, Data.stat.loc_merc, alpha=0.2, width=0.2, edge_color='black', ax=ax)

plt.grid(False)
_ = plt.axis('off')
scalebars = {
    'chic': 5000,
    'london': 5000,
    'madrid': 2000,
    'mexico': 2000,
    'nyc': 5000,
    'sfran': 5000,
    'taipei': 5000,
    'washDC': 5000
    }
scalebar = AnchoredSizeBar(ax.transData, scalebars[city], f'{scalebars[city]//1000:d} km', 'lower right', 
                           pad=0.2, color='black', frameon=False, size_vertical=50,
                           )
ax.add_artist(scalebar)
plt.savefig("figures/Louvain_{0}_{1}.png".format(city, 'September'), dpi = 150,
            bbox_inches = 'tight', pad_inches = 0)

#%% The Mexico Conundrum.
# n = len(vec_col)
# keys = []
# for i in range(n):
#     if vec_col[i] == "red": # make sure it is the right colour
#         keys.append(i)

# stations = []
# m = len(keys)
# for i in range(m):
#     stations.append(Data.stat.names[keys[i]])

# ids = [33, 108, 139, 155, 160, 175, 190, 222, 241, 247, 257, 260, 296, 300,
#        327, 360, 391, 434, 445, 449, 454, 457, 459, 464, 467, 469, 472, 479,]

# d = []
# print("sid #wsid #inlist")
# for _id in ids:
#     a = df[df['start_stat_id'] == _id]['end_stat_id']
#     b = a.isin(ids)
#     print(f'{_id:3d} {len(a):5d} {sum(b):5d}')
