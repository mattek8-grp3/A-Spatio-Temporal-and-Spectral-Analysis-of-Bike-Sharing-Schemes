"""
This is script is used to calculate the algebraic connectivity, community
partition, modularity and coverage in mexico city after removing the rides
on electric bicycles in Mexico City.
"""


from itertools import chain
import pandas as pd
import numpy as np
import networkx as nx
import community.community_louvain
import matplotlib.pyplot as plt
import contextily as ctx
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from pyproj import Transformer
import bikeshare as bs


def eigenvalues(laplacian):
    """
    Calculates the Algebraic connectivity of a graph

    Parameters
    ----------
    laplacian : ndarray
        The Laplacian matrix.

    Returns
    -------
    eig : ndarray
        Array of the eigenvalues.

    """

    e, _ = np.linalg.eig(laplacian)
    N = len(e)
    sorter = np.argsort(e.reshape((N, 1)), axis = None)
    eig = np.zeros(N)

    for i in range(N):
        eig[i] = e[sorter[i]]

    return eig


def cleanup_crew(isolated_nodes, adj):
    """
    Parameters
    ----------
    isolated_nodes : list
        List containing the one component nodes.
    adj : ndarray
        Adjacency matrix.

    Returns
    -------
    adj_sub : ndarray
        Adjacency matrix without the isolated nodes.
    """

    g = nx.from_numpy_matrix(adj)
    isolated_nodes = list(set(isolated_nodes))
    g.remove_nodes_from(isolated_nodes)
    adj_sub = nx.convert_matrix.to_numpy_matrix(g)

    return adj_sub


city = "mexico"
year = 2019
month = 9
Data = bs.Data(city, year, month)
df = Data.df
df_filtered = df[df['station_type'].isin(['BIKE', 'BIKE,TPV'])]
total_station_id_filtered = set(df_filtered['start_stat_id']).union(set(df_filtered['end_stat_id']))
n_tot_filtered = len(total_station_id_filtered)
id_index_filtered = dict(zip(sorted(total_station_id_filtered), np.arange(n_tot_filtered)))
day_index_filtererd = bs.days_index(df_filtered)

#%% Algebraic Connectivity

# Month
September = range(1,31)
adj_month = bs.adjacency_filtered(df_filtered, day_index_filtererd, September, n_tot_filtered, id_index_filtered, threshold=0, remove_self_loops=True)
g = nx.from_numpy_matrix(adj_month)
isolated_nodes_month = [e.pop() for e in nx.connected_components(g) if len(e) < 2]

# Weekday
weekdays = [2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 23, 24, 25, 26, 27, 30]
adj_day = bs.adjacency_filtered(df_filtered, day_index_filtererd, weekdays, n_tot_filtered, id_index_filtered, threshold=0, remove_self_loops=True)
g_day = nx.from_numpy_matrix(adj_day)
isolated_nodes_day = [e.pop() for e in nx.connected_components(g_day) if len(e) < 2]

# Weekend
weekends = [1, 7, 8, 14, 15, 21, 22, 28, 29]
adj_end = bs.adjacency_filtered(df_filtered, day_index_filtererd, weekends, n_tot_filtered, id_index_filtered, threshold=0, remove_self_loops=True)
g_end = nx.from_numpy_matrix(adj_end)
isolated_nodes_end = [e.pop() for e in nx.connected_components(g_end) if len(e) < 2]

isolated_nodes = isolated_nodes_month + isolated_nodes_day + isolated_nodes_end

# Month
adj_sub = cleanup_crew(isolated_nodes, adj_month)
deg_matrix = bs.get_degree_matrix(adj_sub)
laplaceian = deg_matrix - adj_sub
eigs = eigenvalues(laplaceian)
connectivity = eigs[1]

# Weekday
adj_sub_day = cleanup_crew(isolated_nodes, adj_day)
deg_matrix_day = bs.get_degree_matrix(adj_sub_day)
laplaceian_day = deg_matrix_day - adj_sub_day
eigs_day = eigenvalues(laplaceian_day)
connectivity_weekday = eigs_day[1]

# Weekend
adj_sub_end = cleanup_crew(isolated_nodes, adj_end)
deg_matrix_end = bs.get_degree_matrix(adj_sub_end)
laplaceian_end = deg_matrix_end - adj_sub_end
eigs_end = eigenvalues(laplaceian_end)
connectivity_weekend = eigs_end[1]

# Weather
precip_threshold = 0
request, rain = bs.get_weather(city, year, month)
df_filtered['hour'] = pd.to_datetime(df_filtered['start_t']).dt.hour
df_filtered['day'] = pd.to_datetime(df_filtered['start_t']).dt.day
df_filtered.reset_index(drop=True, inplace = True)

indexes_w = rain.query('precipMM > {0}'.format(precip_threshold)).index
n = len(indexes_w)
hour_w = rain['hour'][indexes_w].values
day_w = rain['day'][indexes_w].values
hd_w = np.vstack((day_w, hour_w))

wet_indexes = [
    df_filtered.query('day == {0} & hour == {1}'.format(hd_w[0,i], hd_w[1,i])).index for i in range(n)]

wet_indexes = list(chain.from_iterable(wet_indexes))
df_wet = df_filtered.iloc[wet_indexes, :]

adj_wet = bs.adjacency(df_wet, n_tot_filtered, id_index_filtered, threshold=0, remove_self_loops=True)
g_wet = nx.from_numpy_matrix(adj_wet)

isolated_nodes_wet = [e.pop() for e in nx.connected_components(g_wet) if len(e) < 2]

indexes_d = rain.query('precipMM <= {0}'.format(precip_threshold)).index
k = len(indexes_d)
hour_d = rain['hour'][indexes_d].values
day_d = rain['day'][indexes_d].values
hd_d = np.vstack((day_d, hour_d))

dry_indexes = [
    df_filtered.query('day == {0} & hour == {1}'.format(hd_d[0,i], hd_d[1,i])).index for i in range(k)]

dry_indexes = list(chain.from_iterable(dry_indexes))
df_dry = df_filtered.iloc[dry_indexes, :]

adj_dry = bs.adjacency(df_dry, n_tot_filtered, id_index_filtered, threshold=0, remove_self_loops=True)
g_dry = nx.from_numpy_matrix(adj_dry)

isolated_nodes_dry = [e.pop() for e in nx.connected_components(g_dry) if len(e) < 2]

isolated_nodes = isolated_nodes_wet + isolated_nodes_dry

# Wet
adj_sub_wet = cleanup_crew(isolated_nodes, adj_wet)
deg_matrix_wet = bs.get_degree_matrix(adj_sub_wet)
laplaceian_wet = deg_matrix_wet - adj_sub_wet
eigs_wet = eigenvalues(laplaceian_wet)

if len(eigs_wet) != 0:
    connectivity_wet = eigs_wet[1]

else:
    connectivity_wet = 0

# Dry
adj_sub_dry = cleanup_crew(isolated_nodes, adj_dry)
deg_matrix_dry = bs.get_degree_matrix(adj_sub_dry)
laplaceian_dry = deg_matrix_dry - adj_sub_dry
eigs_dry = eigenvalues(laplaceian_dry)
connectivity_dry = eigs_dry[1]

#%% Louvain Clustering
locations = dict()
for e in id_index_filtered.keys():
    if df_filtered[df_filtered['start_stat_id'] == e]['start_stat_lat'].shape[0]:
        locations[id_index_filtered[e]] = (df_filtered[df_filtered['start_stat_id'] == e]['start_stat_long'].iloc[0],
                                  df_filtered[df_filtered['start_stat_id'] == e]['start_stat_lat'].iloc[0])
    else:
        locations[id_index_filtered[e]] = (df_filtered[df_filtered['end_stat_id'] == e]['end_stat_long'].iloc[0],
                                  df_filtered[df_filtered['end_stat_id'] == e]['end_stat_lat'].iloc[0])

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
trans = Transformer.from_crs("EPSG:4326", "EPSG:3857")
locations = bs.station_locations(df_filtered, id_index_filtered)
loc = np.array(list(locations.values()))
loc_merc = np.vstack(trans.transform(loc[:,1], loc[:,0])).T


lat = [loc_merc[i][0] for i in range(n_tot_filtered)]
long = [loc_merc[i][1] for i in range(n_tot_filtered)]

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
nx.draw_networkx_nodes(g, loc_merc, node_size=20, node_color=vec_col, ax=ax)
nx.draw_networkx_edges(g, loc_merc, alpha=0.2, width=0.2, edge_color='black', ax=ax)

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
plt.savefig("figures/Louvain_MexicoInvestigation_{0}_{1}.png".format(city, 'September'), dpi = 150,
            bbox_inches = 'tight', pad_inches = 0)
