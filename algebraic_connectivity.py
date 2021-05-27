"""
This script is used to calculate the Algebraic connectivity of the analysed
cities.
"""

from itertools import chain
import networkx as nx
import numpy as np
import pandas as pd
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


#%%
city = "mexico"
year = 2019
month = 9
Data = bs.Data(city, year, month)

# Month
September = range(1,31)
adj = Data.adjacency(September, threshold = 0, remove_self_loops=True)
g = nx.from_numpy_matrix(adj)

isolated_nodes_month = [e.pop() for e in nx.connected_components(g) if len(e) < 2]

# Weekday
weekdays = np.where(np.array(Data.weekdays) < 5)[0] + 1
adj_day = Data.adjacency(weekdays, threshold = 0, remove_self_loops=True)
g_day = nx.from_numpy_matrix(adj_day)

isolated_nodes_day = [e.pop() for e in nx.connected_components(g_day) if len(e) < 2]

# Weekend
weekends = np.where(np.array(Data.weekdays) >= 5)[0] + 1
adj_end = Data.adjacency(weekends, threshold = 0, remove_self_loops=True)
g_end = nx.from_numpy_matrix(adj_end)

isolated_nodes_end = [e.pop() for e in nx.connected_components(g_end) if len(e) < 2]

isolated_nodes = isolated_nodes_month + isolated_nodes_day + isolated_nodes_end

# Month
adj_sub = cleanup_crew(isolated_nodes, adj)
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

#%% Weather
precip_threshold = 0
request, rain = bs.get_weather(city, year, month)
df = Data.df
df['hour'] = pd.to_datetime(df['start_t']).dt.hour
df['day'] = pd.to_datetime(df['start_t']).dt.day
n_tot = Data.stat.n_tot
id_index = Data.stat.id_index

indexes_w = rain.query('precipMM > {0}'.format(precip_threshold)).index
n = len(indexes_w)
hour_w = rain['hour'][indexes_w].values
day_w = rain['day'][indexes_w].values
hd_w = np.vstack((day_w, hour_w))

wet_indexes = [
    df.query('day == {0} & hour == {1}'.format(hd_w[0,i], hd_w[1,i])).index for i in range(n)]

wet_indexes = list(chain.from_iterable(wet_indexes))
df_wet = df.iloc[wet_indexes, :]

adj_wet = bs.adjacency(df_wet, n_tot, id_index, threshold=0, remove_self_loops=True)
g_wet = nx.from_numpy_matrix(adj_wet)

isolated_nodes_wet = [e.pop() for e in nx.connected_components(g_wet) if len(e) < 2]

indexes_d = rain.query('precipMM <= {0}'.format(precip_threshold)).index
k = len(indexes_d)
hour_d = rain['hour'][indexes_d].values
day_d = rain['day'][indexes_d].values
hd_d = np.vstack((day_d, hour_d))

dry_indexes = [
    df.query('day == {0} & hour == {1}'.format(hd_d[0,i], hd_d[1,i])).index for i in range(k)]

dry_indexes = list(chain.from_iterable(dry_indexes))
df_dry = df.iloc[dry_indexes, :]

adj_dry = bs.adjacency(df_dry, n_tot, id_index, threshold=0, remove_self_loops=True)
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
