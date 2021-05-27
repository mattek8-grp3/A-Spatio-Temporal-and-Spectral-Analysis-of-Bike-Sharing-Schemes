"""
In this script we will perform the spatial analysis using the degrees and the
PageRank algorithm after we have lowpass and highpass filtered the signals.
"""


import numpy as np
import matplotlib.pyplot as plt
import bikeshare as bs


def get_busy_stations(adj, days, stationnames, normalise=True):
    """
    Finds the corresponding degree to each docking station and returns a
    sorted list of the docking stations and their degree.

    Parameters
    ----------
    days : tuple
        Days in consideration.
    normalise : bool, optional
        Normalises the degrees with respect to the number of days if set
       to True, does not if set to False. The default is True.

    Returns
    -------
    list
        List of n tuples with with n being the number of stations. Each
        tuple contains the station ID, the station name and the degree of
        the station. The list is sorted with respect to the degrees in
        descending order.
    """

    degrees = np.sum(adj, axis = 0)
    deg_matrix = np.diag(degrees)
    degrees = np.sum(deg_matrix, axis = 0)

    if normalise:
        degrees = degrees/len(days)

    busy_stations = []
    for i in range(len(degrees)):
        busy_station = stationnames[i]
        busy_stations.append(busy_station)

    temp = list(zip(busy_stations, degrees))
    temp_sorted = sorted(temp, key = lambda x: x[1], reverse = True)

    return temp_sorted, degrees


def adjacency(df, day_index, days, n_tot, id_index, threshold=1, remove_self_loops=True):
    """
    Calculate weighted adjacency matrix (undirected)

    Parameters
    ----------
    df: pandas dataframe
        Contains the data over which the adjacency matrix is calculated.
    day_index: dict
        Contains index of the first trip of the day.
    days : list
        List of days in consideration.
    n_tot: int
        total amount of stations
    id_index: dict
        Contains staion names and indicies
    threshold : int, optional
        Threshold for weights. If an edge has a weight below the threshold
        then the weight is set to zero. The default threshold is 1.
    remove_self_loops : bool, optional
        Does not count trips which start and end at the same station if
        True. The default is True.

    Returns
    -------
    adj : ndarray
        Adjacency matrix of the network.
    """

    adj = np.zeros((n_tot, n_tot))
    si = df['start_stat_id'].map(id_index)
    ei = df['end_stat_id'].map(id_index)
    for day in days:
        if day is max(days):
            for i, j in zip(si[day_index[day]:], ei[day_index[day]:]):
                adj[i, j] += 1
                adj[j, i] += 1

        else:
            for i, j in zip(si[day_index[day]:day_index[day+1]], ei[day_index[day]:day_index[day+1]]):
                adj[i, j] += 1
                adj[j, i] += 1

    adj[adj <= threshold] = 0
    
    if remove_self_loops == True:
        for i in range(n_tot):
            adj[i, i] = 0

    return adj


def diradjacency(df, day_index, days, n_tot, id_index,
                 threshold=1, remove_self_loops=True):
    """
    Calculate the directed adjacency matrix for the network.

    Parameters
    ----------
    df : pandas DataFrame
        bikesharing data.
    day_index : list
        Indices of the first trip per day.
    days : iterable
        Days in consideration.
    n_tot : int
        total amount of verticies in df
    id_index : dict
        dictionary containing the station names and indicies to map them to
    threshold : int, optional
        Threshold for weights. If an edge has a weight below the threshold
        then the weight is set to zero. The default threshold is 1.
    remove_self_loops : bool, optional
        Does not count trips which start and end at the same station if
        True. The default is True.

    Returns
    -------
    d_adj : ndarray
        Array containing the directed adjacency matrix.

    """

    d_adj = np.zeros((n_tot, n_tot))
    for day in days:

        if day is max(days):
            for _, row in df.iloc[day_index[day]:].iterrows():
                d_adj[id_index[row['start_stat_id']],
                      id_index[row['end_stat_id']]] += 1


        else:
            for _, row in df.iloc[day_index[day]:day_index[day+1]].iterrows():
                d_adj[id_index[row['start_stat_id']],
                     id_index[row['end_stat_id']]] += 1

    d_adj[d_adj <= threshold] = 0

    if remove_self_loops:
        for i in range(n_tot):
            d_adj[i, i] = 0

    return d_adj


city = "washDC"
year = 2019
month = 9
cutoff_dict = {"chic": 14.8, "london": 20.5, "madrid": 11.1, "mexico": 15.7, "nyc": 19.1, "sfran": 8.6, "taipei": 9.4, "washDC": 14.3}
Data = bs.Data(city, year = year, month = month)
cutoff = cutoff_dict[city]
weekdays = [2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 23, 24, 25, 26, 27, 30]
weekends = [1, 7, 8, 14, 15, 21, 22, 28, 29]

adj = Data.adjacency(weekends, threshold = 0)
filterarray = bs.TotalVariation(adj, cutoff)
df = Data.df
id_index = Data.stat.id_index
lowpass = True
df_filtered = bs.subframe(filterarray, df, id_index, lowpass)
total_station_id_filtered = set(df_filtered['start_stat_id']).union(set(df_filtered['end_stat_id']))
n_tot_filtered = len(total_station_id_filtered)
id_index_filtered = dict(zip(sorted(total_station_id_filtered), np.arange(n_tot_filtered)))
day_index_filtererd = bs.days_index(df_filtered)

#%% PageRank
d_adj_filtered = diradjacency(df_filtered, day_index_filtererd, weekends, n_tot_filtered, id_index_filtered, threshold=0, remove_self_loops=True)
P = bs.PageRank(d_adj_filtered, d = 0.85, iterations = 100, initialisation = "rdm")
least_popular_stations = np.argsort(P, axis = None)
popular_stations = np.flip(least_popular_stations)
stationnames = bs.station_names(df_filtered, id_index_filtered)
max_rank = 5
popular_station_names = [stationnames[popular_stations[i]] for i in range(max_rank)]

count = 0
print('\nThe {} stations with the highest PageRank:\n'.format(max_rank))
for station in popular_station_names:
    print('Rank {} - {}: PageRank is {}'.format(count+1, station, P[popular_stations[count]][0]))
    count += 1

if city == "chic":
    cityplot = "Chicago"
elif city == "london":
    cityplot = "London"
elif city == "madrid":
    cityplot = "Madrid"
elif city == "mexico":
    cityplot = "Mexico City"
elif city == "nyc":
    cityplot = "New York City"
elif city == "sfran":
    cityplot = "San Francisco"
elif city == "taipei":
    cityplot = "Taipei"
else:
    cityplot = "Washington, D.C."

sorting_page = np.sort(P.reshape(n_tot_filtered))
plt.style.use('seaborn-darkgrid')
plt.plot(sorting_page, '.')
plt.xlabel("Index")
plt.ylabel("PageRank")
plt.title("PageRank_{0}".format(cityplot))
plt.savefig("figures/Pagedist_{0}_{1}_{2}.pdf".format(city, weekends, lowpass))
plt.show()

#%% Degree
adj_filtered = adjacency(df_filtered, day_index_filtererd, weekends, n_tot_filtered, id_index_filtered, threshold=0, remove_self_loops=True)

N = 5
busy_stations, deg_vec = get_busy_stations(adj_filtered, weekends, stationnames, normalise = True)
degrees = [station[1] for station in busy_stations]
mean_degree = np.mean(degrees)

count = 0
print('\n{} stations with the highest degrees:\n'.format(N))
for station in busy_stations[:N]:
    print('Rank {} - {}: degree is {}'.format(
        count+1, station[0], np.round(station[1], decimals = 1)))
    count += 1

avg = np.sum(deg_vec)/n_tot_filtered

sorting_deg = np.sort(deg_vec)
plt.style.use('seaborn-darkgrid')
plt.plot(sorting_deg, '.')
plt.xlabel("Index")
plt.ylabel("Normalised degree")
plt.title("Sorted Degrees for {}".format(cityplot))
plt.savefig("figures/degreedist_{0}_{1}_{2}.pdf".format(city, weekends, lowpass))
plt.show()
