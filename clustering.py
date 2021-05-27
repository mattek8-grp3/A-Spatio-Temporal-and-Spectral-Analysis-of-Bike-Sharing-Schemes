# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 09:30:42 2021

@author: andre
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import networkx as nx
import bikeshare as bs

def spectral_clustering(laplacian):
    """
    Determines the clusters using the spectral method for graph clustering.
    ----------
    laplacian : ndarray
             Array containing the laplacian matrix.

    Returns
    -------
    k : int
     The number of graph clusters obtained from a K-means algorithm.

    kmeans : ndarray
          Array containing the cluster labels for each of the docking stations.
    """

    e, v = np.linalg.eig(laplacian)

    N = len(e)
    sorter = np.argsort(e.reshape((N,1)), axis = None)


    eig = np.zeros(N)
    eig_v = np.zeros((N,N))
    for i in range(N):
        eig[i] = e[sorter[i]]
        eig_v[i] = v[sorter[i]]

    k = 0
    index_list = []
    for i in range(N):
        if eig[i] <= 0.00001:
            index_list.append(i)
            k += 1


    non_zero_vec = []
    if k > N/2:
        for i in range(k+1,N):
            non_zero_vec.append(eig_v[:,i])
    else:
        for i in range(k):
            non_zero_vec.append(eig_v[:,i+k])

    non_zero_vec = np.array(non_zero_vec)
    non_zero_vec = non_zero_vec.T

    y = []

    for i in range(N):
        y.append(non_zero_vec[i,:])

    y_vec = np.array(y)
    if k > N/2:
        kmeans = KMeans(n_clusters = N-k).fit_predict(y_vec)
        print('Gives N-k clusters since k > N/2. According to the algoritm, there exists {} clusters'.format(k))
        return N-k, kmeans

    kmeans = KMeans(n_clusters = k).fit_predict(y_vec)

    return k, kmeans


def cluster_positions(locations, clusters, k):
    """
    Creates a dictionary with station IDs as keys and locations as values for
    a specific graph cluster.
    ----------
    locations : dict
        key : station index (returned from id_index)
        value : tuple (longitude, latitude)
             Dictionary containing locations of all docking stations in the data.

    clusters : ndarray
            Array containing the cluster labels for each of the docking stations.

    k : int
     The number of graph clusters.

    Returns
    -------
    positions : list
        value : tuple (longitude, latitude)
             List containing the locations of the stations for each cluster.

    index_list : list
              List containing the indices for the docking stations for each cluster.
    """
    positions = []
    index_list = []
    for j in range(k):
        positions_j = []
        index_list_j = []
        for i , _ in enumerate(clusters):
            if clusters[i] == j:
                positions_j.append(locations[i])
                index_list_j.append(i)
        positions.append(positions_j)
        index_list.append(index_list_j)
    return positions, index_list


def cluster_coef(adjacency):
    """
    Calculates different clustering/similarity measures for a graph.
    ----------
    adjacency : NetworkX graph
             Graph described by the adjacency matrix of the graph.

    Returns
    -------
    local_cluster_coef : dict
        key : station index (returned from id_index)
        value : local clustering coefficient for the specific station
             Dictionary containing the local clustering coefficients.

    local_average_clustering : float
                            The average local clustering coefficient.

    jaccard_index : float
                  The average Jaccard index.

    pearson : float
           The Pearson correlation coefficient.
    """
    local_cluster_coef = nx.clustering(adjacency)
    local_average_clustering = nx.average_clustering(adjacency)

    jaccard = nx.jaccard_coefficient(adjacency)
    jaccard_list = []
    for _ , _ , p in jaccard:
        jaccard_list.append(p)

    jaccard_index = 1/len(jaccard_list) * sum(jaccard_list)

    pearson = nx.degree_pearson_correlation_coefficient(adjacency)

    return local_cluster_coef, local_average_clustering, jaccard_index, pearson


if __name__ == "__main__":
    city = "nyc"
    year = 2019
    month = 9
    Data = bs.Data("nyc", year = 2019, month = 9)
    locations = Data.stat.locations

    day = [2]
    day_adjacency = Data.adjacency(day)
    adj = nx.from_numpy_matrix(day_adjacency)
    day_Laplacian = Data.get_laplacian(day)

    val = spectral_clustering(day_Laplacian)
    labels = val[1]
    color_vec = []

    pos, index = cluster_positions(locations, labels, val[0])

    for i in range(len(val[1])):
        if labels[i] == 0:
            color_vec.append('blue')

        if labels[i] == 1:
            color_vec.append('red')

        if labels[i] == 2:
            color_vec.append('green')

    nx.draw_networkx_nodes(adj, locations, node_size=1, node_color=color_vec)
    nx.draw_networkx_edges(adj, locations, alpha=0.8, width=0.1, edge_color='black')
    plt.grid(False)
    _ = plt.axis('off')
