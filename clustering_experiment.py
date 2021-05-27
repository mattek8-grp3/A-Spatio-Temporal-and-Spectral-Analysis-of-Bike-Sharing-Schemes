# -*- coding: utf-8 -*-
"""
Created on Fri May  7 09:22:24 2021

@author: andre
"""

import numpy as np
import networkx as nx
import pandas as pd
import Clustering as cl
import bikeshare as bs

#%% Initialising

city = "chic"

Data = bs.Data(city, year = 2019, month = 9)
locations = Data.stat.locations
n_tot = Data.stat.n_tot
id_index = Data.stat.id_index

week_days = [2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 23, 24, 25, 26, 27, 30]
weekend_days =[1, 7, 8, 14, 15, 21, 22, 28, 29]
month = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
         21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

df = Data.df

df['hour'] = pd.to_datetime(df['start_t']).dt.hour
df['day'] = pd.to_datetime(df['start_t']).dt.day

df.duration = df.duration / 60

pd.to_datetime(df['start_t']).dt.day.idxmax()

weekdays_list = np.where(np.array(Data.weekdays) < 5)[0] + 1
weekend_list = np.where(np.array(Data.weekdays) >= 5)[0] + 1

wdays = df.loc[df['day'].isin(weekdays_list)]
wend  = df.loc[df.day.isin(weekend_list)]

wday_hours = dict()
wend_hours = dict()

for hour in range(24):
    wday_hours[hour] = wdays.loc[wdays['hour'] == hour]
    wend_hours[hour] = wend.loc[wend['hour'] == hour]

#%% Survey for whole month
month_adjacency = Data.adjacency(month)
adj_month= nx.from_numpy_matrix(month_adjacency)
month_Laplacian = Data.get_laplacian(month)

number_of_clusters_month, labels_month = cl.spectral_clustering(month_Laplacian)

pos_month, index_list_month = cl.cluster_positions(locations, labels_month,
                                                   number_of_clusters_month)

cluster_coefficients_month = cl.cluster_coef(adj_month)

#%% Survey for weekdays
week_days_adjacency = Data.adjacency(week_days)
adj_week_days = nx.from_numpy_matrix(week_days_adjacency)
week_days_Laplacian = Data.get_laplacian(week_days)

number_of_clusters_week, labels_week = cl.spectral_clustering(week_days_Laplacian)
pos_week, index_list_week = cl.cluster_positions(locations, labels_week, number_of_clusters_week)
color_vec = []

cluster_coefficients_week = cl.cluster_coef(adj_week_days)

#%% Survey for weekends
weekend_days_adjacency = Data.adjacency(weekend_days)
adj_weekend_days = nx.from_numpy_matrix(weekend_days_adjacency)
weekend_days_Laplacian = Data.get_laplacian(weekend_days)

number_of_clusters_weekend, labels_weekend = cl.spectral_clustering(weekend_days_Laplacian)


pos_weekend, index_list_weekend = cl.cluster_positions(locations, labels_weekend,
                                                       number_of_clusters_weekend)

cluster_coefficients_weekend = cl.cluster_coef(adj_weekend_days)

#%% Rush hours vs. non rush hours weekdays
rush_hours_week = pd.DataFrame()

rush_hours_week = pd.concat([wday_hours[7], wday_hours[8],
                             wday_hours[9], wday_hours[16],
                             wday_hours[17], wday_hours[18], wday_hours[19]])

rush_hours_week_adj = bs.adjacency(rush_hours_week, n_tot, id_index)
rush_hours_week_degree = bs.get_degree_matrix(rush_hours_week_adj)
rush_hours_week_laplacian = rush_hours_week_degree - rush_hours_week_adj

number_of_clusters_rush_week, labels_rush_week = cl.spectral_clustering(rush_hours_week_laplacian)

pos_rush_week, index_list_rush_week = cl.cluster_positions(locations,
                                                           labels_rush_week,
                                                           number_of_clusters_rush_week)

adj_rush_week = nx.from_numpy_matrix(rush_hours_week_adj)
cluster_coefficients_rush_week = cl.cluster_coef(adj_rush_week)

non_rush_week = pd.DataFrame()
non_rush_week = pd.concat([wday_hours[10], wday_hours[11], wday_hours[12], wday_hours[13],
                      wday_hours[14], wday_hours[15]])

non_rush_week_adj = bs.adjacency(non_rush_week, n_tot, id_index)
non_rush_week_degree = bs.get_degree_matrix(non_rush_week_adj)
non_rush_week_laplacian = non_rush_week_degree - non_rush_week_adj

number_of_clusters_non_rush_week, labels_non_rush_week = cl.spectral_clustering(non_rush_week_laplacian)

pos_non_rush_week, index_list_non_rush_week = cl.cluster_positions(locations,
                                                                   labels_non_rush_week,
                                                                   number_of_clusters_non_rush_week)

adj_non_rush_week = nx.from_numpy_matrix(non_rush_week_adj)
cluster_coefficients_non_rush_week = cl.cluster_coef(adj_non_rush_week)
