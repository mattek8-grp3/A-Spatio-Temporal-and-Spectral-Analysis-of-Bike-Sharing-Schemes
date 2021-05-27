"""
In this script we have implemented functions to calculate and plot the mean
number of hourly trips for the lowpass- and highpass-filtered data.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bikeshare as bs


def trips_pr_hour_filter(days, df):
    """
    Calculates the mean amount of trips pr. hour for the high and
    low frequency dataframes.

    Parameters
    ----------
    days : list
        list containing days of interrest.
    df : pandas dataframe
        citydata.

    Returns
    -------
    trips_pr_hour : ndarray
        containing trips pr. hour pr. day.

    """

    trips_pr_hour = np.zeros((len(days)+1, 24))
    df_check = df[df['day'].isin(days)]
    id_index_filtered = dict(zip(days, np.arange(len(days))))
    day = df_check['day'].map(id_index_filtered)
    hour = df_check['hour']
    for d, h in zip(day, hour):
        trips_pr_hour[d, h] += 1
    return trips_pr_hour


def Usertype(df, days, city):
    """
    Calculates how many trips are taken by subscribers/customers

    Parameters
    ----------
    df : pandas dataframe
        citydata.
    days : list
        list containing days of interrest.
    city : str
        str describing which city we are wroking with.

    Returns
    -------
    CS : ndarray
        contains amount of customer and subscriber trips.

    """

    df = df[df['day'].isin(days)]
    citylist = ["chic", "nyc", "sfran"]
    if city in citylist:
        CS = np.zeros(2)
        usertype = df['user_type']
        for u in usertype:
            if u == "Customer":
                CS[0] +=1
            else:
                CS[1] += 1
    elif city == "washDC":
        CS = np.zeros(2)
        usertype = df['user_type']
        for u in usertype:
            if u == "Casual":
                CS[0] +=1
            else:
                CS[1] += 1
    elif city == "madrid":
        CS = np.zeros(3)
        usertype = df['user_type']
        for u in usertype:
            if u == 2:
                CS[0] +=1
            elif u == 1:
                CS[1] += 1
            else:
                CS[2] += 1
    else:
        CS = np.zeros(2)
        print("City does not contain demographic data")
    return CS


#%% Initialise
city = "chic"
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
df['hour'] = pd.to_datetime(df['start_t']).dt.hour
df['day'] = pd.to_datetime(df['start_t']).dt.day
id_index = Data.stat.id_index

df_filtered_l = bs.subframe(filterarray, df, id_index, low = True)
df_filtered_l['hour'] = pd.to_datetime(df_filtered_l['start_t']).dt.hour
df_filtered_l['day'] = pd.to_datetime(df_filtered_l['start_t']).dt.day

df_filtered_h = bs.subframe(filterarray, df, id_index, low = False)
df_filtered_h['hour'] = pd.to_datetime(df_filtered_h['start_t']).dt.hour
df_filtered_h['day'] = pd.to_datetime(df_filtered_h['start_t']).dt.day

#%% Trips pr. Hour
trips_pr_hour_l_d = trips_pr_hour_filter(weekends, df_filtered_l)
trips_pr_hour_h_d = trips_pr_hour_filter(weekends, df_filtered_h)
m = len(weekends)
trips_pr_hour_h = np.zeros(24)
trips_pr_hour_l = np.zeros(24)
for i in range(24):
    trips_pr_hour_h[i] = np.sum(trips_pr_hour_h_d[:,i])/m
    trips_pr_hour_l[i] = np.sum(trips_pr_hour_l_d[:,i])/m

hours = range(24)
plt.style.use('seaborn-darkgrid')
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8,4))
ax[0].bar(hours, trips_pr_hour_l)
ax[1].bar(hours, trips_pr_hour_h)
ax[0].set_title("Lowpass Filtered")
ax[1].set_title("Highpass Filtered")
ax[0].set_xlabel("Hour")
ax[1].set_xlabel("Hour")
ax[0].set_ylabel("Number of Rides")
ax[0].set_xticks([0,3,6,9,12,15,18,21])
ax[1].set_xticks([0,3,6,9,12,15,18,21])
fig.subplots_adjust(hspace=0.1, wspace=0.1)
plt.savefig("figures/filter_{0}_{1}_{2}.pdf".format(city, cutoff_dict[city], weekends))
plt.show()

#%% Usertype
CS_l = Usertype(df_filtered_l, weekends, city)
CS_h = Usertype(df_filtered_h, weekends, city)
citylist = ["chic", "nyc", "sfran", "washDC"]

if city in citylist:
    names = ["Customer", "Subscriber"]
else:
    names = ["Customer", "Subscriber", "Workforce"]

if CS_l[0] != 0:
    plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8,4))
    ax[0].bar(names, CS_l)
    ax[1].bar(names, CS_h)
    ax[0].set_title("Lowpass Filtered")
    ax[1].set_title("Highpass Filtered")
    ax[0].set_xlabel("User type")
    ax[1].set_xlabel("User type")
    ax[0].set_ylabel("# of types")
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.savefig("figures/filter_usertype_{0}_{1}_{2}.pdf".format(city, cutoff_dict[city], weekends))
    plt.show()

#%% Trips pr. hour based on usertype
customer = ["Customer"]
df_customer = df[df['user_type'].isin(customer)]
df_subscriber = df[~df['user_type'].isin(customer)]
trips_pr_hour_customer_d = trips_pr_hour_filter(weekends, df_customer)
trips_pr_hour_subscriber_d = trips_pr_hour_filter(weekends, df_subscriber)
m = len(weekends)
trips_pr_hour_customer = np.zeros(24)
trips_pr_hour_subscriber = np.zeros(24)
for i in range(24):
    trips_pr_hour_customer[i] = np.sum(trips_pr_hour_customer_d[:,i])/m
    trips_pr_hour_subscriber[i] = np.sum(trips_pr_hour_subscriber_d[:,i])/m

hours = range(24)
plt.style.use('seaborn-darkgrid')
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8,4))
ax[0].bar(hours, trips_pr_hour_customer)
ax[1].bar(hours, trips_pr_hour_subscriber)
ax[0].set_title("Customer Rides")
ax[1].set_title("Subscriber Rides")
ax[0].set_xlabel("Hour")
ax[1].set_xlabel("Hour")
ax[0].set_ylabel("Number of Rides")
ax[0].set_xticks([0,3,6,9,12,15,18,21])
ax[1].set_xticks([0,3,6,9,12,15,18,21])
fig.subplots_adjust(hspace=0.1, wspace=0.1)
plt.savefig("figures/userfilter_{0}_{1}_{2}.pdf".format(city, cutoff_dict[city], weekends))
plt.show()
