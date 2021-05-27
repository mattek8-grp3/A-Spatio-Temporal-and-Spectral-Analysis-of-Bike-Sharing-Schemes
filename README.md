# A Spatio-Temporal and Spectral Analysis of Bike-Sharing Schemes

Scripts for the analysis of bike-sharing data. These scipts have been developed for the purpose of a 2nd semester project of the Master in Mathematical Engineering at Aalborg University:

A. A. Andersen, D. B. van Diepen, M. S. Kaaber, and N. A. Weinreich. "A Spatio-Temporal and Spectral Analysis of Bike-Sharing Schemes", Aalborg University, Denmark, 2021

Internally published in the project catalogue at Aalborg University.


## Dependencies
This project is created with `Python 3.9`

Dependencies:
```
matplotlib 3.3.4
mplleaflet 0.0.5
numpy 1.20.1
networkx 2.5
pandas 1.2.2
pickle 1.0.2
pygsp 0.5.1
python-louvain 0.15
scipy 1.20.1
sklearn 0.24.1
contextily 1.1.0
pyproj 3.1.0
```

### Conda environment
If working with Conda, you could for example make a conda environment as follows. (You can ignore spyder if you wish to use another development environment)

```
conda create -n bike_env numpy matplotlib pandas scipy spyder

conda activate bike_env
```
For Windows users, follow the instructions on https://rasterio.readthedocs.io/en/latest/installation.html#windows to install `rasterio`

Then install the remaining packages
```
pip install networkx python-louvain sklearn contextily pygsp mplleaflet pyproj
```

## Directory structure

The data should be organised as follows. Please create directories `data`, `python_variables`, `python_variables/big_data`, and `figures` as necessary.

```
./data
├── (Put data .csv/.json files here)
├── Divvy_Trips_2019_Q3.csv
│
├── 177JourneyDataExtract28Aug2019-03Sep2019.csv
├── 178JourneyDataExtract04Sep2019-10Sep2019.csv
├── 179JourneyDataExtract11Sep2019-17Sep2019.csv
├── 180JourneyDataExtract18Sep2019-24Sep2019.csv
├── 181JourneyDataExtract25Sep2019-01Oct2019.csv
├── london_stations.csv
│
├── 201908_movements.json
├── 201909_movements.json
├── 201909_stations_madrid.json
│
├── 2019-09-mexico.csv
├── stations_mexico.json
│
├── 201909-citibike-tripdata.csv
│
├── 201909-baywheels-tripdata.csv
│
├── 201909-taipei.csv
├── stations_taipei.csv
│
├── 201909-capitalbikeshare-tripdata.csv
└── Capital_Bike_Share_Locations.csv


./python_variables
├── big_data
│   └── (Dataframe pickles will be here)
│
└── (Pickle files will be here)


./figures
└── (Figures will be here)
```

## Scripts:

`bikeshare.py`
	- Module script containing various functions used in the other scripts. Includes importing data, computing adjacency matrices etc.

`dataframe_key.py`
	- Dictionaries for converting data from proprietary formats to a common format.

`daily_patterns.py`
	- Analyse and plot temporal patterns and the influence of demographic information.

`algebraic_connectivity.py`
	- Calculate the algebraic connectivity of each city.

`clustering.py`
	- Contains functions for calculating clustering.

`clustering_experiment.py`
	- Perform clustering analysis for a given city.

`louvain_clustering.py`
	- Calculate and plot Louvain clusters and coverage.

`distance.py`
	- Calculate the distance between stations using the Haversine formula.

`hourly.py`
	- Calculate and plot filtered trips per hour.

`hypotese_test.py`
	- Calculates the significanse of precipitation by making a t-test.

`spatial_trip_analysis.py`
	- Calculates the PageRank and the average degrees and determines the five most important stations. The network is also plotted with PageRank and degree. Also determines the most common trips in the network.

`spatial_trip_analysis_filtered.py`
	- Calculates the PageRank and the average degrees after filtering and determines the five most important stations.

# Data Sources
Trip data can be accessed at the following locations

| City             | Link                                                                                   |
|------------------|----------------------------------------------------------------------------------------|
| Chicago          | https://www.divvybikes.com/system-data                                                 |
|                  | https://data.cityofchicago.org/Transportation/Divvy-Bicycle-Stations-All-Map/bk89-9dk7 |
| London           | https://cycling.data.tfl.gov.uk/                                                       |
| Madrid           | https://opendata.emtmadrid.es/Datos-estaticos/Datos-generales-(1)                      |
| Mexico City      | https://www.ecobici.cdmx.gob.mx/en/informacion-del-servicio/open-data                  |
| New York City    | https://www.citibikenyc.com/system-data                                                |
| San Francisco    | https://www.lyft.com/bikes/bay-wheels/system-data                                      |
| Taipei           | https://drive.google.com/drive/folders/1QsROgp8AcER6qkTJDxpuV8Mt1Dy6lGQO               |
| Washington, D.C. | https://www.capitalbikeshare.com/system-data                                           |

