# ICR_CRIME_DATA_ANALYSIS

## CREATE GRID   

Requires: .geojson of area of interest, fiona v1.9.5

geohexgrid.py https://github.com/mrcagney/geohexgrid/blob/master/geohexgrid/main.py contains the functions used to create the grid.

DENhexgrids.py creates a hexagonal grid over Denver, CO using a given inradius, and saves it as a .geojson. Can be adapted to other areas.

## CLEAN TO TENSOR

Requires: hexagonal grid with douyble height coordinates .geojson as it produced by DENhexgrids.py, crime event .csv file with column 'FIRST_OCCURANCE_DATE' as date column, and 'GEO_LON'  and 'GEO_LAT' as location columns. 

converttoarray.py stores all the functions.

dataset_to_tensor.py takes in a grid .geojson  and crime event .csv file and returns a tensor that has a grid of crime counts for each day in the data set.

## HEXCNN_BASE_MODEL

Requires: train, validation and test tensors of crime data with heagonal grid represenation of crimes per day.

HEXCNN_network.py contains a function to create pairs of past data and forecast data, defines the base model, and trains the model when run.

validate_conv_model.py takes the model produced by HEXCNN_network.py, predicts values for the validation set and calculates MSE. 
