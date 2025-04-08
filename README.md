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

validate_conv_model.py takes the model produced by HEXCNN_network.py, predicts values for the validation set and calculates RMSE and MAE. 

## TFT_BASE_MODEL

Requires: .csv where each crime is a row with assigned cell_ids

TFTdataprep.py prepares the data for use and prediction in TFT base model.

TFT_model.py trains a TFT base model and saves the parameters.

validate_tft_model.py takes in a test data set and the trained TFT base model, makes and saves predictions, and calculatse RMSE and MAE.

## PLOTS

Requires: .geojson corresponding to hexagonal grid, .csv file containing the predictions on the test dataset in the form date, forecast_day (1-7), predicted_crimes, actual_crimes, absolute_error

MAEgridheatmap.py creates a heatmap of absolute error on a hexagonal grid.

crimesperdayplot.py creates a time series plot of total crimes predicted per day over a predefined time period.

residualplot.py creates a histogram of residuals for the test data, gives skew and standard deviation, and a line that represents the mean bias.




