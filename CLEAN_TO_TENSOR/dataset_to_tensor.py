import geopandas as gpd
import pandas as pd
import converttoarray as ca
import numpy as np
import torch

grid_file = "grid.geojson" #.geojson
hex_grid = gpd.read_file(grid_file)

crime_data = "val_data.csv" #.csv
crime_df = pd.read_csv(crime_data)

save_name = "val_tensor.pth" #.pth

#get unique dates
crime_df['FIRST_OCCURRENCE_DATE'] = pd.to_datetime(crime_df['FIRST_OCCURRENCE_DATE'], errors='coerce')
crime_df['date'] = crime_df['FIRST_OCCURRENCE_DATE'].dt.date

unique_dates = sorted(crime_df['date'].dropna().unique())

#print(unique_dates[0:5])

crime_grids = []

#create grid of crime counts for each day
for date in unique_dates:
    
    grid = ca.assign_crimes_to_grid(crime_df, hex_grid, date)
    crime_grids.append(grid)
    
crime_array = np.array(crime_grids)

crime_tensor = torch.tensor(crime_array, dtype=torch.float32)

torch.save(crime_tensor, save_name)