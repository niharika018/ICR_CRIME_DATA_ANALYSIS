import pandas as pd
import torch
import json
import geopandas as gpd
import numpy as np

# Load the GeoJSON file
hex_gdf = gpd.read_file('grid.geojson')

# Ensure it's sorted by cell_id
hex_gdf = hex_gdf.sort_values(by='cell_id').reset_index(drop=True)

# Get the number of hexes
num_hexes = len(hex_gdf)
print(f"Number of hexes: {num_hexes}")

# Save the ordered list of cell_ids
ordered_cell_ids = hex_gdf['cell_id'].tolist()

#save for use later
with open('ordered_cell_ids.json', 'w') as f:
    json.dump(ordered_cell_ids, f)

# Load crime
crime_df = pd.read_csv('testset.csv')
crime_df['date'] = pd.to_datetime(crime_df['date']).dt.date

# Load ordered cell IDs from the geojson
with open('ordered_cell_ids.json', 'r') as f:
    ordered_cell_ids = json.load(f)

# Build the full list of dates in dataset
all_dates = pd.date_range(crime_df['date'].min(), crime_df['date'].max(), freq='D').date

# Pivot data to get daily crime matrix
daily_crime_matrix = crime_df.pivot_table(
    index='date',
    columns='cell_id',
    values='crime'
).reindex(columns=ordered_cell_ids)

daily_crime_array = daily_crime_matrix.values

daily_crime_tensor = torch.tensor(daily_crime_matrix.values, dtype=torch.float32)
torch.save(daily_crime_tensor, 'testloss_crime_tensor.pth')
# Convert to tensor

print(f"Daily crime tensor shape: {daily_crime_tensor.shape}")  # (n_days, num_hexes)