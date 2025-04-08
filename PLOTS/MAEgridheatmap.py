import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

grid_file = "grid.geojson" #.geojson
prediction_file = "hex_predictions.csv" #.csv

#load data
hex_grid = gpd.read_file(grid_file) 
predictions_df = pd.read_csv(prediction_file)

# calculate MAE per hex cell
mae_per_cell = predictions_df.groupby('cell_id').apply(
    lambda x: np.mean(np.abs(x['predicted_crimes'] - x['actual_crimes']))
).reset_index(name='MAE')

#merge with geospatial data
if 'cell_id' in hex_grid.columns:
    hex_grid = hex_grid.merge(mae_per_cell, on='cell_id')
elif 'properties' in hex_grid.columns:
    hex_grid['cell_id'] = hex_grid['properties'].apply(lambda x: x['cell_id'])
    hex_grid = hex_grid.merge(mae_per_cell, on='cell_id')
else:
    print("GeoJSON columns:", hex_grid.columns)
    print("First feature:", hex_grid.iloc[0])
    hex_grid = hex_grid.reset_index().merge(mae_per_cell, left_on='index', right_on='cell_id')

#red-yellow-green color map
cmap = LinearSegmentedColormap.from_list('error_map', ['green', 'yellow', 'red'])

#initialize plot
fig, ax = plt.subplots(figsize=(15, 10))

#plot heat map
hex_grid.plot(
    column='MAE',
    cmap=cmap,
    legend=True,
    ax=ax,
    edgecolor='none',
    legend_kwds={
        'label': "Mean Absolute Error (MAE)",
        'shrink': 0.7
    }
)

#plot customization
ax.set_title("Crime Prediction Error by Hexagonal Grid Cell", fontsize=16)
ax.set_axis_off()
plt.tight_layout()

#save and show
plt.savefig("denver_crime_prediction_error_map.png", dpi=300, bbox_inches='tight')
plt.show()
