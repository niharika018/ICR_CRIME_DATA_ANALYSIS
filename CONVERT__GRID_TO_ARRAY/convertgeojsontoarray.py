# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 21:29:57 2025

@author: Daniel Long
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import converttoarray as ca

grid_file = "gridcounts.geojson"
grid_counts = gpd.read_file(grid_file)

data = []

for _, row in grid_counts.iterrows():
        cell_id = row["cell_id"]
        crime_count = row["crime_count"]
        
        # Parse the cell_id string into x and y.
        x, y = ca.parse_cell_id(cell_id)
        data.append({
            "x": x,
            "y": y,
            "crime_count": int(crime_count)
        })

df = pd.DataFrame(data)

crime_array = ca.doubleheightdf_to_array(df, x_col='x', y_col='y', value_col='crime_count')


# =============================================================================
print(crime_array)
# 
np.savetxt("crime_array.csv", crime_array, delimiter=",", fmt="%d")
# =============================================================================
