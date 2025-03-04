# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 21:29:57 2025

@author: Daniel Long
"""

import geopandas as gpd
import pandas as pd
import converttoarray as ca

grid_file = "gridcounts.geojson"
grid_counts = gpd.read_file(grid_file)

data = []

for _, row in grid_counts.iterrows():
    
    cell_id = row["cell_id"]
    crime_count = row["crime_count"]
    
    x,y = ca.parse_cell_id(cell_id)
    q, r = ca.doubleheight_to_axial(x, y)
    s = -q-r
    
    data.append({"q": int(q), "r": int(r), "s": int(s), "crime_count": int(crime_count)})
    

df = pd.DataFrame(data)

crime_array = ca.axialdf_to_array(df)


# =============================================================================
# print(crime_array)
# 
# np.savetxt("crime_array.csv", crime_array, delimiter=",", fmt="%d")
# =============================================================================
