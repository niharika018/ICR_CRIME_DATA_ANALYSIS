import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import matplotlib.pyplot as plt


grid_file = "grid.geojson" #.geojson
subset_crimedata = "final_cleaned_crime_data.csv" #.csv
savefilename = "gridcounts.geojson" #.geojson
hex_grid = gpd.read_file(grid_file)

#convert dataframe to GeoDataFrame
crime_df = pd.read_csv(subset_crimedata)
crime_df['geometry'] = crime_df.apply(lambda row: Point(row['GEO_LON'], row['GEO_LAT']), axis=1)
crime_gdf = gpd.GeoDataFrame(crime_df, geometry='geometry', crs="EPSG:4326")

#ensure both GeoDataFrames have the same CRS
hex_grid = hex_grid.to_crs(epsg=4326)

#Assign crimes to a hexagon
crimes_with_hex = gpd.sjoin(crime_gdf, hex_grid, how="left", predicate="within")

#crimes per hexagon count
crime_counts = crimes_with_hex.groupby('cell_id').size().reset_index(name='crime_count')

#merge crime counts back into the hex grid
count_grid = hex_grid.merge(crime_counts, on='cell_id', how='left')
print(count_grid)
count_grid['crime_count'] = count_grid['crime_count'].fillna(0)  

count_grid.to_file(savefilename, driver='GeoJSON') 

# =============================================================================
# #test plotting 
# den = gpd.read_file("DENVER.geojson")
# 
# fig, ax = plt.subplots(figsize=(10, 10))
# base = den.plot(color="black", figsize=(10, 10), aspect="equal", alpha = 0.5)
# count_grid.plot(column = "crime_count", cmap = "coolwarm", ax=base, edgecolor="red", alpha=0.5)
# =============================================================================
