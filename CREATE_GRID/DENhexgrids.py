#need fiona 1.9.5

import geopandas as gpd
import geohexgrid as ghg
import matplotlib.pyplot as plt

r = 0.005
den = gpd.read_file("DENVER.geojson")
grid = ghg.make_grid_from_gdf(den, R=r)

base = den.plot(color="black", figsize=(10, 10), aspect="equal", alpha = 0.5)
grid.plot(ax=base, color="white", edgecolor="red", alpha=0.5)

plt.show()
