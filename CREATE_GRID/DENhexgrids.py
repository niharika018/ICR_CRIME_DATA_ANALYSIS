# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import geopandas as gpd
import geohexgrid as ghg

den = gpd.read_file("DENVER.geojson")
grid = ghg.make_grid_from_gdf(den, R=100)