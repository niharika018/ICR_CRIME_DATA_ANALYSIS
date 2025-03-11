import json
import numpy as np
import pandas as pd

def parse_cell_id(cell_id):
    
    x,y = map(int, cell_id.split(","))
    
    return x, y

def double_height_to_hexagdly(x, y):

    # Rotate 90° counterclockwise
    col = x  # x-coordinates map to column indices
    row = y // 2  # Every two y-values correspond to one row

    # Shift odd columns upwards by 1 unit
    if col % 2 != 0:
        row += 1

    return row, col

def doubleheightdf_to_array(df, x_col='x', y_col='y', value_col='crime_count', default_value=0):
    
    # Convert each double-height coordinate to odd-q offset coordinates.
    df['row'], df['col'] = zip(*df.apply(lambda row: double_height_to_hexagdly(row[x_col], row[y_col]), axis=1))
    
    # Shift coordinates so that the minimum row and col become 0.
    min_row = df['row'].min()
    min_col = df['col'].min()
    df['row'] = df['row'] - min_row
    df['col'] = df['col'] - min_col
    
    # Determine grid dimensions.
    max_row = df['row'].max()
    max_col = df['col'].max()
    
    # Initialize the grid with the default value.
    grid = [[default_value for _ in range(max_col + 1)] for _ in range(max_row + 1)]
    
    # Populate the grid with the value from each DataFrame row.
    for _, r in df.iterrows():
        grid[int(r['row'])][int(r['col'])] = r[value_col]
    
    return grid