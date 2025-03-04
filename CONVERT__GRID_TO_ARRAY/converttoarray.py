import json
import numpy as np
import pandas as pd

def parse_cell_id(cell_id):
    
    x,y = map(int, cell_id.split(","))
    
    return x, y

def doubleheight_to_axial(x,y):
    
    q = x
    
    r = (y-x)/2
    
    return (q,r)

def axialdf_to_array(df):
    
    min_q, max_q = df["q"].min(), df["q"].max()
    min_r, max_r = df["r"].min(), df["r"].max()
    
    rows = max_r - min_r + 1  
    cols = max_q - min_q + 1  
    
    crime_array = np.full((rows, cols), fill_value = 0, dtype = int)
    
    for _, row in df.iterrows():
        q, r, crime_count = int(row["q"]), int(row["r"]), int(row["crime_count"])
    

        array_row = r - min_r  
        array_col = q - min_q  


        crime_array[array_row, array_col] = crime_count
    
    return crime_array