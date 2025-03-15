import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

#parses cell id from grid

def parse_cell_id(cell_id):
    
    x,y = map(int, cell_id.split(","))
    
    return x, y

#converts double height coordinates to hexagdly coordinates
def double_height_to_hexagdly(x, y):

    # Rotate 90° counterclockwise
    col = x  # x-coordinates map to column indices
    row = y // 2  # Every two y-values correspond to one row

    # Shift odd columns upwards by 1 unit
    if col % 2 != 0:
        row += 1

    return row, col

#converts a df of double height to an array
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

def assign_crimes(crime_df, hex_grid, date):
    
    #convert dataframe to GeoDataFrame
    crime_df['FIRST_OCCURRENCE_DATE'] = pd.to_datetime(crime_df['FIRST_OCCURRENCE_DATE'], errors='coerce')
    crime_df['date'] = crime_df['FIRST_OCCURRENCE_DATE'].dt.date
    daily_crime_df = crime_df[crime_df['date'] == date]

    daily_crime_df = daily_crime_df.copy()
    daily_crime_df.loc[:, 'geometry'] = daily_crime_df.apply(lambda row: Point(row['GEO_LON'], row['GEO_LAT']), axis=1)
    crime_gdf = gpd.GeoDataFrame(daily_crime_df, geometry='geometry', crs="EPSG:4326")

    #ensure both GeoDataFrames have the same CRS
    hex_grid = hex_grid.to_crs(epsg=4326)

    #Assign crimes to a hexagon
    crimes_with_hex = gpd.sjoin(crime_gdf, hex_grid, how="left", predicate="within")

    #crimes per hexagon count
    crime_counts = crimes_with_hex.groupby('cell_id').size().reset_index(name='crime_count')

    #merge crime counts back into the hex grid
    count_grid = hex_grid.merge(crime_counts, on='cell_id', how='left')

    count_grid['crime_count'] = count_grid['crime_count'].fillna(0)

    return count_grid

#assigns crimes to the grid and then converts it to hexagdly coordinates
def assign_crimes_to_grid(crime_data, hex_grid, date):
    # Assign crimes to hexagons
    crime_counts = assign_crimes(crime_data, hex_grid, date)

    # Extract x, y coordinates from cell_id
    crime_counts[['x', 'y']] = crime_counts['cell_id'].apply(lambda cid: pd.Series(parse_cell_id(cid)))

    # Convert to grid format
    grid = doubleheightdf_to_array(crime_counts, x_col='x', y_col='y', value_col='crime_count', default_value=0)
    
    return grid