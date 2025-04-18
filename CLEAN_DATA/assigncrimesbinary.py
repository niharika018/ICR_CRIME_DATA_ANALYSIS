import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from datetime import timedelta

def assign_crimes_no_sum(crime_df, hex_grid, date):
    #convert df to geodataframe
    daily_crime_df = crime_df[crime_df['date'] == date].copy()
    daily_crime_df.loc[:, 'geometry'] = daily_crime_df.apply(
        lambda row: Point(row['GEO_LON'], row['GEO_LAT']), axis=1)
    crime_gdf = gpd.GeoDataFrame(daily_crime_df, geometry='geometry', crs="EPSG:4326")

    #ensure both GeoDataFrames have the same CRS
    hex_grid = hex_grid.to_crs(epsg=4326)

    #assign crimes to hexagon
    return gpd.sjoin(crime_gdf, hex_grid, how="left", predicate="within")

#load data
crime_df = pd.read_csv("test_data.csv")
hex_grid = gpd.read_file("grid.geojson")

#filter for assault like crimes
offenses_to_keep = [
    'indecent-exposure',
    'explosive-incendiary-dev-use',
    'weapon-flourishing',
    'harassment',
    'harassment-dv',
    'public-fighting',
    'homicide-family',
    'homicide-other',
    'robbery-purse-snatch-w-force',
    'assault-simple',
    'assault-dv',
    'weapon-fire-into-occ-veh',
    'aggravated-assault',
    'aggravated-assault-dv',
    'menacing-felony-w-weap',
    'agg-aslt-shoot',
    'homicide-negligent',
    'threats-to-injure'
]
crime_df = crime_df[crime_df['OFFENSE_TYPE_ID'].isin(offenses_to_keep)]

#get unique dates
crime_df['FIRST_OCCURRENCE_DATE'] = pd.to_datetime(crime_df['FIRST_OCCURRENCE_DATE'], errors='coerce')
crime_df['date'] = crime_df['FIRST_OCCURRENCE_DATE'].dt.date
unique_dates = sorted(crime_df['date'].dropna().unique())

#get all possible cell ids
all_cell_ids = hex_grid['cell_id'].unique()

#create date range covering all dates in the dataset
date_range = pd.date_range(start=min(unique_dates), end=max(unique_dates), freq='D')

#initialize dataframe with all dates and cell ids for each date
full_grid = pd.MultiIndex.from_product(
    [all_cell_ids, date_range],
    names=['cell_id', 'date']
).to_frame(index=False)

#fill df
crime_grids = []
for date in unique_dates:
    grid = assign_crimes_no_sum(crime_df, hex_grid, date)
    if not grid.empty:
        #get unique cell_ids with crimes for this day
        crime_cells = grid['cell_id'].dropna().unique()
        #create a temporary DataFrame with 1s for these cells
        temp_df = pd.DataFrame({
            'cell_id': crime_cells,
            'date': pd.to_datetime(date),
            'crime': 1
        })
        #append to crime grids
        crime_grids.append(temp_df)

#combine all crime occurances
if crime_grids:
    crimes_occurred = pd.concat(crime_grids)
    #merge with full df to mark 1s and 0s
    result = full_grid.merge(
        crimes_occurred,
        on=['cell_id', 'date'],
        how='left'
    )
    result['crime'] = result['crime'].fillna(0).astype(int)
else:
    #if no crimes at all, just mark all as 0
    result = full_grid.copy()
    result['crime'] = 0


save
result.to_csv("testset.csv", index=False) 

#debug code
print("Processing complete:")
print(f"- Total cells: {len(all_cell_ids)}")
print(f"- Date range: {min(date_range)} to {max(date_range)}")
print(f"- Crimes recorded: {result['crime'].sum()} occurrences")