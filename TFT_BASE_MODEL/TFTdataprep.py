import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

#load raw data with assigned hex
df = pd.read_csv("trainassigned.csv")

#extract date
df['date'] = pd.to_datetime(df['FIRST_OCCURRENCE_DATE']).dt.date

#get crime counts per hex per day
daily_counts = df.groupby(['date', 'cell_id']).size().reset_index(name='crime_count')

#ensure that date is in correct format
daily_counts['date'] = pd.to_datetime(daily_counts['date'])

#create complete date-cell grid
all_dates = pd.date_range(
    start=daily_counts['date'].min(),
    end=daily_counts['date'].max(),
    freq='D'
)
all_cell_ids = daily_counts['cell_id'].unique()

full_grid = pd.DataFrame([
    (date.date(), cell_id)  #using .date() to match types
    for date in all_dates
    for cell_id in all_cell_ids
], columns=['date', 'cell_id'])

#convert full_grid date to datetime to match daily_counts
full_grid['date'] = pd.to_datetime(full_grid['date'])

#merge with crime counts
full_data = full_grid.merge(
    daily_counts,
    on=['date', 'cell_id'],
    how='left'
)

#fill missing values with 0 indicating no crime happened in that cell on that date
full_data['crime_count'] = full_data['crime_count'].fillna(0).astype(int)

#add temporal features (day of week, month)
full_data['day_of_week'] = full_data['date'].dt.dayofweek
full_data['month'] = full_data['date'].dt.month

#create time_idx (days since start) MUST STAY IN ORDER, DAY 0 IS FIRST DAY OF TRAIN SET
day_index_start = 0
full_data = full_data.sort_values('date')
full_data['time_idx'] = (full_data['date'] - full_data['date'].min()).dt.days + day_index_start

#encode cell_id
le = LabelEncoder()
full_data['cell_id_enc'] = le.fit_transform(full_data['cell_id'])

#sort by cell id and date
full_data = full_data.sort_values(["cell_id", "date"])

#save to csv
full_data.to_csv("TFTtrain_data.csv", index=False)