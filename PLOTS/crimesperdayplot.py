import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

prediction_file = "hextest_predictions.csv"
num_days = 365
startday = "2024-04-04" #2019-02-01 for test, 2023-09-07 for val, and 2024-04-04 for test

#load predictions
predictions_df = pd.read_csv(prediction_file)

#filter the amount of days to see
predictions_df['date'] = pd.to_datetime(predictions_df['period_start'])
start_date = pd.to_datetime(startday)
end_date = start_date + timedelta(days=num_days)
date_mask = (predictions_df['date'] >= start_date) & (predictions_df['date'] <= end_date)
daily_data = predictions_df[date_mask].copy()

#count hexes with any crime
daily_totals = daily_data.groupby('date').agg({
    'predicted_class': lambda x: (x > 0).sum(),  # Count hexes with predicted crime
    'actual_outcome': lambda x: (x > 0).sum()       # Count hexes with actual crime
}).reset_index()

#initialize plot
plt.figure(figsize=(14, 7))

#plot actual amount of hexes with crime
plt.plot(daily_totals['date'], daily_totals['actual_outcome'], 
         label='Hexes with Actual Crime', color='#2ca02c', linewidth=2, marker='o')

#plot predicted amount of hexes with crime
plt.plot(daily_totals['date'], daily_totals['predicted_class'], 
         label='Hexes with Predicted Crime', color='#1f77b4', linewidth=2, marker='s')

#other plot customizations
plt.title(f'Hexagons with Crime (Predicted vs Actual)\n{start_date.date()} to {end_date.date()}', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of Hexagons with Crime', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

#save
plt.tight_layout()
plt.savefig('hexes_with_crime_predictions.png', dpi=300, bbox_inches='tight')
plt.show()
