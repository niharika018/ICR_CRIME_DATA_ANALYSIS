import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta


prediction_file = "hex_predictions.csv"
num_days = 28
startday = "2024-03-05"

#load predictions
predictions_df = pd.read_csv(prediction_file )

#filter for period to splot
start_date = pd.to_datetime(startday) 
end_date = start_date + timedelta(days=num_days)
date_mask = (pd.to_datetime(predictions_df['date']) >= start_date) & (pd.to_datetime(predictions_df['date']) <= end_date)
daily_data = predictions_df[date_mask].copy()

#aggregate to get totals for Denver
daily_totals = daily_data.groupby('date').agg({
    'predicted_crimes': 'sum',
    'actual_crimes': 'sum'
}).reset_index()

#initialize plot
plt.figure(figsize=(14, 7))

#plot actual values
plt.plot(daily_totals['date'], daily_totals['actual_crimes'], 
         label='Actual Crimes', color='#2ca02c', linewidth=2, marker='o')

#plot predicted values
plt.plot(daily_totals['date'], daily_totals['predicted_crimes'], 
         label='Predicted Crimes', color='#1f77b4', linewidth=2, linestyle='--', marker='s')

#other plot customizations
plt.title('Daily Crime Predictions vs. Actuals (28-Day Period)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Total Crimes', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

#save and show
plt.tight_layout()
plt.savefig('daily_crime_predictions_28days.png', dpi=300, bbox_inches='tight')
plt.show()