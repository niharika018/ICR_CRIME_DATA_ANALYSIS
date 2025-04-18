import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import matplotlib.dates as mdates

#load and label date split.
def load_predictions(file, split):
    df = pd.read_csv(file)
    df['date'] = pd.to_datetime(df['period_start'])
    df['split'] = split
    return df

train_df = load_predictions("hexlstm_predictions_train.csv", 'Train')
test_df = load_predictions("hexlstm_predictions_test.csv", 'Test')
val_df = load_predictions("hexlstm_predictions_val.csv", 'Val')


all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)

#get predictions by period start date
daily_totals = (
    all_data
    .groupby(['split', 'date'])
    .agg({
        'predicted_class': lambda x: (x > 0).sum(),
        'actual_outcome': lambda x: (x > 0).sum()
    })
    .reset_index()
)

#sort by date
daily_totals = daily_totals.sort_values("date")

#split boundaries
split_boundaries = (
    daily_totals
    .groupby('split')['date']
    .agg(['min', 'max'])
    .sort_values('min')
).reset_index()

#initialize plot
plt.figure(figsize=(15, 7))

#plot actual and predicted outcomes for each split
plt.plot(daily_totals['date'], daily_totals['actual_outcome'], 
         label='Actual Crime Hexes', color='#2ca02c', marker='o', linewidth=1, alpha = 0.5,markersize = 0.5)

plt.plot(daily_totals['date'], daily_totals['predicted_class'], 
         label='Predicted Crime Hexes', color='#1f77b4', marker='s', linewidth=1, alpha= 0.5, markersize=0.5)

#add vertical lines for splits
for i in range(1, len(split_boundaries)):
    boundary_date = split_boundaries['min'].iloc[i]
    split_label = split_boundaries['split'].iloc[i]
    plt.axvline(boundary_date, color='gray', linestyle='--', linewidth=1.5)
    plt.text(boundary_date + timedelta(days=1), plt.ylim()[1]*0.95, 
             f'{split_label} Start', color='gray', fontsize=10)

#other customizations
plt.title('Hexagons with Crime: Predicted vs Actual Across Splits', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of Hexagons with Crime', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig("hexes_with_crime_predictions_unified.png", dpi=300, bbox_inches='tight')
plt.show()