import pandas as pd

df = pd.read_csv("trainset.csv", parse_dates=["date"])
df.sort_values(by=["cell_id", "date"], inplace=True)

df_full = df.groupby('cell_id').apply(
    lambda g: g.set_index('date').asfreq('D').fillna(0)
).reset_index(drop=True)

print(df_full['crime'].sum())
#aggregate over 7-day rolling window (past and future)
df_full['past_7day_sum'] = df_full.groupby('cell_id')['crime'].transform(
    lambda x: x.rolling(window=7, min_periods=7).sum().shift(1)
)

df_full['next_7day_sum'] = df_full.groupby('cell_id')['crime'].transform(
    lambda x: x.rolling(window=7, min_periods=7).sum().shift(-7)
)

#create binary target
df_full['future_crime'] = (df_full['next_7day_sum'] > 0).astype(int)

df_valid = df_full.dropna(subset=['past_7day_sum', 'future_crime'])

df_valid['prediction'] = (df_valid['past_7day_sum'] > 0).astype(int)

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

y_true = df_valid['future_crime']
y_pred = df_valid['prediction']
y_scores = df_valid['past_7day_sum'] / 7

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_scores)

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"AUC: {auc:.3f}")