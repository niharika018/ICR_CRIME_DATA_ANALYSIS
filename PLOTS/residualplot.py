import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


predictions_file = "hex_predictions.csv" #csv

#load predictions
predictions_df = pd.read_csv(predictions_file)

#calculate residuals
predictions_df['residual'] = predictions_df['predicted_crimes'] - predictions_df['actual_crimes']

#plot window limits
x_min, x_max = -3, 3

#initialize plot
plt.figure(figsize=(12, 7))

#histogram with density curve
sns.histplot(data=predictions_df, x='residual', 
             bins=300, kde=True, 
             color='#4e79a7',
             edgecolor='white',
             linewidth=0.5)

#set axis limits
plt.xlim(x_min, x_max)

#add line for mean bias
mean_residual = predictions_df['residual'].mean()
plt.axvline(x=mean_residual, color='#f28e2b', linestyle='-', linewidth=2, label=f'Mean Bias: {mean_residual:.2f}')

#add standard deviation and skew.
std_dev = predictions_df['residual'].std()
skewness = predictions_df['residual'].skew()

stats_text = (f"Std Dev: {std_dev:.2f}\n"
              f"Skewness: {skewness:.2f}"
)

plt.annotate(stats_text, xy=(1, 0.95), xycoords='axes fraction',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

#other plot customizations
plt.title('Distribution of Prediction Errors', fontsize=16, pad=20)
plt.xlabel('Prediction Error (Predicted - Actual Crimes)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(loc='upper left', fontsize=10)
sns.despine()
plt.grid(axis='y', alpha=0.3)

#save and show
plt.tight_layout()
plt.savefig('prediction_residuals_histogram.png', dpi=300, bbox_inches='tight')
plt.show()
