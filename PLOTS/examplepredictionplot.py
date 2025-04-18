import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Normalize

#initialize figure
fig = plt.figure(figsize=(10, 12))
gs = fig.add_gridspec(nrows=4, ncols=2, 
                     height_ratios=[1,1,1,0.08],
                     hspace=0.3, wspace=0.1)

#create subplots
axes = [[fig.add_subplot(gs[i,0]) for i in range(3)],
        [fig.add_subplot(gs[i,1]) for i in range(3)]]
cax = fig.add_subplot(gs[3,:])  # Colorbar axis

#load data
grid = gpd.read_file('grid.geojson')
predictions = pd.read_csv('hexlstm_predictions_test.csv')
predictions['period_start'] = pd.to_datetime(predictions['period_start'])

#get three chosen days
valid_dates = predictions[predictions['actual_outcome'].notna()]['period_start'].unique()[[27,75,143]]
print(valid_dates)
plot_data = grid.merge(predictions[predictions['period_start'].isin(valid_dates)], on='cell_id')

#for color scaling
max_val = max(plot_data['predicted_prob'].quantile(0.95), 
             plot_data['actual_outcome'].quantile(0.95))
norm = Normalize(vmin=0, vmax=max_val)

#plot each of the days that were chosen
for i, date in enumerate(sorted(valid_dates)):
    day_data = plot_data[plot_data['period_start'] == date]
    year = date.year
    
    #left column, predicted
    ax_pred = axes[0][i]
    day_data.plot(column='predicted_prob', cmap='YlOrRd', norm=norm, ax=ax_pred, edgecolor='black', linewidth=0.2)
    ax_pred.set_title(f"Forecast Week Starting - {date.strftime('%b %d')}, {year}", pad=5, fontsize=10)
    ax_pred.axis('off')
    
    #right column, actual
    ax_actual = axes[1][i]
    day_data.plot(column='actual_outcome', cmap='YlOrRd', norm=norm, ax=ax_actual, edgecolor='black', linewidth=0.2)
    ax_actual.set_title(f"Actual Week Starting - {date.strftime('%b %d')}, {year}", pad=5, fontsize=10)
    ax_actual.axis('off')

#other customizations
sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=norm)
cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
cbar.set_label('Crime Probability', labelpad=5, fontsize=10)
cbar.ax.tick_params(labelsize=8)
plt.suptitle('Denver Crime Forecast vs Actual', y=0.98, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.subplots_adjust(top=0.93, bottom=0.08)

#save
plt.savefig('forecast_vs_actual.png', dpi=300, bbox_inches='tight')
plt.show()