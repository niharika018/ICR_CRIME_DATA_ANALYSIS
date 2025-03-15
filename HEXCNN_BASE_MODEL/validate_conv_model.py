import torch
import torch.nn as nn
from HexCNN_network import TemporalHexCNN, create_sequences

#these need to match what model used during training
window_size = 30  
forecast_horizon = 7 

#specify device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#read in validation data
val_tensor = torch.load('val_tensor.pth')

#create input data and output data for validation set
x_val, y_val = create_sequences(val_tensor, window_size, forecast_horizon)
x_val = x_val.unsqueeze(2)
x_val = x_val.to(device)
y_val = y_val.to(device)

#initialize the model with the same architecture as training
model = TemporalHexCNN(input_channels=1, output_channels=8, kernel_size=1, stride=2,
                        grid_height=36, grid_width=70, seq_len=window_size, forecast_horizon=forecast_horizon)

# Load the trained weights
model.load_state_dict(torch.load('temporal_hexcnn_model.pth', map_location=device))
model.to(device)
model.eval() 

mse_loss = nn.MSELoss()

#perform predictions
with torch.no_grad():  
    y_pred = model(x_val)

#compute MSE
mse = mse_loss(y_pred.view(y_val.size()), y_val)
print(f"Validation MSE: {mse.item()}")


######### Convert back so we can plot

# Assuming you have the predicted tensor y_pred with shape [batch_size, forecast_horizon, height, width]
y_pred_reshaped = y_pred.view(forecast_horizon, -1)  # Reshape it to 2D for easy handling

def predictions_to_grid(predictions, x_col='x', y_col='y', forecast_horizon=7):
    grids = []
    for t in range(forecast_horizon):
        # Reshape predictions for each time step (predictions for each time step will be a 2D array)
        pred_grid = predictions[t].reshape(-1, 1)  # Assuming predictions are flattened for each grid cell
        # Convert the prediction grid back to the hexagonal grid (like the crime grid was before)
        pred_grid_hex = doubleheightdf_to_array(pred_grid, x_col=x_col, y_col=y_col, value_col='prediction', default_value=0)
        grids.append(pred_grid_hex)
    
    return grids

import matplotlib.pyplot as plt

def plot_predictions(prediction_grids, forecast_horizon):
    for t in range(forecast_horizon):
        plt.imshow(prediction_grids[t], cmap='hot', interpolation='nearest')  # or any other colormap
        plt.title(f'Predictions for Day {t+1}')
        plt.colorbar()
        plt.show()

# Assuming y_pred_reshaped is the predictions tensor you obtained
prediction_grids = predictions_to_grid(y_pred_reshaped, forecast_horizon=forecast_horizon)
plot_predictions(prediction_grids, forecast_horizon=forecast_horizon)