import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.metrics import RMSE, MAE
from torch.utils.data import DataLoader
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
import numpy as np

# load test data
test_df = pd.read_csv("TFTtest_data.csv")  # Ensure same columns/structure as training
test_df['crime_count'] = test_df['crime_count'].astype(float)  # Ensure same dtype

#load dataset paramaters
training = TimeSeriesDataSet.load("dataset_params_final.json")

#prepare test dataset using same parameters
test_dataset = TimeSeriesDataSet.from_dataset(
    training, 
    test_df,
    predict=True,
    stop_randomization=True
)

#create test dataloader
test_dataloader = test_dataset.to_dataloader(
    train=False,
    batch_size=128,  # Match training batch size
    num_workers=0,   # Keep 0 for Windows
    shuffle=False    # Important for consistent results
)

#load trained model
model = TemporalFusionTransformer.load_from_checkpoint("tft_final_model.ckpt")
model.eval()  # Set to evaluation mode

#make predictions and store actual values
predictions = []
actuals = []

with torch.no_grad():
    for batch in test_dataloader:
        x, y = batch
        out = model(x)["prediction"]
        predictions.append(out.cpu())
        actuals.append(y[0].cpu())  #actual values are first element

#convert to numpy array
predictions = torch.cat(predictions).numpy()
actuals = torch.cat(actuals).numpy()

#calculate RMSE and MAE
def calculate_metrics(preds, targets):
    rmse = np.sqrt(np.mean((preds - targets)**2))
    mae = np.mean(np.abs(preds - targets))
    return rmse, mae

rmse, mae = calculate_metrics(predictions, actuals)

print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")

#save predictions with original data
test_df["predicted_crime_count"] = predictions
test_df.to_csv("TFT_predictions.csv", index=False)