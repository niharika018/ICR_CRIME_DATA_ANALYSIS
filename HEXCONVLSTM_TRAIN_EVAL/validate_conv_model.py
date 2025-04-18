import torch
import numpy as np
import json
import pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, average_precision_score
)
import torch

from ConvLSTM import HexConvLSTM

def add_temporal_channels(sequences, start_date_str):
    # num windows, days in window, height, width
    N, T, H, W = sequences.shape
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    
    # Creating the temporal channels for sin and cos of the day of the year
    sin_channel = torch.zeros((N, T, H, W), dtype=torch.float32)
    cos_channel = torch.zeros((N, T, H, W), dtype=torch.float32)

    for i in range(N):
        for t in range(T):
            #calculate the current date based on the start date and the time step
            current_date = start_date + timedelta(days=(i * 30 + t))
            day_of_year = current_date.timetuple().tm_yday
            angle = 2 * np.pi * day_of_year / 365.0  # Normalize to a [0, 2π] range

            #compute sin and cos values for the current day
            sin_channel[i, t] = np.sin(angle)
            cos_channel[i, t] = np.cos(angle)

    # Add temporal channels (sin and cos of the day of the year) to the original sequence
    return torch.stack([sequences, sin_channel, cos_channel], dim=2)


def validate_model(model_path='hex_convlstm_final.pth',
                  data_path='test_tensor.pth',
                  target_path='test_tensor_target.pth',
                  cell_ids_path='ordered_cell_ids.json',
                  start_date="2024-03-05",  # 2023-08-08 for val, 2024-03-05 for test 2019-02-01 for train
                  forecast_days=7):       
    
    #use gpu is available 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=== Using device: {device} ===\n")

    #load in data
    print("1. Loading data...")
    x_val = torch.load(data_path)
    y_val = torch.load(target_path)

    #add channels to x tensor for day of year 
    if x_val.dim() == 4:
        x_val = add_temporal_channels(x_val, start_date_str=start_date)

    #initialize model and set to eval mode
    print("\n2. Loading model...")
    model = HexConvLSTM(
        input_channels=3,
        hidden_channels=8,
        num_hexes=820,
        height=36,
        width=70
    ).to(device)
    
    model.load_state_dict(torch.load(model_path))
    model.eval()

    #make predictions
    print("\n3. Making predictions...")
    with torch.no_grad():
        x_val = x_val.float().to(device)
        y_pred_probs = model(x_val)
        y_pred_probs = y_pred_probs.cpu().numpy()

    #calculating metrics
    print("\n4. Calculating metrics...")
    y_true = y_val.numpy().flatten()
    y_probs = y_pred_probs.flatten()
    y_pred = (y_probs > 0.3).astype(int)
    
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROC AUC": roc_auc_score(y_true, y_probs),
        "Avg Precision": average_precision_score(y_true, y_probs)
    }

    #print results
    print("\n5. Final results:")
    print("┌────────────────────┬──────────────┐")
    for name, value in metrics.items():
        print(f"│ {name:<18} │ {value:>12.4f} │")
    print("└────────────────────┴──────────────┘")

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(f"True Neg: {cm[0, 0]} | False Pos: {cm[0, 1]}")
    print(f"False Neg: {cm[1, 0]} | True Pos: {cm[1, 1]}")

    #building dataframe and saving
    print("\n6. Saving predictions...")
    with open(cell_ids_path) as f:
        cell_ids = json.load(f)
    
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    
    #create date ranges for each prediction period
    date_ranges = []
    for i in range(x_val.shape[0]):
        period_start = start_date + timedelta(days=i)
        period_end = period_start + timedelta(days=forecast_days-1)
        date_ranges.append((period_start, period_end))
    
    results = []
    for sample_idx, (period_start, period_end) in enumerate(date_ranges):
        for cell_idx, cell_id in enumerate(cell_ids):
            results.append({
                'period_start': period_start.strftime("%Y-%m-%d"),
                'period_end': period_end.strftime("%Y-%m-%d"),
                'cell_id': cell_id,
                'predicted_prob': y_pred_probs[sample_idx, cell_idx],
                'actual_outcome': y_val[sample_idx, cell_idx].item(),
                'predicted_class': int(y_pred_probs[sample_idx, cell_idx] > 0.3)
            })
    
    results_df = pd.DataFrame(results)
    
    #save predictions
    csv_path = "hexlstm_predictions_test.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nPredictions saved to {csv_path}")
    print(f"Date range: {date_ranges[0][0].date()} to {date_ranges[-1][1].date()}")
    print(f"Total predictions: {len(results_df)}")

if __name__ == '__main__':
    validate_model()