import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datetime import timedelta

# parameters
LOOKBACK_DAYS = 30
FORECAST_DAYS = 7
BATCH_SIZE = 64
HIDDEN_SIZE = 32
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load data
train_df = pd.read_csv("trainset.csv", parse_dates=["date"])
test_df = pd.read_csv("testset.csv", parse_dates=["date"])

#create sequences
def create_sequences(df, lookback=30, forecast=7):
    sequences = []
    grouped = df.sort_values("date").groupby("cell_id")

    for cell_id, group in grouped:
        group = group.set_index("date").asfreq("D", fill_value=0).reset_index()
        for i in range(len(group) - lookback - forecast + 1):
            past_seq = group["crime"].iloc[i : i + lookback].values
            future_window = group["crime"].iloc[i + lookback : i + lookback + forecast].values
            label = int(future_window.sum() > 0)  # any crime in next 7 days
            sequences.append((cell_id, group["date"].iloc[i + lookback - 1], past_seq, label))
    
    return sequences

train_sequences = create_sequences(train_df, LOOKBACK_DAYS, FORECAST_DAYS)
test_sequences = create_sequences(test_df, LOOKBACK_DAYS, FORECAST_DAYS)

#create dataset and dataloader
class CrimeSequenceDataset(Dataset):
    def __init__(self, sequences):
        self.X = torch.tensor([s[2] for s in sequences], dtype=torch.float32)
        self.y = torch.tensor([s[3] for s in sequences], dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(-1), self.y[idx]  # shape: (seq_len, 1)

train_dataset = CrimeSequenceDataset(train_sequences)
test_dataset = CrimeSequenceDataset(test_sequences)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# LSTM model
class CrimeLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return torch.sigmoid(out).squeeze()

model = CrimeLSTM(hidden_size=HIDDEN_SIZE).to(DEVICE)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

#evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(DEVICE)
        preds = model(X_batch).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

predicted_labels = [1 if p > 0.5 else 0 for p in all_preds]

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

accuracy = accuracy_score(all_labels, predicted_labels)
precision = precision_score(all_labels, predicted_labels)
recall = recall_score(all_labels, predicted_labels)
auc = roc_auc_score(all_labels, all_preds)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"AUC: {auc:.4f}")