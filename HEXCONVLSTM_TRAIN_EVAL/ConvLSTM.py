import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from hexagdly import Conv2d
from datetime import datetime, timedelta
import numpy as np

#custom SE block
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_pool(x).view(b, c)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y.expand_as(x)

#hexagonal conv LSTM
class HexConvLSTM(nn.Module):
    def __init__(self, input_channels=3, hidden_channels=8, num_hexes=820, height=36, width=70, dropout_prob = 0.75):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.height = height
        self.width = width
        
        #initial convolutions
        self.conv_x = Conv2d(input_channels, hidden_channels, kernel_size=1)
        self.conv_h = Conv2d(hidden_channels, hidden_channels, kernel_size=1)
        
        #SE blocks
        self.se_x = SEBlock(hidden_channels)
        self.se_h = SEBlock(hidden_channels)

        #gates with convolution
        self.conv_f = Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size=1)
        self.conv_i = Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size=1)
        self.conv_o = Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size=1)
        self.conv_c = Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size=1)

        #dropout to avoid overfitting
        self.dropout = nn.Dropout(p=dropout_prob)
        
        #predictions
        self.fc = nn.Linear(hidden_channels * height * width, num_hexes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [batch, seq_len=30, 1, height, width]
        batch_size = x.size(0)
        h = torch.zeros(batch_size, self.hidden_channels, self.height, self.width, device=x.device)
        c = torch.zeros(batch_size, self.hidden_channels, self.height, self.width, device=x.device)
        
        for t in range(x.size(1)):

            x_t = x[:, t, :, :, :]  # [batch, 1, height, width]

            #LSTM operations
            combined = torch.cat([x_t, self.se_h(h)], dim=1)
            forget = torch.sigmoid(self.conv_f(combined))
            input_gate = torch.sigmoid(self.conv_i(combined))
            output_gate = torch.sigmoid(self.conv_o(combined))
            cell_candidate = torch.tanh(self.conv_c(combined))
            
            #cell
            c = forget * c + input_gate * cell_candidate

            #hidden state which is sent to fully connected and then sigmoid for predictions
            h = output_gate * torch.tanh(c)

            h = self.dropout(h)
        
        return self.sigmoid(self.fc(h.view(batch_size, -1)))

#add channels for day of year. 
def add_temporal_channels(sequences, start_date_str):
    # num windows, days in window, height, width
    N, T, H, W = sequences.shape
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    
    #temporal channels for sin and cos of the day of the year
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

    #add temporal channels (sin and cos of the day of the year) to the original sequence
    return torch.stack([sequences, sin_channel, cos_channel], dim=2)

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #load data
    sequences = torch.load('train_tensor.pth')  # [1643, 30, 36, 70]
    targets = torch.load('train_tensor_target.pth')  # [1643, 820]

    #add temporal channels
    sequences = add_temporal_channels(sequences, start_date_str="2019-01-02").float()  # [batch, 30, 3, height, width]
    
    #dataloaders
    dataset = TensorDataset(sequences, targets)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    #load dataset for validation
    val_sequences = torch.load('val_tensor.pth')  # [n_val, 30, 36, 70]
    val_targets = torch.load('val_tensor_target.pth')  # [n_val, 820]
    val_sequences = add_temporal_channels(val_sequences, start_date_str="2023-08-08").float()
    val_dataset = TensorDataset(val_sequences, val_targets)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    #initialize model and loss
    model = HexConvLSTM().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    #early stopping params to avoid overfitting
    best_val_loss = float('inf')
    patience = 3  #how many epochs to wait for improvement
    epochs_no_improve = 0
    
    #training loop
    for epoch in range(25):
        model.train()
        total_loss = 0
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        #validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        #checkl for early stopping
        print(f'Epoch {epoch+1}, Training Loss: {total_loss:.4f}, Validation Loss: {val_loss:.4f}')
        
        #if val loss improved save model.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'hex_convlstm_final.pth')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        #early training stop
        if epochs_no_improve >= patience:
            print("Early stopping triggered. Training stopped.")
            break

    #save the final model at the end of training
    torch.save(model.state_dict(), 'hex_convlstm_final.pth')
    print("Model saved at the end of training.")

if __name__ == '__main__':
    train_model()
