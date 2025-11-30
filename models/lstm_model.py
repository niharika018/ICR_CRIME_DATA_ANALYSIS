import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    A simple LSTM regression model for spatial-temporal crime forecasting.

    Args:
        input_size (int): Number of input features per timestep.
        hidden_size (int): Dimensionality of the LSTM hidden state.

    The model expects input of shape (batch_size, input_size)
    and internally adds a sequence dimension for a 1-step sequence.
    """

    def __init__(self, input_size: int, hidden_size: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Reshape (batch_size, features) → (batch_size, seq_len=1, features)
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
