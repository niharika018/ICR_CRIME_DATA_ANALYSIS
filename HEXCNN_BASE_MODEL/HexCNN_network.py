import torch
import torch.nn as nn
import torch.optim as optim
import hexagdly
from torch.utils.data import DataLoader, TensorDataset
from torchvision.ops import SqueezeExcitation  
import time

#params
num_epochs = 50
window_size = 30  #Use the previous window_size days for forecasting
forecast_horizon = 7 #Days to forcast
learn = 0.01 #learning rate
batch = 128 #batchsize

#function creates sequences and their corresponding targets for given window size and forecast horizon.
def create_sequences(data, window_size, forecast_horizon):
    sequences = []
    targets = []
    for i in range(len(data) - window_size - forecast_horizon):
        seq = data[i:i + window_size]  # Create a sequence of 'window_size' time steps
        sequences.append(seq)
        target = data[i + window_size:i + window_size + forecast_horizon] #create a target of 'forecast_horizon' time steps
        targets.append(target)
    return torch.stack(sequences), torch.stack(targets)

class TemporalHexCNN(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, grid_height, grid_width, seq_len, forecast_horizon):
        super(TemporalHexCNN, self).__init__()      

        #hex convolution
        self.spatial_cnn = hexagdly.Conv2d(in_channels=input_channels, 
                                   out_channels=output_channels, 
                                   kernel_size=kernel_size, 
                                   stride=stride)
        
        #SE block
        self.se = SqueezeExcitation(output_channels, squeeze_channels=8)

        #hex pooling
        self.pool = hexagdly.MaxPool2d(kernel_size = 1, stride = 2)

        #RELU activation
        self.relu = nn.ReLU()

        #fully connected layers 1 and 2
        self.fc1 = nn.Linear(self.flattened_size(), 128)
        self.fc2 = nn.Linear(128, grid_height * grid_width * forecast_horizon)

    def flattened_size(self):
        # This function calculates the flattened size after passing through the spatial CNN and pooling layers
        #adjust this based on the output size of the CNN layer
        return 38880
        
    def forward(self, x):

        batch_size, seq_len, channels, height, width = x.size()

        #apply convolutions for each time step.
        spatial_features = []
        for t in range(seq_len):
            cnn_output = self.spatial_cnn(x[:, t, :, :, :])  #apply to each time step
            se_output = self.se(cnn_output)
            pooled_output = self.pool(se_output)  #apply pooling after CNN
            activated_output = self.relu(pooled_output)  #apply sigmoid after pooling
            spatial_features.append(activated_output)
        
        #outputs from the convolutions, pooling.
        spatial_features = torch.stack(spatial_features, dim=1)

        #flatten the output for fully connected lauyers
        flattened = spatial_features.view(batch_size, -1)

        #goes into first fully connect layer
        fc1_out = self.fc1(flattened)

        #apply RELU after first fully connected layer
        fc1_out = self.relu(fc1_out)

        #output layer
        out = self.fc2(fc1_out)

        return out

#load training tensor   
crime_tensor = torch.load('train_tensor.pth')
crime_tensor = crime_tensor.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
crime_tensor = crime_tensor.unsqueeze(1)

#create the sequences and the targets
sequences, targets = create_sequences(crime_tensor, window_size, forecast_horizon)

#create the DataLoader with sequences and targets
dataset = TensorDataset(sequences, targets)
dataloader = DataLoader(dataset, batch_size=batch, shuffle=False)

def train_model():
    #initialize model, loss, and optimizer
    model = TemporalHexCNN(input_channels=1, output_channels=8, kernel_size=1, stride=2,
                            grid_height=36, grid_width=70, seq_len=window_size, forecast_horizon=forecast_horizon)
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learn)

    #cuDNN autotuner
    torch.backends.cudnn.benchmark = True

    #train for the amount of epochs
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()  # Start timing the epoch

        #run through each set of inputs and targets
        for i, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.float().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))  
            targets = targets.float().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

            #zero the gradients
            optimizer.zero_grad()

            #forward pass
            outputs = model(inputs)

            #calculate loss
            loss = criterion(outputs.view(targets.size()), targets)

            #backpropagation
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}, Time: {epoch_time:.2f} seconds")

    #save the trained model
    torch.save(model.state_dict(), 'temporal_hexcnn_model.pth')

if __name__ == "__main__":
    # Only run training if this script is executed directly
    train_model()