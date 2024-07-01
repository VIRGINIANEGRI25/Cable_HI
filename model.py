import torch 
from torch import nn, relu 
import numpy as np
import torch.nn.functional as F

class Cnn(torch.nn.Module):
    def __init__(self, inputs, outputs, batch_size):
        super(Cnn, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.batch_size = batch_size
        self.multiplier = 12
        self.conv = nn.Conv1d(1, 256, kernel_size=1)
        
        self.output_layer = nn.Linear(outputs * inputs * 256, 1)

    def forward(self, x):
        x = x.reshape((self.batch_size, 1, self.inputs))  # Reshape input for Conv1d
        x = F.relu(self.conv(x))
        x = x.view(self.batch_size, -1)  # Flatten for the fully connected layer
        x = self.output_layer(x)
        x = x.view(self.batch_size)  # Flatten to shape (batch_size,)
        return x
    
class Convlstm(torch.nn.Module):
    def __init__(self, inputs, outputs, batch_size, lstm_hidden_size= 32, num_lstm_layers=1, dropout_prob=0.3):
        super(Convlstm, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.batch_size = batch_size
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers
        
        self.conv = nn.Conv1d(1, outputs, kernel_size=1)
        
        self.batch_norm = nn.BatchNorm1d(outputs)
        
        self.dropout = nn.Dropout(dropout_prob)
        
        self.fc1 = nn.Linear(outputs, outputs)
        
        self.bilstm = nn.LSTM(input_size=outputs, hidden_size=lstm_hidden_size, 
                              num_layers=num_lstm_layers, batch_first=True, bidirectional=True)
        
        self.fc2 = nn.Linear(lstm_hidden_size * 2, batch_size)

    def forward(self, x):
        x = x.reshape((self.batch_size, 1, self.inputs))
        x = self.conv(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x.view(self.batch_size, -1)
        x = self.fc1(x)
        x = x.reshape((self.batch_size, 1, -1))
        lstm_out, _ = self.bilstm(x)
        lstm_out = lstm_out[:, -1, :]
        x = F.relu(x)
        x = self.fc2(lstm_out)
        
        return x
    
def train(dataloader, model, loss_fn, optimizer, verbose=True):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
          X, y = X, y

          # Compute prediction error
          pred = model(X)
          loss = loss_fn(pred, y)

          # Backpropagation
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()

          if verbose == True:
            if batch % 500 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return loss

def test(dataloader, model, loss_fn, verbose=True):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X, y
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    if verbose == True:
        print(f"Test Avg loss: {test_loss:>8f} \n")


def evaluate_model(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    return all_predictions, all_targets
