import torch 
from torch import nn, relu 
import numpy as np

class Cnn(torch.nn.Module):
  # defined the initialization method
  def __init__(self, inputs, outputs, batch_size):
    # initialization of the superclass
    super(Cnn, self).__init__()
    # store the parameters
    self.inputs = inputs
    self.outputs = outputs
    self.batch_size = batch_size
    # define the convolutional layer
    self.conv = nn.Conv1d(inputs, outputs, 1)
   
    # define the output layer
    self.output_layer = nn.Linear(1, outputs)

  def forward(self, x):
    x = x.reshape((self.batch_size, self.inputs, 1))
    x = relu(self.conv(x))
    x = self.output_layer(x)
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