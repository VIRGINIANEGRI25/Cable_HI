from preprocessing import load_data, scale_data, HI_computation, one_hot_encoding
from model import Cnn, train, test
from dataloader import train_test_datasets, train_test_dataloaders
import torch
from torch import nn

# load data
data = load_data('C:/Users/utente/Desktop/Cable_HI/XLPE_data.xlsx')
print(data)
scaled_data = scale_data(data)
data = HI_computation(data, scaled_data)
# data = one_hot_encoding(data)
print(data)

batch_size = 1
train_dataset, test_dataset = train_test_datasets(data)
train_dataloader, test_dataloader = train_test_dataloaders(train_dataset, test_dataset, batch_size)

for X, y in train_dataloader:
  print(f"Shape of X: {X.shape}")
  print(f"Shape of y: {y.shape} {y.dtype}")
  break
for X, y in test_dataloader:
  print(f"Shape of X: {X.shape}")
  print(f"Shape of y: {y.shape} {y.dtype}")
  break

model = Cnn(X.shape[1], y.shape[0], batch_size)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 1

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
                            