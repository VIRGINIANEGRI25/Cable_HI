from preprocessing import load_data, scale_data, HI_computation, one_hot_encoding
from model import Cnn, train, test
from dataloader import train_test_datasets, train_test_dataloaders
import torch
from torch import nn

# load data
data = load_data('XLPE_data.xlsx')
print(data)
scaled_data = scale_data(data)
data = HI_computation(data, scaled_data)
# data = one_hot_encoding(data)
# print(data)

batch_size = 2
train_dataset, test_dataset = train_test_datasets(data)
train_dataloader, test_dataloader = train_test_dataloaders(train_dataset, test_dataset, batch_size)

for X, y in train_dataloader:
  print(f"Shape of X: {X.shape}")
  print(f"Shape of y: {y.shape} {y.dtype}")
  break

model = Cnn(X.shape[1], y.shape[0], batch_size)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 100
patience = 10
best_loss = float('inf')
patience_counter = 0

for epoch in range(epochs):
    # Training phase
    model.train()
    for batch in train_dataloader:
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

    # Validation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in test_dataloader:
            inputs, targets = batch
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            val_loss += loss.item()
    val_loss /= len(test_dataloader)

    print(f'Epoch {epoch+1}, Validation Loss: {val_loss}')

    # Early stopping logic
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping triggered")
        break