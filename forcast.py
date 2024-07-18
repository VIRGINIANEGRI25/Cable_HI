import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.optim import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def custom_collate(batch):
    graph_data_list = [item[0] for item in batch]
    outputs = torch.stack([item[1] for item in batch])
    return graph_data_list, outputs

class CustomTensorDataset(Dataset):
    def __init__(self, directory, train=True, validation_split=0.2, random_seed=42):
        self.file_paths = [os.path.join(directory, fname) for fname in os.listdir(directory) if fname.endswith('.pt')]
        
        # Split dataset into train and validation sets
        train_files, val_files = train_test_split(self.file_paths, test_size=validation_split, random_state=random_seed)
        
        if train:
            self.data = [torch.load(fp) for fp in train_files]
        else:
            self.data = [torch.load(fp) for fp in val_files]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tensor_dict = self.data[idx]
        x = tensor_dict['input']
        y = tensor_dict['output']
        
        # Define edges (specific connections: 1st-2nd, 1st-3rd, 2nd-3rd)
        edge_index = torch.tensor([[0, 0, 1],
                                   [1, 2, 2]], dtype=torch.long)
        
        graph_data = Data(x=x, edge_index=edge_index)
        
        return graph_data, y

# Directory containing the tensor files
data_directory = 'data_processed'
train_dataset = CustomTensorDataset(data_directory, train=True)
val_dataset = CustomTensorDataset(data_directory, train=False)

batch_size = 512 
shuffle = True
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc_gnn = nn.Linear(hidden_dim, hidden_dim)
        self.fc_guidance = nn.Linear(1, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim - 1)  # Output dimension is one less

    def forward(self, data, guidance):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = torch.mean(x, dim=0)  # Aggregate node features

        x_gnn = self.fc_gnn(x)
        
        guidance = guidance.view(-1, 1)  # Ensure guidance is the correct shape
        x_guidance = F.relu(self.fc_guidance(guidance))

        x_combined = x_gnn + x_guidance
        x_out = self.fc_out(x_combined)
        
        return x_out

class GNNLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(GNNLSTMModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc_gnn = nn.Linear(hidden_dim, hidden_dim)
        self.fc_guidance = nn.Linear(1, hidden_dim)
        self.bidirectional_lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(2 * hidden_dim, output_dim - 1)  # Output dimension is one less

    def forward(self, data, guidance):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Aggregate node features
        x = torch.mean(x, dim=0).unsqueeze(0)  # Add batch dimension (1, num_nodes, hidden_dim)
        
        # Bidirectional LSTM on node features
        lstm_out, _ = self.bidirectional_lstm(x)
        lstm_out = torch.mean(lstm_out, dim=1)  # Average across nodes, keep bidirectional output
        
        x_gnn = self.fc_gnn(lstm_out)
        x_gnn = self.dropout(x_gnn)
        
        guidance = guidance.view(-1, 1)  # Ensure guidance is the correct shape
        x_guidance = F.relu(self.fc_guidance(guidance))
        x_guidance = self.dropout(x_guidance)

        x_combined = torch.cat((x_gnn, x_guidance), dim=1)
        x_out = self.fc_out(x_combined)
        
        return x_out
    
# Check if CUDA is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_dim = train_dataset[0][0].x.size(1)  # Assuming all inputs have the same dimension
hidden_dim = 16  # Number of hidden units
output_dim = train_dataset[0][1].size(0)  # Assuming all outputs have the same dimension

model = GNNModel(input_dim, hidden_dim, output_dim).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 200
train_losses = []
val_losses = []
best_val_loss = float('inf')
best_model_path = 'best_model.pth'

for epoch in range(num_epochs):
    # Training
    model.train()
    for batch in train_loader:
        graph_data_list, outputs = batch
        
        for i, graph_data in enumerate(graph_data_list):
            inputs = graph_data.to(device)
            guidance_vector = outputs[i][0].view(1).to(device)
            targets = outputs[i][1:].to(device)
            
            # Forward pass
            outputs_pred = model(inputs, guidance_vector)
            
            # Ensure the dimensions match for MSE loss calculation
            outputs_pred = outputs_pred.view(-1)
            targets = targets.view(-1)
            
            loss = criterion(outputs_pred, targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    train_losses.append(loss.item())
    print(f'Epoch {epoch+1}, Train Loss: {loss.item()}')
    
    # Validation
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            graph_data_list, outputs = batch
            for i, graph_data in enumerate(graph_data_list):
                inputs = graph_data.to(device)
                guidance_vector = outputs[i][0].view(1).to(device)
                targets = outputs[i][1:].to(device)
                
                # Forward pass
                outputs_pred = model(inputs, guidance_vector)
                
                # Ensure the dimensions match for MSE loss calculation
                outputs_pred = outputs_pred.view(-1)
                targets = targets.view(-1)
                
                loss = criterion(outputs_pred, targets)
                val_losses.append(loss.item())
    
    avg_val_loss = torch.tensor(val_losses).mean().item()
    print(f'Epoch {epoch+1}, Validation Loss: {avg_val_loss}')
    
    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f'Saving best model with validation loss: {best_val_loss}')

# Load best model for evaluation on the test set
best_model = GNNModel(input_dim, hidden_dim, output_dim).to(device)
best_model.load_state_dict(torch.load(best_model_path))

# Evaluation on test set
test_losses = []
predictions_test = []
gt_values_test = []

best_model.eval()
with torch.no_grad():
    for batch in val_loader:  # Using val_loader for test evaluation here, adjust as needed
        graph_data_list, outputs = batch
        for i, graph_data in enumerate(graph_data_list):
            inputs = graph_data.to(device)
            guidance_vector = outputs[i][0].view(1).to(device)
            targets = outputs[i][1:].to(device)
            
            # Forward pass
            outputs_pred = best_model(inputs, guidance_vector)
            
            # Collect predictions and GT values
            predictions_test.append(outputs_pred.cpu().numpy())
            gt_values_test.append(targets.cpu().numpy())
            
            # Compute loss
            loss = criterion(outputs_pred.view(-1), targets.view(-1))
            test_losses.append(loss.item())

# Compute average test loss
avg_test_loss = sum(test_losses) / len(test_losses)
print(f'Avg Test Loss: {avg_test_loss}')

# Plot GT vs Predictions for test set
# Flatten lists
predictions_test = [item for sublist in predictions_test for item in sublist]
gt_values_test = [item for sublist in gt_values_test for item in sublist]

# Plot GT vs Predictions
plt.figure(figsize=(8, 8))
plt.scatter(gt_values_test, predictions_test, alpha=0.5)
plt.xlabel('Ground Truth')
plt.ylabel('Predictions')
plt.title('Ground Truth vs Predictions (Test Set)')
plt.grid(True)
plt.show()
