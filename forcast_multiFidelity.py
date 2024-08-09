import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset
from torch_geometric.nn import GATConv, global_mean_pool

# Function to compute global minima and maxima
def compute_global_min_max(directory):
    input_min, input_max = float('inf'), float('-inf')
    output_min, output_max = float('inf'), float('-inf')
    
    for fname in os.listdir(directory):
        if fname.endswith('.pt'):
            tensor_dict = torch.load(os.path.join(directory, fname))
            input_min = min(input_min, tensor_dict['input'].min().item())
            input_max = max(input_max, tensor_dict['input'].max().item())
            output_min = min(output_min, tensor_dict['output'].min().item())
            output_max = max(output_max, tensor_dict['output'].max().item())
    
    return input_min, input_max, output_min, output_max

# Normalize function
def normalize(tensor, min_val, max_val):
    return (tensor - min_val) / (max_val - min_val)

# Calculate global minima and maxima
data_directory = 'data_processed'
input_min, input_max, output_min, output_max = compute_global_min_max(data_directory)

def custom_collate(batch):
    graph_data_list = [item[0] for item in batch]
    outputs = torch.stack([item[1] for item in batch])
    return graph_data_list, outputs

class CustomTensorDataset(Dataset):
    def __init__(self, directory, train=True, validation_split=0.2, random_seed=42):
        self.file_paths = [os.path.join(directory, fname) for fname in os.listdir(directory) if fname.endswith('_2.pt')]
        train_files, val_files = train_test_split(self.file_paths, test_size=validation_split, random_state=random_seed)
        self.data = [torch.load(fp) for fp in (train_files if train else val_files)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tensor_dict = self.data[idx]
        x = tensor_dict['input']
        y = tensor_dict['output']
        
        # Normalize inputs and outputs
        x = normalize(x, input_min, input_max)
        y = normalize(y, output_min, output_max)

        num_nodes = x.size(0)
        if num_nodes == 2:
            edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        elif num_nodes == 3:
            edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long)
        else:
            #edge_index = torch.empty((2, 0), dtype=torch.long)  # No edges for 1 node
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)


        graph_data = Data(x=x, edge_index=edge_index)
        return graph_data, y

train_dataset = CustomTensorDataset(data_directory, train=True)
val_dataset = CustomTensorDataset(data_directory, train=False)

batch_size = 128
shuffle = True
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate, num_workers=4)

'''
class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc_gnn = nn.Linear(hidden_dim, hidden_dim)
        self.fc_guidance = nn.Linear(1, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim - 1)

    def forward(self, data, guidance):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = torch.mean(x, dim=0)
        x_gnn = self.fc_gnn(x)
        guidance = guidance.view(-1, 1)
        x_guidance = F.relu(self.fc_guidance(guidance))
        x_combined = x_gnn + x_guidance
        x_out = self.fc_out(x_combined)
        return x_out

class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.att_conv1 = GATConv(input_dim, hidden_dim, heads=4, concat=True)
        self.att_conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True)  # 4 heads in each layer
        self.fc_gnn = nn.Linear(hidden_dim * 4, hidden_dim)  # Adjusted input dimension
        self.fc_guidance = nn.Linear(1, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim - 1)

    def forward(self, data, guidance):
        x, edge_index = data.x, data.edge_index
        x = self.att_conv1(x, edge_index)
        x = F.elu(x)
        x = self.att_conv2(x, edge_index)
        x = F.elu(x)
        x = torch.mean(x, dim=0)  # Aggregate node features
        x_gnn = self.fc_gnn(x)
        guidance = guidance.view(-1, 1)
        x_guidance = F.relu(self.fc_guidance(guidance))
        x_combined = x_gnn + x_guidance
        x_out = self.fc_out(x_combined)
        return x_out
'''   
class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4):
        super(GNNModel, self).__init__()
        self.att_conv1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True)
        self.att_conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True)
        self.fc_gnn = nn.Linear(hidden_dim * heads, hidden_dim)  # Adjusted input dimension
        self.fc_guidance = nn.Linear(1, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim - 1)  # Assuming output_dim-1 is intentional

    def forward(self, data, guidance):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.att_conv1(x, edge_index)
        x = F.elu(x)
        x = self.att_conv2(x, edge_index)
        x = F.elu(x)
        x = global_mean_pool(x, batch)  # Aggregate node features for each graph in the batch
        x_gnn = self.fc_gnn(x)
        guidance = guidance.view(-1, 1)
        x_guidance = F.relu(self.fc_guidance(guidance))
        x_combined = x_gnn + x_guidance
        x_out = self.fc_out(x_combined)
        return x_out
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_dim = train_dataset[0][0].x.size(1)
hidden_dim = 16
output_dim = train_dataset[0][1].size(0)

model = GNNModel(input_dim, hidden_dim, output_dim).to(device)

criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

num_epochs = 500
train_losses = []
val_losses = []
best_val_loss = float('inf')
best_model_path = 'best_model_multi.pth'
early_stopping_patience = 25
early_stopping_counter = 0
'''
'''
for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0
    for batch in train_loader:
        graph_data_list, outputs = batch
        for i, graph_data in enumerate(graph_data_list):
            inputs = graph_data.to(device)
            guidance_vector = outputs[i][0].view(1).to(device)
            targets = outputs[i][1:].to(device)
            outputs_pred = model(inputs, guidance_vector)
            outputs_pred = outputs_pred.view(-1)
            targets = targets.view(-1)
            loss = criterion(outputs_pred, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss}')

    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            graph_data_list, outputs = batch
            for i, graph_data in enumerate(graph_data_list):
                inputs = graph_data.to(device)
                guidance_vector = outputs[i][0].view(1).to(device)
                targets = outputs[i][1:].to(device)
                outputs_pred = model(inputs, guidance_vector)
                outputs_pred = outputs_pred.view(-1)
                targets = targets.view(-1)
                loss = criterion(outputs_pred, targets)
                epoch_val_loss += loss.item()

    avg_val_loss = epoch_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f'Epoch {epoch+1}, Validation Loss: {avg_val_loss}')

    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f'Saving best model with validation loss: {best_val_loss}')
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1

    if early_stopping_counter >= early_stopping_patience:
        print('Early stopping triggered')
        break