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
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
import numpy as np
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

# Denormalize function
def denormalize(tensor, min_val, max_val):
    return tensor * (max_val - min_val) + min_val

# Calculate global minima and maxima
data_directory = 'test_set'
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

val_dataset = CustomTensorDataset(data_directory, train=False)

batch_size = 1
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

input_dim = val_dataset[0][0].x.size(1)
hidden_dim = 16
output_dim = val_dataset[0][1].size(0)

model = GNNModel(input_dim, hidden_dim, output_dim).to(device)

criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)


train_losses = []
val_losses = []
best_val_loss = float('inf')
best_model_path = 'best_model_multi.pth'

best_model = GNNModel(input_dim, hidden_dim, output_dim).to(device)
best_model.load_state_dict(torch.load(best_model_path))


best_model.eval()
predictions_test = []
gt_values_test = []
test_losses = []

with torch.no_grad():
    for batch in val_loader:
        graph_data_list, outputs = batch
        for i, graph_data in enumerate(graph_data_list):
            inputs = graph_data.to(device)
            guidance_vector = outputs[i][0].view(1).to(device)
            targets = outputs[i][1:].to(device)
            outputs_pred = best_model(inputs, guidance_vector)
            predictions_test.append(outputs_pred.cpu().numpy().flatten())
            gt_values_test.append(targets.cpu().numpy().flatten())
            loss = criterion(outputs_pred.view(-1), targets.view(-1))
            test_losses.append(loss.item())

avg_test_loss = sum(test_losses) / len(test_losses)
print(f'Avg Test Loss: {avg_test_loss}')

predictions_test_flat = np.concatenate(predictions_test, axis=0)
gt_values_test_flat = np.concatenate(gt_values_test, axis=0)


predictions_test_flat_denorm = denormalize(torch.tensor(predictions_test_flat), output_min, output_max).numpy()
gt_values_test_flat_denorm = denormalize(torch.tensor(gt_values_test_flat), output_min, output_max).numpy()



r2_HI = r2_score(gt_values_test_flat_denorm[4:], predictions_test_flat_denorm[4:])
print(f'R2 score for HI: {r2_HI}')
r2_measurement = r2_score(gt_values_test_flat_denorm[:4], predictions_test_flat_denorm[:4])
print(f'R2 score for measurements: {r2_measurement}')
mape_HI = mean_absolute_percentage_error(gt_values_test_flat_denorm[4:], predictions_test_flat_denorm[4:])
print(f'MAPE for the HI: {mape_HI}')
mape_measurement = mean_absolute_percentage_error(gt_values_test_flat_denorm[:4], predictions_test_flat_denorm[:4])
print(f'MAPE score for measurements: {mape_measurement}')
rmse_HI = np.sqrt(mean_squared_error(gt_values_test_flat_denorm[:4], predictions_test_flat_denorm[:4]))
print(f'RMSE for the HI: {rmse_HI}')
rmse_measurement = np.sqrt(mean_squared_error(gt_values_test_flat_denorm[4:], predictions_test_flat_denorm[4:]))
print(f'RMSE score for measurements: {rmse_measurement}')
