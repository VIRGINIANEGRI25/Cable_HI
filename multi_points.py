from preprocessing import load_data, scale_data, HI_computation, visual_condition_inversion, one_hot_encoding
from model import Cnn, Convlstm, evaluate_model
from dataloader import train_test_datasets, train_test_dataloaders
import torch
from torch import nn
import matplotlib.pyplot as plt

# Check for CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Load data
data = load_data('XLPE_data.xlsx')
#print(data)
data = visual_condition_inversion(data)
scaled_data = scale_data(data)
data = HI_computation(data, scaled_data)
unique_ids = data['ID'].unique()

for unique_id in unique_ids:
    # Filter the DataFrame for the specific ID
    filtered_df = data[data['ID'] == unique_id]
    filtered_df = filtered_df.drop(columns=['ID'])
    # Save the filtered DataFrame to a CSV file named by the ID
    filtered_df.to_csv(f'data_processed/{unique_id}.csv', index=False)
    print(f'DataFrame for ID {unique_id} saved successfully.')