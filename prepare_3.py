import os
import glob
import pandas as pd
import torch

# Define the function to process each CSV file
def process_csv(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Extract input (first 3 columns) and output (4th column)
    input_data = df.iloc[:, :3].values
    output_data = df.iloc[:, 3].values
    
    # Convert to PyTorch tensors
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    output_tensor = torch.tensor(output_data, dtype=torch.float32)
    
    # Create a tensor dictionary
    tensor_dict = {
        'input': input_tensor,
        'output': output_tensor
    }
    
    # Determine the new file name
    base_name = os.path.basename(file_path)
    name, ext = os.path.splitext(base_name)
    new_file_name = f"{name}_3.pt"
    
    # Save the tensor dictionary
    output_file_path = os.path.join(os.path.dirname(file_path), new_file_name)
    torch.save(tensor_dict, output_file_path)

    print(f"Processed {file_path} -> {output_file_path}")

# Define the directory containing the CSV files
directory = 'data_processed/'  # Change this to your directory

# Find all CSV files in the directory
csv_files = glob.glob(os.path.join(directory, '*.csv'))

# Process each CSV file
for csv_file in csv_files:
    process_csv(csv_file)
