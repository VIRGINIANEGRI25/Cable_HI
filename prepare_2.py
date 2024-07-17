import os
import glob
import pandas as pd
import torch

# Define the function to process each CSV file
def process_csv(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Extract input-output sets
    input1 = df.iloc[:, :2].values  # First 2 columns as input
    output1 = df.iloc[:, 2].values  # 3rd column as output
    
    input2 = df.iloc[:, :2].values  # First 2 columns as input
    output2 = df.iloc[:, 3].values  # 4th column as output
    
    input3 = df.iloc[:, 1:3].values  # Second 2 columns as input
    output3 = df.iloc[:, 3].values  # 4th column as output
    
    # Convert to PyTorch tensors
    input1_tensor = torch.tensor(input1, dtype=torch.float32)
    output1_tensor = torch.tensor(output1, dtype=torch.float32)
    
    input2_tensor = torch.tensor(input2, dtype=torch.float32)
    output2_tensor = torch.tensor(output2, dtype=torch.float32)
    
    input3_tensor = torch.tensor(input3, dtype=torch.float32)
    output3_tensor = torch.tensor(output3, dtype=torch.float32)
    
    # Create tensor dictionaries
    tensor_dict1 = {'input': input1_tensor, 'output': output1_tensor}
    tensor_dict2 = {'input': input2_tensor, 'output': output2_tensor}
    tensor_dict3 = {'input': input3_tensor, 'output': output3_tensor}
    
    # Determine the base file name
    base_name = os.path.basename(file_path)
    name, ext = os.path.splitext(base_name)
    
    # Save each tensor dictionary
    output_file_path1 = os.path.join(os.path.dirname(file_path), f"{name}_v1_2.pt")
    output_file_path2 = os.path.join(os.path.dirname(file_path), f"{name}_v2_2.pt")
    output_file_path3 = os.path.join(os.path.dirname(file_path), f"{name}_v3_2.pt")
    
    torch.save(tensor_dict1, output_file_path1)
    torch.save(tensor_dict2, output_file_path2)
    torch.save(tensor_dict3, output_file_path3)
    
    print(f"Processed {file_path} -> {output_file_path1}, {output_file_path2}, {output_file_path3}")

# Define the directory containing the CSV files
directory = 'data_processed/'  # Change this to your directory

# Find all CSV files in the directory
csv_files = glob.glob(os.path.join(directory, '*.csv'))

# Process each CSV file
for csv_file in csv_files:
    process_csv(csv_file)
