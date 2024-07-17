import os
import glob
import pandas as pd
import torch

# Define the function to process each CSV file
def process_csv(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Extract input-output sets
    input1 = df.iloc[:, 0].values  # 1st column as input
    output1 = df.iloc[:, 1].values  # 2nd column as output
    
    input2 = df.iloc[:, 0].values  # 1st column as input
    output2 = df.iloc[:, 2].values  # 3rd column as output
    
    input3 = df.iloc[:, 0].values  # 1st column as input
    output3 = df.iloc[:, 3].values  # 4th column as output
    
    input4 = df.iloc[:, 1].values  # 2nd column as input
    output4 = df.iloc[:, 2].values  # 3rd column as output
    
    input5 = df.iloc[:, 1].values  # 2nd column as input
    output5 = df.iloc[:, 3].values  # 4th column as output
    
    input6 = df.iloc[:, 2].values  # 3rd column as input
    output6 = df.iloc[:, 3].values  # 4th column as output
    
    # Convert to PyTorch tensors
    input1_tensor = torch.tensor(input1, dtype=torch.float32)
    output1_tensor = torch.tensor(output1, dtype=torch.float32)
    
    input2_tensor = torch.tensor(input2, dtype=torch.float32)
    output2_tensor = torch.tensor(output2, dtype=torch.float32)
    
    input3_tensor = torch.tensor(input3, dtype=torch.float32)
    output3_tensor = torch.tensor(output3, dtype=torch.float32)
    
    input4_tensor = torch.tensor(input4, dtype=torch.float32)
    output4_tensor = torch.tensor(output4, dtype=torch.float32)
    
    input5_tensor = torch.tensor(input5, dtype=torch.float32)
    output5_tensor = torch.tensor(output5, dtype=torch.float32)
    
    input6_tensor = torch.tensor(input6, dtype=torch.float32)
    output6_tensor = torch.tensor(output6, dtype=torch.float32)
    
    # Create tensor dictionaries
    tensor_dicts = [
        {'input': input1_tensor, 'output': output1_tensor},
        {'input': input2_tensor, 'output': output2_tensor},
        {'input': input3_tensor, 'output': output3_tensor},
        {'input': input4_tensor, 'output': output4_tensor},
        {'input': input5_tensor, 'output': output5_tensor},
        {'input': input6_tensor, 'output': output6_tensor}
    ]
    
    # Determine the base file name
    base_name = os.path.basename(file_path)
    name, ext = os.path.splitext(base_name)
    
    # Save each tensor dictionary
    for i, tensor_dict in enumerate(tensor_dicts, start=1):
        output_file_path = os.path.join(os.path.dirname(file_path), f"{name}_v{i}_1.pt")
        torch.save(tensor_dict, output_file_path)
        print(f"Processed {file_path} -> {output_file_path}")

# Define the directory containing the CSV files
directory = 'data_processed/'  # Change this to your directory

# Find all CSV files in the directory
csv_files = glob.glob(os.path.join(directory, '*.csv'))

# Process each CSV file
for csv_file in csv_files:
    process_csv(csv_file)
