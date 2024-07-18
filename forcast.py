import os
import glob

# Define the directory containing the .pt files
directory = 'data_processed/'  # Change this to your directory

# Find all .pt files in the directory
pt_files = glob.glob(os.path.join(directory, '*.pt'))

# Delete each .pt file
for pt_file in pt_files:
    os.remove(pt_file)
    print(f"Deleted {pt_file}")

print("All .pt files have been deleted.")
