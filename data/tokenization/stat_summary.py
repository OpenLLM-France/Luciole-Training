import os
import json
import pandas as pd
import glob

# Specify the directory path where your JSON files are located
data_path = "/lustre/fsn1/projects/rech/qgz/commun/OpenLLM-BPI-output/data/tokens_ablation/stats"

# Get only JSON filenames (not full paths)
json_files = [file for file in os.listdir(data_path) if file.lower().endswith('.json')]

# List to store data
data_list = []

# Process each JSON file
for json_file in json_files:
    json_file_path = os.path.join(data_path, json_file)
    
    with open(json_file_path, 'r') as f:
        data = json.load(f)  # Load JSON data
        
        # Ensure 'file_name' is the first column by creating an ordered dictionary
        data = {'name': os.path.splitext(json_file)[0], **data}
        
        data_list.append(data)

# Create a DataFrame from the list of dictionaries
df = pd.DataFrame(data_list)
df = df.sort_values('name')

# Save the DataFrame to a CSV file
df.to_csv(os.path.join(data_path, 'output.csv'), index=False)

print("CSV file 'output.csv' created successfully.")
