import re
import pandas as pd
import os
import json

rehydratation_mapping = dict(
    zip(["1", "2", "3", "4", "5-100", "100-1000", "1000+"], [1, 2, 3, 3, 5, 8, 1])
)

def catch_name_and_cluster_size(name):
    # The regex pattern
    pattern = r"(fineweb2_.*)_cluster_(.*)"
    # Perform the search
    match = re.search(pattern, name)
    # Check if a match was found
    if match:
        return match.group(1), match.group(2)
    else:
        return name, None

def apply_rehydratation(df, column_name='tokens'):
    # Apply the mapping to create the new columns
    df[['name_pro', 'cluster_size']] = df['name'].apply(lambda x: pd.Series(catch_name_and_cluster_size(x)))
    df['rehydratation_weight'] = df.apply(lambda x: rehydratation_mapping.get(x['cluster_size'], 1), axis=1)
    df[column_name + '_rehydrated'] = df[column_name] * df['rehydratation_weight'] 
    return df

def read_stats(data_path):
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
    return df
