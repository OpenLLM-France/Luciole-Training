import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import squarify
import seaborn as sb

# Specify the directory path where your JSON files are located
data_path = os.path.join(os.getenv("OpenLLM_OUTPUT"), "data/tokens_ablation/stats")

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

###
rehydratation_mapping = dict(
    zip(["1", "2", "3", "4", "5-100", "100-1000", "1000+"], [1, 2, 3, 3, 5, 8, 1])
)

import re
def catch_cluster_size(name):
    # The regex pattern
    pattern = r"fineweb2_.*atn_cluster_(.*)"
    # Perform the search
    match = re.search(pattern, name)
    # Check if a match was found
    if match:
        return match.group(1)
    else:
        return None

df['rehydratation_weight'] = df.apply(lambda x: rehydratation_mapping.get(catch_cluster_size(x['name']), 1), axis=1)
df['total_tokens'] = df['total_tokens'] * df['rehydratation_weight'] * 20

patterns = [
    'fineweb2_fra',
    'fineweb2_ita',
    'fineweb2_deu',
    'fineweb2_spa',
    'wik',
    'gallica'
]
    
# Create a new column to store extracted values
df['group'] = df['name'].str.extract(f"({'|'.join(patterns)})")
df['group'] = df['group'].fillna(df['name'])

grouped_df = df.groupby('group')['total_tokens'].sum().reset_index()
# grouped_df = grouped_df.sort_values('total_tokens', ascending=True)

labels = [f"{group}\n{tokens/ 1e9:.1f} B" for group, tokens in zip(grouped_df['group'], grouped_df['total_tokens'])]

# Treemap
plt.figure(figsize=(10, 6))
squarify.plot(sizes = grouped_df['total_tokens'], label = labels,
              pad = 0.05, alpha=0.7,
              text_kwargs = {'fontsize': 6, 'color': 'black'},
              color = sb.color_palette("rocket", len(grouped_df)))

# Remove the axis:
plt.axis("off")
plt.title(f"Tokens per dataset (estimation, with rehydratation)\n Total tokens: {df['total_tokens'].sum()/ 1e9:.1f} B")
plt.savefig(os.path.join(os.getenv("OpenLLM_OUTPUT"), "data/tokens_ablation/figs/treemap.png"), dpi=300, bbox_inches='tight')
# plt.show() 
