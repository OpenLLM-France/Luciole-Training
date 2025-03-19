import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import os

path = f"{os.getenv('DATA')}/fineweb2/logs/fra_Latn/clusters"
os.makedirs(f"{path}/fig", exist_ok=True)

with open(f"{path}/stats/merged_stats.json", "r", encoding="utf-8") as f:
    data = json.load(f)

out = []
for k, v in data[-1]['stats'].items():
    if 'cluster' in k:
        out.append({
            'cluster_size': re.search(r"cluster_size:(.*)/", k).group(1),
            'documents': v['total']
        })

order = ["1", "2", "3", "4", "5-100", "100-1000", "1000+"]
weights = [1, 2, 3, 3, 5, 8, 1]

df = pd.DataFrame(out)
df['cluster_size'] = pd.Categorical(df['cluster_size'], categories=order, ordered=True)
df = df.sort_values('cluster_size')
df['weights'] = weights

total_docs = df['documents'].sum()
total_docs_weighted = (df['documents'] * df['weights']).sum()
print(f"Number of documents: {total_docs:,}")
print(f'Number of documents after rehydratation: {total_docs_weighted:,}')
print(f'Ratio: x {total_docs_weighted / total_docs:.2f}')

# Plotting 
plt.figure(figsize=(10, 5))
plt.bar(df['cluster_size'], df['documents'])
plt.yscale("log")  # Set y-axis to logarithmic scale
plt.xlabel("Cluster Size")
plt.ylabel("Number of Documents")
plt.title("Number of Documents per Cluster Size")
plt.savefig(f"{path}/fig/cluster_size_documents.png", dpi=300)

plt.figure(figsize=(10, 5))
plt.bar(df['cluster_size'], df['weights'])
plt.xlabel("Cluster Size")
plt.ylabel("Upsampling Weight")
plt.title("Upsampling Weight per Cluster Size")
plt.savefig(f"{path}/fig/upsampling_weight_documents.png", dpi=300)
