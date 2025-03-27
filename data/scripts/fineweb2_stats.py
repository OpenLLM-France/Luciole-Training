import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps

MAIN_PATH = os.getenv('OpenLLM_OUTPUT')
os.makedirs(f"{MAIN_PATH}/fig", exist_ok=True)
os.makedirs(f"{MAIN_PATH}/stats", exist_ok=True)

def read_data(data_path):
    with open(f"{data_path}/stats/merged_stats.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    out = []
    for k, v in data[-1]['stats'].items():
        if 'cluster' in k:
            out.append({
                'cluster_size': re.search(r"cluster_size:(.*)/", k).group(1),
                'documents': v['total']
            })
    return pd.DataFrame(out)

def order(df):
    order = ["1", "2", "3", "4", "5-100", "100-1000", "1000+"]
    df['cluster_size'] = pd.Categorical(df['cluster_size'], categories=order, ordered=True)
    df = df.sort_values('cluster_size')
    return df

if __name__ == "__main__":
    languages = ['fra_Latn', 'spa_Latn', 'ita_Latn', 'deu_Latn']

    colormap = colormaps['tab20']  # Using tab20 colormap
    mapping = dict(zip(
        ["1", "2", "3", "4", "5-100", "100-1000", "1000+"],
        [1, 2, 3, 3, 5, 8, 1]
    ))
 
    all_data = []
    for language in languages:
        data_path = f"{MAIN_PATH}/datasets/fineweb2/logs/{language}/clusters"
        df = read_data(data_path)
        df['language'] = language
        all_data.append(df)

    df = pd.concat(all_data)
    df = order(df)
    df['weight'] = df['cluster_size'].map(mapping)
    df['upsampled_doc'] = df['weight'] * df['documents']
    df.to_csv(f'{MAIN_PATH}/stats/fineweb2.csv')

    # Plot 1: Total number of documents by language
    plt.figure(figsize=(12, 6))
    df_grouped = df.groupby('language').agg({'documents': 'sum', 'upsampled_doc': 'sum'})
    ax = df_grouped.plot.bar(rot=0)
    plt.yscale("log")

    # Add values on top of each bar in scientific notation with 2 decimal places
    for container in ax.containers:
        ax.bar_label(container, labels=[f'{v.get_height():.2e}' for v in container], label_type='edge', fontsize=10, color='black')
    plt.savefig(f"{MAIN_PATH}/fig/fineweb2_total_num_documents.png")

    # Plot 2: Number of documents by cluster size and language
    plt.figure(figsize=(12, 6))
    df_pivot = df.pivot(index='cluster_size', columns='language', values='documents')
    ax = df_pivot.plot.bar(rot=0)
    plt.yscale("log")
    plt.savefig(f"{MAIN_PATH}/fig/fineweb2_num_documents_by_cluster.png")
