import re
import pandas as pd
import os
import argparse
from collections import OrderedDict

rehydratation_mapping = OrderedDict(
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

def apply_rehydratation(df):
    # Apply the mapping to create the new columns
    df[['dataset', 'cluster_size']] = df['name'].apply(lambda x: pd.Series(catch_name_and_cluster_size(x)))
    df['rehydratation_weight'] = df.apply(lambda x: rehydratation_mapping.get(x['cluster_size'], 1), axis=1)
    df['total_tokens_rehydrated'] = df['total_tokens'] * df['rehydratation_weight'] 

    df["cluster_size"] = pd.Categorical(
        df["cluster_size"], categories=list(rehydratation_mapping.keys()), ordered=True
    )
    df = df.sort_values('cluster_size')
    return df

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    main_path = os.getenv("OpenLLM_OUTPUT")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", 
        default=f"{main_path}/data/tokens_ablation/stats/all_stats_merged.csv"
        )
    parser.add_argument(
        "--output_path", 
        default=f"{main_path}/data/tokens_ablation/figs"
        )
    parser.add_argument('--multiplier', type=float, default=20, help="Multiplier for total tokens")
    args = parser.parse_args()
    data_path = args.data_path
    output_path = args.output_path

    os.makedirs(output_path, exist_ok=True)

    df = pd.read_csv(data_path)
    df['total_tokens'] *= args.multiplier
    df = apply_rehydratation(df)
    
    for dataset, group in df.groupby('dataset'):
        if "fineweb2" in dataset:
            before = group['total_tokens'].sum()/ 1e9
            after = group['total_tokens_rehydrated'].sum()/ 1e9
            plt.figure(figsize=(12, 6))
            group.plot.bar(x="cluster_size", y=['total_tokens', 'total_tokens_rehydrated'], rot=0)
            plt.yscale('log')
            plt.title(f"Tokens per cluster size in {dataset}\n Total tokens: from {before:.1f} B to {after:.1f} B (x{after/before:.1f})")
            plt.savefig(os.path.join(output_path, f"rehydration_{dataset}.png"))