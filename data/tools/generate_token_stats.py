import json
import os
import numpy as np
import re
import glob
import pandas as pd
import argparse
from collections import OrderedDict

def extract_token_lengths(data_path, name):
    from nemo.collections.nlp.data.language_modeling.megatron import indexed_dataset
    suffix = "_text_document"
    dataset = indexed_dataset.MMapIndexedDataset(os.path.join(data_path, name + suffix))
    return [len(data) for data in dataset]

def stats_summary(token_lengths):
    stats = {
        "total_tokens": np.sum(token_lengths).item(),
        "total_sequences": len(token_lengths),
        "min_tokens": min(token_lengths),
        "max_tokens": max(token_lengths),
        "mean_tokens": np.mean(token_lengths),
        "Q1_tokens": np.percentile(token_lengths, 25),
        "Q2_tokens": np.median(token_lengths),
        "Q3_tokens": np.percentile(token_lengths, 75),
    }
    return stats

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

def merge_stats(data_path):
    # Get only JSON filenames
    json_files = [file for file in os.listdir(data_path) if file.lower().endswith('.json')]

    data_list = []
    for json_file in json_files:
        with open(os.path.join(data_path, json_file), 'r') as f:
            stats = json.load(f)
            stats = {'name': os.path.splitext(json_file)[0], **stats}
            data_list.append(stats)

    df = pd.DataFrame(data_list)
    return df


if __name__ == "__main__":
    main_path = os.getenv("OpenLLM_OUTPUT")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", 
        default=f"{main_path}/data/tokens_ablation"
        )
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Force re-generation of stats even if they already exist."
        )
    args = parser.parse_args()
    data_path = args.data_path

    # Look for all files matching the pattern *_text_document.idx
    files = glob.glob(os.path.join(data_path, "*_text_document.idx"))
    names = [re.match(r"(.*?)_text_document\.idx", os.path.basename(f)).group(1) for f in files]

    for name in names:
        stats_path = os.path.join(data_path, 'stats', f"{name}.json")

        if not os.path.exists(stats_path) or args.force:
            print(f"Extracting and saving stats for {name}...")
            token_lengths = extract_token_lengths(data_path, name)
            os.makedirs(os.path.join(data_path, 'stats'), exist_ok=True)
            stats = stats_summary(token_lengths)
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=4)
        else:
            print(f"Stats for {name} already exist, skipping.")

    print("All stats saved.")

    # Merge all stats into a single CSV file
    merged_df = merge_stats(os.path.join(data_path, 'stats'))
    merged_df = apply_rehydratation(merged_df)
    merged_df = merged_df.sort_values(by="name")

    output_csv_path = os.path.join(data_path, "stats/all_stats_merged.csv")
    merged_df.to_csv(output_csv_path, index=False)
    print(f"Merged stats saved to {output_csv_path}")
